# main.py
"""
Main FastAPI application for Space Detective v2.0
Integrates all modules: Database Manager, RAG System, and Chat API
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

# Import our modules
from config import AppConfig
from database_manager import DatabaseManager, DatabaseMigration, create_database_manager
from rag_system import RAGSystem, create_rag_system, migrate_documents_from_files
from chat_api import ChatAPI, create_chat_api, sanitize_message, validate_h3_index

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global configuration
config = AppConfig()
config.app_start_time = datetime.now()

# Global components
db_manager: Optional[DatabaseManager] = None
rag_system: Optional[RAGSystem] = None
chat_api: Optional[ChatAPI] = None


# Pydantic Models
class DatabaseInitRequest(BaseModel):
    """Request model for database initialization"""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    database: str = Field(default="space_detective", description="Database name")
    username: str = Field(default="postgres", description="Database username")
    password: str = Field(default="password", description="Database password")

class ModelLoadRequest(BaseModel):
    """Request model for loading model from database"""
    model_name: str = Field(default="isolation_forest_v1", description="Model name to load")
    dataset_name: str = Field(default="main_dataset", description="Dataset name to load")
    alibaba_api_key: Optional[str] = Field(default=None, description="Alibaba API key for enhanced chat")

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., description="User message", max_length=5000)
    session_id: str = Field(default="default", description="Chat session ID")
    use_rag: bool = Field(default=True, description="Use RAG for enhanced responses")
    max_context_docs: int = Field(default=3, description="Maximum context documents from RAG")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload"""
    title: str = Field(..., description="Document title", max_length=500)
    content: str = Field(..., description="Document content", max_length=1000000)
    document_type: str = Field(default="general", description="Document type")
    author: Optional[str] = Field(default=None, description="Document author")
    source_url: Optional[str] = Field(default=None, description="Source URL")
    tags: Optional[List[str]] = Field(default=None, description="Document tags")

class RAGSearchRequest(BaseModel):
    """Request model for RAG search"""
    query: str = Field(..., description="Search query", max_length=1000)
    max_results: int = Field(default=5, description="Maximum search results")
    similarity_threshold: float = Field(default=0.7, description="Similarity threshold")
    document_types: Optional[List[str]] = Field(default=None, description="Filter by document types")

class LocationAnalysisRequest(BaseModel):
    """Request model for location analysis"""
    h3_indexes: List[str] = Field(..., description="List of H3 indexes to analyze")
    session_id: str = Field(default="default", description="Chat session ID")

class MigrationRequest(BaseModel):
    """Request model for data migration"""
    csv_file_path: Optional[str] = Field(default=None, description="CSV file path for migration")
    model_file_path: Optional[str] = Field(default=None, description="Model file path for migration")
    model_name: str = Field(default="isolation_forest_v1", description="Model name")
    dataset_name: str = Field(default="main_dataset", description="Dataset name")


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global db_manager, rag_system, chat_api
    
    try:
        logger.info("🚀 Starting Space Detective v2.0...")
        
        # Initialize components will happen via API calls
        # This allows for flexible configuration
        
        logger.info("✅ Space Detective v2.0 startup completed")
        yield
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Cleanup
        logger.info("🛑 Shutting down Space Detective v2.0...")
        
        if chat_api:
            logger.info("Cleaning up chat sessions...")
        
        if rag_system:
            logger.info("Closing RAG system...")
            await rag_system.close()
        
        if db_manager:
            logger.info("Closing database connections...")
            await db_manager.close()
        
        logger.info("✅ Shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Space Detective v2.0 API",
    description="AI-Driven Satellite Intelligence for Combating Money Laundering with PostgreSQL & RAG",
    version="2.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure as needed for production
)


# Dependency functions
async def get_db_manager() -> DatabaseManager:
    """Get database manager dependency"""
    if not db_manager or not db_manager.is_connected:
        raise HTTPException(status_code=503, detail="Database not connected. Please initialize database first.")
    return db_manager

async def get_rag_system() -> RAGSystem:
    """Get RAG system dependency"""
    if not rag_system or not rag_system.initialized:
        raise HTTPException(status_code=503, detail="RAG system not initialized. Please initialize system first.")
    return rag_system

async def get_chat_api() -> ChatAPI:
    """Get chat API dependency"""
    if not chat_api:
        raise HTTPException(status_code=503, detail="Chat API not available. Please initialize system first.")
    return chat_api


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with web interface"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Space Detective v2.0</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .feature { margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .endpoint { background: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 3px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .status.ok { background: #d4edda; color: #155724; }
        .status.warning { background: #fff3cd; color: #856404; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛰️ Space Detective v2.0</h1>
            <p>AI-Driven Satellite Intelligence with PostgreSQL & RAG</p>
        </div>
        
        <div class="status ok">
            <strong>✅ API Server Running</strong><br>
            Version: 2.0.0 | Start Time: """ + config.app_start_time.strftime("%Y-%m-%d %H:%M:%S") + """
        </div>
        
        <div class="feature">
            <h3>🔧 System Initialization</h3>
            <div class="endpoint">POST /api/initialize - Initialize database and RAG system</div>
            <div class="endpoint">GET /api/status - Check system status</div>
        </div>
        
        <div class="feature">
            <h3>🗄️ Database Operations</h3>
            <div class="endpoint">POST /api/database/load-model - Load ML model from database</div>
            <div class="endpoint">GET /api/database/models - List all models</div>
            <div class="endpoint">GET /api/database/datasets - List all datasets</div>
            <div class="endpoint">GET /api/database/statistics - Get system statistics</div>
        </div>
        
        <div class="feature">
            <h3>🤖 Chat Interface</h3>
            <div class="endpoint">POST /api/chat - AI-powered chat with RAG</div>
            <div class="endpoint">POST /api/chat/analyze-location - Location-specific analysis</div>
            <div class="endpoint">POST /api/chat/compare-locations - Compare multiple locations</div>
        </div>
        
        <div class="feature">
            <h3>📚 RAG Knowledge Base</h3>
            <div class="endpoint">POST /api/rag/upload-document - Upload document</div>
            <div class="endpoint">POST /api/rag/search - Search documents</div>
            <div class="endpoint">GET /api/rag/documents - List all documents</div>
        </div>
        
        <div class="feature">
            <h3>📊 Analysis & Investigation</h3>
            <div class="endpoint">GET /api/anomaly-statistics - Comprehensive anomaly statistics</div>
            <div class="endpoint">POST /api/shap-analysis/{h3_index} - SHAP explainability</div>
            <div class="endpoint">GET /api/investigation-history - Investigation logs</div>
        </div>
        
        <div class="feature">
            <h3>📖 Documentation</h3>
            <p><a href="/docs" target="_blank">📋 Interactive API Documentation (Swagger)</a></p>
            <p><a href="/redoc" target="_blank">📚 Alternative Documentation (ReDoc)</a></p>
        </div>
    </div>
</body>
</html>
    """


# System Management Endpoints
@app.post("/api/initialize")
async def initialize_system(request: DatabaseInitRequest):
    """Initialize database and RAG system"""
    global db_manager, rag_system, chat_api
    
    try:
        logger.info("Initializing Space Detective v2.0 system...")
        
        # Update database config
        config.database.host = request.host
        config.database.port = request.port
        config.database.database = request.database
        config.database.username = request.username
        config.database.password = request.password
        
        # Initialize database manager
        logger.info("Initializing database manager...")
        db_manager = await create_database_manager(config.database)
        
        # Initialize RAG system
        logger.info("Initializing RAG system...")
        rag_system = await create_rag_system(config.rag, config.database)
        
        # Initialize chat API
        logger.info("Initializing chat API...")
        chat_api = await create_chat_api(db_manager, rag_system, config)
        
        logger.info("✅ System initialization completed successfully")
        
        return {
            "status": "success",
            "message": "Space Detective v2.0 initialized successfully",
            "components": {
                "database": db_manager.is_connected,
                "rag_system": rag_system.initialized,
                "chat_api": chat_api is not None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise HTTPException(status_code=500, detail=f"System initialization failed: {str(e)}")


@app.get("/api/status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        db_status = db_manager.is_connected if db_manager else False
        rag_status = rag_system.initialized if rag_system else False
        chat_status = chat_api is not None
        
        # Get detailed statistics if available
        db_stats = {}
        rag_stats = {}
        chat_stats = {}
        
        if db_manager and db_status:
            try:
                db_stats = await db_manager.get_system_stats()
            except:
                pass
        
        if rag_system and rag_status:
            try:
                rag_stats = await rag_system.get_statistics()
            except:
                pass
        
        if chat_api:
            try:
                chat_stats = chat_api.get_system_statistics()
            except:
                pass
        
        return {
            "status": "operational" if all([db_status, rag_status, chat_status]) else "partial",
            "components": {
                "database": {
                    "connected": db_status,
                    "statistics": db_stats
                },
                "rag_system": {
                    "initialized": rag_status,
                    "statistics": rag_stats
                },
                "chat_api": {
                    "available": chat_status,
                    "statistics": chat_stats
                }
            },
            "config": {
                "version": "2.0.0",
                "features_enabled": {
                    "rag": config.enable_rag,
                    "shap": config.enable_shap,
                    "caching": config.enable_caching,
                    "auth": config.enable_auth
                }
            },
            "uptime": str(datetime.now() - config.app_start_time),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "status": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Database Management Endpoints
@app.post("/api/database/load-model")
async def load_model_from_database(
    request: ModelLoadRequest,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Load ML model and data from database"""
    try:
        # Load model
        model, model_info = await db.load_model(request.model_name)
        logger.info(f"Model '{request.model_name}' loaded successfully")
        
        # Load training data
        training_data, dataset_info = await db.load_training_data(request.dataset_name)
        logger.info(f"Dataset '{request.dataset_name}' loaded successfully")
        
        # Load SHAP explainer if available
        explainer, explainer_info = await db.load_shap_explainer(f"{request.model_name}_explainer")
        
        # Configure external API if provided
        if request.alibaba_api_key:
            config.external_apis.alibaba_api_key = request.alibaba_api_key
        
        # Calculate statistics
        anomaly_count = len(training_data[training_data['anomaly_score'] == -1]) if 'anomaly_score' in training_data.columns else 0
        total_count = len(training_data)
        
        return {
            "status": "success",
            "message": "Model and data loaded successfully from database",
            "model_info": {
                "model_name": request.model_name,
                "model_type": model_info.get('model_type'),
                "version": model_info.get('version'),
                "loaded": True,
                "shap_available": explainer is not None
            },
            "dataset_info": {
                "dataset_name": request.dataset_name,
                "total_records": total_count,
                "anomaly_count": anomaly_count,
                "anomaly_percentage": (anomaly_count / total_count * 100) if total_count > 0 else 0,
                "feature_columns": dataset_info.get('feature_columns', []),
                "provinces": list(training_data['nmprov'].unique()) if 'nmprov' in training_data.columns else []
            },
            "external_api": {
                "alibaba_configured": bool(request.alibaba_api_key)
            }
        }
        
    except Exception as e:
        logger.error(f"Error loading model from database: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.get("/api/database/models")
async def list_database_models(db: DatabaseManager = Depends(get_db_manager)):
    """List all models in database"""
    try:
        models = await db.list_models(active_only=True)
        return {
            "models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")


@app.get("/api/database/datasets")
async def list_database_datasets(db: DatabaseManager = Depends(get_db_manager)):
    """List all datasets in database"""
    try:
        datasets = await db.list_datasets(active_only=True)
        return {
            "datasets": datasets,
            "count": len(datasets),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@app.get("/api/database/statistics")
async def get_database_statistics(db: DatabaseManager = Depends(get_db_manager)):
    """Get comprehensive database statistics"""
    try:
        stats = await db.get_system_stats()
        return {
            "database_statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


# Chat API Endpoints
@app.post("/api/chat")
async def chat_with_ai(
    request: ChatRequest,
    chat: ChatAPI = Depends(get_chat_api)
):
    """AI-powered chat interface with RAG support"""
    try:
        # Sanitize message
        clean_message = sanitize_message(request.message)
        if not clean_message:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process message
        response = await chat.process_message(
            message=clean_message,
            session_id=request.session_id,
            use_rag=request.use_rag,
            max_context_docs=request.max_context_docs
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


@app.post("/api/chat/analyze-location")
async def analyze_location(
    request: LocationAnalysisRequest,
    chat: ChatAPI = Depends(get_chat_api)
):
    """Analyze specific locations"""
    try:
        # Validate H3 indexes
        valid_indexes = [h3 for h3 in request.h3_indexes if validate_h3_index(h3)]
        if not valid_indexes:
            raise HTTPException(status_code=400, detail="No valid H3 indexes provided")
        
        if len(valid_indexes) == 1:
            response = await chat.analyze_location(valid_indexes[0], request.session_id)
        else:
            response = await chat.compare_locations(valid_indexes, request.session_id)
        
        return response
        
    except Exception as e:
        logger.error(f"Location analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Location analysis failed: {str(e)}")


@app.get("/api/chat/sessions/{session_id}")
async def get_chat_session(
    session_id: str,
    chat: ChatAPI = Depends(get_chat_api)
):
    """Get chat session information"""
    try:
        session_info = await chat.get_session_info(session_id)
        return session_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")


@app.delete("/api/chat/sessions/{session_id}")
async def clear_chat_session(
    session_id: str,
    chat: ChatAPI = Depends(get_chat_api)
):
    """Clear specific chat session"""
    try:
        success = await chat.clear_session(session_id)
        return {
            "success": success,
            "message": f"Session {session_id} cleared" if success else "Session not found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")


@app.get("/api/chat/sessions/{session_id}/history")
async def get_conversation_history(
    session_id: str,
    limit: int = 20,
    chat: ChatAPI = Depends(get_chat_api)
):
    """Get conversation history"""
    try:
        history = await chat.get_conversation_history(session_id, limit)
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")


@app.get("/api/chat/sessions/{session_id}/export", response_class=PlainTextResponse)
async def export_conversation(
    session_id: str,
    chat: ChatAPI = Depends(get_chat_api)
):
    """Export conversation as text file"""
    try:
        export_text = await chat.export_conversation(session_id)
        if not export_text:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return PlainTextResponse(
            content=export_text,
            headers={"Content-Disposition": f"attachment; filename=conversation_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting conversation: {str(e)}")


# RAG System Endpoints
@app.post("/api/rag/upload-document")
async def upload_document_to_rag(
    request: DocumentUploadRequest,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Upload document to RAG knowledge base"""
    try:
        result = await rag.add_document(
            title=request.title,
            content=request.content,
            document_type=request.document_type,
            author=request.author,
            source_url=request.source_url,
            tags=request.tags
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@app.post("/api/rag/upload-file")
async def upload_file_to_rag(
    file: UploadFile = File(...),
    document_type: str = "general",
    author: Optional[str] = None,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Upload file to RAG knowledge base"""
    try:
        # Read file content
        content = await file.read()
        
        # Handle different file types
        if file.filename.endswith(('.txt', '.md')):
            text_content = content.decode('utf-8')
        elif file.filename.endswith('.json'):
            json_data = json.loads(content.decode('utf-8'))
            text_content = json.dumps(json_data, indent=2)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use .txt, .md, or .json files.")
        
        # Store in RAG system
        result = await rag.add_document(
            title=file.filename,
            content=text_content,
            document_type=document_type,
            author=author,
            additional_metadata={
                "file_size": len(content),
                "original_filename": file.filename,
                "upload_date": datetime.now().isoformat()
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"File upload error: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.post("/api/rag/search")
async def search_rag_documents(
    request: RAGSearchRequest,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Search RAG knowledge base"""
    try:
        results = await rag.search_documents(
            query=request.query,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            document_types=request.document_types
        )
        
        return {
            "query": request.query,
            "results_count": len(results),
            "results": [
                {
                    "id": result.id,
                    "title": result.title,
                    "content": result.content,
                    "document_type": result.document_type,
                    "similarity": result.similarity,
                    "metadata": result.metadata,
                    "created_at": result.created_at
                }
                for result in results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"RAG search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/api/rag/documents")
async def list_rag_documents(
    document_type: Optional[str] = None,
    limit: int = 100,
    rag: RAGSystem = Depends(get_rag_system)
):
    """List documents in RAG knowledge base"""
    try:
        documents = await rag.list_documents(document_type, limit)
        return {
            "documents": documents,
            "count": len(documents),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")


@app.get("/api/rag/documents/{doc_id}")
async def get_rag_document(
    doc_id: int,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Get specific document from RAG knowledge base"""
    try:
        document = await rag.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "document": {
                "id": document.id,
                "title": document.title,
                "content": document.content,
                "document_type": document.document_type,
                "metadata": document.metadata,
                "created_at": document.created_at
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting document: {str(e)}")


@app.delete("/api/rag/documents/{doc_id}")
async def delete_rag_document(
    doc_id: int,
    delete_all_chunks: bool = True,
    rag: RAGSystem = Depends(get_rag_system)
):
    """Delete document from RAG knowledge base"""
    try:
        success = await rag.delete_document(doc_id, delete_all_chunks)
        return {
            "success": success,
            "message": f"Document {doc_id} deleted" if success else "Document not found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")


@app.get("/api/rag/statistics")
async def get_rag_statistics(rag: RAGSystem = Depends(get_rag_system)):
    """Get RAG system statistics"""
    try:
        stats = await rag.get_statistics()
        return {
            "rag_statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting RAG statistics: {str(e)}")


# Analysis and Investigation Endpoints
@app.get("/api/anomaly-statistics")
async def get_anomaly_statistics(
    dataset_name: str = "main_dataset",
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get comprehensive anomaly statistics"""
    try:
        from database_manager import get_anomaly_statistics
        
        stats = await get_anomaly_statistics(db, dataset_name)
        return {
            "anomaly_statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting anomaly statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")


@app.post("/api/shap-analysis/{h3_index}")
async def get_shap_analysis(
    h3_index: str,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get SHAP analysis for specific location"""
    try:
        # Validate H3 index
        if not validate_h3_index(h3_index):
            raise HTTPException(status_code=400, detail="Invalid H3 index format")
        
        # Load model and data
        model, model_info = await db.load_model("isolation_forest_v1")
        training_data, dataset_info = await db.load_training_data("main_dataset")
        explainer, explainer_info = await db.load_shap_explainer("isolation_forest_v1_explainer")
        
        if not explainer:
            raise HTTPException(status_code=404, detail="SHAP explainer not available")
        
        # Find location data
        location_data = training_data[training_data['h3_index'] == h3_index]
        if location_data.empty:
            raise HTTPException(status_code=404, detail=f"Location {h3_index} not found")
        
        row = location_data.iloc[0]
        
        # Get feature columns
        feature_columns = [col for col in training_data.columns 
                          if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
        
        if not feature_columns:
            raise HTTPException(status_code=400, detail="No feature columns found")
        
        # Prepare features for SHAP
        import pandas as pd
        features = row[feature_columns].fillna(0).values.reshape(1, -1)
        
        # Get SHAP values
        shap_values = explainer.shap_values(features)
        
        # Prepare response
        shap_data = []
        for i, feature in enumerate(feature_columns):
            shap_data.append({
                "feature": feature,
                "value": float(row[feature]) if not pd.isna(row[feature]) else 0,
                "shap_value": float(shap_values[0][i]),
                "impact": "increases anomaly likelihood" if shap_values[0][i] > 0 else "decreases anomaly likelihood",
                "abs_importance": abs(float(shap_values[0][i]))
            })
        
        # Sort by absolute SHAP value
        shap_data.sort(key=lambda x: x['abs_importance'], reverse=True)
        
        return {
            "h3_index": h3_index,
            "location": f"{row.get('nmdesa', 'Unknown')}, {row.get('nmkab', 'Unknown')}, {row.get('nmprov', 'Unknown')}",
            "is_anomaly": row.get('anomaly_score', 1) == -1,
            "shap_analysis": shap_data,
            "base_value": float(explainer.expected_value),
            "prediction_explanation": "SHAP values show how each feature contributed to the anomaly prediction. Positive values push toward anomaly, negative values toward normal.",
            "top_factors": shap_data[:5]  # Top 5 most important factors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
        raise HTTPException(status_code=500, detail=f"SHAP analysis failed: {str(e)}")


@app.get("/api/investigation-history")
async def get_investigation_history(
    h3_index: Optional[str] = None,
    investigator: Optional[str] = None,
    limit: int = 100,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Get investigation history"""
    try:
        investigations = await db.get_investigation_history(h3_index, investigator, limit)
        return {
            "investigations": investigations,
            "count": len(investigations),
            "filters": {
                "h3_index": h3_index,
                "investigator": investigator,
                "limit": limit
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting investigation history: {str(e)}")


@app.post("/api/investigation-log")
async def log_investigation(
    h3_index: str,
    investigator: str,
    investigation_type: str = "anomaly_analysis",
    findings: Optional[str] = None,
    status: str = "open",
    priority_level: str = "medium",
    confidence_score: Optional[float] = None,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Log investigation activity"""
    try:
        if not validate_h3_index(h3_index):
            raise HTTPException(status_code=400, detail="Invalid H3 index format")
        
        log_id = await db.log_investigation(
            h3_index=h3_index,
            investigator=investigator,
            investigation_type=investigation_type,
            findings=findings,
            status=status,
            priority_level=priority_level,
            confidence_score=confidence_score,
            metadata={"logged_via": "api", "timestamp": datetime.now().isoformat()}
        )
        
        return {
            "success": True,
            "log_id": log_id,
            "message": f"Investigation logged for H3 {h3_index}"
        }
        
    except Exception as e:
        logger.error(f"Error logging investigation: {e}")
        raise HTTPException(status_code=500, detail=f"Investigation logging failed: {str(e)}")


# Data Migration Endpoints
@app.post("/api/migrate")
async def migrate_data(
    request: MigrationRequest,
    background_tasks: BackgroundTasks,
    db: DatabaseManager = Depends(get_db_manager)
):
    """Migrate data from files to database"""
    try:
        migration = DatabaseMigration(db)
        results = {"migrations": []}
        
        # Migrate CSV data if provided
        if request.csv_file_path:
            background_tasks.add_task(
                _migrate_csv_background,
                migration,
                request.csv_file_path,
                request.dataset_name
            )
            results["migrations"].append({
                "type": "csv_data",
                "file": request.csv_file_path,
                "status": "started"
            })
        
        # Migrate model if provided
        if request.model_file_path:
            background_tasks.add_task(
                _migrate_model_background,
                migration,
                request.model_file_path,
                request.model_name
            )
            results["migrations"].append({
                "type": "ml_model",
                "file": request.model_file_path,
                "status": "started"
            })
        
        if not results["migrations"]:
            raise HTTPException(status_code=400, detail="No migration files specified")
        
        return {
            "message": "Migration tasks started",
            "results": results,
            "note": "Check logs for migration progress",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Migration error: {e}")
        raise HTTPException(status_code=500, detail=f"Migration failed: {str(e)}")


# Background task functions
async def _migrate_csv_background(migration: DatabaseMigration, csv_path: str, dataset_name: str):
    """Background task for CSV migration"""
    try:
        success = await migration.migrate_from_csv(csv_path, dataset_name)
        if success:
            logger.info(f"✅ CSV migration completed: {csv_path}")
        else:
            logger.error(f"❌ CSV migration failed: {csv_path}")
    except Exception as e:
        logger.error(f"❌ CSV migration error: {e}")


async def _migrate_model_background(migration: DatabaseMigration, model_path: str, model_name: str):
    """Background task for model migration"""
    try:
        success = await migration.migrate_model(model_path, model_name)
        if success:
            logger.info(f"✅ Model migration completed: {model_path}")
        else:
            logger.error(f"❌ Model migration failed: {model_path}")
    except Exception as e:
        logger.error(f"❌ Model migration error: {e}")


# Bulk Operations Endpoints
@app.post("/api/rag/bulk-upload")
async def bulk_upload_documents(
    documents: List[DocumentUploadRequest],
    rag: RAGSystem = Depends(get_rag_system)
):
    """Bulk upload documents to RAG system"""
    try:
        if len(documents) > 50:  # Limit bulk uploads
            raise HTTPException(status_code=400, detail="Maximum 50 documents per bulk upload")
        
        # Convert to format expected by RAG system
        doc_list = []
        for doc in documents:
            doc_list.append({
                "title": doc.title,
                "content": doc.content,
                "document_type": doc.document_type,
                "author": doc.author,
                "source_url": doc.source_url,
                "tags": doc.tags,
                "metadata": {"bulk_upload": True, "upload_date": datetime.now().isoformat()}
            })
        
        result = await rag.bulk_add_documents(doc_list)
        return result
        
    except Exception as e:
        logger.error(f"Bulk upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk upload failed: {str(e)}")


# Health Check Endpoints
@app.get("/api/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/api/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status"""
    try:
        checks = {
            "api": True,
            "database": False,
            "rag_system": False,
            "chat_api": False
        }
        
        # Check database
        if db_manager:
            checks["database"] = await db_manager.health_check()
        
        # Check RAG system
        if rag_system:
            checks["rag_system"] = rag_system.initialized
        
        # Check chat API
        if chat_api:
            checks["chat_api"] = True
        
        overall_healthy = all(checks.values())
        
        return {
            "status": "healthy" if overall_healthy else "degraded",
            "components": checks,
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0"
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.logging.level.lower()
    )