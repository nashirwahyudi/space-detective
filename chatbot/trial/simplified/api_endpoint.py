# api_endpoints.py
"""
FastAPI endpoints for Space Detective Chat API
RESTful API to serve chat responses in JSON format for frontend consumption
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our chat API
from chat_api import ChatAPI, validate_h3_index, extract_h3_indexes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global chat API instance
chat_api_instance: Optional[ChatAPI] = None


# Pydantic models for request/response
class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str = Field(..., description="User message", min_length=1, max_length=5000)
    session_id: str = Field(default="default", description="Chat session ID")
    include_analysis: bool = Field(default=False, description="Include intent analysis in response")

class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    response: str = Field(..., description="AI assistant response")
    timestamp: str = Field(..., description="Response timestamp")
    session_id: str = Field(..., description="Chat session ID")
    data_loaded: bool = Field(..., description="Whether training data is loaded")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Message analysis (if requested)")
    error: bool = Field(default=False, description="Whether an error occurred")

class LocationAnalysisRequest(BaseModel):
    """Request model for location analysis"""
    h3_index: str = Field(..., description="H3 index to analyze", min_length=15, max_length=15)
    include_shap: bool = Field(default=False, description="Include SHAP analysis")

class LocationAnalysisResponse(BaseModel):
    """Response model for location analysis"""
    success: bool = Field(..., description="Whether analysis was successful")
    h3_index: str = Field(..., description="Analyzed H3 index")
    location_info: Optional[Dict[str, Any]] = Field(None, description="Location information")
    analysis: Optional[Dict[str, Any]] = Field(None, description="Analysis results")
    shap_analysis: Optional[Dict[str, Any]] = Field(None, description="SHAP analysis (if requested)")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class SystemInfoResponse(BaseModel):
    """Response model for system information"""
    system_status: Dict[str, bool] = Field(..., description="System component status")
    data_info: Dict[str, Any] = Field(..., description="Training data information")
    conversation_stats: Dict[str, int] = Field(..., description="Conversation statistics")
    capabilities: List[str] = Field(..., description="System capabilities")
    uptime: str = Field(..., description="System uptime")

class ShapAnalysisRequest(BaseModel):
    """Request model for SHAP analysis"""
    h3_index: str = Field(..., description="H3 index for SHAP analysis")
    top_features: int = Field(default=10, description="Number of top features to return", ge=1, le=20)

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Health check timestamp")
    components: Dict[str, bool] = Field(..., description="Component health status")


# App lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global chat_api_instance
    
    # Startup
    logger.info("🚀 Starting Space Detective API...")
    
    # Configuration - Edit these paths for your files
    config = {
        "csv_path": os.getenv("CSV_PATH", "sample_data.csv"),
        "model_path": os.getenv("MODEL_PATH", "model.pkl"),
        "explainer_path": os.getenv("EXPLAINER_PATH", "explainer.pkl"),
        "alibaba_api_key": os.getenv("ALIBABA_API_KEY", 'sk-0d3d6181c41a40559b9e014269aa15c3')
    }
    
    # Initialize chat API
    chat_api_instance = ChatAPI(
        csv_path=config["csv_path"],
        model_path=config["model_path"],
        explainer_path=config["explainer_path"],
        alibaba_api_key=config["alibaba_api_key"]
    )
    
    # Load data
    if chat_api_instance.initialize():
        logger.info("✅ Chat API initialized successfully")
    else:
        logger.error("❌ Chat API initialization failed")
        logger.error(f"Check file paths:")
        logger.error(f"  CSV: {config['csv_path']}")
        logger.error(f"  Model: {config['model_path']}")
        logger.error(f"  Explainer: {config['explainer_path']}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Space Detective API...")


# Create FastAPI app
app = FastAPI(
    title="Space Detective Chat API",
    description="RESTful API for AI-powered money laundering detection chat",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store startup time
startup_time = datetime.now()


# Helper function to ensure chat API is ready
def get_chat_api() -> ChatAPI:
    """Get chat API instance, raise HTTPException if not ready"""
    if chat_api_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat API not initialized"
        )
    
    if not chat_api_instance.data_loader.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Training data not loaded. Check file paths and restart the service."
        )
    
    return chat_api_instance


# API Endpoints

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🛰️ Space Detective Chat API v2.0",
        "description": "AI-powered money laundering detection through satellite intelligence",
        "status": "operational",
        "endpoints": {
            "chat": "/api/chat",
            "location_analysis": "/api/analyze-location",
            "shap_analysis": "/api/shap-analysis",
            "system_info": "/api/system-info",
            "health": "/api/health"
        },
        "documentation": "/docs",
        "version": "2.0.0"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for AI conversation"""
    try:
        chat_api = get_chat_api()
        
        # Process chat message
        response_data = await chat_api.chat(request.message, request.session_id)
        
        # Format response
        chat_response = ChatResponse(
            response=response_data["response"],
            timestamp=response_data["timestamp"],
            session_id=request.session_id,
            data_loaded=response_data.get("data_loaded", False),
            error=response_data.get("error", False)
        )
        
        # Include analysis if requested
        if request.include_analysis and "analysis" in response_data:
            chat_response.analysis = response_data["analysis"]
        
        return chat_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        return ChatResponse(
            response=f"🚨 I encountered an error: {str(e)}",
            timestamp=datetime.now().isoformat(),
            session_id=request.session_id,
            data_loaded=False,
            error=True
        )


@app.post("/api/analyze-location", response_model=LocationAnalysisResponse)
async def analyze_location_endpoint(request: LocationAnalysisRequest):
    """Analyze specific location by H3 index"""
    try:
        # Validate H3 index
        if not validate_h3_index(request.h3_index):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid H3 index format. Must be 15-character hexadecimal string starting with '8'."
            )
        
        chat_api = get_chat_api()
        
        # Get location analysis
        analysis_result = chat_api.analyze_location(request.h3_index)
        
        if analysis_result.get("error"):
            return LocationAnalysisResponse(
                success=False,
                h3_index=request.h3_index,
                error_message=analysis_result["message"]
            )
        
        response = LocationAnalysisResponse(
            success=True,
            h3_index=request.h3_index,
            location_info=analysis_result["location_info"],
            analysis=analysis_result["analysis"]
        )
        
        # Include SHAP analysis if requested
        if request.include_shap:
            shap_result = chat_api.get_shap_analysis(request.h3_index)
            if shap_result.get("success"):
                response.shap_analysis = shap_result
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Location analysis error: {e}")
        return LocationAnalysisResponse(
            success=False,
            h3_index=request.h3_index,
            error_message=str(e)
        )


@app.post("/api/shap-analysis", response_model=Dict[str, Any])
async def shap_analysis_endpoint(request: ShapAnalysisRequest):
    """Get SHAP analysis for specific location"""
    try:
        # Validate H3 index
        if not validate_h3_index(request.h3_index):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid H3 index format"
            )
        
        chat_api = get_chat_api()
        
        # Get SHAP analysis
        shap_result = chat_api.get_shap_analysis(request.h3_index)
        
        if shap_result.get("error"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=shap_result["message"]
            )
        
        # Limit features if requested
        if "shap_analysis" in shap_result:
            shap_result["shap_analysis"] = shap_result["shap_analysis"][:request.top_features]
            shap_result["top_factors"] = shap_result["shap_analysis"][:5]
        
        return shap_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SHAP analysis failed: {str(e)}"
        )


@app.get("/api/system-info", response_model=SystemInfoResponse)
async def system_info_endpoint():
    """Get comprehensive system information"""
    try:
        if chat_api_instance is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Chat API not initialized"
            )
        
        system_info = chat_api_instance.get_system_info()
        uptime = str(datetime.now() - startup_time)
        
        return SystemInfoResponse(
            system_status=system_info["system_status"],
            data_info=system_info["data_info"],
            conversation_stats=system_info["conversation_stats"],
            capabilities=system_info["capabilities"],
            uptime=uptime
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"System info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get system info: {str(e)}"
        )


@app.get("/api/anomaly-statistics")
async def anomaly_statistics_endpoint():
    """Get anomaly statistics from training data"""
    try:
        chat_api = get_chat_api()
        
        stats = chat_api.data_loader.get_anomaly_statistics()
        
        if not stats:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No anomaly statistics available"
            )
        
        return {
            "success": True,
            "statistics": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anomaly statistics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get anomaly statistics: {str(e)}"
        )


@app.get("/api/conversation-history")
async def conversation_history_endpoint(
    session_id: Optional[str] = None,
    limit: int = 20
):
    """Get conversation history"""
    try:
        chat_api = get_chat_api()
        
        history = chat_api.get_conversation_history(session_id, limit)
        
        return {
            "success": True,
            "session_id": session_id,
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation history error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get conversation history: {str(e)}"
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    components = {
        "chat_api": chat_api_instance is not None,
        "data_loaded": chat_api_instance.data_loader.loaded if chat_api_instance else False,
        "model_available": chat_api_instance.data_loader.model is not None if chat_api_instance else False,
        "explainer_available": chat_api_instance.data_loader.explainer is not None if chat_api_instance else False
    }
    
    overall_health = all(components.values())
    
    return HealthResponse(
        status="healthy" if overall_health else "degraded",
        timestamp=datetime.now().isoformat(),
        components=components
    )


@app.get("/api/sample-h3-indexes")
async def sample_h3_indexes_endpoint(limit: int = 10):
    """Get sample H3 indexes from training data"""
    try:
        chat_api = get_chat_api()
        
        df = chat_api.data_loader.training_data
        
        if df is None or df.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No training data available"
            )
        
        # Get sample of H3 indexes with info
        sample_data = df.head(limit)[['h3_index', 'nmprov', 'nmkab', 'nmdesa', 'anomaly_score', 'built_growth_pct_22_24']].to_dict('records')
        
        return {
            "success": True,
            "sample_h3_indexes": sample_data,
            "total_available": len(df),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sample H3 indexes error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sample H3 indexes: {str(e)}"
        )


@app.post("/api/batch-analyze")
async def batch_analyze_endpoint(h3_indexes: List[str], include_shap: bool = False):
    """Analyze multiple locations in batch"""
    try:
        if len(h3_indexes) > 20:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 20 H3 indexes per batch request"
            )
        
        # Validate all H3 indexes
        invalid_indexes = [h3 for h3 in h3_indexes if not validate_h3_index(h3)]
        if invalid_indexes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid H3 indexes: {invalid_indexes}"
            )
        
        chat_api = get_chat_api()
        
        results = []
        for h3_index in h3_indexes:
            try:
                analysis_result = chat_api.analyze_location(h3_index)
                
                result = {
                    "h3_index": h3_index,
                    "success": not analysis_result.get("error", False),
                    "analysis": analysis_result
                }
                
                # Add SHAP if requested and successful
                if include_shap and not analysis_result.get("error"):
                    shap_result = chat_api.get_shap_analysis(h3_index)
                    result["shap_analysis"] = shap_result
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    "h3_index": h3_index,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "batch_results": results,
            "total_processed": len(results),
            "successful": len([r for r in results if r["success"]]),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analyze error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch analysis failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested endpoint was not found",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Configuration and startup
def create_app_config():
    """Create app configuration file"""
    config_content = '''# app_config.py
"""
Configuration for Space Detective API
Edit these settings for your deployment
"""

import os

# File paths for your data
CSV_PATH = os.getenv("CSV_PATH", "sample_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
EXPLAINER_PATH = os.getenv("EXPLAINER_PATH", "explainer.pkl")

# Optional: Alibaba API key for enhanced responses
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", None)

# API server settings
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# CORS settings (configure for production)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
'''
    
    with open("app_config.py", "w") as f:
        f.write(config_content)
    
    print("📄 Created app_config.py - Edit file paths there")


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Space Detective Chat API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--create-config", action="store_true", help="Create configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_app_config()
        print("✅ Configuration file created. Edit app_config.py with your file paths.")
        exit(0)
    
    print("🛰️ Starting Space Detective Chat API Server")
    print(f"📡 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"🔧 Health Check: http://{args.host}:{args.port}/api/health")
    print(f"💬 Chat: POST http://{args.host}:{args.port}/api/chat")
    print(f"🛑 Stop with Ctrl+C")
    
    # Run server
    uvicorn.run(
        "api_endpoints:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )