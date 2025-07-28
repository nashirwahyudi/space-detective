# rag_system.py
"""
RAG (Retrieval-Augmented Generation) System for Space Detective v2.0
Handles document storage, vector embeddings, and similarity search using pgvector
"""

import asyncio
import asyncpg
import numpy as np
import hashlib
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
import tiktoken
from dataclasses import dataclass
import re
from collections import Counter

from config import RAGConfig, DatabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    content: str
    chunk_index: int
    total_chunks: int
    title: str
    document_type: str
    metadata: Dict[str, Any]


@dataclass
class SearchResult:
    """Search result with similarity score"""
    id: int
    title: str
    content: str
    document_type: str
    similarity: float
    metadata: Dict[str, Any]
    created_at: str


class TextProcessor:
    """Advanced text processing utilities"""
    
    def __init__(self, max_chunk_size: int = 500, chunk_overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = None
        
        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoder: {e}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\/\@\#\$\%]', '', text)
        
        # Fix common issues
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.strip()
        
        return text
    
    def chunk_text(self, text: str, preserve_sentences: bool = True) -> List[str]:
        """Split text into chunks with intelligent boundaries"""
        if not text:
            return []
        
        text = self.clean_text(text)
        
        if self.tokenizer:
            return self._chunk_with_tokenizer(text)
        else:
            return self._chunk_by_words(text, preserve_sentences)
    
    def _chunk_with_tokenizer(self, text: str) -> List[str]:
        """Chunk text using tokenizer for precise token counting"""
        try:
            tokens = self.tokenizer.encode(text)
            chunks = []
            
            for i in range(0, len(tokens), self.max_chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i:i + self.max_chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens)
                chunks.append(chunk_text.strip())
            
            return chunks
        except Exception as e:
            logger.warning(f"Tokenizer chunking failed: {e}, falling back to word-based chunking")
            return self._chunk_by_words(text)
    
    def _chunk_by_words(self, text: str, preserve_sentences: bool = True) -> List[str]:
        """Chunk text by words with sentence preservation"""
        if preserve_sentences:
            # Split by sentences first
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = [text]
        
        chunks = []
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            words = sentence.split()
            sentence_word_count = len(words)
            
            # If adding this sentence would exceed chunk size
            if current_word_count + sentence_word_count > self.max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_words = current_chunk.split()[-self.chunk_overlap:] if self.chunk_overlap > 0 else []
                current_chunk = " ".join(overlap_words + words)
                current_word_count = len(overlap_words) + sentence_word_count
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_word_count += sentence_word_count
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Extract metadata from text content"""
        metadata = {}
        
        # Basic statistics
        metadata['word_count'] = len(text.split())
        metadata['char_count'] = len(text)
        metadata['paragraph_count'] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Extract keywords
        metadata['keywords'] = self.extract_keywords(text)
        
        # Detect language (simple heuristic)
        metadata['language'] = self.detect_language(text)
        
        # Content type detection
        metadata['content_type'] = self.detect_content_type(text)
        
        return metadata
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        # Convert to lowercase and extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'what', 'when',
            'where', 'why', 'how', 'who', 'which', 'said', 'than', 'them',
            'other', 'time', 'very', 'her', 'now', 'find', 'long', 'here',
            'see', 'him', 'two', 'more', 'go', 'no', 'way', 'she', 'many',
            'its', 'make', 'get', 'use', 'her', 'our', 'out', 'day', 'year'
        }
        
        # Filter words
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Count frequency and return most common
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(max_keywords)]
    
    def detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Very basic heuristic - in production, use a proper language detection library
        indonesian_words = ['dan', 'atau', 'yang', 'untuk', 'dengan', 'pada', 'dalam', 'adalah', 'akan', 'telah']
        english_words = ['and', 'or', 'the', 'for', 'with', 'on', 'in', 'is', 'will', 'have']
        
        text_lower = text.lower()
        indonesian_count = sum(1 for word in indonesian_words if word in text_lower)
        english_count = sum(1 for word in english_words if word in text_lower)
        
        if indonesian_count > english_count:
            return 'id'
        else:
            return 'en'
    
    def detect_content_type(self, text: str) -> str:
        """Detect content type based on text patterns"""
        text_lower = text.lower()
        
        # Check for investigation-related content
        investigation_keywords = ['investigation', 'investigasi', 'money laundering', 'suspicious', 'anomaly', 'evidence']
        if any(keyword in text_lower for keyword in investigation_keywords):
            return 'investigation_guide'
        
        # Check for technical content
        technical_keywords = ['algorithm', 'model', 'api', 'database', 'shap', 'h3', 'technical']
        if any(keyword in text_lower for keyword in technical_keywords):
            return 'technical_guide'
        
        # Check for procedural content
        procedural_keywords = ['procedure', 'step', 'process', 'checklist', 'guideline']
        if any(keyword in text_lower for keyword in procedural_keywords):
            return 'procedure'
        
        # Check for regulatory content
        regulatory_keywords = ['regulation', 'law', 'compliance', 'legal', 'policy']
        if any(keyword in text_lower for keyword in regulatory_keywords):
            return 'regulatory'
        
        return 'general'


class EmbeddingManager:
    """Manages text embeddings using Sentence Transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embedding_dim = None
        
    def initialize(self) -> bool:
        """Initialize embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"Embedding model '{self.model_name}' initialized (dim: {self.embedding_dim})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            return False
    
    def encode(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            # Clean text before encoding
            text = text.strip()
            if not text:
                return np.zeros(self.embedding_dim)
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.zeros(self.embedding_dim)
    
    def encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts"""
        if not self.model:
            raise RuntimeError("Embedding model not initialized")
        
        try:
            # Clean texts
            cleaned_texts = [text.strip() if text.strip() else "empty" for text in texts]
            
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            return [emb.astype(np.float32) for emb in embeddings]
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            return [np.zeros(self.embedding_dim) for _ in texts]


class RAGDatabase:
    """Database operations for RAG system"""
    
    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.pool = None
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.db_config.host,
                port=self.db_config.port,
                database=self.db_config.database,
                user=self.db_config.username,
                password=self.db_config.password,
                min_size=self.db_config.min_connections,
                max_size=self.db_config.max_connections
            )
            
            await self.create_tables()
            logger.info("RAG database initialized")
            return True
            
        except Exception as e:
            logger.error(f"RAG database initialization failed: {e}")
            return False
    
    async def create_tables(self):
        """Create RAG-specific tables"""
        async with self.pool.acquire() as conn:
            # Enable pgvector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # RAG Documents table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id SERIAL PRIMARY KEY,
                    document_uuid UUID DEFAULT uuid_generate_v4(),
                    title VARCHAR(500) NOT NULL,
                    content TEXT NOT NULL,
                    content_hash VARCHAR(64) UNIQUE,
                    document_type VARCHAR(100) DEFAULT 'general',
                    source_url VARCHAR(1000),
                    author VARCHAR(255),
                    embedding vector(384),
                    chunk_index INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 1,
                    parent_document_id INTEGER,
                    language VARCHAR(10) DEFAULT 'en',
                    tags TEXT[],
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    access_level VARCHAR(50) DEFAULT 'public',
                    word_count INTEGER,
                    char_count INTEGER
                );
            """)
            
            # Create vector index for similarity search
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding 
                ON rag_documents USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """)
            
            # Create other indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_type 
                ON rag_documents(document_type);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_created_at 
                ON rag_documents(created_at DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_tags 
                ON rag_documents USING gin(tags);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_content_hash 
                ON rag_documents(content_hash);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rag_documents_parent 
                ON rag_documents(parent_document_id);
            """)
    
    async def store_document_chunks(self, chunks: List[DocumentChunk], 
                                  embeddings: List[np.ndarray]) -> List[int]:
        """Store document chunks with embeddings"""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        document_ids = []
        parent_doc_id = None
        
        async with self.pool.acquire() as conn:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Generate content hash
                content_hash = hashlib.sha256(
                    f"{chunk.title}_{chunk.content}_{chunk.chunk_index}".encode()
                ).hexdigest()
                
                # Check if chunk already exists
                existing = await conn.fetchrow(
                    "SELECT id FROM rag_documents WHERE content_hash = $1",
                    content_hash
                )
                
                if existing:
                    document_ids.append(existing['id'])
                    if i == 0:
                        parent_doc_id = existing['id']
                    continue
                
                # Insert chunk
                doc_id = await conn.fetchval("""
                    INSERT INTO rag_documents 
                    (title, content, content_hash, document_type, embedding, 
                     chunk_index, total_chunks, parent_document_id, language,
                     tags, metadata, word_count, char_count)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                    RETURNING id
                """, 
                chunk.title, chunk.content, content_hash, chunk.document_type,
                embedding.tolist(), chunk.chunk_index, chunk.total_chunks, parent_doc_id,
                chunk.metadata.get('language', 'en'),
                chunk.metadata.get('keywords', []),
                json.dumps(chunk.metadata),
                chunk.metadata.get('word_count', 0),
                chunk.metadata.get('char_count', 0)
                )
                
                document_ids.append(doc_id)
                
                # Set parent document ID for first chunk
                if i == 0:
                    parent_doc_id = doc_id
                    await conn.execute(
                        "UPDATE rag_documents SET parent_document_id = $1 WHERE id = $1",
                        doc_id
                    )
        
        return document_ids
    
    async def search_similar_documents(self, query_embedding: np.ndarray, 
                                     max_results: int = 5,
                                     similarity_threshold: float = 0.7,
                                     document_types: List[str] = None,
                                     language: str = None) -> List[SearchResult]:
        """Search for similar documents using vector similarity"""
        try:
            async with self.pool.acquire() as conn:
                # Build query conditions
                conditions = ["1 - (embedding <=> $1) > $2"]
                params = [query_embedding.tolist(), similarity_threshold]
                param_count = 2
                
                if document_types:
                    param_count += 1
                    conditions.append(f"document_type = ANY(${param_count})")
                    params.append(document_types)
                
                if language:
                    param_count += 1
                    conditions.append(f"language = ${param_count}")
                    params.append(language)
                
                # Add active condition
                conditions.append("is_active = true")
                
                # Construct final query
                where_clause = " AND ".join(conditions)
                query = f"""
                    SELECT 
                        id, title, content, document_type, metadata, created_at, language, tags,
                        1 - (embedding <=> $1) as similarity
                    FROM rag_documents
                    WHERE {where_clause}
                    ORDER BY embedding <=> $1
                    LIMIT ${param_count + 1}
                """
                params.append(max_results)
                
                results = await conn.fetch(query, *params)
                
                # Format results
                search_results = []
                for row in results:
                    search_results.append(SearchResult(
                        id=row['id'],
                        title=row['title'],
                        content=row['content'],
                        document_type=row['document_type'],
                        similarity=float(row['similarity']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else {},
                        created_at=row['created_at'].isoformat()
                    ))
                
                return search_results
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_document_by_id(self, doc_id: int) -> Optional[SearchResult]:
        """Get document by ID"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT id, title, content, document_type, metadata, created_at
                    FROM rag_documents
                    WHERE id = $1 AND is_active = true
                """, doc_id)
                
                if result:
                    return SearchResult(
                        id=result['id'],
                        title=result['title'],
                        content=result['content'],
                        document_type=result['document_type'],
                        similarity=1.0,
                        metadata=json.loads(result['metadata']) if result['metadata'] else {},
                        created_at=result['created_at'].isoformat()
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting document by ID: {e}")
            return None
    
    async def list_documents(self, document_type: str = None, 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """List documents with metadata"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, title, document_type, chunk_index, total_chunks,
                           parent_document_id, language, created_at, word_count, char_count,
                           CASE WHEN parent_document_id IS NULL THEN true ELSE false END as is_parent
                    FROM rag_documents
                    WHERE is_active = true
                """
                params = []
                
                if document_type:
                    query += " AND document_type = $1"
                    params.append(document_type)
                
                query += " ORDER BY created_at DESC, chunk_index ASC LIMIT $" + str(len(params) + 1)
                params.append(limit)
                
                results = await conn.fetch(query, *params)
                
                documents = []
                for row in results:
                    documents.append({
                        "id": row['id'],
                        "title": row['title'],
                        "document_type": row['document_type'],
                        "chunk_index": row['chunk_index'],
                        "total_chunks": row['total_chunks'],
                        "parent_document_id": row['parent_document_id'],
                        "is_parent": row['is_parent'],
                        "language": row['language'],
                        "word_count": row['word_count'],
                        "char_count": row['char_count'],
                        "created_at": row['created_at'].isoformat()
                    })
                
                return documents
                
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    async def delete_document(self, doc_id: int, delete_all_chunks: bool = True) -> bool:
        """Delete document(s)"""
        try:
            async with self.pool.acquire() as conn:
                if delete_all_chunks:
                    # Delete all chunks of the parent document
                    result = await conn.execute("""
                        UPDATE rag_documents 
                        SET is_active = false, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1 OR parent_document_id = $1
                    """, doc_id)
                else:
                    # Delete only specific chunk
                    result = await conn.execute("""
                        UPDATE rag_documents 
                        SET is_active = false, updated_at = CURRENT_TIMESTAMP
                        WHERE id = $1
                    """, doc_id)
                
                return result.split()[-1] != '0'  # Check if any rows affected
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Overall statistics
                overall_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(DISTINCT parent_document_id) as unique_documents,
                        COUNT(*) FILTER (WHERE is_active = true) as active_documents,
                        AVG(word_count) as avg_word_count,
                        SUM(word_count) as total_words
                    FROM rag_documents
                """)
                
                # Document type statistics
                type_stats = await conn.fetch("""
                    SELECT document_type, COUNT(*) as count
                    FROM rag_documents
                    WHERE is_active = true
                    GROUP BY document_type
                    ORDER BY count DESC
                """)
                
                # Language statistics
                language_stats = await conn.fetch("""
                    SELECT language, COUNT(*) as count
                    FROM rag_documents
                    WHERE is_active = true
                    GROUP BY language
                    ORDER BY count DESC
                """)
                
                return {
                    "overall": {
                        "total_documents": overall_stats['total_documents'],
                        "unique_documents": overall_stats['unique_documents'],
                        "active_documents": overall_stats['active_documents'],
                        "avg_word_count": float(overall_stats['avg_word_count']) if overall_stats['avg_word_count'] else 0,
                        "total_words": overall_stats['total_words'] or 0
                    },
                    "by_type": [{"type": row['document_type'], "count": row['count']} for row in type_stats],
                    "by_language": [{"language": row['language'], "count": row['count']} for row in language_stats]
                }
                
        except Exception as e:
            logger.error(f"Error getting document statistics: {e}")
            return {}
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()


class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, rag_config: RAGConfig, db_config: DatabaseConfig):
        self.config = rag_config
        self.text_processor = TextProcessor(
            max_chunk_size=rag_config.max_chunk_size,
            chunk_overlap=rag_config.chunk_overlap
        )
        self.embedding_manager = EmbeddingManager(rag_config.embedding_model)
        self.database = RAGDatabase(db_config)
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize all RAG components"""
        try:
            # Initialize embedding model
            if not self.embedding_manager.initialize():
                return False
            
            # Initialize database
            if not await self.database.initialize():
                return False
            
            self.initialized = True
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"RAG system initialization failed: {e}")
            return False
    
    def _validate_inputs(self, title: str, content: str) -> bool:
        """Validate input parameters"""
        if not title or not title.strip():
            raise ValueError("Document title cannot be empty")
        
        if not content or not content.strip():
            raise ValueError("Document content cannot be empty")
        
        if len(content) > 1000000:  # 1MB limit
            raise ValueError("Document content too large (max 1MB)")
        
        return True
    
    async def add_document(self, title: str, content: str, 
                         document_type: str = "general",
                         author: str = None,
                         source_url: str = None,
                         tags: List[str] = None,
                         additional_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add document to RAG system"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        self._validate_inputs(title, content)
        
        try:
            # Process text and create chunks
            chunks_text = self.text_processor.chunk_text(content)
            total_chunks = len(chunks_text)
            
            # Create document chunks with metadata
            chunks = []
            for i, chunk_content in enumerate(chunks_text):
                # Extract metadata for each chunk
                chunk_metadata = self.text_processor.extract_metadata(chunk_content)
                
                # Add additional metadata
                if additional_metadata:
                    chunk_metadata.update(additional_metadata)
                
                chunk_metadata.update({
                    'author': author,
                    'source_url': source_url,
                    'tags': tags or [],
                    'original_title': title,
                    'processing_date': datetime.now().isoformat()
                })
                
                chunks.append(DocumentChunk(
                    content=chunk_content,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    title=title,
                    document_type=document_type,
                    metadata=chunk_metadata
                ))
            
            # Generate embeddings for all chunks
            chunk_contents = [chunk.content for chunk in chunks]
            embeddings = self.embedding_manager.encode_batch(chunk_contents)
            
            # Store in database
            document_ids = await self.database.store_document_chunks(chunks, embeddings)
            
            logger.info(f"Document '{title}' added successfully: {total_chunks} chunks, {len(document_ids)} stored")
            
            return {
                "success": True,
                "message": f"Document '{title}' added successfully",
                "document_ids": document_ids,
                "chunks_count": total_chunks,
                "total_words": sum(chunk.metadata.get('word_count', 0) for chunk in chunks),
                "document_type": document_type
            }
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    async def search_documents(self, query: str, 
                             max_results: int = None,
                             similarity_threshold: float = None,
                             document_types: List[str] = None,
                             language: str = None) -> List[SearchResult]:
        """Search documents using semantic similarity"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        if not query or not query.strip():
            return []
        
        try:
            # Use config defaults if not provided
            max_results = max_results or self.config.max_search_results
            similarity_threshold = similarity_threshold or self.config.similarity_threshold
            
            # Generate query embedding
            query_embedding = self.embedding_manager.encode(query)
            
            # Search in database
            results = await self.database.search_similar_documents(
                query_embedding=query_embedding,
                max_results=max_results,
                similarity_threshold=similarity_threshold,
                document_types=document_types,
                language=language
            )
            
            logger.info(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    async def get_document(self, doc_id: int) -> Optional[SearchResult]:
        """Get specific document by ID"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        return await self.database.get_document_by_id(doc_id)
    
    async def list_documents(self, document_type: str = None,
                           limit: int = 100) -> List[Dict[str, Any]]:
        """List all documents"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        return await self.database.list_documents(document_type, limit)
    
    async def delete_document(self, doc_id: int, 
                            delete_all_chunks: bool = True) -> bool:
        """Delete document from RAG system"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        success = await self.database.delete_document(doc_id, delete_all_chunks)
        
        if success:
            logger.info(f"Document {doc_id} deleted successfully")
        
        return success
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        stats = await self.database.get_document_statistics()
        stats['embedding_model'] = self.embedding_manager.model_name
        stats['embedding_dimension'] = self.embedding_manager.embedding_dim
        stats['chunk_settings'] = {
            'max_chunk_size': self.config.max_chunk_size,
            'chunk_overlap': self.config.chunk_overlap
        }
        
        return stats
    
    async def update_document(self, doc_id: int, title: str = None,
                            content: str = None, document_type: str = None,
                            tags: List[str] = None) -> Dict[str, Any]:
        """Update existing document (re-processes and re-embeds)"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        try:
            # Get existing document
            existing_doc = await self.get_document(doc_id)
            if not existing_doc:
                raise ValueError(f"Document {doc_id} not found")
            
            # Use existing values if not provided
            new_title = title or existing_doc.title
            new_content = content or existing_doc.content
            new_type = document_type or existing_doc.document_type
            
            # Delete old document
            await self.delete_document(doc_id, delete_all_chunks=True)
            
            # Add updated document
            result = await self.add_document(
                title=new_title,
                content=new_content,
                document_type=new_type,
                tags=tags,
                additional_metadata={'updated_from': doc_id, 'update_date': datetime.now().isoformat()}
            )
            
            logger.info(f"Document {doc_id} updated successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            raise
    
    async def bulk_add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add multiple documents in batch"""
        if not self.initialized:
            raise RuntimeError("RAG system not initialized")
        
        results = {
            "success": [],
            "failed": [],
            "total_processed": 0,
            "total_chunks": 0
        }
        
        for i, doc in enumerate(documents):
            try:
                result = await self.add_document(
                    title=doc.get('title', f'Document {i+1}'),
                    content=doc['content'],
                    document_type=doc.get('document_type', 'general'),
                    author=doc.get('author'),
                    source_url=doc.get('source_url'),
                    tags=doc.get('tags'),
                    additional_metadata=doc.get('metadata')
                )
                
                results["success"].append({
                    "index": i,
                    "title": doc.get('title'),
                    "chunks": result["chunks_count"]
                })
                results["total_chunks"] += result["chunks_count"]
                
            except Exception as e:
                results["failed"].append({
                    "index": i,
                    "title": doc.get('title', f'Document {i+1}'),
                    "error": str(e)
                })
            
            results["total_processed"] += 1
        
        logger.info(f"Bulk upload completed: {len(results['success'])} success, {len(results['failed'])} failed")
        return results
    
    async def close(self):
        """Close RAG system and cleanup resources"""
        if self.database:
            await self.database.close()
        
        logger.info("RAG system closed")


# Utility functions
async def create_rag_system(rag_config: RAGConfig, db_config: DatabaseConfig) -> RAGSystem:
    """Factory function to create and initialize RAG system"""
    rag_system = RAGSystem(rag_config, db_config)
    
    if await rag_system.initialize():
        return rag_system
    else:
        raise RuntimeError("Failed to initialize RAG system")


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity (fallback method)"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


async def migrate_documents_from_files(rag_system: RAGSystem, 
                                     file_paths: List[str],
                                     document_type: str = "general") -> Dict[str, Any]:
    """Migrate documents from text files"""
    results = []
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title = file_path.split('/')[-1].replace('.txt', '')
            
            result = await rag_system.add_document(
                title=title,
                content=content,
                document_type=document_type,
                additional_metadata={'source_file': file_path}
            )
            
            results.append({
                "file": file_path,
                "success": True,
                "chunks": result["chunks_count"]
            })
            
        except Exception as e:
            results.append({
                "file": file_path,
                "success": False,
                "error": str(e)
            })
    
    return {
        "processed": len(results),
        "successful": len([r for r in results if r["success"]]),
        "failed": len([r for r in results if not r["success"]]),
        "details": results
    }


if __name__ == "__main__":
    # Example usage
    async def main():
        from config import RAGConfig, DatabaseConfig
        
        # Create configurations
        rag_config = RAGConfig()
        db_config = DatabaseConfig(
            host="localhost",
            database="space_detective",
            username="postgres",
            password="password"
        )
        
        # Create and initialize RAG system
        try:
            rag_system = await create_rag_system(rag_config, db_config)
            
            # Test document addition
            result = await rag_system.add_document(
                title="Test Investigation Guide",
                content="This is a test document for money laundering investigation procedures. Include steps for analyzing satellite imagery and detecting anomalies.",
                document_type="investigation_guide",
                author="Test User"
            )
            print(f"Added document: {result}")
            
            # Test search
            search_results = await rag_system.search_documents("investigation procedures")
            print(f"Search results: {len(search_results)}")
            
            # Get statistics
            stats = await rag_system.get_statistics()
            print(f"RAG statistics: {stats}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            if 'rag_system' in locals():
                await rag_system.close()
    
    # Run example
    import asyncio
    asyncio.run(main())