# config.py
"""
Configuration management for Space Detective v2.0
"""

import os
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    host: str = Field(default="localhost", env="DATABASE_HOST")
    port: int = Field(default=5432, env="DATABASE_PORT")
    database: str = Field(default="space_detective", env="DATABASE_NAME")
    username: str = Field(default="postgres", env="DATABASE_USER")
    password: str = Field(default="password", env="DATABASE_PASSWORD")
    url: Optional[str] = Field(default=None, env="DATABASE_URL")
    
    # Connection pool settings
    min_connections: int = Field(default=5, env="DB_MIN_CONNECTIONS")
    max_connections: int = Field(default=20, env="DB_MAX_CONNECTIONS")
    
    @property
    def connection_url(self) -> str:
        if self.url:
            return self.url
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
    
    @property
    def async_connection_url(self) -> str:
        if self.url:
            return self.url.replace("postgresql://", "postgresql+asyncpg://")
        return f"postgresql+asyncpg://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


class RAGConfig(BaseSettings):
    """RAG system configuration"""
    embedding_model: str = Field(default="all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    max_chunk_size: int = Field(default=500, env="MAX_CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, env="CHUNK_OVERLAP")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    max_search_results: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    
    # Vector index settings
    vector_dimensions: int = 384  # for all-MiniLM-L6-v2
    index_lists: int = 100  # for IVFFlat index


class APIConfig(BaseSettings):
    """API configuration"""
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")
    reload: bool = Field(default=False, env="API_RELOAD")
    
    # Security
    secret_key: str = Field(default="your-secret-key-change-this", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # CORS
    cors_origins: List[str] = Field(default=["*"], env="CORS_ORIGINS")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")


class MLConfig(BaseSettings):
    """Machine learning configuration"""
    default_model_name: str = Field(default="isolation_forest_v1", env="DEFAULT_MODEL_NAME")
    default_dataset_name: str = Field(default="main_dataset", env="DEFAULT_DATASET_NAME")
    
    # Model parameters
    contamination: float = Field(default=0.1, env="MODEL_CONTAMINATION")
    random_state: int = Field(default=42, env="RANDOM_STATE")
    
    # Feature engineering
    feature_prefixes: List[str] = Field(default=["built_", "RWI", "ntl_", "veg_", "urban_"])
    
    # SHAP configuration
    shap_sample_size: int = Field(default=100, env="SHAP_SAMPLE_SIZE")
    shap_background_samples: int = Field(default=50, env="SHAP_BACKGROUND_SAMPLES")


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", env="LOG_FORMAT")
    file: Optional[str] = Field(default=None, env="LOG_FILE")
    max_file_size: int = Field(default=10485760, env="LOG_MAX_FILE_SIZE")  # 10MB
    backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")


class ExternalAPIConfig(BaseSettings):
    """External API configuration"""
    alibaba_api_key: Optional[str] = Field(default=None, env="DASHSCOPE_API_KEY")
    alibaba_model: str = Field(default="qwen-turbo", env="ALIBABA_MODEL")
    alibaba_temperature: float = Field(default=0.7, env="ALIBABA_TEMPERATURE")
    alibaba_max_tokens: int = Field(default=1000, env="ALIBABA_MAX_TOKENS")


class RedisConfig(BaseSettings):
    """Redis configuration for caching"""
    url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    max_connections: int = Field(default=10, env="REDIS_MAX_CONNECTIONS")
    
    # Cache settings
    default_ttl: int = Field(default=3600, env="CACHE_DEFAULT_TTL")  # 1 hour
    model_cache_ttl: int = Field(default=86400, env="MODEL_CACHE_TTL")  # 1 day


class AppConfig(BaseSettings):
    """Main application configuration"""
    app_name: str = "Space Detective v2.0"
    app_version: str = "2.0.0"
    description: str = "AI-Driven Satellite Intelligence with PostgreSQL & RAG"
    
    # Feature flags
    enable_rag: bool = Field(default=True, env="ENABLE_RAG")
    enable_shap: bool = Field(default=True, env="ENABLE_SHAP")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_auth: bool = Field(default=False, env="ENABLE_AUTH")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    rag: RAGConfig = RAGConfig()
    api: APIConfig = APIConfig()
    ml: MLConfig = MLConfig()
    logging: LoggingConfig = LoggingConfig()
    external_apis: ExternalAPIConfig = ExternalAPIConfig()
    redis: RedisConfig = RedisConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global configuration instance
config = AppConfig()

