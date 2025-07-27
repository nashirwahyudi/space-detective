
# utils.py
"""
Utility functions for Space Detective v2.0
"""

import hashlib
import json
import logging
import pickle
import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from functools import wraps
import aioredis
import asyncpg

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self, redis_url: str, default_ttl: int = 3600):
        self.redis_url = redis_url
        self.default_ttl = default_ttl
        self.redis = None
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Redis connection established")
            return True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            return False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis:
            return None
        
        try:
            value = await self.redis.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self.redis:
            return False
        
        try:
            serialized = pickle.dumps(value)
            await self.redis.setex(key, ttl or self.default_ttl, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.redis:
            return False
        
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern"""
        if not self.redis:
            return 0
        
        try:
            keys = await self.redis.keys(pattern)
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear pattern error: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()


def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key
            for arg in args:
                if isinstance(arg, (str, int, float, bool)):
                    key_parts.append(str(arg))
                else:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:8])
            
            # Add kwargs to key
            for k, v in sorted(kwargs.items()):
                key_parts.append(f"{k}:{v}")
            
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if cache_manager:
                cached_result = await cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if cache_manager and result is not None:
                await cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


class DataValidator:
    """Data validation utilities"""
    
    @staticmethod
    def validate_h3_index(h3_index: str) -> bool:
        """Validate H3 index format"""
        if not isinstance(h3_index, str):
            return False
        
        # H3 index should be 15 characters long and start with '8'
        if len(h3_index) != 15 or not h3_index.startswith('8'):
            return False
        
        # Should contain only hexadecimal characters
        try:
            int(h3_index[1:], 16)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_coordinates(lat: float, lng: float) -> bool:
        """Validate latitude and longitude"""
        return -90 <= lat <= 90 and -180 <= lng <= 180
    
    @staticmethod
    def validate_anomaly_score(score: Union[int, float]) -> bool:
        """Validate anomaly score"""
        return score in [-1, 1] or isinstance(score, float)
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
        """Validate DataFrame structure"""
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": {}
        }
        
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Missing required columns: {missing_columns}")
        
        # Check for empty DataFrame
        if df.empty:
            validation_result["valid"] = False
            validation_result["errors"].append("DataFrame is empty")
        
        # Check for duplicate H3 indexes if present
        if 'h3_index' in df.columns:
            duplicates = df['h3_index'].duplicated().sum()
            if duplicates > 0:
                validation_result["warnings"].append(f"Found {duplicates} duplicate H3 indexes")
        
        # Check data types
        validation_result["info"]["shape"] = df.shape
        validation_result["info"]["dtypes"] = df.dtypes.to_dict()
        validation_result["info"]["memory_usage"] = df.memory_usage(deep=True).sum()
        
        return validation_result


class ModelManager:
    """Utilities for model management"""
    
    @staticmethod
    def serialize_model(model: Any) -> bytes:
        """Serialize model to bytes"""
        return pickle.dumps(model)
    
    @staticmethod
    def deserialize_model(model_data: bytes) -> Any:
        """Deserialize model from bytes"""
        return pickle.loads(model_data)
    
    @staticmethod
    def get_model_info(model: Any) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "type": type(model).__name__,
            "module": type(model).__module__,
            "size_bytes": len(pickle.dumps(model))
        }
        
        # Try to get model-specific parameters
        if hasattr(model, 'get_params'):
            info["parameters"] = model.get_params()
        
        if hasattr(model, 'feature_importances_'):
            info["has_feature_importance"] = True
        
        return info
    
    @staticmethod
    async def evaluate_model_performance(
        model: Any, 
        X_test: pd.DataFrame, 
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model performance"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, confusion_matrix
        )
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            # ROC AUC for binary classification
            if len(np.unique(y_test)) == 2:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(X_test)
                else:
                    y_proba = y_pred
                
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {}


class TextProcessor:
    """Text processing utilities for RAG"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text"""
        import re
        from collections import Counter
        
        # Simple keyword extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
            'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may',
            'might', 'can', 'this', 'that', 'these', 'those', 'what', 'when',
            'where', 'why', 'how', 'who', 'which'
        }
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Return most common keywords
        return [word for word, count in Counter(filtered_words).most_common(max_keywords)]
    
    @staticmethod
    def similarity_score(text1: str, text2: str) -> float:
        """Calculate text similarity score"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


class PerformanceMonitor:
    """Performance monitoring utilities"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """End timing and return duration"""
        if operation in self.start_times:
            duration = (datetime.now() - self.start_times[operation]).total_seconds()
            
            if operation not in self.metrics:
                self.metrics[operation] = []
            
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            
            return duration
        return 0.0
    
    def get_average_time(self, operation: str) -> float:
        """Get average time for operation"""
        if operation in self.metrics and self.metrics[operation]:
            return sum(self.metrics[operation]) / len(self.metrics[operation])
        return 0.0
    
    def get_metrics_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        summary = {}
        
        for operation, times in self.metrics.items():
            if times:
                summary[operation] = {
                    "count": len(times),
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        
        return summary


def time_operation(operation_name: str):
    """Decorator to time operations"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            monitor = getattr(wrapper, '_monitor', PerformanceMonitor())
            
            monitor.start_timer(operation_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(operation_name)
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
        
        return wrapper
    return decorator


class DatabaseUtils:
    """Database utility functions"""
    
    @staticmethod
    async def check_connection(pool: asyncpg.Pool) -> bool:
        """Check database connection"""
        try:
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    @staticmethod
    async def get_table_info(pool: asyncpg.Pool, table_name: str) -> Dict[str, Any]:
        """Get table information"""
        try:
            async with pool.acquire() as conn:
                # Get table schema
                schema_query = """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_name = $1
                ORDER BY ordinal_position;
                """
                columns = await conn.fetch(schema_query, table_name)
                
                # Get row count
                count_query = f"SELECT COUNT(*) FROM {table_name}"
                row_count = await conn.fetchval(count_query)
                
                return {
                    "table_name": table_name,
                    "columns": [dict(row) for row in columns],
                    "row_count": row_count
                }
        except Exception as e:
            logger.error(f"Error getting table info: {e}")
            return {}
    
    @staticmethod
    async def optimize_table(pool: asyncpg.Pool, table_name: str):
        """Optimize table performance"""
        try:
            async with pool.acquire() as conn:
                # Analyze table for better query planning
                await conn.execute(f"ANALYZE {table_name}")
                
                # Vacuum table to reclaim space
                await conn.execute(f"VACUUM {table_name}")
                
                logger.info(f"Table {table_name} optimized")
        except Exception as e:
            logger.error(f"Error optimizing table {table_name}: {e}")


# Global instances
cache_manager = None
performance_monitor = PerformanceMonitor()

async def initialize_utils(config: AppConfig):
    """Initialize utility components"""
    global cache_manager
    
    if config.enable_caching:
        cache_manager = CacheManager(config.redis.url, config.redis.default_ttl)
        await cache_manager.connect()


async def cleanup_utils():
    """Cleanup utility components"""
    global cache_manager
    
    if cache_manager:
        await cache_manager.close()