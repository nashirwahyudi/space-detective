# database_manager.py
"""
Database Manager for Space Detective v2.0
Handles all PostgreSQL operations including model storage, data management, and queries
"""

import asyncio
import asyncpg
import psycopg2
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import pandas as pd
import numpy as np
import pickle
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from io import StringIO
import hashlib

from config import DatabaseConfig

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Main database manager for Space Detective"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = None
        self.engine = None
        self.async_engine = None
        self._connected = False
    
    async def initialize(self):
        """Initialize database connections and create tables"""
        try:
            # AsyncPG connection pool
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=self.config.min_connections,
                max_size=self.config.max_connections
            )
            
            # SQLAlchemy engines
            self.engine = create_engine(self.config.connection_url)
            self.async_engine = create_async_engine(self.config.async_connection_url)
            
            # Create tables if not exist
            await self.create_tables()
            await self.create_indexes()
            
            self._connected = True
            logger.info("Database connections initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise Exception(f"Database connection failed: {e}")
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected"""
        return self._connected and self.pool is not None
    
    async def health_check(self) -> bool:
        """Perform database health check"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def create_tables(self):
        """Create all necessary tables"""
        async with self.pool.acquire() as conn:
            # Enable extensions
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
            
            # ML Models table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ml_models (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255) UNIQUE NOT NULL,
                    model_data BYTEA NOT NULL,
                    model_type VARCHAR(100) NOT NULL,
                    algorithm VARCHAR(100),
                    hyperparameters JSONB,
                    performance_metrics JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    created_by VARCHAR(100),
                    version VARCHAR(50),
                    is_active BOOLEAN DEFAULT true,
                    metadata JSONB,
                    file_size_bytes BIGINT,
                    checksum VARCHAR(64)
                );
            """)

            # Training Data table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id SERIAL PRIMARY KEY,
                    dataset_name VARCHAR(255) NOT NULL,
                    data_json JSONB NOT NULL,
                    feature_columns TEXT[],
                    target_column VARCHAR(100),
                    data_source VARCHAR(255),
                    preprocessing_steps JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    version VARCHAR(50),
                    is_active BOOLEAN DEFAULT true,
                    metadata JSONB,
                    record_count INTEGER,
                    anomaly_count INTEGER,
                    normal_count INTEGER,
                    data_hash VARCHAR(64)
                );
            """)

            # SHAP Explainers table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS shap_explainers (
                    id SERIAL PRIMARY KEY,
                    explainer_name VARCHAR(255) UNIQUE NOT NULL,
                    explainer_data BYTEA NOT NULL,
                    explainer_type VARCHAR(100),
                    model_name VARCHAR(255),
                    feature_names TEXT[],
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT true,
                    metadata JSONB,
                    FOREIGN KEY (model_name) REFERENCES ml_models(model_name) ON DELETE CASCADE
                );
            """)

            # Model Performance table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(255),
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    dataset_name VARCHAR(255),
                    metrics JSONB,
                    precision_score FLOAT,
                    recall_score FLOAT,
                    f1_score FLOAT,
                    accuracy_score FLOAT,
                    auc_score FLOAT,
                    confusion_matrix JSONB,
                    feature_importance JSONB,
                    evaluation_time_seconds FLOAT,
                    notes TEXT,
                    FOREIGN KEY (model_name) REFERENCES ml_models(model_name) ON DELETE CASCADE
                );
            """)

            # Investigation Logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS investigation_logs (
                    id SERIAL PRIMARY KEY,
                    h3_index VARCHAR(20) NOT NULL,
                    investigator VARCHAR(100),
                    investigation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    investigation_type VARCHAR(100),
                    findings TEXT,
                    status VARCHAR(50) DEFAULT 'open',
                    priority_level VARCHAR(20) DEFAULT 'medium',
                    follow_up_date DATE,
                    evidence_links TEXT[],
                    confidence_score FLOAT,
                    metadata JSONB
                );
            """)

            # System Logs table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    log_level VARCHAR(20),
                    module VARCHAR(100),
                    operation VARCHAR(100),
                    message TEXT,
                    user_id VARCHAR(100),
                    execution_time_ms INTEGER,
                    metadata JSONB
                );
            """)

            logger.info("Database tables created successfully")

    async def create_indexes(self):
        """Create optimized indexes"""
        async with self.pool.acquire() as conn:
            # ML Models indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_models_name 
                ON ml_models(model_name);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_models_created_at 
                ON ml_models(created_at DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ml_models_is_active 
                ON ml_models(is_active);
            """)

            # Training Data indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_data_dataset_name 
                ON training_data(dataset_name);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_data_created_at 
                ON training_data(created_at DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_training_data_is_active 
                ON training_data(is_active);
            """)

            # Investigation Logs indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_investigation_logs_h3_index 
                ON investigation_logs(h3_index);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_investigation_logs_status 
                ON investigation_logs(status);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_investigation_logs_date 
                ON investigation_logs(investigation_date DESC);
            """)

            # System Logs indexes
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp 
                ON system_logs(timestamp DESC);
            """)
            
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_system_logs_module 
                ON system_logs(module);
            """)

            logger.info("Database indexes created successfully")

    # Model Management Methods
    async def save_model(self, model_obj: Any, model_name: str, model_type: str, 
                        algorithm: str = None, hyperparameters: Dict = None,
                        performance_metrics: Dict = None, metadata: Dict = None,
                        created_by: str = None, version: str = "1.0") -> int:
        """Save ML model to database"""
        try:
            # Serialize model
            model_data = pickle.dumps(model_obj)
            file_size = len(model_data)
            checksum = hashlib.sha256(model_data).hexdigest()
            
            async with self.pool.acquire() as conn:
                model_id = await conn.fetchval("""
                    INSERT INTO ml_models 
                    (model_name, model_data, model_type, algorithm, hyperparameters, 
                     performance_metrics, created_by, version, metadata, file_size_bytes, checksum)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (model_name) 
                    DO UPDATE SET 
                        model_data = EXCLUDED.model_data,
                        model_type = EXCLUDED.model_type,
                        algorithm = EXCLUDED.algorithm,
                        hyperparameters = EXCLUDED.hyperparameters,
                        performance_metrics = EXCLUDED.performance_metrics,
                        metadata = EXCLUDED.metadata,
                        file_size_bytes = EXCLUDED.file_size_bytes,
                        checksum = EXCLUDED.checksum,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, model_name, model_data, model_type, algorithm, 
                json.dumps(hyperparameters or {}), json.dumps(performance_metrics or {}),
                created_by, version, json.dumps(metadata or {}), file_size, checksum)
            
            logger.info(f"Model '{model_name}' saved successfully (ID: {model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    async def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load ML model from database"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT model_data, model_type, algorithm, hyperparameters, 
                           performance_metrics, metadata, version, created_at
                    FROM ml_models 
                    WHERE model_name = $1 AND is_active = true
                """, model_name)
                
                if not result:
                    raise ValueError(f"Model '{model_name}' not found or inactive")
                
                # Deserialize model
                model_obj = pickle.loads(result['model_data'])
                
                # Prepare metadata
                model_info = {
                    "model_type": result['model_type'],
                    "algorithm": result['algorithm'],
                    "hyperparameters": json.loads(result['hyperparameters']) if result['hyperparameters'] else {},
                    "performance_metrics": json.loads(result['performance_metrics']) if result['performance_metrics'] else {},
                    "metadata": json.loads(result['metadata']) if result['metadata'] else {},
                    "version": result['version'],
                    "created_at": result['created_at'].isoformat()
                }
                
                logger.info(f"Model '{model_name}' loaded successfully")
                return model_obj, model_info
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    async def list_models(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all models in database"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT model_name, model_type, algorithm, version, created_at, 
                           updated_at, file_size_bytes, is_active
                    FROM ml_models
                """
                if active_only:
                    query += " WHERE is_active = true"
                query += " ORDER BY updated_at DESC"
                
                results = await conn.fetch(query)
                
                models = []
                for row in results:
                    models.append({
                        "model_name": row['model_name'],
                        "model_type": row['model_type'],
                        "algorithm": row['algorithm'],
                        "version": row['version'],
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat(),
                        "file_size_mb": round(row['file_size_bytes'] / (1024*1024), 2) if row['file_size_bytes'] else 0,
                        "is_active": row['is_active']
                    })
                
                return models
                
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def delete_model(self, model_name: str, soft_delete: bool = True) -> bool:
        """Delete model from database"""
        try:
            async with self.pool.acquire() as conn:
                if soft_delete:
                    # Soft delete - mark as inactive
                    result = await conn.execute("""
                        UPDATE ml_models 
                        SET is_active = false, updated_at = CURRENT_TIMESTAMP
                        WHERE model_name = $1
                    """, model_name)
                else:
                    # Hard delete
                    result = await conn.execute("""
                        DELETE FROM ml_models WHERE model_name = $1
                    """, model_name)
                
                success = result.split()[-1] != '0'  # Check if any rows affected
                if success:
                    action = "deactivated" if soft_delete else "deleted"
                    logger.info(f"Model '{model_name}' {action} successfully")
                
                return success
                
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False

    # Training Data Management Methods
    async def save_training_data(self, df: pd.DataFrame, dataset_name: str,
                               feature_columns: List[str] = None,
                               target_column: str = None,
                               data_source: str = None,
                               preprocessing_steps: Dict = None,
                               metadata: Dict = None,
                               version: str = "1.0") -> int:
        """Save training data to database"""
        try:
            # Convert DataFrame to JSON
            data_json = df.to_dict(orient='records')
            data_hash = hashlib.sha256(json.dumps(data_json, sort_keys=True).encode()).hexdigest()
            
            # Calculate statistics
            record_count = len(df)
            anomaly_count = len(df[df['anomaly_score'] == -1]) if 'anomaly_score' in df.columns else 0
            normal_count = record_count - anomaly_count
            
            # Auto-detect feature columns if not provided
            if feature_columns is None:
                feature_columns = [col for col in df.columns 
                                 if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            
            async with self.pool.acquire() as conn:
                dataset_id = await conn.fetchval("""
                    INSERT INTO training_data 
                    (dataset_name, data_json, feature_columns, target_column, data_source,
                     preprocessing_steps, metadata, version, record_count, anomaly_count, 
                     normal_count, data_hash)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    ON CONFLICT (dataset_name)
                    DO UPDATE SET 
                        data_json = EXCLUDED.data_json,
                        feature_columns = EXCLUDED.feature_columns,
                        target_column = EXCLUDED.target_column,
                        preprocessing_steps = EXCLUDED.preprocessing_steps,
                        metadata = EXCLUDED.metadata,
                        record_count = EXCLUDED.record_count,
                        anomaly_count = EXCLUDED.anomaly_count,
                        normal_count = EXCLUDED.normal_count,
                        data_hash = EXCLUDED.data_hash,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, dataset_name, json.dumps(data_json), feature_columns, target_column,
                data_source, json.dumps(preprocessing_steps or {}), json.dumps(metadata or {}),
                version, record_count, anomaly_count, normal_count, data_hash)
            
            logger.info(f"Training data '{dataset_name}' saved successfully (ID: {dataset_id}, {record_count} records)")
            return dataset_id
            
        except Exception as e:
            logger.error(f"Error saving training data: {e}")
            raise

    async def load_training_data(self, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load training data from database"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT data_json, feature_columns, target_column, data_source,
                           preprocessing_steps, metadata, version, record_count,
                           anomaly_count, normal_count, created_at
                    FROM training_data 
                    WHERE dataset_name = $1 AND is_active = true
                """, dataset_name)
                
                if not result:
                    raise ValueError(f"Dataset '{dataset_name}' not found or inactive")
                
                # Convert JSON back to DataFrame
                data_json = result['data_json']
                df = pd.read_json(StringIO(json.dumps(data_json)), orient='records')
                
                # Prepare metadata
                dataset_info = {
                    "feature_columns": result['feature_columns'],
                    "target_column": result['target_column'],
                    "data_source": result['data_source'],
                    "preprocessing_steps": json.loads(result['preprocessing_steps']) if result['preprocessing_steps'] else {},
                    "metadata": json.loads(result['metadata']) if result['metadata'] else {},
                    "version": result['version'],
                    "record_count": result['record_count'],
                    "anomaly_count": result['anomaly_count'],
                    "normal_count": result['normal_count'],
                    "created_at": result['created_at'].isoformat()
                }
                
                logger.info(f"Training data '{dataset_name}' loaded successfully ({len(df)} records)")
                return df, dataset_info
                
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise

    async def list_datasets(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """List all datasets in database"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT dataset_name, data_source, version, record_count, 
                           anomaly_count, normal_count, created_at, updated_at, is_active
                    FROM training_data
                """
                if active_only:
                    query += " WHERE is_active = true"
                query += " ORDER BY updated_at DESC"
                
                results = await conn.fetch(query)
                
                datasets = []
                for row in results:
                    datasets.append({
                        "dataset_name": row['dataset_name'],
                        "data_source": row['data_source'],
                        "version": row['version'],
                        "record_count": row['record_count'],
                        "anomaly_count": row['anomaly_count'],
                        "normal_count": row['normal_count'],
                        "anomaly_percentage": (row['anomaly_count'] / row['record_count'] * 100) if row['record_count'] > 0 else 0,
                        "created_at": row['created_at'].isoformat(),
                        "updated_at": row['updated_at'].isoformat(),
                        "is_active": row['is_active']
                    })
                
                return datasets
                
        except Exception as e:
            logger.error(f"Error listing datasets: {e}")
            return []

    # SHAP Explainer Management
    async def save_shap_explainer(self, explainer_obj: Any, explainer_name: str,
                                explainer_type: str, model_name: str,
                                feature_names: List[str] = None,
                                metadata: Dict = None) -> int:
        """Save SHAP explainer to database"""
        try:
            explainer_data = pickle.dumps(explainer_obj)
            
            async with self.pool.acquire() as conn:
                explainer_id = await conn.fetchval("""
                    INSERT INTO shap_explainers 
                    (explainer_name, explainer_data, explainer_type, model_name, 
                     feature_names, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (explainer_name)
                    DO UPDATE SET 
                        explainer_data = EXCLUDED.explainer_data,
                        explainer_type = EXCLUDED.explainer_type,
                        model_name = EXCLUDED.model_name,
                        feature_names = EXCLUDED.feature_names,
                        metadata = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """, explainer_name, explainer_data, explainer_type, model_name,
                feature_names, json.dumps(metadata or {}))
            
            logger.info(f"SHAP explainer '{explainer_name}' saved successfully (ID: {explainer_id})")
            return explainer_id
            
        except Exception as e:
            logger.error(f"Error saving SHAP explainer: {e}")
            raise

    async def load_shap_explainer(self, explainer_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load SHAP explainer from database"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT explainer_data, explainer_type, model_name, 
                           feature_names, metadata, created_at
                    FROM shap_explainers 
                    WHERE explainer_name = $1 AND is_active = true
                """, explainer_name)
                
                if not result:
                    logger.warning(f"SHAP explainer '{explainer_name}' not found")
                    return None, {}
                
                # Deserialize explainer
                explainer_obj = pickle.loads(result['explainer_data'])
                
                # Prepare metadata
                explainer_info = {
                    "explainer_type": result['explainer_type'],
                    "model_name": result['model_name'],
                    "feature_names": result['feature_names'],
                    "metadata": json.loads(result['metadata']) if result['metadata'] else {},
                    "created_at": result['created_at'].isoformat()
                }
                
                logger.info(f"SHAP explainer '{explainer_name}' loaded successfully")
                return explainer_obj, explainer_info
                
        except Exception as e:
            logger.error(f"Error loading SHAP explainer: {e}")
            return None, {}

    # Investigation Logging
    async def log_investigation(self, h3_index: str, investigator: str = None,
                              investigation_type: str = "anomaly_analysis",
                              findings: str = None, status: str = "open",
                              priority_level: str = "medium",
                              confidence_score: float = None,
                              metadata: Dict = None) -> int:
        """Log investigation activity"""
        try:
            async with self.pool.acquire() as conn:
                log_id = await conn.fetchval("""
                    INSERT INTO investigation_logs 
                    (h3_index, investigator, investigation_type, findings, status,
                     priority_level, confidence_score, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    RETURNING id
                """, h3_index, investigator, investigation_type, findings,
                status, priority_level, confidence_score, json.dumps(metadata or {}))
            
            logger.info(f"Investigation logged for H3 {h3_index} (ID: {log_id})")
            return log_id
            
        except Exception as e:
            logger.error(f"Error logging investigation: {e}")
            raise

    async def get_investigation_history(self, h3_index: str = None, 
                                      investigator: str = None,
                                      limit: int = 100) -> List[Dict[str, Any]]:
        """Get investigation history"""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT h3_index, investigator, investigation_date, investigation_type,
                           findings, status, priority_level, confidence_score
                    FROM investigation_logs
                    WHERE 1=1
                """
                params = []
                param_count = 0
                
                if h3_index:
                    param_count += 1
                    query += f" AND h3_index = ${param_count}"
                    params.append(h3_index)
                
                if investigator:
                    param_count += 1
                    query += f" AND investigator = ${param_count}"
                    params.append(investigator)
                
                query += f" ORDER BY investigation_date DESC LIMIT ${param_count + 1}"
                params.append(limit)
                
                results = await conn.fetch(query, *params)
                
                investigations = []
                for row in results:
                    investigations.append({
                        "h3_index": row['h3_index'],
                        "investigator": row['investigator'],
                        "investigation_date": row['investigation_date'].isoformat(),
                        "investigation_type": row['investigation_type'],
                        "findings": row['findings'],
                        "status": row['status'],
                        "priority_level": row['priority_level'],
                        "confidence_score": row['confidence_score']
                    })
                
                return investigations
                
        except Exception as e:
            logger.error(f"Error getting investigation history: {e}")
            return []

    # System Monitoring
    async def log_system_activity(self, log_level: str, module: str, operation: str,
                                message: str, user_id: str = None,
                                execution_time_ms: int = None,
                                metadata: Dict = None):
        """Log system activity"""
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO system_logs 
                    (log_level, module, operation, message, user_id, execution_time_ms, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                """, log_level, module, operation, message, user_id, 
                execution_time_ms, json.dumps(metadata or {}))
                
        except Exception as e:
            logger.error(f"Error logging system activity: {e}")

    async def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            async with self.pool.acquire() as conn:
                # Models statistics
                models_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total_models,
                           COUNT(*) FILTER (WHERE is_active = true) as active_models,
                           SUM(file_size_bytes) as total_size_bytes
                    FROM ml_models
                """)
                
                # Datasets statistics
                datasets_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total_datasets,
                           COUNT(*) FILTER (WHERE is_active = true) as active_datasets,
                           SUM(record_count) as total_records,
                           SUM(anomaly_count) as total_anomalies
                    FROM training_data
                """)
                
                # Investigation statistics
                investigations_stats = await conn.fetchrow("""
                    SELECT COUNT(*) as total_investigations,
                           COUNT(DISTINCT h3_index) as unique_locations,
                           COUNT(*) FILTER (WHERE status = 'open') as open_investigations
                    FROM investigation_logs
                """)
                
                return {
                    "models": {
                        "total": models_stats['total_models'],
                        "active": models_stats['active_models'],
                        "total_size_mb": round((models_stats['total_size_bytes'] or 0) / (1024*1024), 2)
                    },
                    "datasets": {
                        "total": datasets_stats['total_datasets'],
                        "active": datasets_stats['active_datasets'],
                        "total_records": datasets_stats['total_records'] or 0,
                        "total_anomalies": datasets_stats['total_anomalies'] or 0,
                        "anomaly_rate": (datasets_stats['total_anomalies'] / datasets_stats['total_records'] * 100) 
                                       if datasets_stats['total_records'] else 0
                    },
                    "investigations": {
                        "total": investigations_stats['total_investigations'],
                        "unique_locations": investigations_stats['unique_locations'],
                        "open": investigations_stats['open_investigations']
                    },
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}

    async def cleanup_old_logs(self, days_to_keep: int = 30):
        """Clean up old system logs"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    DELETE FROM system_logs 
                    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '%s days'
                """, days_to_keep)
                
                deleted_count = int(result.split()[-1])
                logger.info(f"Cleaned up {deleted_count} old log entries")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Error cleaning up logs: {e}")
            return 0

    async def close(self):
        """Close all database connections"""
        try:
            if self.pool:
                await self.pool.close()
            if self.async_engine:
                await self.async_engine.dispose()
            
            self._connected = False
            logger.info("Database connections closed")
            
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")


class DatabaseMigration:
    """Database migration utilities"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    async def migrate_from_csv(self, csv_file_path: str, dataset_name: str = "main_dataset",
                             chunk_size: int = 1000) -> bool:
        """Migrate data from CSV file to PostgreSQL with chunking for large files"""
        try:
            logger.info(f"Starting migration from {csv_file_path}")
            
            # Read CSV file in chunks
            chunk_list = []
            for chunk_df in pd.read_csv(csv_file_path, chunksize=chunk_size):
                chunk_list.append(chunk_df)
            
            # Combine all chunks
            df = pd.concat(chunk_list, ignore_index=True)
            
            # Clean and prepare data
            df = df.fillna(0)  # Handle missing values
            
            # Save to database
            dataset_id = await self.db_manager.save_training_data(
                df=df,
                dataset_name=dataset_name,
                data_source=csv_file_path,
                metadata={
                    "migration_date": datetime.now().isoformat(),
                    "source_file": csv_file_path,
                    "original_shape": df.shape
                }
            )
            
            logger.info(f"Successfully migrated {len(df)} records from {csv_file_path} (Dataset ID: {dataset_id})")
            return True
            
        except Exception as e:
            logger.error(f"Migration from CSV failed: {e}")
            return False

    async def migrate_model(self, model_file_path: str, model_name: str,
                          algorithm: str = None, version: str = "1.0") -> bool:
        """Migrate model from pickle file to PostgreSQL"""
        try:
            logger.info(f"Starting model migration from {model_file_path}")
            
            # Load model from file
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Extract model information
            model_type = str(type(model).__name__)
            hyperparameters = {}
            
            # Try to get model parameters
            if hasattr(model, 'get_params'):
                hyperparameters = model.get_params()
            
            # Save to database
            model_id = await self.db_manager.save_model(
                model_obj=model,
                model_name=model_name,
                model_type=model_type,
                algorithm=algorithm or model_type.lower(),
                hyperparameters=hyperparameters,
                version=version,
                metadata={
                    "migration_date": datetime.now().isoformat(),
                    "source_file": model_file_path
                }
            )
            
            logger.info(f"Successfully migrated model {model_name} (Model ID: {model_id})")
            return True
            
        except Exception as e:
            logger.error(f"Model migration failed: {e}")
            return False

    async def migrate_shap_explainer(self, explainer_file_path: str, explainer_name: str,
                                   model_name: str, explainer_type: str = "TreeExplainer") -> bool:
        """Migrate SHAP explainer from pickle file to PostgreSQL"""
        try:
            logger.info(f"Starting SHAP explainer migration from {explainer_file_path}")
            
            # Load explainer from file
            with open(explainer_file_path, 'rb') as f:
                explainer = pickle.load(f)
            
            # Save to database
            explainer_id = await self.db_manager.save_shap_explainer(
                explainer_obj=explainer,
                explainer_name=explainer_name,
                explainer_type=explainer_type,
                model_name=model_name,
                metadata={
                    "migration_date": datetime.now().isoformat(),
                    "source_file": explainer_file_path
                }
            )
            
            logger.info(f"Successfully migrated SHAP explainer {explainer_name} (Explainer ID: {explainer_id})")
            return True
            
        except Exception as e:
            logger.error(f"SHAP explainer migration failed: {e}")
            return False

    async def export_to_csv(self, dataset_name: str, output_path: str) -> bool:
        """Export dataset from PostgreSQL to CSV"""
        try:
            df, dataset_info = await self.db_manager.load_training_data(dataset_name)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Dataset '{dataset_name}' exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Export to CSV failed: {e}")
            return False

    async def backup_database(self, backup_path: str) -> bool:
        """Create database backup"""
        try:
            import subprocess
            import os
            
            # Create backup directory if not exists
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Use pg_dump to create backup
            cmd = [
                'pg_dump',
                '-h', self.db_manager.config.host,
                '-p', str(self.db_manager.config.port),
                '-U', self.db_manager.config.username,
                '-d', self.db_manager.config.database,
                '-f', backup_path,
                '--verbose'
            ]
            
            # Set password as environment variable
            env = os.environ.copy()
            env['PGPASSWORD'] = self.db_manager.config.password
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Database backup created successfully: {backup_path}")
                return True
            else:
                logger.error(f"Database backup failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False


# Factory function to create database manager
async def create_database_manager(config: DatabaseConfig) -> DatabaseManager:
    """Create and initialize database manager"""
    db_manager = DatabaseManager(config)
    await db_manager.initialize()
    return db_manager


# Utility functions for common database operations
async def get_location_data(db_manager: DatabaseManager, h3_index: str) -> Optional[Dict[str, Any]]:
    """Get location data for specific H3 index"""
    try:
        # Load the main dataset
        df, _ = await db_manager.load_training_data("main_dataset")
        
        # Filter for specific location
        location_data = df[df['h3_index'] == h3_index]
        if location_data.empty:
            return None
        
        row = location_data.iloc[0]
        return {
            'h3_index': h3_index,
            'location_name': f"{row.get('nmdesa', 'Unknown')}, {row.get('nmkab', 'Unknown')}, {row.get('nmprov', 'Unknown')}",
            'coordinates': {
                'latitude': row.get('latitude'),
                'longitude': row.get('longitude')
            },
            'anomaly_info': {
                'is_anomaly': row.get('anomaly_score', 1) == -1,
                'anomaly_score': row.get('anomaly_score', 0),
                'confidence': row.get('confidence_score', 0)
            },
            'indicators': {
                'built_growth': row.get('built_growth_pct_22_24', 0),
                'wealth_index': row.get('RWI', 0),
                'night_lights': row.get('ntl_sumut_monthly_mean', 0),
                'population': row.get('WPOP2020_sum', 0)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting location data: {e}")
        return None


async def get_anomaly_statistics(db_manager: DatabaseManager, dataset_name: str = "main_dataset") -> Dict[str, Any]:
    """Get comprehensive anomaly statistics"""
    try:
        df, dataset_info = await db_manager.load_training_data(dataset_name)
        
        total_records = len(df)
        anomaly_records = len(df[df['anomaly_score'] == -1]) if 'anomaly_score' in df.columns else 0
        normal_records = total_records - anomaly_records
        
        # Province-wise statistics
        province_stats = []
        if 'nmprov' in df.columns:
            for province in df['nmprov'].unique():
                province_data = df[df['nmprov'] == province]
                province_anomalies = len(province_data[province_data['anomaly_score'] == -1])
                province_total = len(province_data)
                
                province_stats.append({
                    "province": province,
                    "total_locations": province_total,
                    "anomaly_count": province_anomalies,
                    "normal_count": province_total - province_anomalies,
                    "anomaly_percentage": (province_anomalies / province_total * 100) if province_total > 0 else 0
                })
        
        # Top anomalous areas
        top_anomalies = []
        if not df.empty and 'anomaly_score' in df.columns:
            anomalous_areas = df[df['anomaly_score'] == -1].copy()
            if not anomalous_areas.empty and 'built_growth_pct_22_24' in anomalous_areas.columns:
                anomalous_areas = anomalous_areas.sort_values('built_growth_pct_22_24', ascending=False)
                
                for _, row in anomalous_areas.head(15).iterrows():
                    top_anomalies.append({
                        "h3_index": row['h3_index'],
                        "location": f"{row.get('nmdesa', 'Unknown')}, {row.get('nmkec', 'Unknown')}, {row.get('nmkab', 'Unknown')}",
                        "province": row.get('nmprov', 'Unknown'),
                        "built_growth_percentage": row.get('built_growth_pct_22_24', 0),
                        "rwi_score": row.get('RWI', 0),
                        "night_light": row.get('ntl_sumut_monthly_mean', 0),
                        "anomaly_score": row.get('anomaly_score', 0)
                    })
        
        return {
            "overall_statistics": {
                "total_locations": total_records,
                "anomaly_count": anomaly_records,
                "normal_count": normal_records,
                "anomaly_percentage": (anomaly_records / total_records * 100) if total_records > 0 else 0
            },
            "province_statistics": sorted(province_stats, key=lambda x: x['anomaly_percentage'], reverse=True),
            "top_anomalous_areas": top_anomalies,
            "dataset_info": dataset_info
        }
        
    except Exception as e:
        logger.error(f"Error getting anomaly statistics: {e}")
        return {}


if __name__ == "__main__":
    # Example usage
    async def main():
        from config import DatabaseConfig
        
        # Create database configuration
        db_config = DatabaseConfig(
            host="localhost",
            database="space_detective",
            username="postgres",
            password="password"
        )
        
        # Create and initialize database manager
        db_manager = await create_database_manager(db_config)
        
        try:
            # Test database connection
            health = await db_manager.health_check()
            print(f"Database health: {health}")
            
            # Get system statistics
            stats = await db_manager.get_system_stats()
            print(f"System stats: {stats}")
            
            # List models and datasets
            models = await db_manager.list_models()
            datasets = await db_manager.list_datasets()
            
            print(f"Models: {len(models)}")
            print(f"Datasets: {len(datasets)}")
            
        finally:
            await db_manager.close()
    
    # Run example
    import asyncio
    asyncio.run(main())