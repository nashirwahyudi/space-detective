# database_setup.py
"""
Database setup and migration scripts for Space Detective v2.0
Run this script to initialize PostgreSQL database with required tables and extensions
"""

import asyncio
import asyncpg
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import logging
from sklearn.ensemble import IsolationForest
from sentence_transformers import SentenceTransformer
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    def __init__(self, host="localhost", port=5432, database="space_detective", 
                 username="postgres", password="password"):
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.pool = None

    async def connect(self):
        """Establish database connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.username,
                password=self.password,
                min_size=2,
                max_size=10
            )
            logger.info("Database connection pool created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    async def setup_extensions(self):
        """Setup required PostgreSQL extensions"""
        try:
            async with self.pool.acquire() as conn:
                # Enable pgvector extension for vector operations
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Enable uuid extension for UUID generation
                await conn.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")
                
                # Enable btree_gin for faster JSON queries
                await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
                
                logger.info("PostgreSQL extensions enabled successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to setup extensions: {e}")
            return False

    async def create_tables(self):
        """Create all required tables"""
        try:
            async with self.pool.acquire() as conn:
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

                # RAG Documents table with enhanced schema
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

                logger.info("All tables created successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            return False

    async def create_indexes(self):
        """Create indexes for better performance"""
        try:
            async with self.pool.acquire() as conn:
                # RAG Documents indexes
                await conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding 
                    ON rag_documents USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
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

                logger.info("All indexes created successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            return False

    async def insert_sample_data(self):
        """Insert sample data for testing"""
        try:
            async with self.pool.acquire() as conn:
                # Insert sample model
                sample_model = IsolationForest(contamination=0.1, random_state=42)
                # Create some dummy training data for the model
                X_sample = np.random.rand(100, 5)
                sample_model.fit(X_sample)
                
                model_data = pickle.dumps(sample_model)
                
                await conn.execute("""
                    INSERT INTO ml_models 
                    (model_name, model_data, model_type, algorithm, version, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (model_name) DO NOTHING
                """, 
                "sample_isolation_forest", 
                model_data, 
                "IsolationForest", 
                "isolation_forest",
                "1.0",
                json.dumps({"contamination": 0.1, "random_state": 42, "sample": True})
                )

                # Insert sample training data
                sample_df = pd.DataFrame({
                    'h3_index': [f'87{i:012d}ffffff' for i in range(10)],
                    'built_growth_pct_22_24': np.random.uniform(0, 200, 10),
                    'RWI': np.random.uniform(-2, 2, 10),
                    'ntl_sumut_monthly_mean': np.random.uniform(0, 10, 10),
                    'anomaly_score': np.random.choice([-1, 1], 10),
                    'nmprov': np.random.choice(['Sumatera Utara', 'DKI Jakarta', 'Jawa Barat'], 10),
                    'nmkab': ['Sample City'] * 10,
                    'nmdesa': [f'Sample Village {i}' for i in range(10)]
                })
                
                await conn.execute("""
                    INSERT INTO training_data 
                    (dataset_name, data_json, record_count, anomaly_count, normal_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (dataset_name) DO NOTHING
                """,
                "sample_dataset",
                json.dumps(sample_df.to_dict(orient='records')),
                len(sample_df),
                len(sample_df[sample_df['anomaly_score'] == -1]),
                len(sample_df[sample_df['anomaly_score'] == 1]),
                json.dumps({"source": "sample_data", "purpose": "testing"})
                )

                # Insert sample RAG documents
                try:
                    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    sample_docs = [
                        {
                            "title": "Money Laundering Investigation Procedures",
                            "content": "This document outlines the standard procedures for investigating suspected money laundering activities using satellite imagery analysis. Key indicators include rapid building construction in low-wealth areas, unusual night-time activity patterns, and mismatched economic indicators.",
                            "document_type": "investigation_guide"
                        },
                        {
                            "title": "H3 Spatial Analysis Best Practices",
                            "content": "H3 hexagonal hierarchical spatial indexing provides efficient geospatial analysis capabilities. This guide covers best practices for using H3 indexes in anomaly detection, including resolution selection, neighbor analysis, and temporal pattern recognition.",
                            "document_type": "technical_guide"
                        },
                        {
                            "title": "SHAP Interpretation Guidelines",
                            "content": "SHAP (SHapley Additive exPlanations) values provide interpretable explanations for machine learning model predictions. This guide explains how to interpret SHAP values in the context of anomaly detection for anti-money laundering investigations.",
                            "document_type": "technical_guide"
                        }
                    ]
                    
                    for doc in sample_docs:
                        embedding = embedding_model.encode(doc['content'])
                        content_hash = str(hash(doc['content']))
                        
                        await conn.execute("""
                            INSERT INTO rag_documents 
                            (title, content, content_hash, document_type, embedding, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (content_hash) DO NOTHING
                        """,
                        doc['title'],
                        doc['content'],
                        content_hash,
                        doc['document_type'],
                        embedding.tolist(),
                        json.dumps({"source": "sample_data", "language": "en"})
                        )
                
                except Exception as e:
                    logger.warning(f"Could not insert sample RAG documents (sentence-transformers may not be available): {e}")

                logger.info("Sample data inserted successfully")
                return True
        except Exception as e:
            logger.error(f"Failed to insert sample data: {e}")
            return False

    async def setup_database(self, include_sample_data=True):
        """Complete database setup"""
        logger.info("Starting database setup...")
        
        if not await self.connect():
            return False
        
        if not await self.setup_extensions():
            return False
        
        if not await self.create_tables():
            return False
        
        if not await self.create_indexes():
            return False
        
        if include_sample_data:
            if not await self.insert_sample_data():
                logger.warning("Failed to insert sample data, but continuing...")
        
        logger.info("Database setup completed successfully!")
        return True

    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()


class DatabaseMigration:
    """Database migration utilities"""
    
    def __init__(self, db_setup: DatabaseSetup):
        self.db_setup = db_setup

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
            
            # Calculate statistics
            anomaly_count = len(df[df['anomaly_score'] == -1]) if 'anomaly_score' in df.columns else 0
            normal_count = len(df) - anomaly_count
            
            # Feature columns detection
            feature_columns = [col for col in df.columns if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            
            async with self.db_setup.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO training_data 
                    (dataset_name, data_json, feature_columns, record_count, anomaly_count, normal_count, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (dataset_name) 
                    DO UPDATE SET 
                        data_json = EXCLUDED.data_json,
                        feature_columns = EXCLUDED.feature_columns,
                        record_count = EXCLUDED.record_count,
                        anomaly_count = EXCLUDED.anomaly_count,
                        normal_count = EXCLUDED.normal_count,
                        updated_at = CURRENT_TIMESTAMP
                """,
                dataset_name,
                json.dumps(df.to_dict(orient='records')),
                feature_columns,
                len(df),
                anomaly_count,
                normal_count,
                json.dumps({
                    "source_file": csv_file_path,
                    "migration_date": datetime.now().isoformat(),
                    "columns": list(df.columns)
                })
                )
            
            logger.info(f"Successfully migrated {len(df)} records from {csv_file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            return False

    async def migrate_model(self, model_file_path: str, model_name: str):
        """Migrate model from pickle file to PostgreSQL"""
        try:
            logger.info(f"Starting model migration from {model_file_path}")
            
            # Load model from file
            with open(model_file_path, 'rb') as f:
                model = pickle.load(f)
            
            # Serialize model
            model_data = pickle.dumps(model)
            model_type = str(type(model).__name__)
            
            # Get model parameters if available
            hyperparameters = {}
            if hasattr(model, 'get_params'):
                hyperparameters = model.get_params()
            
            async with self.db_setup.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO ml_models 
                    (model_name, model_data, model_type, algorithm, hyperparameters, version, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (model_name) 
                    DO UPDATE SET 
                        model_data = EXCLUDED.model_data,
                        model_type = EXCLUDED.model_type,
                        hyperparameters = EXCLUDED.hyperparameters,
                        version = EXCLUDED.version,
                        updated_at = CURRENT_TIMESTAMP
                """,
                model_name,
                model_data,
                model_type,
                model_type.lower(),
                json.dumps(hyperparameters),
                "1.0",
                json.dumps({
                    "source_file": model_file_path,
                    "migration_date": datetime.now().isoformat()
                })
                )
            
            logger.info(f"Successfully migrated model {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Model migration failed: {e}")
            return False


# Command line interface
async def main():
    """Main function for database setup"""
    parser = argparse.ArgumentParser(description="Space Detective Database Setup")
    parser.add_argument("--host", default="localhost", help="Database host")
    parser.add_argument("--port", type=int, default=5432, help="Database port")
    parser.add_argument("--database", default="space_detective", help="Database name")
    parser.add_argument("--username", default="postgres", help="Database username")
    parser.add_argument("--password", default="password", help="Database password")
    parser.add_argument("--no-sample-data", action="store_true", help="Skip sample data insertion")
    parser.add_argument("--migrate-csv", help="Path to CSV file to migrate")
    parser.add_argument("--migrate-model", help="Path to model file to migrate")
    parser.add_argument("--model-name", default="isolation_forest_v1", help="Model name for migration")
    
    args = parser.parse_args()
    
    # Initialize database setup
    db_setup = DatabaseSetup(
        host=args.host,
        port=args.port,
        database=args.database,
        username=args.username,
        password=args.password
    )
    
    try:
        # Setup database
        success = await db_setup.setup_database(include_sample_data=not args.no_sample_data)
        if not success:
            logger.error("Database setup failed")
            return
        
        # Perform migrations if requested
        if args.migrate_csv or args.migrate_model:
            migration = DatabaseMigration(db_setup)
            
            if args.migrate_csv:
                await migration.migrate_from_csv(args.migrate_csv)
            
            if args.migrate_model:
                await migration.migrate_model(args.migrate_model, args.model_name)
        
        logger.info("🎉 Database setup completed successfully!")
        
    finally:
        await db_setup.close()


# Interactive setup function
async def interactive_setup():
    """Interactive database setup"""
    print("🛰️ SPACE DETECTIVE v2.0 - DATABASE SETUP")
    print("=" * 50)
    
    # Get database credentials
    print("\n📊 Enter your PostgreSQL credentials:")
    host = input("Host (default: localhost): ").strip() or "localhost"
    port = input("Port (default: 5432): ").strip() or "5432"
    database = input("Database name (default: space_detective): ").strip() or "space_detective"
    username = input("Username (default: postgres): ").strip() or "postgres"
    password = input("Password: ").strip()
    
    # Sample data option
    include_sample = input("\nInclude sample data for testing? (y/n, default: y): ").strip().lower()
    include_sample_data = include_sample != 'n'
    
    # Initialize setup
    db_setup = DatabaseSetup(
        host=host,
        port=int(port),
        database=database,
        username=username,
        password=password
    )
    
    try:
        print("\n🔄 Setting up database...")
        success = await db_setup.setup_database(include_sample_data=include_sample_data)
        
        if success:
            print("✅ Database setup completed successfully!")
            print(f"📊 Database: {host}:{port}/{database}")
            print("🎯 Next steps:")
            print("   1. Run: python main.py")
            print("   2. Initialize API: POST /api/initialize")
            print("   3. Load your data: POST /api/database/load-model")
        else:
            print("❌ Database setup failed!")
            
    except Exception as e:
        print(f"❌ Setup error: {e}")
    finally:
        await db_setup.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments, run interactive setup
        asyncio.run(interactive_setup())
    else:
        # Run with command line arguments
        asyncio.run(main())