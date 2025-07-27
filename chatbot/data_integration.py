# data_integration.py
"""
Data Integration Script for Space Detective v2.0
Supports both database and file-based approaches for your existing data
"""

import asyncio
import pandas as pd
import pickle
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import os

from config import AppConfig, DatabaseConfig
from database_manager import DatabaseManager, create_database_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrator:
    """Handles integration of existing training data, model, and explainer"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.db_manager: Optional[DatabaseManager] = None
        
    async def initialize_database(self, db_config: DatabaseConfig = None):
        """Initialize database connection"""
        if db_config is None:
            db_config = self.config.database
            
        try:
            self.db_manager = await create_database_manager(db_config)
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False

    def analyze_training_data(self, csv_path: str) -> Dict[str, Any]:
        """Analyze the structure and content of training data"""
        try:
            # Load and analyze data
            df = pd.read_csv(csv_path)
            
            analysis = {
                "file_path": csv_path,
                "shape": df.shape,
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "missing_values": df.isnull().sum().to_dict(),
                "sample_data": df.head(3).to_dict('records')
            }
            
            # Analyze Space Detective specific columns
            space_detective_analysis = self._analyze_space_detective_features(df)
            analysis.update(space_detective_analysis)
            
            logger.info(f"✅ Data analysis completed: {df.shape[0]} rows, {df.shape[1]} columns")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Data analysis failed: {e}")
            return {}
    
    def _analyze_space_detective_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Space Detective specific features"""
        analysis = {}
        
        # Check for required columns
        required_columns = ['h3_index', 'anomaly_score', 'built_growth_pct_22_24', 'RWI']
        missing_required = [col for col in required_columns if col not in df.columns]
        analysis['missing_required_columns'] = missing_required
        
        # Anomaly statistics
        if 'anomaly_score' in df.columns:
            anomaly_stats = {
                "total_records": len(df),
                "anomaly_count": len(df[df['anomaly_score'] == -1]),
                "normal_count": len(df[df['anomaly_score'] == 1]),
                "anomaly_percentage": (len(df[df['anomaly_score'] == -1]) / len(df) * 100)
            }
            analysis['anomaly_statistics'] = anomaly_stats
        
        # Geographic coverage
        if 'nmprov' in df.columns:
            provinces = df['nmprov'].value_counts().to_dict()
            analysis['geographic_coverage'] = {
                "provinces": list(provinces.keys()),
                "province_distribution": provinces
            }
        
        # Feature analysis
        feature_columns = [col for col in df.columns if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
        analysis['ml_features'] = {
            "feature_columns": feature_columns,
            "feature_count": len(feature_columns),
            "feature_statistics": df[feature_columns].describe().to_dict() if feature_columns else {}
        }
        
        # H3 index validation
        if 'h3_index' in df.columns:
            h3_validation = self._validate_h3_indexes(df['h3_index'])
            analysis['h3_validation'] = h3_validation
        
        return analysis
    
    def _validate_h3_indexes(self, h3_series: pd.Series) -> Dict[str, Any]:
        """Validate H3 index format"""
        import re
        
        total_count = len(h3_series)
        valid_pattern = r'^87[0-9a-f]{12}$'
        
        valid_count = 0
        invalid_examples = []
        
        for h3_idx in h3_series.head(100):  # Check first 100 for performance
            if isinstance(h3_idx, str) and re.match(valid_pattern, h3_idx, re.IGNORECASE):
                valid_count += 1
            else:
                if len(invalid_examples) < 5:
                    invalid_examples.append(str(h3_idx))
        
        return {
            "total_checked": min(100, total_count),
            "valid_count": valid_count,
            "valid_percentage": (valid_count / min(100, total_count) * 100) if total_count > 0 else 0,
            "invalid_examples": invalid_examples
        }

    async def migrate_to_database(self, csv_path: str, model_path: str, explainer_path: str) -> Dict[str, Any]:
        """Migrate all data to PostgreSQL database"""
        if not self.db_manager:
            raise RuntimeError("Database not initialized. Call initialize_database() first.")
        
        results = {
            "training_data": {"status": "pending"},
            "model": {"status": "pending"},
            "explainer": {"status": "pending"}
        }
        
        try:
            # 1. Migrate training data
            logger.info("🔄 Migrating training data...")
            df = pd.read_csv(csv_path)
            
            # Prepare metadata
            metadata = {
                "source_file": csv_path,
                "migration_date": datetime.now().isoformat(),
                "original_shape": df.shape,
                "data_source": "user_provided",
                "feature_columns": [col for col in df.columns if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))]
            }
            
            dataset_id = await self.db_manager.save_training_data(
                df=df,
                dataset_name="main_dataset",
                feature_columns=metadata["feature_columns"],
                target_column="anomaly_score",
                data_source="CSV file migration",
                metadata=metadata,
                version="1.0"
            )
            
            results["training_data"] = {
                "status": "success",
                "dataset_id": dataset_id,
                "records_count": len(df),
                "features_count": len(metadata["feature_columns"])
            }
            logger.info(f"✅ Training data migrated successfully (ID: {dataset_id})")
            
            # 2. Migrate model
            logger.info("🔄 Migrating ML model...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Get model parameters
            model_params = {}
            if hasattr(model, 'get_params'):
                model_params = model.get_params()
            
            model_id = await self.db_manager.save_model(
                model_obj=model,
                model_name="isolation_forest_v1",
                model_type=str(type(model).__name__),
                algorithm="isolation_forest",
                hyperparameters=model_params,
                metadata={
                    "source_file": model_path,
                    "migration_date": datetime.now().isoformat(),
                    "trained_on": "user_data"
                },
                version="1.0"
            )
            
            results["model"] = {
                "status": "success",
                "model_id": model_id,
                "model_type": str(type(model).__name__),
                "parameters": model_params
            }
            logger.info(f"✅ Model migrated successfully (ID: {model_id})")
            
            # 3. Migrate SHAP explainer
            logger.info("🔄 Migrating SHAP explainer...")
            with open(explainer_path, 'rb') as f:
                explainer = pickle.load(f)
            
            explainer_id = await self.db_manager.save_shap_explainer(
                explainer_obj=explainer,
                explainer_name="isolation_forest_v1_explainer",
                explainer_type=str(type(explainer).__name__),
                model_name="isolation_forest_v1",
                feature_names=metadata["feature_columns"],
                metadata={
                    "source_file": explainer_path,
                    "migration_date": datetime.now().isoformat(),
                    "model_reference": "isolation_forest_v1"
                }
            )
            
            results["explainer"] = {
                "status": "success",
                "explainer_id": explainer_id,
                "explainer_type": str(type(explainer).__name__),
                "feature_count": len(metadata["feature_columns"])
            }
            logger.info(f"✅ SHAP explainer migrated successfully (ID: {explainer_id})")
            
            # Summary
            results["summary"] = {
                "migration_completed": True,
                "total_components": 3,
                "successful_components": sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "success"),
                "migration_timestamp": datetime.now().isoformat()
            }
            
            logger.info("🎉 Complete migration to database successful!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            # Update failed component status
            for component, result in results.items():
                if isinstance(result, dict) and result.get("status") == "pending":
                    result["status"] = "failed"
                    result["error"] = str(e)
            
            return results

    def setup_file_based_config(self, csv_path: str, model_path: str, explainer_path: str) -> Dict[str, str]:
        """Setup file-based configuration (alternative to database)"""
        
        # Validate file paths
        files_to_check = {
            "training_data": csv_path,
            "model": model_path,
            "explainer": explainer_path
        }
        
        validation_results = {}
        for file_type, file_path in files_to_check.items():
            if os.path.exists(file_path):
                validation_results[file_type] = {
                    "path": file_path,
                    "exists": True,
                    "size_mb": round(os.path.getsize(file_path) / (1024*1024), 2)
                }
                logger.info(f"✅ {file_type}: {file_path} (exists, {validation_results[file_type]['size_mb']} MB)")
            else:
                validation_results[file_type] = {
                    "path": file_path,
                    "exists": False,
                    "error": "File not found"
                }
                logger.error(f"❌ {file_type}: {file_path} (not found)")
        
        # Generate config file
        config_content = f"""# Space Detective v2.0 - File-based Configuration
# Generated on: {datetime.now().isoformat()}

# File paths for your data
TRAINING_DATA_PATH = "{csv_path}"
MODEL_PATH = "{model_path}"
EXPLAINER_PATH = "{explainer_path}"

# Usage in your application:
# 1. Load training data: pd.read_csv(TRAINING_DATA_PATH)
# 2. Load model: pickle.load(open(MODEL_PATH, 'rb'))
# 3. Load explainer: pickle.load(open(EXPLAINER_PATH, 'rb'))
"""
        
        # Save config file
        config_file_path = "space_detective_file_config.py"
        with open(config_file_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"📝 File-based configuration saved to: {config_file_path}")
        
        return {
            "config_file": config_file_path,
            "validation_results": validation_results,
            "setup_completed": all(r["exists"] for r in validation_results.values())
        }

    async def create_hybrid_loader(self, csv_path: str, model_path: str, explainer_path: str) -> str:
        """Create a hybrid loader that works with both database and file-based approaches"""
        
        loader_code = f'''# hybrid_data_loader.py
"""
Hybrid Data Loader for Space Detective v2.0
Automatically detects and loads from database or files
"""

import pandas as pd
import pickle
import logging
from typing import Tuple, Any, Optional, Dict
import os

logger = logging.getLogger(__name__)

class SpaceDetectiveDataLoader:
    """Hybrid loader for Space Detective data"""
    
    def __init__(self):
        # File paths (your current setup)
        self.csv_path = "{csv_path}"
        self.model_path = "{model_path}"
        self.explainer_path = "{explainer_path}"
        
        # Database manager (will be set if database is available)
        self.db_manager = None
    
    def set_database_manager(self, db_manager):
        """Set database manager for database-based loading"""
        self.db_manager = db_manager
        logger.info("Database manager configured for hybrid loading")
    
    async def load_training_data(self, prefer_database: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """Load training data from database or file"""
        
        # Try database first if preferred and available
        if prefer_database and self.db_manager:
            try:
                df, dataset_info = await self.db_manager.load_training_data("main_dataset")
                logger.info(f"✅ Training data loaded from database: {{df.shape[0]}} records")
                return df, dataset_info
            except Exception as e:
                logger.warning(f"Database loading failed, falling back to file: {{e}}")
        
        # Fallback to file-based loading
        if os.path.exists(self.csv_path):
            df = pd.read_csv(self.csv_path)
            
            # Create dataset info similar to database format
            dataset_info = {{
                "source": "file",
                "file_path": self.csv_path,
                "record_count": len(df),
                "anomaly_count": len(df[df['anomaly_score'] == -1]) if 'anomaly_score' in df.columns else 0,
                "feature_columns": [col for col in df.columns if col.startswith(('built_', 'RWI', 'ntl_', 'veg_', 'urban_'))],
                "loaded_at": pd.Timestamp.now().isoformat()
            }}
            
            logger.info(f"✅ Training data loaded from file: {{df.shape[0]}} records")
            return df, dataset_info
        else:
            raise FileNotFoundError(f"Training data not found: {{self.csv_path}}")
    
    async def load_model(self, prefer_database: bool = True) -> Tuple[Any, Dict]:
        """Load ML model from database or file"""
        
        # Try database first if preferred and available
        if prefer_database and self.db_manager:
            try:
                model, model_info = await self.db_manager.load_model("isolation_forest_v1")
                logger.info(f"✅ Model loaded from database: {{model_info.get('model_type')}}")
                return model, model_info
            except Exception as e:
                logger.warning(f"Database model loading failed, falling back to file: {{e}}")
        
        # Fallback to file-based loading
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Create model info similar to database format
            model_info = {{
                "source": "file",
                "file_path": self.model_path,
                "model_type": str(type(model).__name__),
                "loaded_at": pd.Timestamp.now().isoformat()
            }}
            
            # Add parameters if available
            if hasattr(model, 'get_params'):
                model_info["hyperparameters"] = model.get_params()
            
            logger.info(f"✅ Model loaded from file: {{model_info['model_type']}}")
            return model, model_info
        else:
            raise FileNotFoundError(f"Model not found: {{self.model_path}}")
    
    async def load_explainer(self, prefer_database: bool = True) -> Tuple[Optional[Any], Dict]:
        """Load SHAP explainer from database or file"""
        
        # Try database first if preferred and available
        if prefer_database and self.db_manager:
            try:
                explainer, explainer_info = await self.db_manager.load_shap_explainer("isolation_forest_v1_explainer")
                if explainer:
                    logger.info(f"✅ SHAP explainer loaded from database")
                    return explainer, explainer_info
            except Exception as e:
                logger.warning(f"Database explainer loading failed, falling back to file: {{e}}")
        
        # Fallback to file-based loading
        if os.path.exists(self.explainer_path):
            with open(self.explainer_path, 'rb') as f:
                explainer = pickle.load(f)
            
            # Create explainer info similar to database format
            explainer_info = {{
                "source": "file",
                "file_path": self.explainer_path,
                "explainer_type": str(type(explainer).__name__),
                "loaded_at": pd.Timestamp.now().isoformat()
            }}
            
            logger.info(f"✅ SHAP explainer loaded from file: {{explainer_info['explainer_type']}}")
            return explainer, explainer_info
        else:
            logger.warning(f"SHAP explainer not found: {{self.explainer_path}}")
            return None, {{"source": "file", "error": "file_not_found"}}
    
    async def load_all(self, prefer_database: bool = True) -> Dict[str, Any]:
        """Load all components (data, model, explainer)"""
        try:
            # Load all components
            training_data, dataset_info = await self.load_training_data(prefer_database)
            model, model_info = await self.load_model(prefer_database)
            explainer, explainer_info = await self.load_explainer(prefer_database)
            
            logger.info("🎉 All components loaded successfully!")
            
            return {{
                "training_data": training_data,
                "dataset_info": dataset_info,
                "model": model,
                "model_info": model_info,
                "explainer": explainer,
                "explainer_info": explainer_info,
                "loading_summary": {{
                    "data_source": dataset_info.get("source"),
                    "model_source": model_info.get("source"),
                    "explainer_source": explainer_info.get("source"),
                    "total_records": len(training_data),
                    "anomaly_count": dataset_info.get("anomaly_count", 0),
                    "feature_count": len(dataset_info.get("feature_columns", [])),
                    "loaded_at": pd.Timestamp.now().isoformat()
                }}
            }}
            
        except Exception as e:
            logger.error(f"❌ Failed to load components: {{e}}")
            raise


# Global instance for easy import
data_loader = SpaceDetectiveDataLoader()

# Convenience functions
async def load_all_data(prefer_database: bool = True) -> Dict[str, Any]:
    """Convenience function to load all data"""
    return await data_loader.load_all(prefer_database)

async def setup_with_database(db_manager):
    """Setup with database manager"""
    data_loader.set_database_manager(db_manager)
    logger.info("Hybrid loader configured with database support")

# Example usage:
if __name__ == "__main__":
    import asyncio
    
    async def test_loader():
        # Test file-based loading
        try:
            result = await load_all_data(prefer_database=False)
            print(f"Loaded {{result['loading_summary']['total_records']}} records")
            print(f"Found {{result['loading_summary']['anomaly_count']}} anomalies")
            print(f"Using {{result['loading_summary']['feature_count']}} features")
        except Exception as e:
            print(f"Loading failed: {{e}}")
    
    # Run test
    asyncio.run(test_loader())
'''
        
        # Save the hybrid loader
        loader_file_path = "hybrid_data_loader.py"
        with open(loader_file_path, 'w') as f:
            f.write(loader_code)
        
        logger.info(f"📁 Hybrid data loader created: {loader_file_path}")
        return loader_file_path


async def main():
    """Main function to demonstrate data integration options"""
    
    # Example file paths (adjust these to your actual file locations)
    csv_path = "sample_data.csv"  # Your training data
    model_path = "isolation_forest_model.pkl"  # Your trained model
    explainer_path = "shap_explainer.pkl"  # Your SHAP explainer
    
    print("🛰️ Space Detective v2.0 - Data Integration")
    print("=" * 50)
    
    # Initialize integrator
    config = AppConfig()
    integrator = DataIntegrator(config)
    
    # Analyze your data
    print("\n📊 STEP 1: Analyzing your training data...")
    if os.path.exists(csv_path):
        analysis = integrator.analyze_training_data(csv_path)
        print(f"✅ Data shape: {analysis.get('shape')}")
        print(f"✅ Anomaly rate: {analysis.get('anomaly_statistics', {}).get('anomaly_percentage', 0):.2f}%")
        print(f"✅ Provinces: {len(analysis.get('geographic_coverage', {}).get('provinces', []))}")
        print(f"✅ ML features: {analysis.get('ml_features', {}).get('feature_count', 0)}")
    else:
        print(f"❌ Training data file not found: {csv_path}")
        return
    
    # Option selection
    print("\n🔧 STEP 2: Choose integration approach:")
    print("1. 🗄️  Migrate to PostgreSQL database (recommended for production)")
    print("2. 📁  Keep file-based (simpler setup)")
    print("3. 🔄  Create hybrid loader (best of both worlds)")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        print("\n🗄️ MIGRATING TO DATABASE...")
        
        # Initialize database
        db_initialized = await integrator.initialize_database()
        if not db_initialized:
            print("❌ Database initialization failed. Check your PostgreSQL setup.")
            return
        
        # Migrate all data
        migration_results = await integrator.migrate_to_database(csv_path, model_path, explainer_path)
        
        print("\n📋 MIGRATION RESULTS:")
        for component, result in migration_results.items():
            if isinstance(result, dict):
                status = result.get('status', 'unknown')
                if status == 'success':
                    print(f"✅ {component}: {status}")
                else:
                    print(f"❌ {component}: {status}")
        
        if migration_results.get('summary', {}).get('migration_completed'):
            print("\n🎉 Migration completed! Your data is now in PostgreSQL.")
            print("📋 Next steps:")
            print("   1. Start the Space Detective API: python main.py")
            print("   2. Initialize system: POST /api/initialize")
            print("   3. Load from database: POST /api/database/load-model")
        
    elif choice == "2":
        print("\n📁 SETTING UP FILE-BASED CONFIGURATION...")
        
        config_result = integrator.setup_file_based_config(csv_path, model_path, explainer_path)
        
        print(f"\n✅ Configuration file created: {config_result['config_file']}")
        
        if config_result['setup_completed']:
            print("🎉 File-based setup completed!")
            print("📋 Next steps:")
            print("   1. Use the generated config file in your application")
            print("   2. Load data using standard pandas/pickle methods")
        else:
            print("⚠️ Some files are missing. Please check file paths.")
    
    elif choice == "3":
        print("\n🔄 CREATING HYBRID LOADER...")
        
        loader_path = await integrator.create_hybrid_loader(csv_path, model_path, explainer_path)
        
        print(f"✅ Hybrid loader created: {loader_path}")
        print("🎉 Hybrid setup completed!")
        print("📋 Next steps:")
        print("   1. Import: from hybrid_data_loader import load_all_data")
        print("   2. Use: data = await load_all_data(prefer_database=False)")
        print("   3. Optionally setup database later for production")
    
    else:
        print("❌ Invalid choice. Please run the script again.")
    
    print("\n🚀 Setup completed! Your Space Detective v2.0 is ready to use.")


if __name__ == "__main__":
    asyncio.run(main())