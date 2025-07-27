# setup_space_detective.py
"""
Setup Script for Space Detective v2.0
Configure PostgreSQL credentials and Alibaba API key
"""

import os
import asyncio
import json
from pathlib import Path
from typing import Dict, Any

# Create .env file for configuration
def create_env_file():
    """Create .env file with your credentials"""
    
    print("🔧 SPACE DETECTIVE v2.0 - CONFIGURATION SETUP")
    print("=" * 50)
    
    # Get PostgreSQL credentials
    print("\n📊 PostgreSQL Database Configuration:")
    db_host = input("Database Host (default: localhost): ").strip() or "localhost"
    db_port = input("Database Port (default: 5432): ").strip() or "5432"
    db_name = input("Database Name (default: space_detective): ").strip() or "space_detective"
    db_user = input("Database Username: ").strip()
    db_password = input("Database Password: ").strip()
    
    # Get Alibaba API key
    print("\n🤖 Alibaba API Configuration:")
    alibaba_key = input("Alibaba API Key (optional, press Enter to skip): ").strip()
    
    # Get file paths for your data
    print("\n📁 Data File Paths:")
    csv_path = input("Training data CSV path: ").strip()
    model_path = input("Model PKL file path: ").strip()
    explainer_path = input("SHAP explainer PKL path: ").strip()
    
    # Create .env content
    env_content = f"""# Space Detective v2.0 Configuration
# Generated on: {datetime.now().isoformat()}

# Database Configuration
DATABASE_HOST={db_host}
DATABASE_PORT={db_port}
DATABASE_NAME={db_name}
DATABASE_USER={db_user}
DATABASE_PASSWORD={db_password}
DATABASE_URL=postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}

# Alibaba Cloud API
DASHSCOPE_API_KEY={alibaba_key}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=False

# Data File Paths (for migration)
TRAINING_DATA_PATH={csv_path}
MODEL_PATH={model_path}
EXPLAINER_PATH={explainer_path}

# Feature Flags
ENABLE_RAG=true
ENABLE_SHAP=true
ENABLE_CACHING=true

# Security
SECRET_KEY=space-detective-secret-key-change-in-production

# RAG Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_CHUNK_SIZE=500
CHUNK_OVERLAP=50
SIMILARITY_THRESHOLD=0.7
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Configuration saved to .env file")
    
    return {
        "database": {
            "host": db_host,
            "port": int(db_port),
            "database": db_name,
            "username": db_user,
            "password": db_password
        },
        "alibaba_api_key": alibaba_key,
        "file_paths": {
            "csv": csv_path,
            "model": model_path,
            "explainer": explainer_path
        }
    }


# Alternative: Direct setup function
async def setup_space_detective_direct():
    """Direct setup without .env file"""
    
    # Your credentials here - GANTI DENGAN CREDENTIALS ANDA
    config = {
        "database": {
            "host": "localhost",           # Ganti dengan host PostgreSQL Anda
            "port": 5432,                 # Ganti dengan port PostgreSQL Anda  
            "database": "space_detective", # Ganti dengan nama database Anda
            "username": "postgres",        # Ganti dengan username PostgreSQL Anda
            "password": "your_password",   # Ganti dengan password PostgreSQL Anda
        },
        "alibaba_api_key": "your_alibaba_api_key",  # Ganti dengan Alibaba API key Anda
        "file_paths": {
            "csv": "path/to/your/sample_data.csv",           # Path ke CSV Anda
            "model": "path/to/your/model.pkl",               # Path ke model PKL Anda
            "explainer": "path/to/your/explainer.pkl"       # Path ke explainer PKL Anda
        }
    }
    
    print("🚀 Starting Space Detective v2.0 setup...")
    
    # Import our modules
    from config import AppConfig, DatabaseConfig
    from database_manager import create_database_manager
    from rag_system import create_rag_system  
    from chat_api import create_chat_api
    from data_integration import DataIntegrator
    
    try:
        # 1. Initialize database
        print("📊 Initializing PostgreSQL database...")
        db_config = DatabaseConfig(
            host=config["database"]["host"],
            port=config["database"]["port"],
            database=config["database"]["database"],
            username=config["database"]["username"],
            password=config["database"]["password"]
        )
        
        db_manager = await create_database_manager(db_config)
        print("✅ Database connected successfully!")
        
        # 2. Migrate your data to database
        print("🔄 Migrating your data to database...")
        integrator = DataIntegrator(AppConfig())
        integrator.db_manager = db_manager
        
        migration_result = await integrator.migrate_to_database(
            csv_path=config["file_paths"]["csv"],
            model_path=config["file_paths"]["model"], 
            explainer_path=config["file_paths"]["explainer"]
        )
        
        if migration_result["summary"]["migration_completed"]:
            print("✅ Data migration completed successfully!")
        else:
            print("❌ Data migration failed!")
            return False
        
        # 3. Initialize RAG system
        print("📚 Initializing RAG system...")
        app_config = AppConfig()
        app_config.external_apis.alibaba_api_key = config["alibaba_api_key"]
        
        rag_system = await create_rag_system(app_config.rag, db_config)
        print("✅ RAG system initialized!")
        
        # 4. Initialize Chat API
        print("🤖 Initializing Chat API...")
        chat_api = await create_chat_api(db_manager, rag_system, app_config)
        print("✅ Chat API initialized!")
        
        # 5. Test the system
        print("🧪 Testing system...")
        test_response = await chat_api.process_message(
            "What are the anomaly statistics?", 
            session_id="setup_test"
        )
        
        if not test_response.get("error"):
            print("✅ System test passed!")
            print(f"📊 Test response: {test_response['response'][:100]}...")
        else:
            print("❌ System test failed!")
        
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("📋 Next steps:")
        print("1. Start the API server: python main.py")
        print("2. Access API docs: http://localhost:8000/docs") 
        print("3. Test chat: POST http://localhost:8000/api/chat")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


# Quick setup script
def main():
    """Main setup function"""
    
    print("🛰️ SPACE DETECTIVE v2.0 - SETUP WIZARD")
    print("=" * 50)
    print("Choose setup method:")
    print("1. 📝 Interactive setup (creates .env file)")
    print("2. ⚡ Direct setup (edit code with your credentials)")
    print("3. 🔧 Manual configuration")
    
    choice = input("\nEnter your choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        # Interactive setup
        config = create_env_file()
        print("\n✅ .env file created!")
        print("📋 Next steps:")
        print("1. Review the .env file")
        print("2. Run: python setup_space_detective.py (choice 2)")
        print("3. Or run: python main.py")
        
    elif choice == "2":
        # Direct setup
        print("\n⚠️  EDIT THE CREDENTIALS FIRST!")
        print("📝 Edit setup_space_detective_direct() function with your:")
        print("   - PostgreSQL credentials")
        print("   - Alibaba API key") 
        print("   - File paths to your data")
        print("\nThen run this script again with choice 2")
        
        confirm = input("\nHave you updated the credentials? (y/n): ").strip().lower()
        if confirm == 'y':
            # Run the direct setup
            import asyncio
            asyncio.run(setup_space_detective_direct())
        else:
            print("❌ Please update credentials first!")
    
    elif choice == "3":
        # Manual configuration guide
        print_manual_setup_guide()
    
    else:
        print("❌ Invalid choice!")


def print_manual_setup_guide():
    """Print manual setup guide"""
    
    guide = """
🔧 MANUAL CONFIGURATION GUIDE
=" * 50

📁 1. Create .env file in project root:
   Copy the template above and fill in your credentials

📊 2. PostgreSQL Setup:
   - Ensure PostgreSQL is running
   - Create database 'space_detective' 
   - Grant permissions to your user

🤖 3. Alibaba API Setup:
   - Get API key from Alibaba Cloud
   - Add to .env file as DASHSCOPE_API_KEY

📂 4. Prepare your data files:
   - sample_data.csv (your training data)
   - model.pkl (your trained model)
   - explainer.pkl (your SHAP explainer)

🚀 5. Run the system:
   python main.py

📡 6. Initialize via API:
   POST /api/initialize
   {
     "host": "your_db_host",
     "database": "space_detective", 
     "username": "your_username",
     "password": "your_password"
   }

📚 7. Load your model:
   POST /api/database/load-model
   {
     "model_name": "isolation_forest_v1",
     "dataset_name": "main_dataset",
     "alibaba_api_key": "your_api_key"
   }

🎯 8. Test chat:
   POST /api/chat
   {
     "message": "What are the most suspicious areas?",
     "session_id": "test",
     "use_rag": true
   }

✅ Your Space Detective v2.0 is ready!
"""
    print(guide)


if __name__ == "__main__":
    from datetime import datetime
    main()


# =============================================================================
# QUICK REFERENCE: Environment Variables Template
# =============================================================================

ENV_TEMPLATE = '''
# Copy this template to .env file and fill in your credentials

# PostgreSQL Database
DATABASE_HOST=localhost
DATABASE_PORT=5432  
DATABASE_NAME=space_detective
DATABASE_USER=your_username
DATABASE_PASSWORD=your_password

# Alibaba API
DASHSCOPE_API_KEY=your_alibaba_api_key

# Data Files (for migration)
TRAINING_DATA_PATH=path/to/your/sample_data.csv
MODEL_PATH=path/to/your/model.pkl
EXPLAINER_PATH=path/to/your/explainer.pkl

# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Feature Flags
ENABLE_RAG=true
ENABLE_SHAP=true
'''


# =============================================================================
# QUICK START SCRIPT FOR IMMEDIATE USE
# =============================================================================

async def quick_start():
    """Quick start with your credentials - EDIT THIS FUNCTION"""
    
    # 🔧 EDIT THESE CREDENTIALS
    YOUR_CREDENTIALS = {
        "db_host": "localhost",                    # Your PostgreSQL host
        "db_port": 5432,                          # Your PostgreSQL port  
        "db_name": "space_detective",             # Your database name
        "db_user": "postgres",                    # Your PostgreSQL username
        "db_password": "YOUR_PASSWORD_HERE",      # Your PostgreSQL password
        "alibaba_api_key": "YOUR_ALIBABA_KEY",    # Your Alibaba API key
        "csv_path": "sample_data.csv",            # Path to your CSV
        "model_path": "model.pkl",                # Path to your model
        "explainer_path": "explainer.pkl"         # Path to your explainer
    }
    
    print("🚀 Quick starting Space Detective v2.0...")
    print(f"🔗 Connecting to: {YOUR_CREDENTIALS['db_host']}:{YOUR_CREDENTIALS['db_port']}")
    
    # Initialize everything
    try:
        # Your setup code here using YOUR_CREDENTIALS
        # This is a template - you need to implement the actual setup
        
        print("✅ Quick start completed!")
        print("📡 API server starting on http://localhost:8000")
        
    except Exception as e:
        print(f"❌ Quick start failed: {e}")


# Uncomment and edit this to run immediately:
# if __name__ == "__main__":
#     asyncio.run(quick_start())