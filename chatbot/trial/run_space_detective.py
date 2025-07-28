# run_space_detective.py
"""
Complete Run Guide for Space Detective v2.0 Modular System
This script helps you understand and run all components
"""

import os
import sys
import asyncio
import subprocess
from pathlib import Path

def print_project_structure():
    """Show the complete project structure"""
    
    structure = """
📁 SPACE DETECTIVE v2.0 - PROJECT STRUCTURE
=" * 60

📦 space-detective-v2/
├── 🔧 config.py                    # Configuration management
├── 🗄️ database_manager.py         # PostgreSQL operations  
├── 📚 rag_system.py               # RAG & vector search
├── 🤖 chat_api.py                 # AI chat interface
├── 🚀 main.py                     # Main FastAPI application
├── 📊 database_setup.py           # Database initialization
├── 🔄 data_integration.py         # Data migration tools
├── ⚙️ setup_space_detective.py    # Setup wizard
├── 🏃 run_space_detective.py      # This file - run guide
├── 📋 requirements.txt            # Dependencies
├── 🐳 docker-compose.yml          # Docker setup
├── 📄 .env                        # Environment config
├── 📊 hybrid_data_loader.py       # Auto-generated loader
├── 📝 README.md                   # Documentation
│
├── 📂 data/                       # Your data files
│   ├── sample_data.csv           # Training data
│   ├── model.pkl                 # Trained model
│   └── explainer.pkl             # SHAP explainer
│
├── 📂 logs/                       # Application logs
└── 📂 tests/                      # Unit tests

🔗 DEPENDENCIES BETWEEN MODULES:
config.py → All modules (provides configuration)
database_manager.py → rag_system.py, chat_api.py  
rag_system.py → chat_api.py
chat_api.py → main.py
main.py → All modules (orchestrates everything)
"""
    print(structure)


def check_requirements():
    """Check if all required files exist"""
    
    required_files = {
        "config.py": "📋 Configuration management",
        "database_manager.py": "🗄️ Database operations", 
        "rag_system.py": "📚 RAG system",
        "chat_api.py": "🤖 Chat API",
        "main.py": "🚀 Main application",
        "requirements.txt": "📦 Dependencies",
        "setup_space_detective.py": "⚙️ Setup wizard"
    }
    
    print("🔍 CHECKING PROJECT FILES:")
    print("-" * 40)
    
    missing_files = []
    for file_name, description in required_files.items():
        if os.path.exists(file_name):
            print(f"✅ {description}: {file_name}")
        else:
            print(f"❌ {description}: {file_name} (MISSING)")
            missing_files.append(file_name)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {missing_files}")
        print("📥 Please create these files from our previous conversation")
        return False
    else:
        print(f"\n✅ All required files present!")
        return True


def check_dependencies():
    """Check if Python dependencies are installed"""
    
    required_packages = [
        "fastapi", "uvicorn", "asyncpg", "pandas", "numpy", 
        "scikit-learn", "sentence-transformers", "h3-py",
        "shap", "plotly", "psycopg2-binary", "pydantic",
        "sqlalchemy", "tiktoken", "aiofiles"
    ]
    
    print("\n🐍 CHECKING PYTHON DEPENDENCIES:")
    print("-" * 40)
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (not installed)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n✅ All dependencies installed!")
        return True


def create_requirements_txt():
    """Create requirements.txt if missing"""
    
    requirements = """# Space Detective v2.0 Requirements

# FastAPI and web framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6

# Database
asyncpg==0.29.0
psycopg2-binary==2.9.9
sqlalchemy[asyncio]==2.0.23

# Vector database and embeddings
sentence-transformers==2.2.2
transformers==4.36.2

# Machine Learning
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.4
joblib==1.3.2
shap==0.43.0

# Geospatial
h3-py==3.7.6
folium==0.15.1

# Visualization
plotly==5.17.0
matplotlib==3.8.2
seaborn==0.13.0

# Text processing
tiktoken==0.5.2
regex==2023.10.3

# Alibaba Cloud (optional)
dashscope==1.14.1

# Utilities
pydantic==2.5.2
python-dotenv==1.0.0
aiofiles==23.2.1

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("📦 requirements.txt created!")


async def run_step_by_step():
    """Run Space Detective step by step"""
    
    print("🚀 RUNNING SPACE DETECTIVE v2.0 STEP BY STEP")
    print("=" * 60)
    
    # Step 1: Check project structure
    print("\n📋 STEP 1: Checking project structure...")
    if not check_requirements():
        print("❌ Please create missing files first!")
        return False
    
    # Step 2: Check dependencies
    print("\n📦 STEP 2: Checking dependencies...")
    if not check_dependencies():
        print("❌ Please install missing dependencies first!")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    # Step 3: Check configuration
    print("\n⚙️ STEP 3: Checking configuration...")
    if not os.path.exists(".env"):
        print("❌ .env file not found!")
        print("💡 Run setup first: python setup_space_detective.py")
        return False
    else:
        print("✅ .env file exists")
    
    # Step 4: Initialize database (optional)
    print("\n🗄️ STEP 4: Database setup...")
    choice = input("Initialize database tables? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            print("🔄 Initializing database...")
            # Import and run database setup
            from database_setup import DatabaseSetup
            from config import DatabaseConfig
            
            # You can adjust these based on your .env
            db_config = DatabaseConfig()
            db_setup = DatabaseSetup(
                host=db_config.host,
                port=db_config.port,
                database=db_config.database,
                username=db_config.username,
                password=db_config.password
            )
            
            success = await db_setup.setup_database(include_sample_data=False)
            if success:
                print("✅ Database initialized successfully!")
            else:
                print("❌ Database initialization failed!")
                return False
                
        except Exception as e:
            print(f"❌ Database setup error: {e}")
            print("💡 You can skip this and run manually later")
    
    # Step 5: Data migration (optional)
    print("\n📊 STEP 5: Data migration...")
    choice = input("Migrate your data to database? (y/n): ").strip().lower()
    if choice == 'y':
        try:
            print("🔄 Migrating data...")
            from data_integration import DataIntegrator
            from config import AppConfig
            
            config = AppConfig()
            integrator = DataIntegrator(config)
            
            # Load paths from environment or ask user
            csv_path = os.getenv("TRAINING_DATA_PATH") or input("CSV file path: ").strip()
            model_path = os.getenv("MODEL_PATH") or input("Model PKL path: ").strip()
            explainer_path = os.getenv("EXPLAINER_PATH") or input("Explainer PKL path: ").strip()
            
            await integrator.initialize_database()
            result = await integrator.migrate_to_database(csv_path, model_path, explainer_path)
            
            if result.get("summary", {}).get("migration_completed"):
                print("✅ Data migration completed!")
            else:
                print("❌ Data migration failed!")
                
        except Exception as e:
            print(f"❌ Migration error: {e}")
            print("💡 You can use file-based loading instead")
    
    # Step 6: Start the application
    print("\n🚀 STEP 6: Starting Space Detective API...")
    
    print("📡 Starting FastAPI server...")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API docs at: http://localhost:8000/docs")
    print("💬 Chat interface at: http://localhost:8000/")
    
    print("\n🎯 Starting server...")
    print("📋 Use Ctrl+C to stop the server")
    
    # Start the main application
    try:
        import uvicorn
        from main import app
        
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False
    
    return True


def run_individual_components():
    """Run individual components for testing"""
    
    print("🧪 TESTING INDIVIDUAL COMPONENTS")
    print("=" * 40)
    
    components = {
        "1": ("📋 Test Config", test_config),
        "2": ("🗄️ Test Database Manager", test_database_manager), 
        "3": ("📚 Test RAG System", test_rag_system),
        "4": ("🤖 Test Chat API", test_chat_api),
        "5": ("🔄 Test Data Integration", test_data_integration),
        "6": ("🚀 Run Full Application", lambda: asyncio.run(run_step_by_step()))
    }
    
    while True:
        print("\nChoose component to test:")
        for key, (name, _) in components.items():
            print(f"{key}. {name}")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "0":
            break
        elif choice in components:
            name, func = components[choice]
            print(f"\n🧪 Running: {name}")
            try:
                if asyncio.iscoroutinefunction(func):
                    asyncio.run(func())
                else:
                    func()
            except Exception as e:
                print(f"❌ Test failed: {e}")
        else:
            print("❌ Invalid choice!")


def test_config():
    """Test configuration module"""
    try:
        from config import AppConfig
        config = AppConfig()
        print(f"✅ Config loaded successfully")
        print(f"📊 Database: {config.database.host}:{config.database.port}")
        print(f"🤖 API: {config.api.host}:{config.api.port}")
        print(f"📚 RAG: {config.rag.embedding_model}")
    except Exception as e:
        print(f"❌ Config test failed: {e}")


async def test_database_manager():
    """Test database manager"""
    try:
        from database_manager import create_database_manager
        from config import DatabaseConfig
        
        db_config = DatabaseConfig()
        db_manager = await create_database_manager(db_config)
        
        # Test health check
        health = await db_manager.health_check()
        print(f"✅ Database manager created")
        print(f"📊 Health check: {health}")
        
        # Test statistics
        stats = await db_manager.get_system_stats()
        print(f"📈 System stats: {stats}")
        
        await db_manager.close()
        
    except Exception as e:
        print(f"❌ Database manager test failed: {e}")


async def test_rag_system():
    """Test RAG system"""
    try:
        from rag_system import create_rag_system
        from config import AppConfig
        
        config = AppConfig()
        rag_system = await create_rag_system(config.rag, config.database)
        
        print(f"✅ RAG system created")
        print(f"📚 Initialized: {rag_system.initialized}")
        
        # Test document upload
        result = await rag_system.add_document(
            title="Test Document",
            content="This is a test document for the RAG system.",
            document_type="test"
        )
        print(f"📄 Document upload: {result['success']}")
        
        # Test search
        results = await rag_system.search_documents("test document")
        print(f"🔍 Search results: {len(results)}")
        
        await rag_system.close()
        
    except Exception as e:
        print(f"❌ RAG system test failed: {e}")


async def test_chat_api():
    """Test chat API"""
    try:
        from chat_api import create_chat_api
        from database_manager import create_database_manager
        from rag_system import create_rag_system
        from config import AppConfig
        
        config = AppConfig()
        
        # Create components
        db_manager = await create_database_manager(config.database)
        rag_system = await create_rag_system(config.rag, config.database)
        chat_api = await create_chat_api(db_manager, rag_system, config)
        
        print(f"✅ Chat API created")
        
        # Test chat
        response = await chat_api.process_message(
            "Hello, how can you help me?",
            session_id="test_session"
        )
        
        print(f"💬 Chat response: {response['response'][:100]}...")
        print(f"🎯 Intent: {response.get('analysis', {}).get('primary_intent')}")
        
        # Cleanup
        await db_manager.close()
        await rag_system.close()
        
    except Exception as e:
        print(f"❌ Chat API test failed: {e}")


async def test_data_integration():
    """Test data integration"""
    try:
        from data_integration import DataIntegrator
        from config import AppConfig
        
        config = AppConfig()
        integrator = DataIntegrator(config)
        
        print(f"✅ Data integrator created")
        
        # Test file analysis (if sample_data.csv exists)
        if os.path.exists("sample_data.csv"):
            analysis = integrator.analyze_training_data("sample_data.csv")
            print(f"📊 Data analysis: {analysis.get('shape')}")
            print(f"🎯 Anomaly rate: {analysis.get('anomaly_statistics', {}).get('anomaly_percentage', 0):.2f}%")
        else:
            print("📄 No sample_data.csv found for analysis")
        
    except Exception as e:
        print(f"❌ Data integration test failed: {e}")


def show_quick_commands():
    """Show quick commands for common tasks"""
    
    commands = """
🚀 QUICK COMMANDS FOR SPACE DETECTIVE v2.0
=" * 50

📦 SETUP & INSTALLATION:
pip install -r requirements.txt              # Install dependencies
python setup_space_detective.py             # Interactive setup wizard
python database_setup.py                    # Initialize database tables

🗄️ DATABASE OPERATIONS:
python -c "from database_manager import *"   # Test database connection
python data_integration.py                  # Migrate data to database

🚀 RUN APPLICATION:
python main.py                              # Start FastAPI server
python run_space_detective.py              # This guided run script
uvicorn main:app --reload --port 8000      # Alternative server start

🧪 TESTING:
python -m pytest tests/                     # Run unit tests
python test_components.py                   # Test individual modules

🐳 DOCKER (Alternative):
docker-compose up -d postgres               # Start PostgreSQL only
docker-compose up                           # Start full stack

📡 API TESTING:
curl http://localhost:8000/                 # Test root endpoint
curl http://localhost:8000/api/status       # Check system status
curl -X POST http://localhost:8000/api/chat # Test chat API

🔧 DEVELOPMENT:
python -c "from config import *; print(AppConfig())"  # Test config
python -m black .                           # Format code
python -m flake8 .                         # Lint code

📚 DOCUMENTATION:
http://localhost:8000/docs                  # Interactive API docs (Swagger)
http://localhost:8000/redoc                # Alternative API docs (ReDoc)
"""
    print(commands)


def main():
    """Main function"""
    
    print("🛰️ SPACE DETECTIVE v2.0 - RUN GUIDE")
    print("=" * 50)
    
    options = {
        "1": ("📋 Show project structure", print_project_structure),
        "2": ("🔍 Check requirements", lambda: check_requirements() and check_dependencies()),
        "3": ("📦 Create requirements.txt", create_requirements_txt),
        "4": ("🚀 Run step-by-step setup", lambda: asyncio.run(run_step_by_step())),
        "5": ("🧪 Test individual components", run_individual_components),
        "6": ("⚡ Show quick commands", show_quick_commands),
        "7": ("🚀 Start application directly", lambda: os.system("python main.py"))
    }
    
    while True:
        print("\n🎯 What would you like to do?")
        for key, (name, _) in options.items():
            print(f"{key}. {name}")
        print("0. Exit")
        
        choice = input("\nEnter your choice: ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice in options:
            name, func = options[choice]
            print(f"\n{name}")
            print("-" * 40)
            try:
                func()
            except KeyboardInterrupt:
                print("\n⏹️ Operation cancelled by user")
            except Exception as e:
                print(f"❌ Error: {e}")
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()


# =============================================================================
# DEPLOYMENT CHECKLIST
# =============================================================================

DEPLOYMENT_CHECKLIST = """
✅ PRODUCTION DEPLOYMENT CHECKLIST
=" * 50

🔧 ENVIRONMENT SETUP:
□ PostgreSQL server running and accessible
□ Database 'space_detective' created
□ All environment variables set in .env
□ Alibaba API key configured
□ Python 3.9+ installed with all dependencies

📊 DATA PREPARATION:
□ Training data CSV file ready
□ ML model PKL file ready
□ SHAP explainer PKL file ready
□ Data migration completed successfully
□ Database tables created and populated

🚀 APPLICATION SETUP:
□ All modular files in place (config.py, database_manager.py, etc.)
□ Dependencies installed (pip install -r requirements.txt)
□ Configuration validated
□ Database connection tested
□ RAG system initialized

🧪 TESTING:
□ Individual component tests passed
□ API endpoints responding
□ Chat functionality working
□ Database queries successful
□ SHAP analysis working

🔒 SECURITY:
□ Database credentials secured
□ API keys protected
□ CORS configured properly
□ Rate limiting enabled (if needed)
□ Logs configured for monitoring

🌐 DEPLOYMENT:
□ Server firewall configured
□ Port 8000 accessible
□ SSL certificate (if HTTPS needed)
□ Process manager (PM2, systemd, etc.)
□ Monitoring and alerting setup

📈 POST-DEPLOYMENT:
□ API documentation accessible (/docs)
□ Health checks responding (/api/health)
□ Chat interface working
□ Database performance monitored
□ Logs being captured

🎯 READY FOR PRODUCTION! 🚀
"""

print(DEPLOYMENT_CHECKLIST)