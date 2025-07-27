# simple_chat_setup.py
"""
Simple setup script for Space Detective Chat API
Edit the file paths below and run this script
"""

import asyncio
from chat_api import ChatAPI

async def run_chat():
    """Run the chat API with your data"""
    
    # 🔧 EDIT THESE PATHS TO YOUR ACTUAL FILES
    CSV_PATH = "sample_data.csv"           # Your training data CSV
    MODEL_PATH = "model.pkl"               # Your trained model PKL
    EXPLAINER_PATH = "explainer.pkl"       # Your SHAP explainer PKL
    ALIBABA_API_KEY = None                 # Optional: Your Alibaba API key
    
    print("🛰️ Starting Space Detective Chat API...")
    
    # Initialize
    chat_api = ChatAPI(CSV_PATH, MODEL_PATH, EXPLAINER_PATH, ALIBABA_API_KEY)
    
    if not chat_api.initialize():
        print("❌ Failed to load data files. Please check your paths:")
        print(f"  CSV: {CSV_PATH}")
        print(f"  Model: {MODEL_PATH}")
        print(f"  Explainer: {EXPLAINER_PATH}")
        return
    
    print("✅ Chat API ready!")
    
    # Interactive chat loop
    while True:
        try:
            user_input = input("\n💬 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            response = await chat_api.chat(user_input)
            print(f"🤖 Assistant: {response['response']}")
            
        except KeyboardInterrupt:
            print("\n👋 Chat ended by user")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(run_chat())
