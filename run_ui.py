"""
Run this from the production-rag/ folder:

    python run_ui.py

Then open your browser at:  http://localhost:8000
"""
import uvicorn

if __name__ == "__main__":
    print("\n🚀 Starting AskMyDocs UI...")
    print("   Open your browser at: http://localhost:8000\n")
    uvicorn.run("ui.server:app", host="0.0.0.0", port=8000, reload=True)
