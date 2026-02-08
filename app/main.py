import uvicorn

if __name__ == "__main__":
    print("="*60)
    print("  STARTING FASTAPI SERVER")
    print("="*60)
    print("\n📍 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop\n")
    print("="*60)
    
    uvicorn.run(
        "api:app",  # Changed: use import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=True
    )