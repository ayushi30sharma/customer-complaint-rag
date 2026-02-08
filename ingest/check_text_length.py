from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sys
import os

# Add parent directory to path to import from rag/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGPipeline

# ============================================
# INITIALIZE FASTAPI APP
# ============================================

app = FastAPI(
    title="Customer Complaint RAG API",
    description="AI-powered customer complaint analysis system",
    version="1.0.0"
)

# ============================================
# CORS MIDDLEWARE
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# GLOBAL VARIABLES
# ============================================

rag_pipeline = None

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class QueryRequest(BaseModel):
    """Request model for asking questions"""
    query: str
    top_k: Optional[int] = 5
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "How do you handle damaged products?",
                "top_k": 5,
                "max_tokens": 500,
                "temperature": 0.7
            }
        }

class Source(BaseModel):
    """Model for a single source document"""
    id: str
    text: str
    source: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    """Response model for query answers"""
    query: str
    answer: str
    sources: List[Source]
    num_sources: int

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    rag_loaded: bool = False

# ============================================
# STARTUP EVENT - Load RAG Pipeline
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup"""
    global rag_pipeline
    
    print("\n" + "="*60)
    print("  INITIALIZING RAG PIPELINE")
    print("="*60)
    
    try:
        # Initialize RAG Pipeline
        rag_pipeline = RAGPipeline()
        rag_pipeline.load()
        
        print("\n" + "="*60)
        print("✅ RAG PIPELINE READY!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error loading RAG pipeline: {e}\n")
        print("⚠️  API will start but /query endpoint will not work")
        print("="*60 + "\n")

# ============================================
# ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Customer Complaint RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if rag_pipeline is not None else "degraded",
        "message": "RAG pipeline loaded" if rag_pipeline is not None else "RAG pipeline not loaded",
        "rag_loaded": rag_pipeline is not None
    }

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_endpoint(request: QueryRequest):
    """
    Answer a question using RAG
    
    Args:
        request (QueryRequest): Query with parameters
    
    Returns:
        QueryResponse: Answer with sources
    """
    # Check if RAG is loaded
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Please check server logs."
        )
    
    try:
        # Query the RAG pipeline
        response = rag_pipeline.query(
            user_query=request.query,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            verbose=False  # Don't print to console in API mode
        )
        
        # Format sources for API response
        formatted_sources = [
            {
                "id": src.get('id', 'unknown'),
                "text": src.get('text', '')[:200] + '...',  # Truncate long text
                "source": src.get('source', 'unknown'),
                "score": src.get('score')
            }
            for src in response['sources']
        ]
        
        return {
            "query": response['query'],
            "answer": response['answer'],
            "sources": formatted_sources,
            "num_sources": response['num_sources']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )