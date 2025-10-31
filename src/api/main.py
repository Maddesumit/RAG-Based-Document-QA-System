"""
FastAPI application for RAG Document QA System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
import shutil
import os
from pathlib import Path

from src.config import settings
from src.database import init_db, get_db
from src.rag import get_ingestor, get_pipeline
from src.api.routes import router

# Initialize FastAPI app
app = FastAPI(
    title="RAG Document QA System",
    description="Retrieval-Augmented Generation system for document-based question answering",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("Starting RAG Document QA System...")
    
    # Initialize database
    init_db()
    print("Database initialized")
    
    # Ensure data directories exist
    settings.docs_dir.mkdir(parents=True, exist_ok=True)
    settings.indexes_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize RAG components
    get_ingestor()
    get_pipeline()
    
    print(f"Server starting on {settings.api_host}:{settings.api_port}")
    print("API documentation available at http://localhost:8000/docs")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RAG Document QA System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check vector store
        from src.vectorstore import get_vector_store
        vector_store = get_vector_store()
        vector_stats = vector_store.get_stats()
        
        return {
            "status": "healthy",
            "database": "connected",
            "vector_store": vector_stats
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
