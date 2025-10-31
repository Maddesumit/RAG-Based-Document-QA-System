"""
API routes for RAG Document QA System
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import StreamingResponse
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os
from pathlib import Path
import json

from src.config import settings
from src.rag import get_ingestor, get_pipeline
from src.database import get_db, SessionLocal, Document, ConversationHistory
from src.utils import create_session_id, validate_file_type

router = APIRouter()


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None
    use_rerank: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    session_id: str
    timestamp: str
    documents_retrieved: int


class DocumentInfo(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int
    total_chunks: int
    upload_date: str


class SessionInfo(BaseModel):
    session_id: str
    history: List[dict]


# Document Upload Endpoints
@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document
    
    - **file**: Document file to upload (PDF, TXT, DOCX, etc.)
    """
    try:
        # Validate file type
        if not validate_file_type(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: PDF, TXT, DOCX, MD, CSV, XLSX, PPTX"
            )
        
        # Save uploaded file
        file_path = settings.docs_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process document
        ingestor = get_ingestor()
        result = ingestor.process_document(str(file_path), file.filename)
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["message"])
        
        return {
            "status": "success",
            "message": result["message"],
            "document_id": result.get("document_id"),
            "filename": file.filename,
            "chunks": result.get("chunks", 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/upload-multiple")
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """
    Upload and process multiple documents
    
    - **files**: List of document files to upload
    """
    results = []
    
    for file in files:
        try:
            # Validate file type
            if not validate_file_type(file.filename):
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "Unsupported file type"
                })
                continue
            
            # Save uploaded file
            file_path = settings.docs_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process document
            ingestor = get_ingestor()
            result = ingestor.process_document(str(file_path), file.filename)
            results.append(result)
        
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(e)
            })
    
    return {"results": results}


@router.get("/documents")
async def list_documents():
    """
    List all uploaded documents
    """
    db = SessionLocal()
    try:
        documents = db.query(Document).all()
        return {
            "total": len(documents),
            "documents": [
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "file_size": doc.file_size,
                    "total_chunks": doc.total_chunks,
                    "upload_date": doc.upload_date.isoformat()
                }
                for doc in documents
            ]
        }
    finally:
        db.close()


@router.get("/documents/{document_id}")
async def get_document(document_id: int):
    """
    Get details of a specific document
    
    - **document_id**: Document ID
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "filename": document.filename,
            "file_type": document.file_type,
            "file_size": document.file_size,
            "total_chunks": document.total_chunks,
            "upload_date": document.upload_date.isoformat(),
            "file_path": document.file_path
        }
    finally:
        db.close()


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """
    Delete a document
    
    - **document_id**: Document ID
    """
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete from database
        db.delete(document)
        db.commit()
        
        # Note: In production, you'd also want to rebuild the vector index
        # without this document's chunks
        
        return {"status": "success", "message": f"Document {document_id} deleted"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@router.get("/documents/stats")
async def get_stats():
    """
    Get statistics about indexed documents
    """
    ingestor = get_ingestor()
    stats = ingestor.get_document_stats()
    return stats


# Query Endpoints
@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents with a question
    
    - **question**: The question to ask
    - **session_id**: Optional session ID for conversation tracking
    - **top_k**: Number of documents to retrieve (default: 5)
    - **use_rerank**: Whether to use document re-ranking (default: true)
    """
    try:
        pipeline = get_pipeline()
        result = pipeline.query(
            question=request.question,
            session_id=request.session_id,
            top_k=request.top_k,
            use_rerank=request.use_rerank
        )
        
        return QueryResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_documents_stream(request: QueryRequest):
    """
    Query documents with streaming response
    
    - **question**: The question to ask
    - **session_id**: Optional session ID for conversation tracking
    - **top_k**: Number of documents to retrieve
    - **use_rerank**: Whether to use document re-ranking
    """
    try:
        pipeline = get_pipeline()
        
        def generate():
            for chunk in pipeline.query_streaming(
                question=request.question,
                session_id=request.session_id,
                top_k=request.top_k,
                use_rerank=request.use_rerank
            ):
                yield chunk
        
        return StreamingResponse(generate(), media_type="text/plain")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Session Management Endpoints
@router.post("/sessions/new")
async def create_session():
    """
    Create a new conversation session
    """
    session_id = create_session_id()
    return {"session_id": session_id}


@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """
    Get conversation history for a session
    
    - **session_id**: Session ID
    """
    try:
        pipeline = get_pipeline()
        history = pipeline.get_session_history(session_id)
        
        return {
            "session_id": session_id,
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """
    Clear conversation history for a session
    
    - **session_id**: Session ID
    """
    try:
        pipeline = get_pipeline()
        pipeline.clear_session(session_id)
        
        return {
            "status": "success",
            "message": f"Session {session_id} cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Index Management
@router.post("/index/rebuild")
async def rebuild_index():
    """
    Rebuild the vector index from all documents in database
    """
    try:
        from src.vectorstore import get_vector_store
        from src.database import DocumentChunk
        
        db = SessionLocal()
        vector_store = get_vector_store()
        
        # Clear existing index
        vector_store.clear()
        
        # Get all chunks
        chunks = db.query(DocumentChunk).all()
        
        if not chunks:
            return {"status": "success", "message": "No documents to index"}
        
        # Prepare texts and metadata
        texts = [chunk.chunk_text for chunk in chunks]
        metadatas = [json.loads(chunk.metadata) for chunk in chunks]
        
        # Rebuild index
        vector_store.add_documents(texts, metadatas)
        vector_store.save_index()
        
        db.close()
        
        return {
            "status": "success",
            "message": f"Index rebuilt with {len(chunks)} chunks"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
