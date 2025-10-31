"""
Document ingestion and processing module
"""
import os
from typing import List, Dict, Any
from pathlib import Path
import hashlib

# Document loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.config import settings
from src.utils import calculate_file_hash, get_file_extension, chunk_text
from src.vectorstore import get_vector_store
from src.database import SessionLocal, Document, DocumentChunk
from datetime import datetime


class DocumentIngestor:
    """Handle document ingestion and processing"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def load_document(self, file_path: str) -> List[str]:
        """
        Load document based on file type
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of text content
        """
        ext = get_file_extension(file_path)
        
        try:
            if ext == 'pdf':
                loader = PyPDFLoader(file_path)
            elif ext == 'txt':
                loader = TextLoader(file_path)
            elif ext in ['doc', 'docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            elif ext == 'md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif ext == 'csv':
                loader = CSVLoader(file_path)
            elif ext in ['xls', 'xlsx']:
                loader = UnstructuredExcelLoader(file_path)
            elif ext in ['ppt', 'pptx']:
                loader = UnstructuredPowerPointLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
            
            documents = loader.load()
            return [doc.page_content for doc in documents]
        
        except Exception as e:
            print(f"Error loading document {file_path}: {e}")
            # Fallback to text loader
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return [f.read()]
            except Exception as e2:
                print(f"Fallback failed: {e2}")
                return []
    
    def process_document(
        self, 
        file_path: str, 
        filename: str = None
    ) -> Dict[str, Any]:
        """
        Process a document: load, chunk, embed, and store
        
        Args:
            file_path: Path to the document
            filename: Original filename (if different from file_path)
            
        Returns:
            Dictionary with processing results
        """
        if filename is None:
            filename = os.path.basename(file_path)
        
        print(f"Processing document: {filename}")
        
        # Calculate file hash
        file_hash = calculate_file_hash(file_path)
        file_size = os.path.getsize(file_path)
        
        # Check if document already exists
        db = SessionLocal()
        try:
            existing_doc = db.query(Document).filter(
                Document.content_hash == file_hash
            ).first()
            
            if existing_doc:
                print(f"Document {filename} already exists in database")
                return {
                    "status": "exists",
                    "document_id": existing_doc.id,
                    "filename": existing_doc.filename,
                    "message": "Document already processed"
                }
            
            # Load document
            texts = self.load_document(file_path)
            if not texts:
                return {
                    "status": "error",
                    "message": "Failed to load document content"
                }
            
            # Combine all text
            full_text = "\n\n".join(texts)
            
            # Split into chunks using LangChain's text splitter
            chunks = self.text_splitter.split_text(full_text)
            print(f"Split document into {len(chunks)} chunks")
            
            # Create document record
            doc_record = Document(
                filename=filename,
                file_path=file_path,
                content_hash=file_hash,
                upload_date=datetime.utcnow(),
                file_type=get_file_extension(filename),
                file_size=file_size,
                total_chunks=len(chunks)
            )
            db.add(doc_record)
            db.commit()
            db.refresh(doc_record)
            
            # Prepare metadata for each chunk
            metadatas = []
            chunk_records = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": doc_record.id,
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": doc_record.file_type
                }
                metadatas.append(metadata)
                
                # Create chunk record
                chunk_record = DocumentChunk(
                    document_id=doc_record.id,
                    chunk_index=i,
                    chunk_text=chunk,
                    metadata=str(metadata)
                )
                chunk_records.append(chunk_record)
            
            # Add chunks to database
            db.bulk_save_objects(chunk_records)
            db.commit()
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadatas)
            self.vector_store.save_index()
            
            print(f"Successfully processed document: {filename}")
            
            return {
                "status": "success",
                "document_id": doc_record.id,
                "filename": filename,
                "chunks": len(chunks),
                "message": f"Document processed successfully with {len(chunks)} chunks"
            }
        
        except Exception as e:
            db.rollback()
            print(f"Error processing document: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            db.close()
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of processing results
        """
        results = []
        directory = Path(directory_path)
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                result = self.process_document(str(file_path))
                results.append(result)
        
        return results
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about indexed documents"""
        db = SessionLocal()
        try:
            total_docs = db.query(Document).count()
            total_chunks = db.query(DocumentChunk).count()
            vector_stats = self.vector_store.get_stats()
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "vector_store": vector_stats
            }
        finally:
            db.close()


# Global ingestor instance
_ingestor = None


def get_ingestor() -> DocumentIngestor:
    """Get or create the global ingestor instance"""
    global _ingestor
    if _ingestor is None:
        _ingestor = DocumentIngestor()
    return _ingestor
