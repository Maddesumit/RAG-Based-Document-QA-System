"""
Utility functions for document processing and text manipulation
"""
import hashlib
import os
from typing import List, Dict, Any
from pathlib import Path
import re


def calculate_file_hash(file_path: str) -> str:
    """
    Calculate SHA-256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename
    
    Args:
        filename: Name of the file
        
    Returns:
        File extension (without dot)
    """
    return Path(filename).suffix.lower().lstrip('.')


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)]', '', text)
    
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < text_length:
            # Look for sentence end within the last 200 chars of the chunk
            last_period = text.rfind('.', end - 200, end)
            last_question = text.rfind('?', end - 200, end)
            last_exclamation = text.rfind('!', end - 200, end)
            
            break_point = max(last_period, last_question, last_exclamation)
            if break_point > start:
                end = break_point + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - chunk_overlap
        
        # Prevent infinite loop
        if start <= 0 and len(chunks) > 0:
            break
    
    return chunks


def format_source_documents(documents: List[Dict[str, Any]]) -> str:
    """
    Format source documents for display
    
    Args:
        documents: List of document dicts
        
    Returns:
        Formatted string
    """
    if not documents:
        return "No source documents found."
    
    formatted = "📚 Source Documents:\n\n"
    for i, doc in enumerate(documents, 1):
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        formatted += f"{i}. "
        if "filename" in metadata:
            formatted += f"**{metadata['filename']}** "
        if "chunk_index" in metadata:
            formatted += f"(Chunk {metadata['chunk_index']}) "
        formatted += f"\n   {text[:200]}...\n\n"
    
    return formatted


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB
    """
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024 * 1024)


def validate_file_type(filename: str, allowed_extensions: List[str] = None) -> bool:
    """
    Validate if file type is allowed
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions
        
    Returns:
        True if valid, False otherwise
    """
    if allowed_extensions is None:
        allowed_extensions = ['pdf', 'txt', 'docx', 'doc', 'md', 'csv', 'xlsx', 'pptx']
    
    ext = get_file_extension(filename)
    return ext in allowed_extensions


def create_session_id() -> str:
    """
    Create a unique session ID
    
    Returns:
        Session ID string
    """
    import uuid
    return str(uuid.uuid4())
