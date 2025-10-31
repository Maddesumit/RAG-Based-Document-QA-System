"""
Database models and connection management for PostgreSQL with pgvector
"""
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from typing import Generator

from src.config import settings

Base = declarative_base()


class Document(Base):
    """Document model for storing document metadata"""
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    content_hash = Column(String(64), unique=True, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    file_type = Column(String(50))
    file_size = Column(Integer)
    total_chunks = Column(Integer)


class DocumentChunk(Base):
    """Document chunk model for storing text chunks with embeddings"""
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(ARRAY(Float), nullable=True)  # Store embeddings in PostgreSQL
    metadata = Column(Text)  # JSON string of metadata
    created_at = Column(DateTime, default=datetime.utcnow)


class ConversationHistory(Base):
    """Conversation history for tracking QA interactions"""
    __tablename__ = "conversation_history"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    source_documents = Column(Text)  # JSON string of source document IDs
    timestamp = Column(DateTime, default=datetime.utcnow)


# Database engine and session
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables"""
    try:
        # Enable pgvector extension
        with engine.connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
    except Exception as e:
        print(f"Note: Could not enable pgvector extension: {e}")
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database initialized successfully")


def get_db() -> Generator:
    """Dependency for getting database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
