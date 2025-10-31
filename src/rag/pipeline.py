"""
Complete RAG pipeline orchestration
"""
from typing import Dict, Any, List, Optional
from datetime import datetime

from src.rag.retriever import get_retriever
from src.rag.generator import get_generator
from src.database import SessionLocal, ConversationHistory
from src.utils import create_session_id
import json


class RAGPipeline:
    """Complete RAG pipeline for question answering"""
    
    def __init__(self):
        """Initialize the RAG pipeline"""
        self.retriever = get_retriever()
        self.generator = get_generator()
    
    def query(
        self, 
        question: str,
        session_id: Optional[str] = None,
        top_k: int = None,
        use_rerank: bool = True
    ) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline
        
        Args:
            question: User's question
            session_id: Session ID for conversation tracking
            top_k: Number of documents to retrieve
            use_rerank: Whether to use re-ranking
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not session_id:
            session_id = create_session_id()
        
        # Step 1: Retrieve relevant documents
        print(f"Retrieving documents for query: {question[:100]}...")
        
        if use_rerank:
            documents = self.retriever.retrieve_with_rerank(question, k=top_k)
        else:
            documents = self.retriever.retrieve(question, k=top_k)
        
        print(f"Retrieved {len(documents)} documents")
        
        # Step 2: Get conversation history
        conversation_history = self._get_conversation_history(session_id)
        
        # Step 3: Generate answer
        print("Generating answer...")
        result = self.generator.generate_answer(
            question, 
            documents,
            conversation_history
        )
        
        # Step 4: Save to conversation history
        self._save_conversation(
            session_id, 
            question, 
            result["answer"],
            result["sources"]
        )
        
        # Add session info to result
        result["session_id"] = session_id
        result["timestamp"] = datetime.utcnow().isoformat()
        result["documents_retrieved"] = len(documents)
        
        return result
    
    def query_streaming(
        self, 
        question: str,
        session_id: Optional[str] = None,
        top_k: int = None,
        use_rerank: bool = True
    ):
        """
        Process a question with streaming response
        
        Args:
            question: User's question
            session_id: Session ID for conversation tracking
            top_k: Number of documents to retrieve
            use_rerank: Whether to use re-ranking
            
        Yields:
            Chunks of the answer
        """
        if not session_id:
            session_id = create_session_id()
        
        # Retrieve documents
        if use_rerank:
            documents = self.retriever.retrieve_with_rerank(question, k=top_k)
        else:
            documents = self.retriever.retrieve(question, k=top_k)
        
        # Generate and stream answer
        full_answer = ""
        for chunk in self.generator.generate_streaming_answer(question, documents):
            full_answer += chunk
            yield chunk
        
        # Save conversation after streaming completes
        self._save_conversation(session_id, question, full_answer, documents)
    
    def _get_conversation_history(
        self, 
        session_id: str, 
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Get recent conversation history for a session
        
        Args:
            session_id: Session ID
            limit: Maximum number of previous turns to retrieve
            
        Returns:
            List of conversation turns
        """
        db = SessionLocal()
        try:
            history = db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).order_by(
                ConversationHistory.timestamp.desc()
            ).limit(limit).all()
            
            return [
                {
                    "question": conv.question,
                    "answer": conv.answer
                }
                for conv in reversed(history)
            ]
        finally:
            db.close()
    
    def _save_conversation(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ):
        """
        Save conversation to database
        
        Args:
            session_id: Session ID
            question: User's question
            answer: Generated answer
            sources: Source documents used
        """
        db = SessionLocal()
        try:
            # Extract document IDs from sources
            source_ids = [
                str(doc.get("metadata", {}).get("document_id", "unknown"))
                for doc in sources
            ]
            
            conversation = ConversationHistory(
                session_id=session_id,
                question=question,
                answer=answer,
                source_documents=json.dumps(source_ids),
                timestamp=datetime.utcnow()
            )
            
            db.add(conversation)
            db.commit()
        except Exception as e:
            print(f"Error saving conversation: {e}")
            db.rollback()
        finally:
            db.close()
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get full conversation history for a session
        
        Args:
            session_id: Session ID
            
        Returns:
            List of conversation entries
        """
        db = SessionLocal()
        try:
            history = db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).order_by(
                ConversationHistory.timestamp.asc()
            ).all()
            
            return [
                {
                    "question": conv.question,
                    "answer": conv.answer,
                    "timestamp": conv.timestamp.isoformat(),
                    "sources": json.loads(conv.source_documents)
                }
                for conv in history
            ]
        finally:
            db.close()
    
    def clear_session(self, session_id: str):
        """
        Clear conversation history for a session
        
        Args:
            session_id: Session ID
        """
        db = SessionLocal()
        try:
            db.query(ConversationHistory).filter(
                ConversationHistory.session_id == session_id
            ).delete()
            db.commit()
            print(f"Cleared history for session: {session_id}")
        except Exception as e:
            print(f"Error clearing session: {e}")
            db.rollback()
        finally:
            db.close()


# Global pipeline instance
_pipeline = None


def get_pipeline() -> RAGPipeline:
    """Get or create the global pipeline instance"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
