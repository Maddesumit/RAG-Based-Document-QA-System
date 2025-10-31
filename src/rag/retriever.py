"""
Document retrieval module for semantic search
"""
from typing import List, Dict, Any, Tuple
from src.vectorstore import get_vector_store
from src.config import settings


class DocumentRetriever:
    """Retrieve relevant documents for a query"""
    
    def __init__(self, top_k: int = None):
        """
        Initialize document retriever
        
        Args:
            top_k: Number of documents to retrieve
        """
        self.vector_store = get_vector_store()
        self.top_k = top_k or settings.top_k_results
    
    def retrieve(
        self, 
        query: str, 
        k: int = None,
        filter_metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            k: Number of documents to retrieve (overrides default)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of relevant documents with metadata
        """
        k = k or self.top_k
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query, k=k)
        
        # Format results
        documents = []
        for doc, distance in results:
            doc_dict = {
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(distance),
                "relevance_score": self._calculate_relevance(distance)
            }
            
            # Apply metadata filters if provided
            if filter_metadata:
                if all(
                    doc["metadata"].get(key) == value 
                    for key, value in filter_metadata.items()
                ):
                    documents.append(doc_dict)
            else:
                documents.append(doc_dict)
        
        return documents
    
    def _calculate_relevance(self, distance: float) -> float:
        """
        Convert distance to relevance score (0-1, higher is better)
        
        Args:
            distance: L2 distance from FAISS
            
        Returns:
            Relevance score
        """
        # Simple conversion: smaller distance = higher relevance
        # Using exponential decay
        import math
        relevance = math.exp(-distance / 10.0)
        return min(max(relevance, 0.0), 1.0)
    
    def retrieve_with_rerank(
        self, 
        query: str, 
        k: int = None,
        rerank_top_n: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with re-ranking
        
        Args:
            query: Search query
            k: Initial number of documents to retrieve
            rerank_top_n: Number of documents to return after re-ranking
            
        Returns:
            List of re-ranked documents
        """
        k = k or self.top_k * 2  # Retrieve more for re-ranking
        rerank_top_n = rerank_top_n or self.top_k
        
        # Initial retrieval
        documents = self.retrieve(query, k=k)
        
        # Simple re-ranking based on keyword matching (can be enhanced with cross-encoder)
        query_terms = set(query.lower().split())
        
        for doc in documents:
            text_terms = set(doc["text"].lower().split())
            overlap = len(query_terms & text_terms)
            doc["keyword_score"] = overlap / len(query_terms) if query_terms else 0
            
            # Combined score
            doc["combined_score"] = (
                doc["relevance_score"] * 0.7 + 
                doc["keyword_score"] * 0.3
            )
        
        # Sort by combined score
        documents.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return documents[:rerank_top_n]


# Global retriever instance
_retriever = None


def get_retriever() -> DocumentRetriever:
    """Get or create the global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = DocumentRetriever()
    return _retriever
