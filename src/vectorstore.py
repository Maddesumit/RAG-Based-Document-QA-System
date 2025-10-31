"""
Vector store implementation using FAISS
"""
import os
import pickle
from typing import List, Tuple, Dict, Any
import numpy as np
import faiss

from src.config import settings
from src.embeddings import get_embedding_generator


class FAISSVectorStore:
    """FAISS-based vector store for semantic search"""
    
    def __init__(self, index_path: str = None):
        """
        Initialize FAISS vector store
        
        Args:
            index_path: Path to save/load FAISS index
        """
        self.index_path = index_path or settings.faiss_index_path
        self.embedding_generator = get_embedding_generator()
        self.dimension = self.embedding_generator.get_embedding_dimension()
        self.index = None
        self.documents = []  # Store document metadata
        self.doc_embeddings = []
        
        # Try to load existing index
        self.load_index()
    
    def create_index(self):
        """Create a new FAISS index"""
        # Using IndexFlatL2 for exact search (can be changed to IndexIVFFlat for faster approximate search)
        self.index = faiss.IndexFlatL2(self.dimension)
        print(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Add documents to the vector store
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dicts for each text
        """
        if self.index is None:
            self.create_index()
        
        # Generate embeddings
        print(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self.embedding_generator.embed_texts(texts)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store documents and metadata
        if metadatas is None:
            metadatas = [{}] * len(texts)
        
        for text, metadata, embedding in zip(texts, metadatas, embeddings):
            self.documents.append({
                "text": text,
                "metadata": metadata
            })
            self.doc_embeddings.append(embedding)
        
        print(f"Added {len(texts)} documents. Total documents: {len(self.documents)}")
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar documents
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of tuples (document_dict, distance)
        """
        if self.index is None or self.index.ntotal == 0:
            print("Warning: Index is empty")
            return []
        
        k = k or settings.top_k_results
        k = min(k, self.index.ntotal)  # Can't return more than we have
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distance)))
        
        return results
    
    def save_index(self):
        """Save FAISS index and documents to disk"""
        if self.index is None:
            print("No index to save")
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{self.index_path}.index")
        
        # Save documents and metadata
        with open(f"{self.index_path}.pkl", "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "doc_embeddings": self.doc_embeddings
            }, f)
        
        print(f"Saved index to {self.index_path}")
    
    def load_index(self):
        """Load FAISS index and documents from disk"""
        index_file = f"{self.index_path}.index"
        pkl_file = f"{self.index_path}.pkl"
        
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load documents and metadata
                with open(pkl_file, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data["documents"]
                    self.doc_embeddings = data.get("doc_embeddings", [])
                
                print(f"Loaded index from {self.index_path} with {len(self.documents)} documents")
            except Exception as e:
                print(f"Error loading index: {e}")
                self.create_index()
        else:
            print("No existing index found, will create new one when documents are added")
    
    def clear(self):
        """Clear the vector store"""
        self.create_index()
        self.documents = []
        self.doc_embeddings = []
        print("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_path": self.index_path
        }


# Global vector store instance
_vector_store = None


def get_vector_store() -> FAISSVectorStore:
    """Get or create the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore()
    return _vector_store
