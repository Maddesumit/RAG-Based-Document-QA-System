"""
Answer generation module using OpenAI GPT
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

from src.config import settings


class AnswerGenerator:
    """Generate answers using OpenAI GPT with retrieved context"""
    
    def __init__(self):
        """Initialize the answer generator"""
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            openai_api_key=settings.openai_api_key
        )
        
        self.system_prompt = """You are a helpful AI assistant that answers questions based on the provided context.

Your responsibilities:
1. Answer questions accurately using ONLY the information from the provided context
2. If the context doesn't contain enough information to answer the question, say so clearly
3. Be concise but comprehensive in your answers
4. Cite specific parts of the context when relevant
5. If asked about something not in the context, acknowledge the limitation

Format your responses clearly and professionally."""
    
    def generate_answer(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]],
        conversation_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer based on query and context
        
        Args:
            query: User's question
            context_documents: Retrieved relevant documents
            conversation_history: Previous conversation turns
            
        Returns:
            Dictionary with answer and metadata
        """
        if not context_documents:
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try rephrasing or ask about something else.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Prepare context from documents
        context = self._format_context(context_documents)
        
        # Build prompt
        prompt = self._build_prompt(query, context, conversation_history)
        
        try:
            # Generate answer
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Calculate confidence based on relevance scores
            avg_relevance = sum(
                doc.get("relevance_score", 0) 
                for doc in context_documents
            ) / len(context_documents)
            
            return {
                "answer": answer,
                "sources": context_documents,
                "confidence": avg_relevance,
                "model": settings.openai_model
            }
        
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while generating the answer: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            
            context_part = f"[Document {i}"
            if "filename" in metadata:
                context_part += f" - {metadata['filename']}"
            context_part += f"]\n{text}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_prompt(
        self, 
        query: str, 
        context: str,
        conversation_history: List[Dict[str, str]] = None
    ) -> List:
        """
        Build the prompt for the LLM
        
        Args:
            query: User's question
            context: Formatted context from documents
            conversation_history: Previous conversation turns
            
        Returns:
            List of messages for the LLM
        """
        messages = [SystemMessage(content=self.system_prompt)]
        
        # Add conversation history if available
        if conversation_history:
            for turn in conversation_history[-3:]:  # Last 3 turns
                messages.append(HumanMessage(content=turn["question"]))
                messages.append(SystemMessage(content=turn["answer"]))
        
        # Add current query with context
        user_message = f"""Context information:
{context}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        messages.append(HumanMessage(content=user_message))
        
        return messages
    
    def generate_streaming_answer(
        self, 
        query: str, 
        context_documents: List[Dict[str, Any]]
    ):
        """
        Generate answer with streaming (for real-time display)
        
        Args:
            query: User's question
            context_documents: Retrieved relevant documents
            
        Yields:
            Chunks of generated text
        """
        if not context_documents:
            yield "I couldn't find any relevant information to answer your question."
            return
        
        context = self._format_context(context_documents)
        prompt = self._build_prompt(query, context)
        
        try:
            for chunk in self.llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"Error: {str(e)}"


# Global generator instance
_generator = None


def get_generator() -> AnswerGenerator:
    """Get or create the global generator instance"""
    global _generator
    if _generator is None:
        _generator = AnswerGenerator()
    return _generator
