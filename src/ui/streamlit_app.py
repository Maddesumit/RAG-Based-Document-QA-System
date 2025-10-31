"""
Streamlit UI for RAG Document QA System
"""
import streamlit as st
import requests
import os
from pathlib import Path
import time
from typing import List
import json

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")

# Page configuration
st.set_page_config(
    page_title="RAG Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-high {
        color: green;
        font-weight: bold;
    }
    .confidence-medium {
        color: orange;
        font-weight: bold;
    }
    .confidence-low {
        color: red;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        response = requests.post(f"{API_BASE_URL}/sessions/new")
        st.session_state.session_id = response.json()["session_id"]
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []


def upload_documents(files):
    """Upload documents to the API"""
    files_data = []
    for file in files:
        files_data.append(("files", (file.name, file, file.type)))
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/documents/upload-multiple",
            files=files_data
        )
        return response.json()
    except Exception as e:
        st.error(f"Error uploading documents: {e}")
        return None


def upload_single_document(file):
    """Upload a single document to the API"""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(
            f"{API_BASE_URL}/documents/upload",
            files=files
        )
        return response.json()
    except Exception as e:
        st.error(f"Error uploading document: {e}")
        return None


def query_documents(question: str, top_k: int = 5, use_rerank: bool = True):
    """Query documents"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={
                "question": question,
                "session_id": st.session_state.session_id,
                "top_k": top_k,
                "use_rerank": use_rerank
            }
        )
        return response.json()
    except Exception as e:
        st.error(f"Error querying documents: {e}")
        return None


def get_document_stats():
    """Get document statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/stats")
        return response.json()
    except Exception as e:
        st.error(f"Error getting stats: {e}")
        return None


def list_documents():
    """List all documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents")
        return response.json()
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        return None


def delete_document(doc_id: int):
    """Delete a document"""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{doc_id}")
        return response.json()
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return None


def display_confidence(confidence: float):
    """Display confidence score with color coding"""
    if confidence >= 0.7:
        color_class = "confidence-high"
        emoji = "🟢"
    elif confidence >= 0.4:
        color_class = "confidence-medium"
        emoji = "🟡"
    else:
        color_class = "confidence-low"
        emoji = "🔴"
    
    return f'{emoji} <span class="{color_class}">{confidence:.2%}</span>'


def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📚 RAG Document QA System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Health Check
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Not Responding")
        except:
            st.error("❌ API Not Available")
        
        st.markdown("---")
        
        # Query Settings
        st.subheader("Query Settings")
        top_k = st.slider("Number of documents to retrieve", 1, 10, 5)
        use_rerank = st.checkbox("Use re-ranking", value=True)
        
        st.markdown("---")
        
        # Document Stats
        st.subheader("📊 Statistics")
        stats = get_document_stats()
        if stats:
            st.metric("Total Documents", stats.get("total_documents", 0))
            st.metric("Total Chunks", stats.get("total_chunks", 0))
            vector_stats = stats.get("vector_store", {})
            st.metric("Indexed Documents", vector_stats.get("total_documents", 0))
        
        st.markdown("---")
        
        # Session Management
        st.subheader("💬 Session")
        st.text(f"ID: {st.session_state.session_id[:8]}...")
        
        if st.button("🔄 New Session"):
            response = requests.post(f"{API_BASE_URL}/sessions/new")
            st.session_state.session_id = response.json()["session_id"]
            st.session_state.chat_history = []
            st.success("New session created!")
            st.rerun()
        
        if st.button("🗑️ Clear History"):
            try:
                requests.delete(f"{API_BASE_URL}/sessions/{st.session_state.session_id}")
                st.session_state.chat_history = []
                st.success("History cleared!")
                st.rerun()
            except:
                st.error("Error clearing history")
    
    # Main content - Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📤 Upload Documents", "📚 Document Library"])
    
    # Tab 1: Chat Interface
    with tab1:
        st.header("Ask Questions About Your Documents")
        
        # Display chat history
        for i, (question, answer, sources, confidence) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**🙋 You:** {question}")
                st.markdown(f"**🤖 Assistant:** {answer}")
                
                # Confidence score
                st.markdown(
                    f"**Confidence:** {display_confidence(confidence)}", 
                    unsafe_allow_html=True
                )
                
                # Sources
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} documents)"):
                        for j, source in enumerate(sources[:3], 1):
                            metadata = source.get("metadata", {})
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {j}:</strong> {metadata.get('filename', 'Unknown')}
                                <br><em>Chunk {metadata.get('chunk_index', 'N/A')}</em>
                                <br><br>{source.get('text', '')[:300]}...
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
        
        # Query input
        with st.form(key="query_form"):
            question = st.text_area(
                "Enter your question:",
                placeholder="What would you like to know about your documents?",
                height=100
            )
            submit_button = st.form_submit_button("🔍 Ask Question")
            
            if submit_button and question:
                with st.spinner("🤔 Thinking..."):
                    result = query_documents(question, top_k, use_rerank)
                    
                    if result:
                        st.session_state.chat_history.append((
                            question,
                            result["answer"],
                            result.get("sources", []),
                            result.get("confidence", 0.0)
                        ))
                        st.rerun()
    
    # Tab 2: Upload Documents
    with tab2:
        st.header("Upload Documents")
        st.markdown("Supported formats: PDF, TXT, DOCX, MD, CSV, XLSX, PPTX")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'doc', 'md', 'csv', 'xlsx', 'pptx']
        )
        
        if uploaded_files:
            if st.button("📤 Upload and Process"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    result = upload_single_document(file)
                    
                    if result and result.get("status") == "success":
                        st.success(f"✅ {file.name}: {result.get('message')}")
                    else:
                        st.error(f"❌ {file.name}: {result.get('message', 'Upload failed')}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("✨ All documents processed!")
                time.sleep(1)
                st.rerun()
    
    # Tab 3: Document Library
    with tab3:
        st.header("Document Library")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        documents = list_documents()
        
        if documents and documents.get("documents"):
            st.write(f"Total documents: {documents['total']}")
            
            for doc in documents["documents"]:
                with st.expander(f"📄 {doc['filename']}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**File Type:** {doc['file_type']}")
                        st.write(f"**File Size:** {doc['file_size']} bytes")
                        st.write(f"**Chunks:** {doc['total_chunks']}")
                        st.write(f"**Uploaded:** {doc['upload_date']}")
                    
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"del_{doc['id']}"):
                            result = delete_document(doc['id'])
                            if result:
                                st.success("Document deleted!")
                                time.sleep(1)
                                st.rerun()
        else:
            st.info("No documents uploaded yet. Go to 'Upload Documents' tab to add some!")
        
        st.markdown("---")
        
        # Rebuild index button
        if st.button("🔧 Rebuild Vector Index"):
            with st.spinner("Rebuilding index..."):
                try:
                    response = requests.post(f"{API_BASE_URL}/index/rebuild")
                    if response.status_code == 200:
                        st.success("✅ Index rebuilt successfully!")
                    else:
                        st.error("❌ Failed to rebuild index")
                except Exception as e:
                    st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
