# 📚 RAG-Based Document QA System

A production-ready **Retrieval-Augmented Generation (RAG)** system for intelligent document-based question answering, built with Python, LangChain, FAISS, OpenAI GPT, FastAPI, and Streamlit.

## 🌟 Features

- **🔍 Intelligent Document Processing**: Automatically processes and indexes PDF, DOCX, TXT, MD, CSV, XLSX, and PPTX documents
- **🧠 Semantic Search**: Uses FAISS vector store with sentence transformers for fast and accurate semantic retrieval
- **💬 Context-Aware QA**: Leverages OpenAI GPT models with retrieved context for accurate answers
- **🗄️ PostgreSQL Integration**: Stores document metadata and embeddings with pgvector support
- **🔄 Conversation Memory**: Maintains conversation history for context-aware multi-turn dialogues
- **🌐 RESTful API**: FastAPI-based REST endpoints for document management and querying
- **🎨 Interactive UI**: Beautiful Streamlit interface for easy document upload and conversational interaction
- **🐳 Docker Support**: Complete containerization with Docker Compose for easy deployment
- **📊 Document Re-ranking**: Advanced re-ranking algorithm for improved retrieval accuracy

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    HTTP │ REST API
         │
┌────────▼────────┐
│  FastAPI Server │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼───────┐
│ RAG  │  │ Database │
│Pipeline│  │PostgreSQL│
└───┬──┘  └──────────┘
    │
┌───┼────────────┐
│   │   │        │
▼   ▼   ▼        ▼
Ingest Retrieve Generate
  │      │        │
  │   ┌──▼──┐     │
  │   │FAISS│     │
  │   └─────┘     │
  │               │
  └───────┬───────┘
          ▼
    Sentence Transformers
          │
          ▼
      OpenAI GPT
```

## 📋 Prerequisites

- Python 3.11+
- PostgreSQL 16+ (with pgvector extension)
- OpenAI API Key
- Docker & Docker Compose (optional, for containerized deployment)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Maddesumit/RAG-Based-Document-QA-System.git
cd RAG-Based-Document-QA-System
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Set Up Database

```bash
# Option A: Using Docker (Recommended)
docker run -d \
  --name rag_postgres \
  -e POSTGRES_DB=rag_qa_system \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Option B: Install PostgreSQL locally and enable pgvector extension
# psql -U postgres
# CREATE DATABASE rag_qa_system;
# CREATE EXTENSION vector;
```

### 5. Run the Application

#### Option A: Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Option B: Run Locally

```bash
# Terminal 1: Start FastAPI server
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Streamlit UI
streamlit run src/ui/streamlit_app.py --server.port 8501
```

### 6. Access the Application

- **Streamlit UI**: http://localhost:8501
- **FastAPI Docs**: http://localhost:8000/docs
- **API Base URL**: http://localhost:8000/api/v1

## 📖 Usage Guide

### Uploading Documents

1. Navigate to the **Upload Documents** tab in the Streamlit UI
2. Select one or multiple documents (PDF, DOCX, TXT, etc.)
3. Click **Upload and Process**
4. Wait for processing to complete

### Asking Questions

1. Go to the **Chat** tab
2. Type your question in the text area
3. Click **Ask Question**
4. View the AI-generated answer with source citations

### Managing Documents

1. Visit the **Document Library** tab
2. View all uploaded documents
3. Delete documents if needed
4. Rebuild the vector index when necessary

## 🔌 API Endpoints

### Document Management

- `POST /api/v1/documents/upload` - Upload a single document
- `POST /api/v1/documents/upload-multiple` - Upload multiple documents
- `GET /api/v1/documents` - List all documents
- `GET /api/v1/documents/{document_id}` - Get document details
- `DELETE /api/v1/documents/{document_id}` - Delete a document
- `GET /api/v1/documents/stats` - Get statistics

### Query Endpoints

- `POST /api/v1/query` - Query documents with a question
- `POST /api/v1/query/stream` - Query with streaming response

### Session Management

- `POST /api/v1/sessions/new` - Create a new session
- `GET /api/v1/sessions/{session_id}/history` - Get conversation history
- `DELETE /api/v1/sessions/{session_id}` - Clear session history

### Index Management

- `POST /api/v1/index/rebuild` - Rebuild the vector index

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `POSTGRES_HOST` | PostgreSQL host | localhost |
| `POSTGRES_PORT` | PostgreSQL port | 5432 |
| `POSTGRES_DB` | Database name | rag_qa_system |
| `POSTGRES_USER` | Database user | postgres |
| `POSTGRES_PASSWORD` | Database password | postgres |
| `EMBEDDING_MODEL` | Sentence transformer model | all-MiniLM-L6-v2 |
| `OPENAI_MODEL` | OpenAI model | gpt-3.5-turbo |
| `CHUNK_SIZE` | Text chunk size | 1000 |
| `CHUNK_OVERLAP` | Chunk overlap | 200 |
| `TOP_K_RESULTS` | Number of results to retrieve | 5 |

## 📦 Project Structure

```
RAG-Based-Document-QA-System/
├── src/
│   ├── api/                    # FastAPI application
│   │   ├── main.py            # API entry point
│   │   └── routes.py          # API endpoints
│   ├── rag/                   # RAG pipeline components
│   │   ├── ingest.py         # Document ingestion
│   │   ├── retriever.py      # Document retrieval
│   │   ├── generator.py      # Answer generation
│   │   └── pipeline.py       # Complete RAG pipeline
│   ├── ui/                    # Streamlit interface
│   │   └── streamlit_app.py  # UI application
│   ├── config.py             # Configuration management
│   ├── database.py           # Database models and connection
│   ├── embeddings.py         # Embedding generation
│   ├── vectorstore.py        # FAISS vector store
│   └── utils.py              # Utility functions
├── data/
│   ├── docs/                 # Uploaded documents
│   └── indexes/              # FAISS indexes
├── docker-compose.yml        # Docker Compose configuration
├── Dockerfile               # Docker image definition
├── requirements.txt         # Python dependencies
├── .env.example            # Example environment variables
└── README.md               # This file
```

## 🧪 Testing

### Using curl

```bash
# Upload a document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"

# Query documents
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "top_k": 5}'
```

### Using Python

```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/documents/upload",
        files={"file": f}
    )
    print(response.json())

# Query
response = requests.post(
    "http://localhost:8000/api/v1/query",
    json={
        "question": "What is the main topic?",
        "top_k": 5
    }
)
print(response.json())
```

## 🔧 Advanced Features

### Custom Embedding Models

Modify `EMBEDDING_MODEL` in `.env` to use different sentence transformer models:

```bash
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

### Different OpenAI Models

Switch between GPT models:

```bash
OPENAI_MODEL=gpt-4
OPENAI_MODEL=gpt-3.5-turbo-16k
```

### Document Re-ranking

Enable/disable re-ranking in API requests:

```json
{
  "question": "Your question here",
  "use_rerank": true,
  "top_k": 10
}
```

## 🐛 Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
docker ps | grep postgres

# View database logs
docker logs rag_postgres
```

### Embedding Model Download

The first run will download the embedding model (~80MB). Ensure internet connectivity.

### Memory Issues

For large document sets, consider:
- Increasing Docker memory limits
- Using smaller embedding models
- Adjusting `CHUNK_SIZE` to reduce the number of chunks

## 📊 Performance Optimization

1. **Vector Index**: Use IVF indexes for large datasets (10,000+ chunks)
2. **Batch Processing**: Process multiple documents in parallel
3. **Caching**: Enable Redis for query caching
4. **GPU Acceleration**: Use `faiss-gpu` for faster similarity search

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sumit Madde**
- GitHub: [@Maddesumit](https://github.com/Maddesumit)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications
- [FAISS](https://github.com/facebookresearch/faiss) - Vector similarity search
- [Sentence Transformers](https://www.sbert.net/) - State-of-the-art sentence embeddings
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Streamlit](https://streamlit.io/) - Data app framework
- [OpenAI](https://openai.com/) - GPT models

## 📮 Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Built with ❤️ using Python, LangChain, FAISS, OpenAI GPT, FastAPI, and Streamlit**