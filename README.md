# PDF Question Answering System

A Retrieval-Augmented Generation (RAG) system for PDF document analysis using hybrid search (semantic + keyword) with query expansion.

## Features

- 📄 PDF document processing with intelligent chunking
- 🔍 Hybrid search combining BM25 keyword search and semantic embeddings
- 💡 Query expansion and results reranking
- 📈 Context-aware response generation
- 🖥️ Gradio web interface

## Technologies

- LangChain (Document processing)
- ChromaDB (Vector storage)
- Sentence Transformers (Embeddings)
- Gradio (UI)
- BM25 (Keyword search)

## Installation

### Prerequisites
- Docker
- Python 3.9+

### Using Docker (Recommended)

```bash
# Build and start the container
docker compose up --build

# Access the web interface at:
# http://localhost:7860
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

## Usage

1. **Upload a PDF document** using the file uploader
2. Wait for processing completion (typically 10-30 seconds)
3. **Ask questions** about the document content
4. View answers with source page references

Example queries:
- "What are the main benefits discussed?"
- "Explain the key concepts from chapter 3"
- "Give examples of practical applications"

## API Details

### Document Processing Pipeline

1. PDF text extraction and cleaning
2. Context-aware chunking (500 chars with 100 overlap)
3. Metadata enrichment (section type, page numbers)
4. Dual indexing (ChromaDB + BM25)

### Search Features

- Query expansion based on question type
- Hybrid search combining semantic and keyword results
- Confidence-based reranking
- Context-aware response formatting

## Configuration

Environment variables:
- `TEMP_PATH`: PDF storage location (default: /app/temp_pdfs)
- `MODEL_PATH`: Model cache directory (default: /app/models)
