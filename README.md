# PDF Question Answering System

This project implements a PDF question-answering chatbot that allows users to upload a PDF document and ask questions about its content. The system leverages both semantic and keyword-based search techniques to provide relevant answers from the PDF.
## Features

- 📄 PDF Extracts and cleans text from PDF documents, document processing with intelligent chunking
- **Chunking:** Splits documents into smaller, manageable chunks with metadata for better retrieval.
- 🔍 Hybrid search Combines semantic search (using vector embeddings) and keyword search (BM25) to find the best matches.
- 💡 Query expansion and results reranking
- 📈 Context-aware response generation
- 🖥️ Gradio web interface A user-friendly Gradio-based interface for interacting with the chatbot.

## Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/pdf-chatbot.git
   cd pdf-chatbot

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
