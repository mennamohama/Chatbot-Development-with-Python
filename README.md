# PDF Question-Answering Chatbot

A sophisticated chatbot application that enables users to upload PDF documents and ask questions about their content. The system uses advanced NLP techniques to provide accurate, context-aware responses based on the document content.

## Features

- PDF document processing and analysis
- Interactive chat interface using Gradio
- Context-aware question answering
- Support for multiple PDF uploads
- Document management system
- Real-time status updates
- Chat history tracking

## Prerequisites

- Python 3.8+
- Hugging Face API token
- Docker

## Required Dependencies

```bash
pip install gradio
pip install langchain
pip install langchain-community
pip install fastembed
pip install chromadb
pip install pypdf
```

## Environment Setup

1. Obtain a Hugging Face API token from [Hugging Face](https://huggingface.co/)
2. Set up your environment variables:
   ```bash
   export HUGGINGFACEHUB_API_TOKEN=your_token_here
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd pdf-chatbot
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. install Docker

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
## Usage
Start the application:
   ```bash
   python main.py
   ```
### Using Docker (Recommended)

```bash
# Build and start the container
docker compose up --build

# Access the web interface at:
# http://localhost:7860
```
2. Access the web interface through your browser (default: http://localhost:7860)

3. Upload PDF documents using the file upload interface

4. Click "Process PDFs" to analyze the documents

5. Start asking questions about the document content

## Features in Detail

### Document Processing
- Supports multiple PDF uploads
- Automatic text extraction and chunking
- Vector embeddings generation using FastEmbed
- Document validation for format and size (10MB limit)

### Chat Interface
- Interactive chat display
- Chat history tracking
- Context-aware responses
- Clear chat functionality

### Vector Store
- ChromaDB for efficient document storage
- Maximum Marginal Relevance (MMR) search
- Configurable chunk size and overlap

### Language Model
- Uses Mistral-7B-Instruct-v0.2
- Configurable temperature and response parameters
- Built-in conversation history management

## Configuration

Key configuration parameters can be adjusted in the code:

- `chunk_size`: Size of text chunks (default: 512)
- `chunk_overlap`: Overlap between chunks (default: 50)
- `max_new_tokens`: Maximum response length (default: 512)
- `temperature`: Response randomness (default: 0.1)

## Data Structures

### ChatMessage
```python
@dataclass
class ChatMessage:
    role: str          # 'user' or 'assistant'
    content: str       # Message content
    timestamp: datetime
```

### UploadedDocument
```python
@dataclass
class UploadedDocument:
    filename: str
    upload_time: datetime
    page_count: int
    status: str
```

## Error Handling

The system includes comprehensive error handling for:
- Invalid file formats
- File size limits
- Processing failures
- Query errors

## Limitations

- Maximum file size: 10MB per PDF
- Supported format: PDF only
- Processing time depends on document size
- Requires active internet connection for model access

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments

- Built with [Gradio](https://gradio.app/)
- Uses [LangChain](https://python.langchain.com/) for document processing
- Powered by [Mistral AI](https://mistral.ai/) language model


## Configuration

Environment variables:
- `TEMP_PATH`: PDF storage location (default: /app/temp_pdfs)
- `MODEL_PATH`: Model cache directory (default: /app/models)

## License

[Apache 2.0]
