# PDF Question-Answering Chatbot
An intelligent chatbot system that allows users to upload PDF documents and ask questions about their content. The system uses advanced language models and vector embeddings to provide accurate, context-aware responses.

## Features
- PDF document processing and analysis
- Intelligent question-answering capabilities
- Vector-based document search
- Real-time chat interface
- Multi-document support
- Conversation history tracking

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Hugging Face API token
- Docker 

### Environment Variables
The following environment variables need to be set:
```
HUGGINGFACEHUB_API_TOKEN=huggingface_token
TEMP_PATH=/app/temp_pdfs  # Path for temporary PDF storage
```

### Installation

1. Clone the repository:
```bash
git clone [https://github.com/mennamohama/Chatbot-Development-with-Python/tree/main]
cd pdf-chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:7860`

## API Documentation

### Core Classes

#### PDFChatbot
Main class that handles document processing and question answering.

```python
PDFChatbot(hf_token: str)
```

##### Key Methods:

###### process_pdfs(pdf_files)
```python
def process_pdfs(pdf_files) -> tuple[str, str]
```
Processes uploaded PDF files and creates vector embeddings.
- **Parameters**: List of PDF file objects
- **Returns**: Tuple of (status_message, document_list)

###### answer_question(question: str, history: List[tuple[str, str]])
```python
def answer_question(question: str, history: List[tuple[str, str]]) -> tuple[List[tuple[str, str]], str]
```
Processes user questions and generates responses.
- **Parameters**:
  - question: User's question
  - history: Current chat history
- **Returns**: Tuple of (updated_history, error_message)

### Data Classes

#### ChatMessage
```python
@dataclass
class ChatMessage:
    role: str          # 'user' or 'assistant'
    content: str       # Message content
    timestamp: datetime  # Creation time
```

#### UploadedDocument
```python
@dataclass
class UploadedDocument:
    filename: str       # File name
    upload_time: datetime
    page_count: int    # Number of pages
    status: str        # Processing status
```

## Usage Guide

### Document Upload
1. Click the "Upload PDF Documents" button
2. Select one or more PDF files (max 10MB each)
3. Click "Process PDFs" to begin document analysis

### Asking Questions
1. Type your question in the input box
2. Click "Ask" or press Enter
3. The system will:
   - Search relevant documents
   - Generate a context-aware response
   - Display the response in the chat interface

### Managing Conversations
- Use the "Clear Chat" button to reset the conversation
- Previous conversations are maintained in the chat history
- The system maintains context from previous messages

## Technical Details

### Vector Store Configuration
- Using Chroma for vector storage
- FastEmbed embeddings (gte-large model)
- Maximum Marginal Relevance (MMR) search
- Retrieves top 4 most relevant document chunks

### Text Processing
- Chunk size: 512 characters
- Chunk overlap: 50 characters
- RecursiveCharacterTextSplitter for document chunking

### Language Model
- Model: Mistral-7B-Instruct-v0.2
- Configuration:
  - Temperature: 0.1 (focused responses)
  - Max tokens: 512
  - Repetition penalty: 1.1
  - Top-p: 0.9

## Error Handling

The system includes robust error handling for:
- Invalid file types
- File size limits
- Processing failures
- Query execution errors
- Concurrent processing prevention

## Security Considerations

- File validation before processing
- Size limits on uploads (10MB per file)
- Restricted file types (PDF only)
- Controlled temporary file storage
- Sanitized user inputs

## Performance Optimization

- Efficient document chunking
- Optimized vector search
- Context window management
- Conversation history limiting
- Concurrent processing prevention

## Limitations

- PDF format only
- Maximum 5 messages in conversation context
- Response length limited to 512 tokens

## License

[Apache 2.0]
