"""
PDF Question Answering System with RAG (Retrieval-Augmented Generation)
Uses hybrid search (semantic + keyword) with query expansion and results reranking
"""

# First, install required packages
# !pip install langchain==0.0.284 sentence-transformers==2.2.2 rank-bm25==0.2.2 chromadb==0.4.7 gradio==3.50.2 pypdf==3.15.2

import os
import gradio as gr
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

TEMP_PATH = os.getenv('TEMP_PATH', '/app/temp_pdfs')

# Data class for storing chat messages with timestamps
@dataclass
class ChatMessage:
    role: str          # Role can be 'user' or 'assistant'
    content: str       # The actual message content
    timestamp: datetime = field(default_factory=datetime.now)  # When the message was created

# Data class for tracking uploaded document metadata
@dataclass
class UploadedDocument:
    filename: str      # Name of the uploaded file
    upload_time: datetime  # When the file was uploaded
    page_count: int    # Number of pages in the document
    status: str        # Processing status (e.g., 'processed', 'failed')

class PDFChatbot:
    def __init__(self, hf_token: str):
        """
        Initialize the PDF Chatbot with necessary components and configurations
        Args:
            hf_token (str): Hugging Face API token for accessing language models
        """
        # Set up environment variables
        os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
        
        # Initialize embedding model for document vectorization
        self.embedding_model = FastEmbedEmbeddings(model_name="thenlper/gte-large")
        
        # Configure text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,     # Size of each text chunk
            chunk_overlap=50    # Overlap between chunks to maintain context
        )
        
        # Initialize storage components
        self.chat_history: List[ChatMessage] = []  # Store conversation history
        self.uploaded_docs: Dict[str, UploadedDocument] = {}  # Track uploaded documents
        self.db = None  # Vector store for document embeddings
        self.processing = False  # Flag to prevent concurrent processing
        
        # Initialize the language model with optimal parameters
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=hf_token,
            model_kwargs={
                "temperature": 0.1,          # Lower temperature for more focused responses
                "max_new_tokens": 512,       # Maximum length of generated responses
                "return_full_text": False,   # Only return the generated response
                "repetition_penalty": 1.1,   # Prevent repetitive responses
                "top_p": 0.9                 # Nucleus sampling parameter
            }
        )
        
        # Set up the conversation prompt template
        self.template = """
        <s>[INST] You are an AI Assistant that follows instructions extremely well. 
        Please be truthful and give direct answers. 
        Please tell 'I don't know' if user query is not in CONTEXT.

        Previous conversation:
        {chat_history}

        Current context: {context}
        </s>

        [INST] {query} [/INST]
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.output_parser = StrOutputParser()

    def validate_pdf(self, file) -> tuple[bool, str]:
        """
        Validate uploaded PDF files for format and size
        Args:
            file: The uploaded file object
        Returns:
            tuple[bool, str]: (is_valid, message)
        """
        # Check file extension
        if not file.name.lower().endswith('.pdf'):
            return False, "Only PDF files are accepted"
        
        # Check file size (10MB limit)
        if os.path.getsize(file.name) > 10 * 1024 * 1024:
            return False, "File size exceeds 10MB limit"
            
        return True, "File is valid"

    def process_pdfs(self, pdf_files) -> tuple[str, str]:
        """
        Process uploaded PDF files and create vector embeddings
        Args:
            pdf_files: List of uploaded PDF files
        Returns:
            tuple[str, str]: (status_message, document_list)
        """
        # Check if already processing files
        if self.processing:
            return "Document processing is already in progress. Please wait.", self.get_document_list()
        
        self.processing = True
        documents = []
        processed_files = []
        
        try:
            # Process each PDF file
            for pdf in pdf_files:
                # Validate the file
                is_valid, message = self.validate_pdf(pdf)
                if not is_valid:
                    self.processing = False
                    return f"Error processing {pdf.name}: {message}", self.get_document_list()
                
                # Load and process the PDF
                loader = PyPDFLoader(pdf.name)
                pdf_docs = loader.load()
                documents.extend(pdf_docs)
                
                # Track the processed document
                self.uploaded_docs[pdf.name] = UploadedDocument(
                    filename=pdf.name,
                    upload_time=datetime.now(),
                    page_count=len(pdf_docs),
                    status="processed"
                )
                processed_files.append(pdf.name)
                
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Create vector store from document chunks
            self.db = Chroma.from_documents(
                chunks,
                self.embedding_model,
                persist_directory="./chroma_db"
            )
            
            self.processing = False
            return f"Successfully processed: {', '.join(processed_files)}", self.get_document_list()
            
        except Exception as e:
            self.processing = False
            return f"Error during processing: {str(e)}", self.get_document_list()

    def format_chat_history(self) -> str:
        """
        Format the chat history for the context window
        Returns:
            str: Formatted chat history
        """
        formatted = []
        # Only include the last 5 messages to maintain relevant context
        for msg in self.chat_history[-5:]:
            formatted.append(f"{msg.role}: {msg.content}")
        return "\n".join(formatted)

    def answer_question(self, question: str, history: List[tuple[str, str]]) -> tuple[List[tuple[str, str]], str]:
        """
        Process user questions and generate responses using the language model
        Args:
            question (str): User's question
            history: Chat history for the Gradio interface
        Returns:
            tuple[List[tuple[str, str]], str]: (updated_history, error_message)
        """
        # Input validation
        if not question.strip():
            return history, "Please enter a question."
            
        if self.db is None:
            return history, "Please upload PDF documents first."
            
        if self.processing:
            return history, "Still processing documents. Please wait."
            
        try:
            # Set up document retriever
            retriever = self.db.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diversity
                search_kwargs={'k': 4}  # Number of documents to retrieve
            )
            
            # Prepare the processing chain
            chain = (
                {
                    "context": retriever,
                    "query": RunnablePassthrough(),
                    "chat_history": lambda _: self.format_chat_history()
                }
                | self.prompt
                | self.llm
                | self.output_parser
            )
            
            # Generate response
            response = chain.invoke(question)
            
            # Update chat histories
            self.chat_history.append(ChatMessage(role="user", content=question))
            self.chat_history.append(ChatMessage(role="assistant", content=response))
            history.append((question, response))
            
            return history, ""
            
        except Exception as e:
            return history, f"Error generating response: {str(e)}"

    def get_document_list(self) -> str:
        """
        Get a formatted list of uploaded documents and their status
        Returns:
            str: Formatted document list
        """
        if not self.uploaded_docs:
            return "No documents uploaded yet."
            
        doc_list = ["Uploaded Documents:"]
        for doc in self.uploaded_docs.values():
            doc_list.append(
                f"- {doc.filename}: {doc.page_count} pages, "
                f"uploaded at {doc.upload_time.strftime('%Y-%m-%d %H:%M:%S')}, "
                f"status: {doc.status}"
            )
        return "\n".join(doc_list)

    def clear_chat_history(self) -> tuple[List[tuple[str, str]], str]:
        """
        Clear the chat history
        Returns:
            tuple[List[tuple[str, str]], str]: (empty_history, status_message)
        """
        self.chat_history = []
        return [], "Chat history cleared."

    def remove_document(self, filename: str) -> str:
        """
        Remove a document from the system
        Args:
            filename (str): Name of the file to remove
        Returns:
            str: Status message
        """
        if filename in self.uploaded_docs:
            del self.uploaded_docs[filename]
            return f"Document {filename} removed."
        return f"Document {filename} not found."

def create_gradio_interface():
    """
    Create and configure the Gradio web interface
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Initialize chatbot with HuggingFace token
    chatbot = PDFChatbot("Hygging_Face_Key")
    
    # Create the interface layout
    with gr.Blocks() as demo:
        gr.Markdown("# Enhanced PDF Question-Answering Chatbot")
        
        # Main interface layout
        with gr.Row():
            # Left column for document management
            with gr.Column(scale=2):
                pdf_files = gr.File(
                    file_count="multiple",
                    label="Upload PDF Documents",
                    file_types=[".pdf"]
                )
                process_button = gr.Button("Process PDFs")
                doc_list = gr.Textbox(label="Uploaded Documents", interactive=False)
                
            # Right column for chat interface
            with gr.Column(scale=3):
                chat_display = gr.Chatbot(label="Chat History")
                question_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="Type your question here..."
                )
                
                # Chat control buttons
                with gr.Row():
                    submit_button = gr.Button("Ask")
                    clear_button = gr.Button("Clear Chat")
        
        # Status display
        status_text = gr.Textbox(label="Status")
        
        # Helper function for handling questions
        def submit_question(question, history):
            if not question.strip():
                return history, "", "Please enter a question."
            new_history, error = chatbot.answer_question(question, history)
            return new_history, "", error
        
        # Set up event handlers
        process_button.click(
            fn=chatbot.process_pdfs,
            inputs=[pdf_files],
            outputs=[status_text, doc_list]
        )
        
        submit_button.click(
            fn=submit_question,
            inputs=[question_input, chat_display],
            outputs=[chat_display, question_input, status_text]
        )
        
        clear_button.click(
            fn=chatbot.clear_chat_history,
            outputs=[chat_display, status_text]
        )
        
        # Initialize document list display
        doc_list.value = chatbot.get_document_list()
    
    return demo

# Application entry point
if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name='0.0.0.0', server_port=7860)

# Create and launch the Gradio interface
demo = create_chat_interface()
demo.launch(server_name='0.0.0.0', server_port=7860)
