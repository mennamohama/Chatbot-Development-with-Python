"""
PDF Question Answering System with RAG (Retrieval-Augmented Generation)
Uses hybrid search (semantic + keyword) with query expansion and results reranking
"""

# First, install required packages
# !pip install langchain==0.0.284 sentence-transformers==2.2.2 rank-bm25==0.2.2 chromadb==0.4.7 gradio==3.50.2 pypdf==3.15.2

import gradio as gr
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import os
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import tempfile
import shutil
from pathlib import Path
import numpy as np  # Added missing import
import os
# Define a temporary path for storing PDFs if not provided in the environment
TEMP_PATH = os.getenv('TEMP_PATH', '/app/temp_pdfs')

class DocumentProcessor:
    """
    Processes PDF documents: loads, cleans, splits into chunks, and creates vector representations.
    """    def __init__(self):
        # Initialize HuggingFace embeddings using a specific sentence-transformers model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        # Initialize text splitter with reduced chunk size for more precise retrieval
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Maximum characters per chunk (reduced from 1000)
            chunk_overlap=100, # Overlap between chunks (reduced from 200)
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""]  # Granular separators to split text more naturally
        )
        # Load a sentence transformer model for computing embeddings
        self.model = SentenceTransformer('all-mpnet-base-v2')
    
    def process_document(self, pdf_path: str):
                """
        Load and process a PDF file:
         - Loads the PDF.
         - Cleans the text by normalizing whitespace.
         - Splits the document into smaller chunks.
         - Adds metadata (page number, source, chunk ID, and section type) to each chunk.
         - Creates a vectorstore for semantic search and builds a BM25 index for keyword search.
        """
        # Load PDF pages using PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        raw_documents = loader.load()
        
        # Clean and preprocess text from each page
        cleaned_documents = []
        for doc in raw_documents:
            text = doc.page_content
            # Normalize whitespace (removes extra spaces)
            text = " ".join(text.split())# Remove redundant whitespace
            doc.page_content = text
            cleaned_documents.append(doc)
        # Split documents into smaller chunks for better retrieval
        split_documents = self.text_splitter.split_documents(cleaned_documents)

        documents = []
        # Add metadata and repackage each chunk as a new Document
        for idx, doc in enumerate(split_documents):
            page_num = doc.metadata.get('page', 0)
            content = doc.page_content
            
             # Identify the section type (e.g., benefits, definition, examples)
            section_type = self._identify_section(content)
            
            metadata = {
                "page_number": page_num,
                "source": pdf_path,
                "chunk_id": f"chunk_{idx}",
                "section_type": section_type
            }
            new_doc = Document(page_content=content, metadata=metadata)
            documents.append(new_doc)
         # Create a semantic vectorstore using Chroma for similarity searches 
        vectorstore = Chroma.from_documents(
            documents,
            self.embeddings,
            collection_name="pdf_collection"
        )
        # Build a BM25 index for keyword-based searches using the text content of each document chunk
        texts = [doc.page_content for doc in documents]
        bm25 = BM25Okapi(texts)
        
        return documents, vectorstore, bm25
    
    def _identify_section(self, text):
        """
        Identify the type of content in a text chunk based on keywords.
        Returns one of: "benefits", "definition", "examples", or "general".
        """
        text_lower = text.lower()
        if any(word in text_lower for word in ["benefit", "advantage", "pro", "strength"]):
            return "benefits"
        elif any(word in text_lower for word in ["definition", "what is", "refers to"]):
            return "definition"
        elif any(word in text_lower for word in ["example", "application", "use case"]):
            return "examples"
        return "general"

# class ChatEngine:
#     """
#     Handles user queries by performing a hybrid search (semantic and keyword search) on the processed PDF documents,
#     then returns the most relevant content chunk.
#     """
#     def __init__(self, doc_processor):
#         self.document_processor = doc_processor
#         self.current_documents = None
#         self.current_vectorstore = None
#         self.current_bm25 = None
    
#     def rerank(self, query: str, documents, top_k=5):
#         """
#         Rerank the given documents based on cosine similarity of embeddings with the query.
#         Returns the top_k most similar documents.
#         """
#         # Expand query based on question type
#         expanded_queries = [query]
#         if "pros" in query.lower() or "benefits" in query.lower():
#             expanded_queries.extend([
#                 f"benefits of {query}",
#                 f"advantages of {query}",
#                 f"why use {query}"
#             ])
        
#         # Encode the query and each document into embeddings
#         query_embeddings = self.document_processor.model.encode(expanded_queries)
#         doc_embeddings = self.document_processor.model.encode([doc.page_content for doc in documents])
        
#         # Compute cosine similarities between query and each document
#         all_similarities = []
#         for query_embedding in query_embeddings:
#             similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
#             all_similarities.append(similarities)
        
#         # Take maximum similarity across all queries
#         final_similarities = np.maximum.reduce(all_similarities)
#         # Get indices of documents sorted by similarity score (highest first)
#         ranked_indices = final_similarities.argsort()[::-1][:top_k]
        
#         return [documents[i] for i in ranked_indices], final_similarities[ranked_indices]
    
#     def get_response(self, query: str) -> tuple:
#         """Generate improved response for user query"""
#         # Identify question type and adjust search
#         query_type = self._identify_question_type(query)
        
#         expanded_query = self._expand_query(query, query_type)
        
#         initial_results = self.hybrid_search(
#             expanded_query,
#             top_k=5  # Increased from 3
#         )
        
#         if not initial_results:
#             return ("I couldn't find a relevant answer in the document. Could you please rephrase your question?", 0)
        
#         final_results, confidences = self.rerank(
#             query,
#             initial_results,
#             top_k=3  # Get top 3 for better context
#         )
        
#         if not final_results:
#             return ("No relevant results found.", 0)
        
#         # Combine relevant information from top results
#         response = self._format_response(query_type, final_results, confidences)
#         page_num = final_results[0].metadata.get('page_number', 0)
        
#         return (response, page_num)
    
#     def _identify_question_type(self, query):
#         """Identify the type of question being asked"""
#         query_lower = query.lower()
#         if any(word in query_lower for word in ["pros", "benefits", "advantages"]):
#             return "benefits"
#         elif any(word in query_lower for word in ["what is", "define", "meaning of"]):
#             return "definition"
#         elif any(word in query_lower for word in ["example", "how to use", "application"]):
#             return "examples"
#         return "general"
    
#     def _expand_query(self, query, query_type):
#         """Expand query based on question type"""
#         if query_type == "benefits":
#             return f"{query} benefits advantages pros positive aspects"
#         elif query_type == "definition":
#             return f"{query} definition meaning concept"
#         elif query_type == "examples":
#             return f"{query} examples applications use cases"
#         return query
    
#     def _format_response(self, query_type, results, confidences):
#         """Format response based on question type and results"""
#         if query_type == "benefits":
#             # Extract and combine benefits from multiple results
#             benefits = []
#             for result in results:
#                 text = result.page_content
#                 # Split into sentences and identify benefit statements
#                 sentences = text.split('.')
#                 for sentence in sentences:
#                     if any(word in sentence.lower() for word in ["benefit", "advantage", "improve", "enhance"]):
#                         benefits.append(sentence.strip())
            
#             if benefits:
#                 return "Key benefits include:\n" + "\n".join(f"- {benefit}" for benefit in benefits)
        
#         # Return the highest confidence result for other types
#         return results[0].page_content

class ChatEngine:
        """
    Handles user queries by performing a hybrid search (semantic and keyword search) on the processed PDF documents,
    then returns the most relevant content chunk.
    """
    def __init__(self, doc_processor):
        self.document_processor = doc_processor
        self.current_documents = None
        self.current_vectorstore = None
        self.current_bm25 = None
    
    def rerank(self, query: str, documents, top_k=5):
        """
        Rerank the given documents based on cosine similarity of embeddings with the query.
        Returns the top_k most similar documents.
        """
        # Encode the query and each document into embeddings
        query_embedding = self.document_processor.model.encode([query])[0]
        doc_embeddings = self.document_processor.model.encode([doc.page_content for doc in documents])
        # Compute cosine similarities between query and each document
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        # Get indices of documents sorted by similarity score (highest first)
        ranked_indices = similarities.argsort()[::-1][:top_k]
        return [documents[i] for i in ranked_indices]
    
    def hybrid_search(self, query: str, top_k=5):
        """
        Perform a hybrid search by combining:
          - Semantic search using the vectorstore.
          - Keyword search using the BM25 index.
        Combines results and returns a list of unique document chunks.
        """
        # Ensure that the PDF has been processed and indices are available
        if not all([self.current_documents, self.current_vectorstore, self.current_bm25]):
            return []
            
        # Get semantic search results from the vectorstore
        semantic_results = self.current_vectorstore.similarity_search(
            query,
            k=top_k
        )
        
        # Get keyword search results using BM25 scores
        keyword_scores = self.current_bm25.get_scores(query.split())
        keyword_indices = keyword_scores.argsort()[::-1][:top_k]
        keyword_results = [self.current_documents[i] for i in keyword_indices]
        
        # Combine results while avoiding duplicates based on unique chunk_id metadata
        seen_chunks = set()
        combined_results = []
        
        for doc in semantic_results + keyword_results:
            chunk_id = doc.metadata.get("chunk_id", None)
            if chunk_id and chunk_id not in seen_chunks:
                combined_results.append(doc)
                seen_chunks.add(chunk_id)
        
        return combined_results[:top_k]
    
    def process_pdf(self, pdf_path):
        """
        Process a new PDF file by extracting documents, building a vectorstore, and preparing a BM25 index.
        """
        self.current_documents, self.current_vectorstore, self.current_bm25 = \
            self.document_processor.process_document(pdf_path)
    
    def get_response(self, query: str) -> tuple:
        """
        Generate a response for a user query:
          - Optionally expands the query if certain keywords are detected.
          - Performs a hybrid search on the processed PDF.
          - Reranks the results and returns the most relevant content along with its source page number.
        """
        # Expand query with extra context if certain keywords are present
        expanded_query = query
        if "chapter" in query.lower():
            expanded_query += " section content"
        if "definition" in query.lower():
            expanded_query += " meaning explanation"
        
        # Get initial search results (top 3)
        initial_results = self.hybrid_search(
            expanded_query,
            top_k=3
        )
        
        if not initial_results:
            return ("I couldn't find a relevant answer in the document. Could you please rephrase your question?", 0)
        
        # Rerank the initial results to get the best match
        final_results = self.rerank(
            query,
            initial_results,
            top_k=1
        )
        
        if not final_results:
            return ("No relevant results found.", 0)
        
        best_result = final_results[0]
        
        return (
            best_result.page_content,
            best_result.metadata.get('page_number', 0)
        )

def create_chat_interface():
        """
    Creates a Gradio interface for the PDF question-answering chatbot.
    - Handles PDF upload and processing.
    - Sets up the chat interface for user queries.
    """
    # Initialize the document processor and chat engine
    doc_processor = DocumentProcessor()
    chat_engine = ChatEngine(doc_processor)
    
    def handle_pdf_upload(file):
        """
        Callback for PDF file upload:
          - Processes the PDF file.
          - Returns a status message.
        """
        if file is None:
            return None, "Please upload a PDF file"
        try:
            chat_engine.process_pdf(file.name)
            return "PDF processed successfully!", "PDF uploaded and processed successfully!"
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def respond(message, chat_history, pdf_status):
         """
        Callback for handling user chat input:
          - Checks if a PDF has been processed.
          - Retrieves the response from the chat engine.
          - Appends the user query and system response to the chat history.
        """
        if not pdf_status or pdf_status != "PDF processed successfully!":
            return "", chat_history + [(None, "Please upload a PDF first")]
        
        try:
            response, page_num = chat_engine.get_response(message)
            full_response = f"{response}\n\n[Source: Page {page_num}]"
            chat_history.append((message, full_response))
            return "", chat_history
        except Exception as e:
            return "", chat_history + [(None, f"Error: {str(e)}")]
    # Build the Gradio Blocks interface
    with gr.Blocks() as interface:
        gr.Markdown("# PDF Question Answering Chat")
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload component for PDF documents
                pdf_file = gr.File(
                    label="Upload PDF Document",
                    file_types=[".pdf"],
                )
                pdf_status = gr.State()# Store processing status
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False
                )
            
            with gr.Column(scale=2):
                # Chatbot interface for Q&A
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=400
                    # Text input for user questions
                )
                
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Enter your question here and press enter",
                )
        
        pdf_file.upload(
            # Set up file upload and chat submission event handlers
            handle_pdf_upload,
            inputs=[pdf_file],
            outputs=[pdf_status, status_text]
        )
        
        txt.submit(
            respond,
            inputs=[txt, chatbot, pdf_status],
            outputs=[txt, chatbot]
        )
        
        gr.Markdown("""
        ### Instructions:
        1. Upload your PDF document using the file upload button
        2. Wait for the document to be processed
        3. Enter your question about the PDF content
        4. Press Enter to submit your question
        5. The system will search the document and provide relevant answers
        """)
    
    return interface

# Create and launch the Gradio interface
demo = create_chat_interface()
demo.launch(server_name='0.0.0.0', server_port=7860)
