"""
Time-Aware RAG Research Assistant
Main application controller and Gradio interface
"""

import gradio as gr
import os
from typing import List, Tuple, Optional
import logging
from dotenv import load_dotenv
from utils import setup_logging, load_config, validate_groq_api_key
from ingestion import DocumentProcessor
from retrieval import TimeAwareRetriever
from fallback import ArxivFallback
from generation import ResponseGenerator
from chat_memory import ConversationManager
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage


# Initialize logging once for the entire application
load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

def gradio_history_to_langchain_messages(history: List[Tuple]) -> List:
    """
    Convert Gradio chat history format to LangChain message format
    
    Args:
        history: Gradio history as list of tuples [(user, assistant), ...]
        
    Returns:
        List of LangChain message objects
    """
    messages = []
    for user, assistant in history:
        if user:
            messages.append(HumanMessage(content=user))
        if assistant:
            messages.append(AIMessage(content=assistant))
    return messages

def convert_to_gradio_format(history: List[Tuple]) -> List[dict]:
    """
    Convert chat history to Gradio's expected format with 'role' and 'content' keys
    
    Args:
        history: List of tuples [(user, assistant), ...]
        
    Returns:
        List of dictionaries with 'role' and 'content' keys
    """
    gradio_history = []
    for user, assistant in history:
        if user:
            gradio_history.append({"role": "user", "content": user})
        if assistant:
            gradio_history.append({"role": "assistant", "content": assistant})
    return gradio_history

def convert_from_gradio_format(history: List) -> List[Tuple]:
    """
    Convert Gradio format (dictionaries with role/content) back to tuples
    
    Args:
        history: List of dictionaries with 'role' and 'content' keys
        
    Returns:
        List of tuples [(user, assistant), ...]
    """
    if not history:
        return []
    
    # If it's already in tuple format, return as is
    if history and isinstance(history[0], tuple):
        return history
    
    # Convert from Gradio format to tuples
    tuple_history = []
    i = 0
    while i < len(history):
        user_msg = None
        assistant_msg = None
        
        if i < len(history) and history[i].get("role") == "user":
            user_msg = history[i].get("content")
            i += 1
        
        if i < len(history) and history[i].get("role") == "assistant":
            assistant_msg = history[i].get("content")
            i += 1
        
        if user_msg or assistant_msg:
            tuple_history.append((user_msg, assistant_msg))
    
    return tuple_history



class RAGAssistant:
    """Main RAG Assistant application controller"""
    
    def __init__(self):
        """Initialize the RAG Assistant with configuration validation"""
        self.config = load_config()
        
        # Validate API keys on startup
        if not validate_groq_api_key():
            raise ValueError("Invalid or missing Groq API key")
        
        # Initialize components
        self.doc_processor = DocumentProcessor(
            faiss_store_path=self.config.get('faiss_index_path', 'faiss_index')
        )
        
        # Initialize retriever with the document processor's vector store and embeddings
        self.retriever = TimeAwareRetriever(
            faiss_store=self.doc_processor.get_vector_store(),
            embeddings=self.doc_processor.embeddings
        )
        
        # Initialize fallback with the same embeddings for vector space compatibility
        self.fallback = ArxivFallback(embeddings=self.doc_processor.embeddings)
        
        # Initialize response generator
        self.generator = ResponseGenerator()
        
        # Initialize conversation manager
        self.conversation_manager = ConversationManager(
            memory_file=self.config.get('chat_history_file', 'chat_history.json')
        )
        
        # Store for tracking current sources
        self.current_sources = []
        
        logger.info("RAG Assistant initialized successfully")
    
    def process_query(self, query: str, time_filter: str, chat_history: List) -> Tuple[str, List, str]:
        """
        Main query processing pipeline
        
        Args:
            query: User's question
            time_filter: Selected time filter option
            chat_history: Current chat history
            
        Returns:
            Tuple of (response, updated_chat_history, sources_display)
        """
        logger.info(f"Processing query: {query} with time filter: {time_filter}")
        
        if not query or not query.strip():
            return "Please enter a question.", chat_history, ""
        
        try:
            # Retrieve documents from local store with time filtering
            local_docs = self.retriever.retrieve_with_time_filter(query, time_filter, k=5)
            
            # Check if results are sufficient
            is_sufficient = self.retriever.is_sufficient_results(local_docs, threshold=0.7)
            
            # If insufficient, trigger arXiv fallback
            all_docs = local_docs
            if not is_sufficient:
                logger.info("Local results insufficient, triggering arXiv fallback")
                arxiv_docs = self.fallback.search_arxiv(query, time_filter, max_results=5)
                
                if arxiv_docs:
                    # Merge results with proper source labeling
                    all_docs = self.fallback.merge_results(local_docs, arxiv_docs, query)
            
            # Generate response using retrieved documents
            if all_docs:
                # Convert Gradio chat history to LangChain messages
                langchain_messages = gradio_history_to_langchain_messages(chat_history)
                response = self.generator.generate_response(
                    query, 
                    all_docs,
                    langchain_messages
                )
            else:
                response = "I couldn't find any relevant documents. Please try uploading documents or adjusting your time filter."
            
            # Store sources for display
            self.current_sources = all_docs
            
            # Add to conversation memory using the same citation format as the response
            sources_list = self.generator.format_citations(all_docs).split('\n')
            self.conversation_manager.add_interaction(query, response, sources_list, time_filter)
            
            # Update chat history for display - ensure it's in the correct format
            if not chat_history:
                chat_history = []
            chat_history.append((query, response))
            
            # Format sources for display
            sources_display = self._format_sources_display(all_docs)
            
            return response, chat_history, sources_display
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            error_response = f"An error occurred while processing your query: {str(e)}"
            if not chat_history:
                chat_history = []
            chat_history.append((query, error_response))
            return error_response, chat_history, ""
    
    def upload_document(self, file) -> str:
        """
        Handle PDF document uploads
        
        Args:
            file: Uploaded file object
            
        Returns:
            Status message
        """
        logger.info(f"Processing file upload: {file}")
        
        if file is None:
            return "No file uploaded"
        
        try:
            files = file if isinstance(file, list) else [file]
            any_success = False
            for file in files:
            # Process and store the document
                success = self.doc_processor.process_and_store(file, "local")
                
                if success:
                    any_success = True
                
            
            
            if any_success:
                self.retriever.faiss_store = self.doc_processor.get_vector_store()
                doc_count = self.doc_processor.get_document_count()
                return f"✓ Document uploaded successfully! Total documents in store: {doc_count}"
            else:
                return "✗ Failed to process document. Please check the file format."
                
        except Exception as e:
            logger.error(f"Error uploading document: {e}")
            return f"✗ Error uploading document: {str(e)}"
    
    def add_url(self, url: str) -> str:
        """
        Process URL-based documents
        
        Args:
            url: URL to process
            
        Returns:
            Status message
        """
        logger.info(f"Processing URL: {url}")
        
        if not url or not url.strip():
            return "Please enter a URL"
        
        try:

            # Process and store the URL document
            success = self.doc_processor.process_and_store(url, "url")
            
            if success:
                # Update retriever with new vector store
                self.retriever.faiss_store = self.doc_processor.get_vector_store()
                
                doc_count = self.doc_processor.get_document_count()
                return f"✓ URL processed successfully! Total documents in store: {doc_count}"
            else:
                return "✗ Failed to process URL. Please check the URL format."
                
        except Exception as e:
            logger.error(f"Error processing URL: {e}")
            return f"✗ Error processing URL: {str(e)}"
    
    def _format_sources_display(self, documents: List[Document]) -> str:
        """
        Format documents for display in the sources accordion with enhanced interactivity
        
        Args:
            documents: List of documents to display
            
        Returns:
            Formatted HTML string for display
        """
        if not documents:
            return "<p style='color: #999; text-align: center;'>No sources retrieved</p>"
        
        # Count sources by type
        local_count = sum(1 for doc in documents if doc.metadata.get('source_type') == 'local')
        arxiv_count = sum(1 for doc in documents if doc.metadata.get('source_type') == 'arxiv')
        
        # Header with source counts
        sources_html = f"""
        <div style="margin-bottom: 16px; padding: 12px; background-color: #f5f5f5; border-radius: 6px;">
            <strong>Retrieved {len(documents)} sources:</strong> 
            <span style="color: #0084ff; margin-right: 12px;">📚 {local_count} Local</span>
            <span style="color: #ff6b6b;">📰 {arxiv_count} arXiv</span>
        </div>
        """
        
        # Individual source cards
        for i, doc in enumerate(documents, 1):
            source_type = doc.metadata.get('source_type', 'local')
            title = doc.metadata.get('title', 'Unknown Title')
            year = doc.metadata.get('year', 'Unknown Year')
            similarity = doc.metadata.get('similarity_score', 0.0)
            source_url = doc.metadata.get('source', '')
            
            # Determine styling based on source type
            if source_type == 'arxiv':
                border_color = '#ff6b6b'
                bg_color = '#fff5f5'
                source_label = "📰 [arXiv]"
                source_icon = "🔗"
            else:
                border_color = '#0084ff'
                bg_color = '#f0f7ff'
                source_label = "📚 [Local]"
                source_icon = "📄"
            
            # Relevance bar visualization
            relevance_percent = int(similarity * 100)
            relevance_bar = f"""
            <div style="width: 100%; height: 4px; background-color: #e0e0e0; border-radius: 2px; margin-top: 6px;">
                <div style="width: {relevance_percent}%; height: 100%; background-color: {border_color}; border-radius: 2px;"></div>
            </div>
            """
            
            sources_html += f"""
            <div style="margin-bottom: 12px; padding: 12px; border-left: 4px solid {border_color}; background-color: {bg_color}; border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: start;">
                    <div style="flex: 1;">
                        <strong style="font-size: 14px;">{i}. {source_label} {title}</strong><br/>
                        <small style="color: #666;">
                            📅 Year: <strong>{year}</strong> | 
                            ⭐ Relevance: <strong>{relevance_percent}%</strong>
                        </small>
                        {relevance_bar}
                    </div>
                </div>
            </div>
            """
        
        return sources_html

    def reset_session(self):
        # 1. Clear in-memory chat memory (LangChain)
        self.conversation_manager.memory.clear()

        # 2. Clear persistent chat history (JSON)
        self.conversation_manager.clear_persistent_history()

        # 3. Clear vector store
        self.doc_processor.reset_vector_store()
        self.retriever.faiss_store = None
    

def create_interface():
    """Create and configure the Gradio interface"""
    
    # Initialize the RAG Assistant
    rag_assistant = RAGAssistant()
    
    # Time filter options
    time_filter_options = [
        "All time",
        "2025 only", 
        "2024+",
        "2023+",
        "Pre-2023"
    ]
    
    with gr.Blocks(title="Time-Aware RAG Research Assistant", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔬 Time-Aware RAG Research Assistant")
        gr.Markdown("Upload research papers and ask questions with temporal filtering capabilities. Get grounded answers with source citations.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Chat interface section
                gr.Markdown("### Chat")
                chatbot = gr.Chatbot(
                    label="Research Assistant",
                    height=400,
                    show_label=True
                )
                
                # Input row with query and controls
                with gr.Row():
                    msg = gr.Textbox(
                        label="Ask a question",
                        placeholder="Enter your research question...",
                        lines=2,
                        scale=4
                    )
                    submit_btn = gr.Button("Send", scale=1, variant="primary")
                
                # Time filter and clear button
                with gr.Row():
                    time_filter = gr.Dropdown(
                        choices=time_filter_options,
                        value="All time",
                        label="Time Filter",
                        scale=2
                    )
                    clear_btn = gr.Button("Clear Chat", scale=1)
                    new_chat_btn = gr.Button("🆕 Create New Chat")
                # Sources display section
                gr.Markdown("### Retrieved Sources")
                with gr.Accordion("Sources", open=False, label="Retrieved Sources"):
                    sources_display = gr.HTML(
                        value="<p style='color: #999;'>Sources will appear here after queries</p>",
                        label="Sources"
                    )
            
            with gr.Column(scale=1):
                # Document upload section
                gr.Markdown("### 📄 Upload Documents")
                file_upload = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                     file_count="multiple",
                    type="filepath"
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    value="Ready for upload"
                )
                
                # URL input section
                gr.Markdown("### 🔗 Add from URL")
                url_input = gr.Textbox(
                    label="Paper URL",
                    placeholder="https://example.com/paper.pdf",
                    lines=2
                )
                url_submit_btn = gr.Button("Add URL", variant="secondary")
                url_status = gr.Textbox(
                    label="URL Status",
                    interactive=False,
                    value="Ready for URL input"
                )
                
                # Info section
                gr.Markdown("### ℹ️ Info")
                info_text = gr.Textbox(
                    label="System Status",
                    interactive=False,
                    value=f"Documents in store: {rag_assistant.doc_processor.get_document_count()}"
                )
        
        # Event handlers for chat submission
        def handle_submit(query, time_filter_val, chat_history):
            """Handle chat submission"""
            if not chat_history:
                chat_history = []
            
            # Convert from Gradio format back to tuples for internal processing
            tuple_history = convert_from_gradio_format(chat_history)
            
            response, updated_history, sources = rag_assistant.process_query(
                query, time_filter_val, tuple_history
            )
            # Convert to Gradio format with role and content keys
            gradio_formatted_history = convert_to_gradio_format(updated_history)
            return "", gradio_formatted_history, sources
        
        # Submit button click
        submit_btn.click(
            handle_submit,
            inputs=[msg, time_filter, chatbot],
            outputs=[msg, chatbot, sources_display]
        )
        
        # Enter key in message box
        msg.submit(
            handle_submit,
            inputs=[msg, time_filter, chatbot],
            outputs=[msg, chatbot, sources_display]
        )
        
        def handle_new_chat():
            rag_assistant.reset_session()
            return []
        
        new_chat_btn.click(
            fn=handle_new_chat,
            outputs=[chatbot]
        )
        # File upload handler
        def handle_file_upload(file):
            """Handle file upload"""
            status = rag_assistant.upload_document(file)
            doc_count = rag_assistant.doc_processor.get_document_count()
            info = f"Documents in store: {doc_count}"
            return status, info
        
        file_upload.change(
            handle_file_upload,
            inputs=[file_upload],
            outputs=[upload_status, info_text]
        )
        
        # URL submission handler
        def handle_url_submit(url):
            """Handle URL submission"""
            status = rag_assistant.add_url(url)
            doc_count = rag_assistant.doc_processor.get_document_count()
            info = f"Documents in store: {doc_count}"
            return status, info
        
        url_submit_btn.click(
            handle_url_submit,
            inputs=[url_input],
            outputs=[url_status, info_text]
        )
        
        # Clear chat handler
        def clear_chat():
            """Clear chat history"""
            return [], "", "<p style='color: #999;'>Sources will appear here after queries</p>"
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot, msg, sources_display]
        )
    
    return interface

def main():
    """Main entry point"""
    try:
        interface = create_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7861,
            share=False
        )
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    main()