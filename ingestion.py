"""
Document Processing Pipeline
Handles document ingestion, text extraction, and year detection
"""

import os
import re
import shutil
from typing import List, Tuple, Dict, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import logging
import PyPDF2
import pdfplumber
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import io

# Import year detection functions from utils
from utils import (
    detect_publication_year_consensus,
    extract_candidate_years,
    calculate_year_consensus
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and ingestion pipeline"""
    
    def __init__(self, faiss_store_path: str = "faiss_index"):
        """Initialize the document processor"""
        self.faiss_store_path = faiss_store_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        self.vector_store = None
        self._load_or_create_vector_store()
        logger.info("DocumentProcessor initialized with HuggingFace embeddings")
    
    def _load_or_create_vector_store(self):
        """Load existing FAISS vector store or create new one"""
        try:
            if os.path.exists(self.faiss_store_path):
                self.vector_store = FAISS.load_local(
                    self.faiss_store_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info(f"Loaded existing FAISS vector store from {self.faiss_store_path}")
            else:
                # Create empty vector store with a dummy document
                dummy_doc = Document(
                    page_content="Initialization document",
                    metadata={"source": "init", "year": 2024, "source_type": "init"}
                )
                self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
                logger.info("Created new FAISS vector store")
        except Exception as e:
            logger.error(f"Error loading/creating vector store: {e}")
            # Create new vector store as fallback
            dummy_doc = Document(
                page_content="Initialization document",
                metadata={"source": "init", "year": 2024, "source_type": "init"}
            )
            self.vector_store = FAISS.from_documents([dummy_doc], self.embeddings)
            logger.info("Created fallback FAISS vector store")
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from PDF file with fallback handling
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from PDF: {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"PDF file not found: {file_path}")
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text = ""
        
        # Primary method: Try pdfplumber first (better for complex layouts)
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text using pdfplumber: {len(text)} characters")
                    return text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        # Fallback method: Try PyPDF2
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text using PyPDF2: {len(text)} characters")
                    return text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # If both methods fail
        if not text.strip():
            logger.error(f"Failed to extract text from PDF: {file_path}")
            raise ValueError(f"Unable to extract text from PDF: {file_path}")
        
        return text.strip()
    
    def extract_text_from_url(self, url: str) -> str:
        """
        Extract text content from URL with error handling
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content
        """
        logger.info(f"Extracting text from URL: {url}")
        
        # Validate URL format
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error(f"Invalid URL format: {url}")
            raise ValueError(f"Invalid URL format: {url}")
        
        try:
            # Set headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Make request with timeout
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Check if it's a PDF URL
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                return self._extract_text_from_pdf_url(response.content)
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract text from common content areas
            text_content = ""
            
            # Try to find main content areas
            main_selectors = [
                'main', 'article', '.content', '.main-content', 
                '.article-content', '.post-content', '#content'
            ]
            
            for selector in main_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    text_content = main_content.get_text(separator='\n', strip=True)
                    break
            
            # If no main content found, extract from body
            if not text_content:
                body = soup.find('body')
                if body:
                    text_content = body.get_text(separator='\n', strip=True)
                else:
                    text_content = soup.get_text(separator='\n', strip=True)
            
            # Clean up the text
            lines = text_content.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and len(line) > 3:  # Filter out very short lines
                    cleaned_lines.append(line)
            
            text = '\n'.join(cleaned_lines)
            
            if not text.strip():
                logger.error(f"No text content extracted from URL: {url}")
                raise ValueError(f"No text content found at URL: {url}")
            
            logger.info(f"Successfully extracted text from URL: {len(text)} characters")
            return text.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            raise ValueError(f"Failed to fetch URL {url}: {e}")
        except Exception as e:
            logger.error(f"Error extracting text from URL {url}: {e}")
            raise ValueError(f"Error extracting text from URL {url}: {e}")
    
    def _extract_text_from_pdf_url(self, pdf_content: bytes) -> str:
        """
        Extract text from PDF content downloaded from URL
        
        Args:
            pdf_content: PDF file content as bytes
            
        Returns:
            Extracted text content
        """
        logger.info("Extracting text from PDF content from URL")
        
        text = ""
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    logger.info(f"Successfully extracted text from PDF URL using pdfplumber: {len(text)} characters")
                    return text.strip()
        except Exception as e:
            logger.warning(f"pdfplumber extraction from URL failed: {e}")
        
        # Fallback to PyPDF2
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if text.strip():
                logger.info(f"Successfully extracted text from PDF URL using PyPDF2: {len(text)} characters")
                return text.strip()
        except Exception as e:
            logger.warning(f"PyPDF2 extraction from URL failed: {e}")
        
        if not text.strip():
            logger.error("Failed to extract text from PDF URL")
            raise ValueError("Unable to extract text from PDF URL")
        
        return text.strip()
    

    
    def chunk_document(self, text: str, metadata: Dict) -> List[Document]:
        """
        Chunk document into manageable segments with metadata preservation
        
        Args:
            text: Document text content
            metadata: Document metadata
            
        Returns:
            List of Document chunks
        """
        logger.info("Chunking document with metadata preservation")
        
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Split text into chunks
        text_chunks = self.text_splitter.split_text(text)
        
        # Create Document objects with preserved metadata
        documents = []
        for i, chunk in enumerate(text_chunks):
            if chunk.strip():  # Only include non-empty chunks
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": i,
                    "total_chunks": len(text_chunks),
                    "chunk_size": len(chunk)
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
        
        logger.info(f"Created {len(documents)} document chunks")
        return documents
    
    def process_and_store(self, source: str, source_type: str) -> bool:
        """
        Process document and store in vector database
        
        Args:
            source: File path or URL
            source_type: "local" or "url"
            
        Returns:
            Success status
        """
        logger.info(f"Processing and storing {source_type} document: {source}")
        
        try:
            # Extract text based on source type
            if source_type == "local":
                text = self.extract_text_from_pdf(source)
                # Extract basic metadata from file
                metadata = {
                    "source": source,
                    "source_type": "local",
                    "title": os.path.basename(source).replace('.pdf', ''),
                }
                # Try to extract PDF metadata
                try:
                    with open(source, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        if pdf_reader.metadata:
                            pdf_title = str(pdf_reader.metadata.get('/Title', '')).strip()
                            if pdf_title and pdf_title != "":
                                metadata["title"] = pdf_title
                            
                            pdf_metadata = {
                                "creation_date": str(pdf_reader.metadata.get('/CreationDate', '')),
                                "modification_date": str(pdf_reader.metadata.get('/ModDate', '')),
                                "author": str(pdf_reader.metadata.get('/Author', ''))
                            }
                            metadata.update(pdf_metadata)
                except Exception as e:
                    logger.warning(f"Could not extract PDF metadata: {e}")
                    
            elif source_type == "url":
                text = self.extract_text_from_url(source)
                metadata = {
                    "source": source,
                    "source_type": "url",
                    "title": self._extract_title_from_url(source),
                }
            else:
                logger.error(f"Unsupported source type: {source_type}")
                return False
            
            # Detect publication year using utils function
            year, confidence = detect_publication_year_consensus(text, metadata)
            metadata.update({
                "year": year,
                "confidence": confidence
            })
            
            logger.info(f"Detected year: {year} with confidence: {confidence:.2f} for document: {metadata.get('title', 'Unknown')}")
            
            # Chunk the document
            document_chunks = self.chunk_document(text, metadata)
            
            if not document_chunks:
                logger.warning(f"No chunks created for document: {source}")
                return False
            
            # Add documents to vector store
            # Add documents to vector store (initialize if needed)
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(
                    document_chunks,
                    self.embeddings
                )
            else:
                self.vector_store.add_documents(document_chunks)
            
            # Save the updated vector store
            self.save_vector_store()
            
            logger.info(f"Successfully processed and stored document: {source} ({len(document_chunks)} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document {source}: {e}")
            return False
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract title from URL or use URL as fallback"""
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag:
                return title_tag.get_text().strip()
        except Exception as e:
            logger.warning(f"Could not extract title from URL: {e}")
        
        # Fallback to URL path
        parsed_url = urlparse(url)
        return parsed_url.path.split('/')[-1] or url
    
    def save_vector_store(self):
        """Save the current vector store to disk"""
        try:
            if self.vector_store:
                self.vector_store.save_local(self.faiss_store_path)
                logger.info(f"Vector store saved to {self.faiss_store_path}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")
    
    def get_vector_store(self) -> FAISS:
        """Get the current vector store instance"""
        return self.vector_store
    
    def get_document_count(self) -> int:
        """Get the number of documents in the vector store"""
        try:
            if self.vector_store:
                return self.vector_store.index.ntotal
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0

    
    def reset_vector_store(self):
        # 1. Delete persisted FAISS index
        if os.path.exists(self.faiss_store_path):
            shutil.rmtree(self.faiss_store_path)

        # 2. Clear in-memory store
        self.vector_store = None

        # 3. (Optional) reinitialize later on first document
        self.document_count = 0