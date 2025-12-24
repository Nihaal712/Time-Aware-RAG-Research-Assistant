"""
arXiv Integration
Intelligent fallback to arXiv when local results are insufficient
"""

from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class ArxivFallback:
    """Handles intelligent fallback to arXiv search"""
    
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        """
        Initialize the arXiv fallback system
        
        Args:
            embeddings: HuggingFace embeddings model (required for vector space compatibility)
        """
        if embeddings is None:
            raise ValueError("Embeddings model is required for ArxivFallback. "
                           "Must use the same embeddings as the local document store.")
        
        self.embeddings = embeddings
        self.temporary_index = None
        
        # Initialize text splitter with same configuration as DocumentProcessor
        # This ensures consistent chunking between local and arXiv documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info("ArxivFallback initialized with provided embeddings")
    
    def search_arxiv(self, query: str, time_filter: str, max_results: int = 5) -> List[Document]:
        """
        Search arXiv for relevant papers and create temporary index for similarity scoring
        
        Args:
            query: Search query
            time_filter: Time filter to apply
            max_results: Maximum number of results
            
        Returns:
            List of arXiv documents with metadata
        """
        logger.info(f"Searching arXiv for query: {query} with time filter: {time_filter}")
        
        try:
            # Use ArxivLoader to search arXiv
            loader = ArxivLoader(query=query, load_max_docs=max_results)
            arxiv_docs = loader.load()
            
            if not arxiv_docs:
                logger.warning(f"No arXiv papers found for query: {query}")
                return []
            
            # Process and filter documents by time
            processed_docs = []
            for doc in arxiv_docs:
                # Extract publication year from arXiv metadata
                year = self._extract_arxiv_year(doc)
                
                # Apply time filter
                if self._passes_time_filter(year, time_filter):
                    # Add metadata for source tracking
                    doc.metadata['source_type'] = 'arxiv'
                    doc.metadata['year'] = year
                    doc.metadata['arxiv_id'] = self._extract_arxiv_id(doc)
                    
                    processed_docs.append(doc)
            
            logger.info(f"Retrieved {len(processed_docs)} arXiv papers after time filtering")
            
            # Create temporary index for similarity scoring
            if processed_docs:
                self.create_temporary_index(processed_docs)
            
            return processed_docs
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _extract_arxiv_year(self, doc: Document) -> int:
        """
        Extract publication year from arXiv document
        
        Args:
            doc: arXiv document
            
        Returns:
            Publication year
        """
        # Try to get from metadata first
        if 'published' in doc.metadata:
            published_str = doc.metadata['published']
            year_match = re.search(r'(\d{4})', published_str)
            if year_match:
                return int(year_match.group(1))
        
        # Try to extract from page_content
        year_match = re.search(r'(\d{4})', doc.page_content[:500])
        if year_match:
            year = int(year_match.group(1))
            if 1900 <= year <= 2030:
                return year
        
        # Default to current year if extraction fails
        return datetime.now().year
    
    def _extract_arxiv_id(self, doc: Document) -> str:
        """
        Extract arXiv ID from document
        
        Args:
            doc: arXiv document
            
        Returns:
            arXiv ID
        """
        # Try to get from metadata
        if 'arxiv_id' in doc.metadata:
            return doc.metadata['arxiv_id']
        
        # Try to extract from source
        if 'source' in doc.metadata:
            source = doc.metadata['source']
            arxiv_match = re.search(r'(\d+\.\d+)', source)
            if arxiv_match:
                return arxiv_match.group(1)
        
        return "unknown"
    
    def _passes_time_filter(self, year: int, time_filter: str) -> bool:
        """
        Check if a year passes the time filter
        
        Args:
            year: Publication year
            time_filter: Time filter option
            
        Returns:
            True if year passes the filter
        """
        if time_filter == "All time":
            return True
        elif time_filter == "2025 only":
            return year == 2025
        elif time_filter == "2024+":
            return year >= 2024
        elif time_filter == "2023+":
            return year >= 2023
        elif time_filter == "Pre-2023":
            return year < 2023 and year > 0
        
        return True
    
    def create_temporary_index(self, arxiv_docs: List[Document]) -> Optional[FAISS]:
        """
        Create temporary FAISS index for arXiv documents with chunking
        
        Args:
            arxiv_docs: List of arXiv documents
            
        Returns:
            Temporary FAISS index or None if creation fails
        """
        if not arxiv_docs:
            logger.warning("No arXiv documents provided for indexing")
            return None
        
        try:
            logger.info(f"Creating temporary FAISS index for {len(arxiv_docs)} arXiv documents")
            
            # Chunk arXiv documents using the same splitter as local documents
            # This ensures consistent processing and better embedding quality
            chunked_docs = []
            for doc in arxiv_docs:
                # Split document into chunks
                text_chunks = self.text_splitter.split_text(doc.page_content)
                
                # Create Document objects for each chunk with preserved metadata
                for i, chunk in enumerate(text_chunks):
                    if chunk.strip():  # Only include non-empty chunks
                        chunk_metadata = doc.metadata.copy()
                        chunk_metadata.update({
                            "chunk_id": i,
                            "total_chunks": len(text_chunks),
                            "chunk_size": len(chunk)
                        })
                        
                        chunked_doc = Document(
                            page_content=chunk,
                            metadata=chunk_metadata
                        )
                        chunked_docs.append(chunked_doc)
            
            logger.info(f"Chunked {len(arxiv_docs)} documents into {len(chunked_docs)} chunks")
            
            # Create temporary FAISS index using the same embeddings as local store
            # This ensures vector space compatibility
            self.temporary_index = FAISS.from_documents(chunked_docs, self.embeddings)
            
            logger.info("Temporary FAISS index created successfully")
            return self.temporary_index
            
        except Exception as e:
            logger.error(f"Error creating temporary FAISS index: {e}")
            return None
    
    def _calculate_arxiv_similarity_scores(self, arxiv_docs: List[Document], query: str) -> List[Document]:
        """
        Calculate similarity scores for arXiv documents using the temporary FAISS index
        
        Args:
            arxiv_docs: List of arXiv documents
            query: Original query string
            
        Returns:
            arXiv documents with similarity scores added to metadata
        """
        if not arxiv_docs or not self.temporary_index:
            logger.warning("Cannot calculate similarity scores: missing arXiv docs or temporary index")
            return arxiv_docs
        
        try:
            logger.info(f"Calculating similarity scores for {len(arxiv_docs)} arXiv documents")
            
            # Search the temporary index with the query to get similarity scores
            docs_with_scores = self.temporary_index.similarity_search_with_score(query, k=len(arxiv_docs))
            
            # Create a mapping of document content to similarity score
            score_map = {}
            for doc, score in docs_with_scores:
                # Convert distance to similarity (same as in retrieval.py)
                similarity = float(1.0 / (1.0 + score))
                # Use page_content as key to match documents
                score_map[doc.page_content[:100]] = similarity
            
            # Add similarity scores to arXiv documents
            for doc in arxiv_docs:
                # Try to find matching score using content prefix
                content_key = doc.page_content[:100]
                if content_key in score_map:
                    doc.metadata['similarity_score'] = score_map[content_key]
                else:
                    # If exact match not found, assign a default lower score
                    # This ensures arXiv docs still get ranked but lower than matched ones
                    doc.metadata['similarity_score'] = 0.5
            
            logger.info("Similarity scores calculated for arXiv documents")
            return arxiv_docs
            
        except Exception as e:
            logger.error(f"Error calculating arXiv similarity scores: {e}")
            # Assign default scores if calculation fails
            for doc in arxiv_docs:
                doc.metadata['similarity_score'] = 0.5
            return arxiv_docs
    
    def merge_results(self, local_docs: List[Document], arxiv_docs: List[Document], query: str = "") -> List[Document]:
        """
        Merge local and arXiv results with proper source labeling and similarity scoring
        
        Args:
            local_docs: Local documents (already have similarity scores)
            arxiv_docs: arXiv documents (will have similarity scores calculated)
            query: Original query string (used to calculate arXiv similarity scores)
            
        Returns:
            Merged and labeled documents sorted by similarity score
        """
        logger.info(f"Merging {len(local_docs)} local and {len(arxiv_docs)} arXiv results")
        
        # Calculate similarity scores for arXiv documents using the query
        if query and arxiv_docs:
            arxiv_docs = self._calculate_arxiv_similarity_scores(arxiv_docs, query)
        
        merged_docs = []
        
        # Add local documents with source labeling
        for doc in local_docs:
            doc.metadata['source_type'] = 'local'
            merged_docs.append(doc)
        
        # Add arXiv documents with source labeling
        for doc in arxiv_docs:
            doc.metadata['source_type'] = 'arxiv'
            merged_docs.append(doc)
        
        # Sort all documents by similarity score (now both have scores)
        merged_docs.sort(
            key=lambda doc: doc.metadata.get('similarity_score', 0.0),
            reverse=True
        )
        
        logger.info(f"Merged results: {len(merged_docs)} total documents (sorted by relevance)")
        return merged_docs