"""
Local Document Retrieval
Handles time-aware retrieval from local FAISS store
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TimeAwareRetriever:
    """Handles time-aware retrieval from local document store"""
    
    def __init__(self, faiss_store: Optional[FAISS] = None, embeddings: Optional[HuggingFaceEmbeddings] = None):
        """
        Initialize the time-aware retriever
        
        Args:
            faiss_store: FAISS vector store instance
            embeddings: HuggingFace embeddings model
        """
        self.faiss_store = faiss_store
        self.embeddings = embeddings
        logger.info("TimeAwareRetriever initialized")
    
    def retrieve_with_time_filter(self, query: str, time_filter: str, k: int = 5) -> List[Document]:
        """
        Retrieve documents with time filtering applied
        
        Args:
            query: Search query
            time_filter: Time filter option ("All time", "2025 only", "2024+", "2023+", "Pre-2023")
            k: Number of documents to retrieve
            
        Returns:
            List of filtered documents with similarity scores
        """
        logger.info(f"Retrieving documents with time filter: {time_filter}")
        
        if not self.faiss_store:
            logger.warning("No FAISS store available for retrieval")
            return []
        
        try:
            # First, retrieve more documents than needed to account for filtering
            # We retrieve k*3 to ensure we have enough after time filtering
            retrieval_k = max(k * 3, 20)  # Retrieve at least 20 to have good filtering options
            
            # Perform similarity search with scores
            docs_with_scores = self.faiss_store.similarity_search_with_score(query, k=retrieval_k)
            
            # Convert to documents with similarity scores in metadata
            documents = []
            for doc, score in docs_with_scores:
                # Add similarity score to metadata (FAISS returns distance, convert to similarity)
                doc.metadata['similarity_score'] = float(1.0 / (1.0 + score))  # Convert distance to similarity
                documents.append(doc)
            
            # Apply time filtering
            filtered_docs = self.apply_time_filter(documents, time_filter)
            
            # Return top k results after filtering
            result_docs = filtered_docs[:k]
            
            logger.info(f"Retrieved {len(result_docs)} documents after time filtering (from {len(documents)} total)")
            return result_docs
            
        except Exception as e:
            logger.error(f"Error during document retrieval: {e}")
            return []
    
    def apply_time_filter(self, documents: List[Document], time_filter: str) -> List[Document]:
        """
        Apply time filter to retrieved documents
        
        Args:
            documents: List of documents to filter
            time_filter: Time filter option ("All time", "2025 only", "2024+", "2023+", "Pre-2023")
            
        Returns:
            Filtered documents sorted by similarity score
        """
        logger.info(f"Applying time filter: {time_filter}")
        
        if time_filter == "All time":
            # Sort by similarity score (highest first) and return all
            return sorted(documents, key=lambda doc: doc.metadata.get('similarity_score', 0), reverse=True)
        
        filtered_docs = []
        for doc in documents:
            year = doc.metadata.get('year', 0)
            
            # Apply the specific time filter logic
            include_doc = False
            if time_filter == "2025 only" and year == 2025:
                include_doc = True
            elif time_filter == "2024+" and year >= 2024:
                include_doc = True
            elif time_filter == "2023+" and year >= 2023:
                include_doc = True
            elif time_filter == "Pre-2023" and year < 2023 and year > 0:  # Exclude year 0 (unknown)
                include_doc = True
            
            if include_doc:
                filtered_docs.append(doc)
        
        # Sort filtered documents by similarity score (highest first)
        filtered_docs.sort(key=lambda doc: doc.metadata.get('similarity_score', 0), reverse=True)
        
        logger.info(f"Filtered to {len(filtered_docs)} documents matching time filter '{time_filter}'")
        return filtered_docs
    
    def calculate_relevance_score(self, documents: List[Document]) -> float:
        """
        Calculate average relevance score for retrieved documents
        
        Args:
            documents: List of documents with similarity scores in metadata
            
        Returns:
            Average relevance score (0.0 to 1.0)
        """
        if not documents:
            logger.info("No documents to calculate relevance score")
            return 0.0
        
        # Extract similarity scores from document metadata
        scores = []
        for doc in documents:
            similarity_score = doc.metadata.get('similarity_score', 0.0)
            scores.append(similarity_score)
        
        if not scores:
            logger.warning("No similarity scores found in document metadata")
            return 0.0
        
        # Calculate weighted average with higher weight for top results
        weights = [1.0 / (i + 1) for i in range(len(scores))]  # 1.0, 0.5, 0.33, 0.25, ...
        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        weight_sum = sum(weights)
        
        avg_relevance = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        
        logger.info(f"Calculated relevance score: {avg_relevance:.3f} from {len(documents)} documents")
        return avg_relevance
    
    def is_sufficient_results(self, documents: List[Document], threshold: float = 0.7) -> bool:
        """
        Determine if retrieved results are sufficient based on relevance and quantity
        
        Args:
            documents: Retrieved documents
            threshold: Minimum relevance threshold (default 0.7)
            
        Returns:
            True if results are sufficient (good relevance and reasonable quantity)
        """
        if not documents:
            logger.info("No documents retrieved - results insufficient")
            return False
        
        # Calculate relevance score
        relevance_score = self.calculate_relevance_score(documents)
        
        # Check if we have enough documents (at least 2 for good coverage)
        min_doc_count = min(2, len(documents))
        has_sufficient_docs = len(documents) >= min_doc_count
        
        # Check if relevance meets threshold
        meets_relevance_threshold = relevance_score >= threshold
        
        # Additional check: ensure top document has good similarity
        top_doc_score = documents[0].metadata.get('similarity_score', 0.0) if documents else 0.0
        top_doc_sufficient = top_doc_score >= (threshold * 0.8)  # Slightly lower threshold for top doc
        
        is_sufficient = meets_relevance_threshold and has_sufficient_docs and top_doc_sufficient
        
        logger.info(f"Result sufficiency check: relevance={relevance_score:.3f}, "
                   f"threshold={threshold}, docs={len(documents)}, "
                   f"top_score={top_doc_score:.3f}, sufficient={is_sufficient}")
        
        return is_sufficient
