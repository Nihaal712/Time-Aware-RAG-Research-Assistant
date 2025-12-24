"""
Utility Functions
Shared utilities for year extraction, text processing, and configuration
"""

import os
import re
import logging
from typing import List, Tuple, Dict, Optional

# Module-specific logger
logger = logging.getLogger(__name__)

def setup_logging() -> None:
    """
    Set up logging configuration for the entire application
    Should only be called once from app.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('rag_assistant.log')
        ]
    )

def load_config() -> Dict:
    """
    Load configuration from environment variables
    
    Returns:
        Configuration dictionary
    """
    config = {
        'groq_api_key': os.getenv('GROQ_API_KEY'),
        'huggingface_model': os.getenv('HUGGINGFACE_MODEL', 'all-MiniLM-L6-v2'),
        'faiss_index_path': os.getenv('FAISS_INDEX_PATH', 'faiss_index'),
        'chat_history_file': os.getenv('CHAT_HISTORY_FILE', 'chat_history.json'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO')
    }
    
    return config

def validate_groq_api_key() -> bool:
    """
    Validate Groq API key is present and properly formatted
    
    Returns:
        True if API key is valid
    """
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
        return False
    
    # Basic validation - Groq API keys typically start with 'gsk_'
    if not api_key.startswith('gsk_'):
        logger.warning("GROQ_API_KEY may not be properly formatted (should start with 'gsk_')")
        return False
    
    if len(api_key) < 20:
        logger.error("GROQ_API_KEY appears to be too short")
        return False
    
    logger.info("Groq API key validation passed")
    return True

def extract_year_candidates_from_text(text: str) -> List[Tuple[int, str]]:
    """
    Extract candidate years from text using regex patterns
    Prioritizes years in the first 2000 characters (abstract/intro) over references
    
    Args:
        text: Text to extract years from
        
    Returns:
        List of (year, source_description) tuples
    """
    candidates = []
    
    # Split text into main content and references
    # References typically start with "References", "Bibliography", "Works Cited", etc.
    reference_markers = r'(?:^|\n)(?:references|bibliography|works cited|citations|references cited)\s*(?:\n|$)'
    reference_match = re.search(reference_markers, text, re.IGNORECASE | re.MULTILINE)
    
    if reference_match:
        main_text = text[:reference_match.start()]
        reference_text = text[reference_match.start():]
    else:
        main_text = text
        reference_text = ""
    
    # Common academic year patterns
    patterns = [
        (r'\b(19|20)\d{2}\b', 'general_year'),
        (r'(?:published|pub\.?)\s+(?:in\s+)?(19|20)\d{2}', 'publication_year'),
        (r'(?:copyright|©)\s+(19|20)\d{2}', 'copyright_year'),
        (r'\((19|20)\d{2}\)', 'parenthetical_year'),
        (r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(19|20)\d{2}', 'month_year')
    ]
    
    # Extract from main content with higher priority
    for pattern, source in patterns:
        matches = re.finditer(pattern, main_text, re.IGNORECASE)
        for match in matches:
            year_str = match.group()
            # Extract just the year digits
            year_match = re.search(r'(19|20)\d{2}', year_str)
            if year_match:
                year = int(year_match.group())
                if 1900 <= year <= 2030:  # Reasonable year range
                    # Add higher weight to main content years
                    candidates.append((year, source + '_main'))
    
    # Extract from references with lower priority (only if main content has no years)
    if not candidates and reference_text:
        for pattern, source in patterns:
            matches = re.finditer(pattern, reference_text, re.IGNORECASE)
            for match in matches:
                year_str = match.group()
                year_match = re.search(r'(19|20)\d{2}', year_str)
                if year_match:
                    year = int(year_match.group())
                    if 1900 <= year <= 2030:
                        candidates.append((year, source + '_ref'))
    
    return candidates

def extract_year_from_metadata(metadata: Dict) -> List[Tuple[int, str]]:
    """
    Extract candidate years from document metadata
    
    Args:
        metadata: Document metadata dictionary
        
    Returns:
        List of (year, source_description) tuples
    """
    candidates = []
    
    # Common metadata fields that might contain years
    year_fields = [
        ('creation_date', 'metadata_creation'),
        ('modification_date', 'metadata_modification'),
        ('publish_date', 'metadata_publish'),
        ('date', 'metadata_date'),
        ('year', 'metadata_year')
    ]
    
    for field, source in year_fields:
        if field in metadata:
            value = str(metadata[field])
            # Extract year from various date formats
            year_match = re.search(r'(19|20)\d{2}', value)
            if year_match:
                year = int(year_match.group())
                if 1900 <= year <= 2030:
                    candidates.append((year, source))
    
    return candidates

def calculate_source_weight(source_type: str) -> float:
    """
    Calculate reliability weight for different source types
    
    Args:
        source_type: Type of source
        
    Returns:
        Weight value between 0 and 1
    """
    weights = {
        'metadata_publish': 0.95,
        'metadata_creation': 0.95,
        'publication_year_main': 0.9,
        'publication_year': 0.2,
        'metadata_year': 0.85,
        'copyright_year_main': 0.8,
        'copyright_year': 0.2,
        'metadata_date': 0.7,
        'parenthetical_year_main': 0.2,
        'parenthetical_year': 0.2,
        'month_year_main': 0.2,
        'month_year': 0.6,
        'metadata_modification': 0.5,
        'general_year_main': 0.5,
        'general_year': 0.3,
        'general_year_ref': 0.1,
        'publication_year_ref': 0.2,
        'copyright_year_ref': 0.15,
        'parenthetical_year_ref': 0.15,
        'month_year_ref': 0.2
    }
    
    return weights.get(source_type, 0.1)

def calculate_year_consensus(candidates: List[Tuple[int, str, float]]) -> Tuple[int, float]:
    """
    Calculate weighted consensus for publication year from candidates
    
    Args:
        candidates: List of (year, source_type, weight) tuples
        
    Returns:
        Tuple of (consensus_year, confidence_score)
    """
    if not candidates:
        return 0, 0.0  # Default to 0 (unknown) with zero confidence
    
    # Group candidates by year and calculate weighted scores
    year_scores = {}
    total_weight = 0
    
    for year, source_type, weight in candidates:
        if year not in year_scores:
            year_scores[year] = 0
        year_scores[year] += weight
        total_weight += weight
    
    if total_weight == 0:
        return 0, 0.0
    
    # Find year with highest weighted score
    best_year = max(year_scores.keys(), key=lambda y: year_scores[y])
    confidence = year_scores[best_year] / total_weight
    
    # Boost confidence if multiple sources agree
    source_types = set(source for year, source, _ in candidates if year == best_year)
    if len(source_types) > 1:
        confidence = min(1.0, confidence * 1.2)
    
    return best_year, confidence

def detect_publication_year_consensus(text: str, metadata: Dict) -> Tuple[int, float]:
    """
    Detect publication year using statistical consensus from multiple sources
    
    Args:
        text: Document text content
        metadata: Document metadata dictionary
        
    Returns:
        Tuple of (publication_year, confidence_score)
    """
    all_candidates = []
    
    # Extract candidates from text
    text_candidates = extract_year_candidates_from_text(text)
    for year, source_type in text_candidates:
        weight = calculate_source_weight(source_type)
        all_candidates.append((year, source_type, weight))
    
    # Extract candidates from metadata
    metadata_candidates = extract_year_from_metadata(metadata)
    for year, source_type in metadata_candidates:
        weight = calculate_source_weight(source_type)
        all_candidates.append((year, source_type, weight))
    
    # Calculate consensus
    return calculate_year_consensus(all_candidates)

def extract_candidate_years(text: str, metadata: Dict) -> List[Tuple[int, str, float]]:
    """
    Extract all candidate years with their sources and weights
    
    Args:
        text: Document text content
        metadata: Document metadata dictionary
        
    Returns:
        List of (year, source_type, weight) tuples
    """
    candidates = []
    
    # Extract from text
    text_candidates = extract_year_candidates_from_text(text)
    for year, source_type in text_candidates:
        weight = calculate_source_weight(source_type)
        candidates.append((year, source_type, weight))
    
    # Extract from metadata
    metadata_candidates = extract_year_from_metadata(metadata)
    for year, source_type in metadata_candidates:
        weight = calculate_source_weight(source_type)
        candidates.append((year, source_type, weight))
    
    # Sort by weight (highest first)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    return candidates

def clean_text(text: str) -> str:
    """
    Clean and preprocess text content
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text


def set_faiss_store(faiss_store) -> None:
    """
    Set or update the FAISS store reference
    
    Args:
        faiss_store: FAISS vector store instance
    """
    logger.info("FAISS store updated")
    return faiss_store

def get_store_info(faiss_store) -> Dict:
    """
    Get information about the current FAISS store
    
    Args:
        faiss_store: FAISS vector store instance
        
    Returns:
        Dictionary with store information
    """
    if not faiss_store:
        return {"status": "no_store", "document_count": 0}
    
    try:
        # Get document count from FAISS index
        doc_count = faiss_store.index.ntotal if hasattr(faiss_store, 'index') else 0
        
        return {
            "status": "ready",
            "document_count": doc_count,
            "embedding_dimension": faiss_store.index.d if hasattr(faiss_store, 'index') else 0
        }
    except Exception as e:
        logger.error(f"Error getting store info: {e}")
        return {"status": "error", "error": str(e), "document_count": 0}