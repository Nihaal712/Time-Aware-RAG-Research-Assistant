"""
Response Generation
Generate grounded responses using retrieved documents and handle temporal comparisons
"""

from typing import List, Optional, Dict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import logging
import os

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles LLM response generation with grounding and citations"""
    
    def __init__(self, llm: Optional[ChatGroq] = None):
        """
        Initialize the response generator
        
        Args:
            llm: ChatGroq LLM instance (will be created if None)
        """
        if llm is None:
            # Initialize Groq LLM with llama3-8b-8192 model
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                logger.warning("GROQ_API_KEY not set, ResponseGenerator will not be fully functional")
            
            try:
                self.llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=1024
                )
            except Exception as e:
                logger.warning(f"Failed to initialize ChatGroq with default params: {e}")
                # Fallback: try with minimal parameters
                self.llm = ChatGroq(model="llama3-8b-8192")
        else:
            self.llm = llm
        
        # Initialize prompt templates
        self._setup_prompts()
        logger.info("ResponseGenerator initialized with Groq LLM")
    
    def _setup_prompts(self) -> None:
        """Set up prompt templates and LCEL chains for different response types"""
        
        # Standard RAG prompt with citation requirements
        self.rag_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a knowledgeable research assistant. Answer the user's question based ONLY on the provided context documents.

IMPORTANT: You MUST cite your sources for every claim. Use the format [Source N] where N is the source number.

Context Documents:
{context}"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                """Question: {question}

Answer: Provide a comprehensive answer based on the context. Include citations for all information."""
            )
        ])
        
        # Create LCEL chain for standard RAG
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        
        # Temporal comparison prompt
        self.comparison_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """You are a research assistant specializing in temporal analysis. Answer the user's question by comparing information across different time periods.

IMPORTANT: You MUST cite your sources for every claim. Use the format [Source N] where N is the source number.

Context Documents (organized by time period):
{context}"""
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                """Question: {question}

Answer: Provide a temporal comparison analysis. Highlight:
1. How the topic has evolved over time
2. Key differences between time periods
3. New developments and trends
Include citations for all information."""
            )
        ])
        
        # Create LCEL chain for temporal comparison
        self.comparison_chain = self.comparison_prompt | self.llm | StrOutputParser()
    
    def generate_response(self, query: str, documents: List[Document], 
                         conversation_memory: Optional[List] = None) -> str:
        """
        Generate grounded response using retrieved documents
        
        Args:
            query: User query
            documents: Retrieved documents for context
            conversation_memory: List of LangChain message objects (optional)
            
        Returns:
            Generated response with citations
        """
        logger.info(f"Generating response for query: {query}")
        
        if not documents:
            logger.warning("No documents provided for response generation")
            return "I don't have enough information to answer your question. Please try uploading relevant documents or adjusting your time filter."
        
        try:
            # Check if this is a temporal comparison query
            is_comparison = self.detect_comparison_intent(query)
            
            if is_comparison:
                logger.info("Detected temporal comparison intent")
                return self.generate_temporal_comparison(query, documents, conversation_memory)
            
            # Standard response generation
            return self._generate_standard_response(query, documents, conversation_memory)
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while generating a response: {str(e)}"
    
    def _generate_standard_response(self, query: str, documents: List[Document],
                                   conversation_memory: Optional[List] = None) -> str:
        """
        Generate standard grounded response using LCEL chain
        
        Args:
            query: User query
            documents: Retrieved documents
            conversation_memory: Conversation history
            
        Returns:
            Generated response with citations
        """
        try:
            # Format context from documents
            context = self._format_context(documents)
            
            # Get chat history messages if available
            chat_history_messages = []
            if conversation_memory:
                try:
                    # conversation_memory is already a list of LangChain message objects
                    chat_history_messages = conversation_memory[-4:] if len(conversation_memory) > 4 else conversation_memory
                except Exception as e:
                    logger.warning(f"Could not extract chat history: {e}")
                    chat_history_messages = []
            
            # Invoke LCEL chain with input variables
            response = self.rag_chain.invoke({
                "context": context,
                "question": query,
                "chat_history": chat_history_messages
            })
            
            # Add source citations at the end
            citations = self.format_citations(documents)
            final_response = f"{response}\n\n**Sources:**\n{citations}"
            
            logger.info("Response generated successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in standard response generation: {e}")
            # Fallback to simple response with citations
            citations = self.format_citations(documents)
            return f"Based on the retrieved documents:\n\n[Unable to generate detailed response due to: {str(e)}]\n\n**Sources:**\n{citations}"
    
    def detect_comparison_intent(self, query: str) -> bool:
        """
        Detect if query involves temporal comparison
        
        Args:
            query: User query
            
        Returns:
            True if comparison intent detected
        """
        comparison_keywords = [
            "compare", "comparison", "versus", "vs", "difference", "changed",
            "evolution", "trend", "over time", "since", "before", "after",
            "how has", "what changed", "progress", "development", "improvement",
            "earlier", "later", "then", "now", "previously", "currently",
            "shift", "transition", "evolution", "growth", "decline"
        ]
        
        query_lower = query.lower()
        detected = any(keyword in query_lower for keyword in comparison_keywords)
        
        logger.info(f"Comparison intent detection: {detected}")
        return detected
    
    def generate_temporal_comparison(self, query: str, documents: List[Document],
                                    conversation_memory: Optional[List] = None) -> str:
        """
        Generate response for temporal comparison queries using LCEL chain
        
        Args:
            query: Comparison query
            documents: Documents from different time periods
            conversation_memory: Conversation history
            
        Returns:
            Temporal comparison response
        """
        logger.info("Generating temporal comparison response")
        
        try:
            # Group documents by year for temporal analysis
            docs_by_year = self._group_documents_by_year(documents)
            
            # Format context with temporal organization
            context = self._format_temporal_context(docs_by_year)
            
            # Get chat history messages if available
            chat_history_messages = []
            if conversation_memory:
                try:
                    # conversation_memory is already a list of LangChain message objects
                    chat_history_messages = conversation_memory[-4:] if len(conversation_memory) > 4 else conversation_memory
                except Exception as e:
                    logger.warning(f"Could not extract chat history: {e}")
                    chat_history_messages = []
            
            # Invoke LCEL chain with input variables
            response = self.comparison_chain.invoke({
                "context": context,
                "question": query,
                "chat_history": chat_history_messages
            })
            
            # Add source citations
            citations = self.format_citations(documents)
            final_response = f"{response}\n\n**Sources:**\n{citations}"
            
            logger.info("Temporal comparison response generated successfully")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in temporal comparison generation: {e}")
            # Fallback response
            citations = self.format_citations(documents)
            return f"Temporal comparison analysis:\n\n[Unable to generate detailed comparison due to: {str(e)}]\n\n**Sources:**\n{citations}"
    
    def _group_documents_by_year(self, documents: List[Document]) -> Dict[int, List[Document]]:
        """
        Group documents by publication year
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary mapping years to documents
        """
        docs_by_year = {}
        for doc in documents:
            year = doc.metadata.get('year', 0)
            if year not in docs_by_year:
                docs_by_year[year] = []
            docs_by_year[year].append(doc)
        
        return docs_by_year
    
    def _format_context(self, documents: List[Document]) -> str:
        """
        Format documents as context for LLM
        
        Args:
            documents: List of documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Unknown Title')
            year = doc.metadata.get('year', 'Unknown Year')
            source_type = doc.metadata.get('source_type', 'local')
            content = doc.page_content[:500]  # Limit content length
            
            source_label = "[arXiv]" if source_type == 'arxiv' else "[Local]"
            
            context_parts.append(
                f"Source {i}: {source_label} {title} ({year})\n"
                f"Content: {content}...\n"
            )
        
        return "\n".join(context_parts)
    
    def _format_temporal_context(self, docs_by_year: Dict[int, List[Document]]) -> str:
        """
        Format documents organized by time period for temporal analysis
        
        Args:
            docs_by_year: Documents grouped by year
            
        Returns:
            Formatted temporal context string
        """
        context_parts = []
        source_counter = 1
        
        # Sort years in ascending order
        for year in sorted(docs_by_year.keys()):
            docs = docs_by_year[year]
            context_parts.append(f"\n=== Year {year} ===")
            
            for doc in docs:
                title = doc.metadata.get('title', 'Unknown Title')
                source_type = doc.metadata.get('source_type', 'local')
                content = doc.page_content[:500]  # Limit content length
                
                source_label = "[arXiv]" if source_type == 'arxiv' else "[Local]"
                
                context_parts.append(
                    f"Source {source_counter}: {source_label} {title}\n"
                    f"Content: {content}...\n"
                )
                source_counter += 1
        
        return "\n".join(context_parts)
    
    def format_citations(self, documents: List[Document]) -> str:
        """
        Format source citations for documents
        
        Args:
            documents: List of documents to cite
            
        Returns:
            Formatted citation string
        """
        if not documents:
            return "No sources available"
        
        citations = []
        for i, doc in enumerate(documents, 1):
            source_type = doc.metadata.get('source_type', 'local')
            title = doc.metadata.get('title', 'Unknown Title')
            year = doc.metadata.get('year', 'Unknown Year')
            
            if source_type == 'arxiv':
                citation = f"{i}. [arXiv] {title} ({year})"
            else:
                citation = f"{i}. [Local] {title} ({year})"
            
            citations.append(citation)
        
        return "\n".join(citations)