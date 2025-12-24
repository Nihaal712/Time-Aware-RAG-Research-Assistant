"""
Conversation Management
Handle both short-term conversational memory and persistent storage
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import BaseMessage
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages conversation memory and persistence"""
    
    def __init__(self, memory_file: str = "chat_history.json"):
        """
        Initialize conversation manager
        
        Args:
            memory_file: Path to persistent memory file
        """
        self.memory_file = memory_file
        self.memory: List[BaseMessage] = []
        logger.info(f"ConversationManager initialized with file: {memory_file}")
        
        # Load existing conversation on startup
        self.load_conversation()
    
    def get_memory(self) -> List:
        """
        Get the current conversation memory
        
        Returns:
            List of messages
        """
        return self.memory
    

    def load_conversation(self) -> List:
        """
        Load conversation from persistent storage
        
        Returns:
            Loaded message list
        """
        try:
            if not os.path.exists(self.memory_file):
                logger.info("No existing conversation file found, starting fresh")
                return self.memory
            
            # Check if file is empty
            if os.path.getsize(self.memory_file) == 0:
                logger.info("Conversation file is empty, starting fresh")
                return self.memory
            
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                history_data = json.load(f)
            
            # Restore conversation history
            conversations = history_data.get("conversations", [])
            
            for conv in conversations:
                human_input = conv.get("human_input", "")
                ai_response = conv.get("ai_response", "")
                
                if human_input and ai_response:
                    self.memory.append(HumanMessage(content=human_input))
                    self.memory.append(AIMessage(content=ai_response))
            
            logger.info(f"Loaded {len(conversations)} conversations from {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to load conversation: {e}")
            logger.info("Starting with fresh conversation memory")
        
        return self.memory
    
    def clear_memory(self) -> None:
        """Clear conversation memory and delete persistent file"""
        try:
            self.memory.clear()
            
            if os.path.exists(self.memory_file):
                os.remove(self.memory_file)
                logger.info(f"Cleared memory and deleted {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
    
    def add_interaction(self, human_input: str, ai_response: str, 
                       sources_used: List[str] = None, time_filter: str = "All time") -> None:
        """
        Add a new interaction to memory and save
        
        Args:
            human_input: User's input
            ai_response: AI's response
            sources_used: List of sources used in response
            time_filter: Time filter applied
        """
        self.memory.append(HumanMessage(content=human_input))
        self.memory.append(AIMessage(content=ai_response))
        
        # Save after each interaction with metadata
        self._save_conversation_with_metadata(human_input, ai_response, sources_used, time_filter)
    
    def _save_conversation_with_metadata(self, human_input: str, ai_response: str,
                                        sources_used: List[str] = None, time_filter: str = "All time") -> None:
        """
        Save conversation with metadata for the latest interaction
        
        Args:
            human_input: User's input
            ai_response: AI's response
            sources_used: List of sources used in response
            time_filter: Time filter applied
        """
        try:
            # Extract conversation history
            messages = self.memory
            conversation_data = []
            
            for i in range(0, len(messages), 2):
                if i + 1 < len(messages):
                    human_msg = messages[i]
                    ai_msg = messages[i + 1]
                    
                    # Check if this is the latest interaction
                    is_latest = (i == len(messages) - 2)
                    
                    conversation_data.append({
                        "timestamp": datetime.now().isoformat(),
                        "human_input": human_msg.content,
                        "ai_response": ai_msg.content,
                        "sources_used": sources_used if is_latest and sources_used else [],
                        "time_filter": time_filter if is_latest else "All time"
                    })
            
            # Create complete history structure
            history_data = {
                "conversations": conversation_data,
                "session_metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "total_interactions": len(conversation_data)
                }
            }
            
            # Save to file
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation saved to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to save conversation: {e}")


    def clear_persistent_history(self):
        empty_history = {
            "conversations": [],
            "session_metadata": {
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_interactions": 0
            }
        }

        with open(self.memory_file, "w") as f:
            json.dump(empty_history, f, indent=2)