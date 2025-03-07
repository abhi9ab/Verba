"""
LLM abstraction layer for the AI assistant.
This module provides a unified interface for different language models.
"""
import os
import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
from functools import lru_cache
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMInterface(ABC):
    """Abstract base class for LLM implementations"""
    
    @abstractmethod
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text based on the prompt"""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> List[float]:
        """Get vector embeddings for the input text"""
        pass

class GeminiLLM(LLMInterface):
    """Google Gemini API implementation"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini LLM"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-8b')
        self.embedding_model = genai.GenerativeModel('embedding-001')
        
    @lru_cache(maxsize=100)
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Generate text using Gemini API"""
        try:
            generation_config = genai.types.GenerationConfig(
                candidate_count=1,
                max_output_tokens=max_length,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                top_k=kwargs.get('top_k', 40),
                stop_sequences=kwargs.get('stop_sequences', [])
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings={
                    "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_MEDIUM_AND_ABOVE",
                    "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_MEDIUM_AND_ABOVE",
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Error in Gemini generation: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings using Gemini API"""
        try:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

class LLMFactory:
    """Factory class to create LLM instances"""
    
    @staticmethod
    def create_llm(llm_type: str = "gemini", config: Dict[str, Any] = None) -> LLMInterface:
        """Create LLM instance"""
        if llm_type.lower() == "gemini":
            api_key = os.getenv('GOOGLE_API_KEY')
            return GeminiLLM(api_key)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")