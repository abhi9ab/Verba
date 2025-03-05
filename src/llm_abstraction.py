"""
LLM abstraction layer for the AI assistant.
This module provides a unified interface for different language models.
"""

import os
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import logging
from functools import lru_cache

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

class HuggingFaceLLM(LLMInterface):
    """Hugging Face implementation of the LLM interface"""
    
    def __init__(self, model_name: str = "google/gemma-2b", 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the Hugging Face LLM.
        
        Args:
            model_name: Name of the model to use for text generation
            embedding_model: Name of the model to use for embeddings
        """
        try:
            logger.info(f"Loading generation model: {model_name}")
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Set device to CPU (recommended for your constraints)
            self.device = "cpu"
            
            # Load tokenizer and model with 4-bit quantization for efficiency
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=self.device,
                load_in_4bit=True,  # 4-bit quantization to reduce memory usage
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info(f"Loading embedding model: {embedding_model}")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
            
        except Exception as e:
            logger.error(f"Error initializing HuggingFaceLLM: {str(e)}")
            raise
    
    @lru_cache(maxsize=100)
    def generate(self, prompt: str, max_length: int = 512, temperature: float = 0.7, **kwargs) -> str:
        """
        Generate text based on prompt.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Generated text response
        """
        try:
            logger.info(f"Generating text with prompt length: {len(prompt)}")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate with specified parameters
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                **kwargs
            )
            
            # Decode the generated tokens
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return only the newly generated text (not including the prompt)
            # This may need adjustment based on your specific model's behavior
            if response.startswith(prompt):
                response = response[len(prompt):]
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Error generating text: {str(e)}"
    
    @lru_cache(maxsize=100)
    def get_embeddings(self, text: str) -> List[float]:
        """
        Get vector embeddings for the input text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        try:
            embeddings = self.embedding_model.encode(text)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

class LLMFactory:
    """Factory class for creating LLM instances"""
    
    @staticmethod
    def create_llm(llm_type: str = "huggingface", config: Dict[str, Any] = None) -> LLMInterface:
        """
        Create an LLM instance based on the specified type and configuration.
        
        Args:
            llm_type: Type of LLM to create ('huggingface', etc.)
            config: Configuration dictionary for LLM initialization
            
        Returns:
            An instance of LLMInterface
        """
        if config is None:
            config = {}
            
        if llm_type.lower() == "huggingface":
            return HuggingFaceLLM(
                model_name=config.get('model_name', 'google/gemma-2b'),
                embedding_model=config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
            )
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")