"""
Data processing module for handling different file formats and preprocessing text.
"""

import os
import csv
import json
import logging
from typing import Dict, List, Any, Union, Optional
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK resources: {str(e)}")

class DataProcessor:
    """Class for handling data ingestion and preprocessing"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            logger.warning("Could not load stopwords, using empty set")
            self.stop_words = set()
    
    def load_file(self, file_path: str) -> Union[List[Dict], List[List], str]:
        """
        Load and parse a file based on its extension.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Parsed content of the file
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.csv':
                return self._load_csv(file_path)
            elif file_ext == '.json':
                return self._load_json(file_path)
            elif file_ext in ['.txt', '.md']:
                return self._load_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
                
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            raise
    
    def _load_csv(self, file_path: str) -> List[Dict]:
        """Load and parse a CSV file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                return list(reader)
        except Exception as e:
            logger.error(f"Error parsing CSV file: {str(e)}")
            raise
    
    def _load_json(self, file_path: str) -> Any:
        """Load and parse a JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file: {str(e)}")
            raise
    
    def _load_text(self, file_path: str) -> str:
        """Load a text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading text file: {str(e)}")
            raise
    
    def preprocess_text(self, text: str, 
                        lowercase: bool = True,
                        remove_stopwords: bool = True,
                        lemmatize: bool = True,
                        remove_special_chars: bool = True) -> str:
        """
        Preprocess text with various NLP techniques.
        
        Args:
            text: Input text to preprocess
            lowercase: Whether to convert text to lowercase
            remove_stopwords: Whether to remove stopwords
            lemmatize: Whether to lemmatize words
            remove_special_chars: Whether to remove special characters
            
        Returns:
            Preprocessed text
        """
        try:
            # Handle empty or non-string input
            if not text or not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            if lowercase:
                text = text.lower()
            
            # Remove special characters
            if remove_special_chars:
                text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords
            if remove_stopwords:
                tokens = [token for token in tokens if token not in self.stop_words]
            
            # Lemmatize
            if lemmatize:
                tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
            
            # Rejoin tokens
            return ' '.join(tokens)
            
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text  # Return original text on error
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for processing.
        
        Args:
            text: Input text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            # If we're not at the end, try to find a good breaking point
            if end < len(text):
                # Look for a space to break at
                while end > start + chunk_size - overlap and text[end] != ' ':
                    end -= 1
                    
                # If we couldn't find a good breaking point, just use the calculated end
                if end == start + chunk_size - overlap:
                    end = start + chunk_size
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else len(text)
        
        return chunks