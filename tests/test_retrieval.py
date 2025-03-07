import unittest
import os
import sys
import tempfile
import pytest
from typing import List
from unittest.mock import Mock, patch

from src.retrieval import RAGSystem
from src.llm_abstraction import LLMInterface
from chromadb.api.types import EmbeddingFunction

class MockEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: str) -> List[float]:
        return [[0.1] * 384] * (len(input) if isinstance(input, list) else 1)

class MockLLM(LLMInterface):
    """Mock LLM for testing"""
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        return "Mock response"
    
    def get_embeddings(self, text: str) -> list:
        return [0.1] * 384  # Mock 384-dimensional embedding

@pytest.mark.usefixtures("test_db_path")
class TestRAGSystem:
    @pytest.fixture(autouse=True)
    def setup(self, test_db_path):
        self.llm = MockLLM()
        self.rag = RAGSystem(
            self.llm,
            persist_directory=test_db_path
        )
    
    def tearDown(self):
        """Clean up test environment"""
        try:
            self.rag.clear_collection()
            # Use shutil to remove directory and contents
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    def test_document_embedding(self):
        """Test document embedding"""
        test_text = "This is a test document."
        doc_id = self.rag.add_document(test_text)
        
        assert doc_id is not None
        assert isinstance(doc_id, str)
    
    def test_document_deletion(self):
        """Test document deletion"""
        doc_id = self.rag.add_document("Test document")
        self.assertTrue(self.rag.delete_document(doc_id))
        
        health = self.rag.check_collection_health()
        self.assertEqual(health["document_count"], 0)
    
    def test_augmented_generation(self):
        """Test RAG generation"""
        # Add context document
        self.rag.add_document("Python is a programming language.")
        
        # Test generation with context
        result = self.rag.augmented_generation("What is Python?")
        self.assertIn("response", result)
        self.assertIn("references", result)
        self.assertTrue(len(result["references"]) > 0)

    def test_collection_health(self):
        """Test health check functionality"""
        health = self.rag.check_collection_health()
        self.assertEqual(health["status"], "healthy")
        self.assertIsNotNone(health.get("document_count"))

if __name__ == '__main__':
    unittest.main()