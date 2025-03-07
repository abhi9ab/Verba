"""
Tests for the LLM abstraction layer.
"""
import unittest
from unittest.mock import patch, MagicMock
from src.llm_abstraction import GeminiLLM, LLMFactory

class TestGeminiLLM(unittest.TestCase):
    """Test cases for the GeminiLLM implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        with patch('google.generativeai.configure'):
            self.llm = GeminiLLM(self.api_key)
    
    @patch('google.generativeai.GenerativeModel')
    def test_generate(self, mock_model):
        """Test text generation"""
        mock_response = MagicMock()
        mock_response.text = "Test response"
        mock_model.return_value.generate_content.return_value = mock_response
        
        with patch('google.generativeai.configure'):
            llm = GeminiLLM("test_key")
            response = llm.generate("Test prompt")
            self.assertEqual(response, "Test response")
    
    @patch('google.generativeai.embed_content')
    def test_get_embeddings(self, mock_embed):
        """Test embedding generation"""
        mock_embed.return_value = {'embedding': [0.1, 0.2, 0.3]}
        
        embeddings = self.llm.get_embeddings("Test text")
        self.assertEqual(embeddings, [0.1, 0.2, 0.3])
    
    def test_error_handling(self):
        """Test error handling"""
        with patch('google.generativeai.GenerativeModel') as mock_model:
            mock_model.return_value.generate_content.side_effect = Exception("API error")
            with patch('google.generativeai.configure'):
                llm = GeminiLLM("test_key")
                response = llm.generate("Test prompt")
                self.assertIn("Error generating text", response)

class TestLLMFactory(unittest.TestCase):
    """Test cases for the LLMFactory"""
    
    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'test_key'})
    def test_create_gemini_llm(self):
        """Test Gemini LLM creation"""
        llm = LLMFactory.create_llm("gemini")
        self.assertIsInstance(llm, GeminiLLM)
    
    def test_invalid_llm_type(self):
        """Test invalid LLM type handling"""
        with self.assertRaises(ValueError):
            LLMFactory.create_llm("invalid_type")

if __name__ == '__main__':
    unittest.main()