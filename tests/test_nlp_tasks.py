import os
import sys
import pytest
from typing import List
import tempfile
from unittest.mock import Mock, patch

from src.nlp_tasks import NLPProcessor
from src.llm_abstraction import LLMInterface

class MockLLM(LLMInterface):
    """Mock LLM implementation for testing"""
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        # Match prompt patterns to return appropriate responses
        if "summarize" in prompt.lower():
            return "This is a summary of the text."
        elif "sentiment" in prompt.lower():
            return """Sentiment: POSITIVE
Confidence: 0.85
Explanation: The text contains positive language."""
        elif "entities" in prompt.lower():
            return """[
    {"entity": "Test", "type": "EXAMPLE"},
    {"entity": "Document", "type": "OBJECT"}
]"""
        elif "code" in prompt.lower():
            return """def test_function():
    print('Hello, World!')"""
        return "Mock response"
    
    def get_embeddings(self, text: str) -> List[float]:
        return [0.1] * 384

@pytest.mark.usefixtures("test_db_path")
class TestNLPProcessor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.llm = MockLLM()
        self.processor = NLPProcessor(self.llm)
        self.test_text = "This is a test document."
    
    def test_summarize_text(self):
        """Test text summarization"""
        summary = self.processor.summarize_text(self.test_text)
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert summary == "This is a summary of the text."
        
        # Test with empty text
        summary = self.processor.summarize_text("")
        assert isinstance(summary, str)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        result = self.processor.analyze_sentiment(self.test_text)
        
        assert isinstance(result, dict)
        assert 'sentiment' in result
        assert 'confidence' in result
        assert 'explanation' in result
        
        assert result['sentiment'] in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        
        # Test with empty text
        result = self.processor.analyze_sentiment("")
        assert isinstance(result, dict)
        assert 'sentiment' in result
    
    def test_extract_entities(self):
        """Test named entity extraction"""
        entities = self.processor.extract_entities(self.test_text)
        
        assert isinstance(entities, list)
        assert len(entities) > 0
        
        for entity in entities:
            assert isinstance(entity, dict)
            assert 'entity' in entity
            assert 'type' in entity
            
        # Test with empty text
        entities = self.processor.extract_entities("")
        assert isinstance(entities, list)
    
    def test_answer_question(self):
        """Test question answering"""
        context = "The capital of France is Paris. It is known for the Eiffel Tower."
        question = "What is the capital of France?"
        
        answer = self.processor.answer_question(context, question)
        assert isinstance(answer, str)
        assert len(answer) > 0
        
        # Test with empty context and question
        answer = self.processor.answer_question("", "")
        assert isinstance(answer, str)
    
    def test_generate_code(self):
        """Test code generation"""
        description = "Write a function that prints 'Hello, World!'"
        
        # Test Python code generation
        python_code = self.processor.generate_code(description, language="python")
        assert isinstance(python_code, str)
        assert "def" in python_code
        assert "test_function" in python_code
        
        # Test with empty description
        code = self.processor.generate_code("")
        assert isinstance(code, str)
        
        # Test with different language
        code = self.processor.generate_code(description, language="javascript")
        assert isinstance(code, str)
    
    @patch('src.nlp_tasks.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in NLP tasks"""
        # Create a failing LLM mock
        failing_llm = Mock(spec=LLMInterface)
        failing_llm.generate.side_effect = Exception("LLM error")
        processor = NLPProcessor(failing_llm)
        
        # Test error handling in each method
        summary = processor.summarize_text(self.test_text)
        assert "Failed to generate" in summary
        
        sentiment = processor.analyze_sentiment(self.test_text)
        assert sentiment["sentiment"] == "NEUTRAL"
        assert sentiment["confidence"] == 0.0
        
        entities = processor.extract_entities(self.test_text)
        assert entities == []
        
        code = processor.generate_code("test")
        assert "Failed to generate" in code
        
        # Verify that errors were logged
        assert mock_logger.error.called

if __name__ == '__main__':
    unittest.main()