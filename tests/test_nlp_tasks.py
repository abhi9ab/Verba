"""
Tests for the NLP tasks module.
"""

import unittest
from unittest.mock import Mock, patch
from src.nlp_tasks import NLPProcessor
from src.llm_abstraction import LLMInterface

class MockLLM(LLMInterface):
    """Mock LLM implementation for testing"""
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        """Mock text generation"""
        if "summarize" in prompt.lower():
            return "This is a summary of the text."
        elif "sentiment" in prompt.lower():
            return """Sentiment analysis:
- Sentiment: POSITIVE
- Confidence: 0.85
- Explanation: The text contains positive language."""
        elif "entities" in prompt.lower():
            return """Named entities (in JSON format):
```json
[
    {"entity": "John Smith", "type": "PERSON"},
    {"entity": "New York", "type": "LOCATION"}
]
```"""
        elif "code" in prompt.lower():
            return """```python
def hello_world():
    print("Hello, World!")
```"""
        return "Mock response"

    def get_embeddings(self, text: str) -> list:
        """Mock embedding generation"""
        return [0.1] * 10

class TestNLPProcessor(unittest.TestCase):
    """Test cases for the NLPProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.llm = MockLLM()
        self.processor = NLPProcessor(self.llm)
        self.test_text = "This is a sample text for testing NLP tasks."
    
    def test_summarize_text(self):
        """Test text summarization"""
        summary = self.processor.summarize_text(self.test_text)
        self.assertIsInstance(summary, str)
        self.assertTrue(len(summary) > 0)
        self.assertEqual(summary, "This is a summary of the text.")
        
        # Test with empty text
        summary = self.processor.summarize_text("")
        self.assertIsInstance(summary, str)
    
    def test_analyze_sentiment(self):
        """Test sentiment analysis"""
        result = self.processor.analyze_sentiment(self.test_text)
        
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
        self.assertIn('confidence', result)
        self.assertIn('explanation', result)
        
        self.assertIn(result['sentiment'], ['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
        self.assertIsInstance(result['confidence'], float)
        self.assertTrue(0 <= result['confidence'] <= 1)
        
        # Test with empty text
        result = self.processor.analyze_sentiment("")
        self.assertIsInstance(result, dict)
        self.assertIn('sentiment', result)
    
    def test_extract_entities(self):
        """Test named entity extraction"""
        entities = self.processor.extract_entities(self.test_text)
        
        self.assertIsInstance(entities, list)
        self.assertTrue(len(entities) > 0)
        
        for entity in entities:
            self.assertIsInstance(entity, dict)
            self.assertIn('entity', entity)
            self.assertIn('type', entity)
            
        # Test with empty text
        entities = self.processor.extract_entities("")
        self.assertIsInstance(entities, list)
    
    def test_answer_question(self):
        """Test question answering"""
        context = "The capital of France is Paris. It is known for the Eiffel Tower."
        question = "What is the capital of France?"
        
        answer = self.processor.answer_question(context, question)
        self.assertIsInstance(answer, str)
        self.assertTrue(len(answer) > 0)
        
        # Test with empty context and question
        answer = self.processor.answer_question("", "")
        self.assertIsInstance(answer, str)
    
    def test_generate_code(self):
        """Test code generation"""
        description = "Write a function that prints 'Hello, World!'"
        
        # Test Python code generation
        python_code = self.processor.generate_code(description, language="python")
        self.assertIsInstance(python_code, str)
        self.assertIn("def", python_code)
        self.assertIn("hello_world", python_code)
        
        # Test with empty description
        code = self.processor.generate_code("")
        self.assertIsInstance(code, str)
        
        # Test with different language
        code = self.processor.generate_code(description, language="javascript")
        self.assertIsInstance(code, str)
    
    @patch('src.nlp_tasks.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in NLP tasks"""
        # Create a failing LLM mock
        failing_llm = Mock(spec=LLMInterface)
        failing_llm.generate.side_effect = Exception("LLM error")
        processor = NLPProcessor(failing_llm)
        
        # Test error handling in each method
        summary = processor.summarize_text(self.test_text)
        self.assertIn("Failed to generate", summary)
        
        sentiment = processor.analyze_sentiment(self.test_text)
        self.assertEqual(sentiment["sentiment"], "NEUTRAL")
        self.assertEqual(sentiment["confidence"], 0.0)
        
        entities = processor.extract_entities(self.test_text)
        self.assertEqual(entities, [])
        
        code = processor.generate_code("test")
        self.assertIn("Failed to generate", code)
        
        # Verify that errors were logged
        self.assertTrue(mock_logger.error.called)

if __name__ == '__main__':
    unittest.main()