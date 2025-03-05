"""
Tests for the data processing module.
"""

import os
import tempfile
import unittest
import json
import csv
from src.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        
        # Create temporary test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test CSV file
        self.csv_path = os.path.join(self.temp_dir.name, 'test.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'value'])
            writer.writerow(['1', 'test1', '100'])
            writer.writerow(['2', 'test2', '200'])
        
        # Create a test JSON file
        self.json_path = os.path.join(self.temp_dir.name, 'test.json')
        with open(self.json_path, 'w') as f:
            json.dump([{'id': 1, 'name': 'test1'}, {'id': 2, 'name': 'test2'}], f)
        
        # Create a test text file
        self.text_path = os.path.join(self.temp_dir.name, 'test.txt')
        with open(self.text_path, 'w') as f:
            f.write("This is a test document. It contains multiple sentences with various words.")
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.temp_dir.cleanup()
    
    def test_load_csv(self):
        """Test loading a CSV file"""
        data = self.processor.load_file(self.csv_path)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['id'], '1')
        self.assertEqual(data[0]['name'], 'test1')
        self.assertEqual(data[0]['value'], '100')
    
    def test_load_json(self):
        """Test loading a JSON file"""
        data = self.processor.load_file(self.json_path)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['id'], 1)
        self.assertEqual(data[0]['name'], 'test1')
    
    def test_load_text(self):
        """Test loading a text file"""
        data = self.processor.load_file(self.text_path)
        self.assertIsInstance(data, str)
        self.assertIn("This is a test document", data)
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises error"""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_file('nonexistent.txt')
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        text = "This is a TEST document! It contains multiple sentences."
        
        # Test with all preprocessing steps
        processed = self.processor.preprocess_text(
            text, lowercase=True, remove_stopwords=True, 
            lemmatize=True, remove_special_chars=True
        )
        self.assertIsInstance(processed, str)
        self.assertNotIn("!", processed)  # Special chars removed
        self.assertNotIn("This", processed)  # Stopword removed
        self.assertEqual(processed.lower(), processed)  # Lowercase
        
        # Test with no preprocessing
        processed = self.processor.preprocess_text(
            text, lowercase=False, remove_stopwords=False, 
            lemmatize=False, remove_special_chars=False
        )
        self.assertEqual(processed, text)
    
    def test_chunk_text(self):
        """Test text chunking"""
        # Create a long text
        text = "Word " * 1000
        
        # Test chunking with no overlap
        chunks = self.processor.chunk_text(text, chunk_size=100, overlap=0)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(len(chunks[0].split()), 20)  # Each "Word " is 5 chars
        
        # Test chunking with overlap
        chunks = self.processor.chunk_text(text, chunk_size=100, overlap=50)
        self.assertGreater(len(chunks), 1)
        
        # Test with empty text
        chunks = self.processor.chunk_text("")
        self.assertEqual(len(chunks), 0)

if __name__ == '__main__':
    unittest.main()