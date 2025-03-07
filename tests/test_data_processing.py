"""
Tests for the data processing module.
"""
import unittest
import os
import tempfile
import json
import csv
from src.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test cases for the DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.create_test_files()
    
    def tearDown(self):
        """Clean up test files"""
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)
    
    def create_test_files(self):
        """Create test files for different formats"""
        # Create test CSV
        self.csv_path = os.path.join(self.test_dir, 'test.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'age'])
            writer.writerow(['John', '30'])
            writer.writerow(['Jane', '25'])
        
        # Create test JSON
        self.json_path = os.path.join(self.test_dir, 'test.json')
        test_data = {
            'people': [
                {'name': 'John', 'age': 30},
                {'name': 'Jane', 'age': 25}
            ]
        }
        with open(self.json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Create test text file
        self.text_path = os.path.join(self.test_dir, 'test.txt')
        with open(self.text_path, 'w') as f:
            f.write("This is a test document.\nIt has multiple lines.\n")
    
    def test_load_csv(self):
        """Test CSV file loading"""
        data = self.processor.load_file(self.csv_path)
        
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)  # 2 rows excluding header
        self.assertIsInstance(data[0], dict)
        self.assertEqual(data[0]['name'], 'John')
        self.assertEqual(data[0]['age'], '30')
    
    def test_load_json(self):
        """Test JSON file loading"""
        data = self.processor.load_file(self.json_path)
        
        self.assertIsInstance(data, dict)
        self.assertIn('people', data)
        self.assertEqual(len(data['people']), 2)
        self.assertEqual(data['people'][0]['name'], 'John')
        self.assertEqual(data['people'][0]['age'], 30)
    
    def test_load_text(self):
        """Test text file loading"""
        data = self.processor.load_file(self.text_path)
        
        self.assertIsInstance(data, str)
        self.assertIn("This is a test document.", data)
        self.assertIn("It has multiple lines.", data)
    
    def test_invalid_file(self):
        """Test handling of invalid files"""
        invalid_path = os.path.join(self.test_dir, 'nonexistent.txt')
        with self.assertRaises(FileNotFoundError):
            self.processor.load_file(invalid_path)
    
    def test_preprocess_text(self):
        """Test text preprocessing"""
        test_text = "This is a sample text! It has some UPPERCASE words."
        processed = self.processor.preprocess_text(test_text)
        
        self.assertIsInstance(processed, str)
        self.assertNotEqual(processed, test_text)  # Should be transformed
        self.assertFalse(any(c.isupper() for c in processed))  # Should be lowercase
    
    def test_empty_inputs(self):
        """Test handling of empty inputs"""
        # Empty text
        self.assertEqual(self.processor.preprocess_text(""), "")
        
        # Empty CSV
        empty_csv = os.path.join(self.test_dir, 'empty.csv')
        with open(empty_csv, 'w') as f:
            pass
        data = self.processor.load_file(empty_csv)
        self.assertEqual(data, [])

if __name__ == '__main__':
    unittest.main()