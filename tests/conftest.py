import os
import shutil
import pytest
from typing import Generator
import nltk
import tempfile

@pytest.fixture(scope="session")
def test_db_path():
    """Provide temporary test database path"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after all tests
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture(autouse=True)
def setup_test_env(test_db_path):
    """Set up test environment variables and NLTK data"""
    # Set environment variables
    os.environ['GOOGLE_API_KEY'] = 'test_key'
    os.environ['ALLOW_RESET'] = 'TRUE'
    os.environ['CHROMA_PERSIST_DIR'] = test_db_path
    
    # Download required NLTK data
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

@pytest.fixture
def temp_file(tmp_path) -> str:
    """Create a temporary file for testing"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Test content")
    return str(test_file)