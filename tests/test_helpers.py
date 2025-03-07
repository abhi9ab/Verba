"""Test helper utilities."""
import os
import sys
from typing import Dict, Any

def get_test_config() -> Dict[str, Any]:
    """Get test configuration"""
    return {
        'model_name': 'gemini-1.5-flash-8b',
        'embedding_model': 'models/embedding-001',
        'temperature': 0.7,
        'max_length': 512
    }