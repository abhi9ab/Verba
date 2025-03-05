# Verba: *Multi-Functional AI Assistant*

This project implements a comprehensive AI assistant leveraging Large Language Models (LLMs) for various natural language processing tasks, including text summarization, sentiment analysis, entity extraction, and code generation, with support for Retrieval-Augmented Generation (RAG).

## Features

- **LLM Abstraction Layer**: Easily switch between different LLM providers
- **Data Ingestion**: Support for CSV, JSON, and text files
- **Core NLP Tasks**:
  - Text summarization
  - Sentiment analysis
  - Named entity recognition
  - Question answering
  - Code generation
- **Retrieval-Augmented Generation**: Enhance responses with document-based context
- **Conversational Interface**: Support for multi-turn dialogue with context retention
- **Persona Switching**: Adapt response styles (Professional, Technical, Casual)
- **Performance Optimization**: Caching, prompt engineering, and logging

## Project Structure

```
/project-root  
│── /src  
│   │── llm_abstraction.py  # LLM interface and implementations
│   │── data_processing.py  # Data loading and preprocessing
│   │── nlp_tasks.py        # Core NLP functionality
│   │── retrieval.py        # RAG implementation
│   │── app.py              # Streamlit UI
│── /tests  
│   │── test_data_processing.py  
│   │── test_nlp_tasks.py  
│── /docs  
│   │── report.pdf  
│   │── demo_video  
│── requirements.txt  
│── README.md  
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/abhi9ab/verba.git
cd verba
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional)
Create a `.env` file in the project root with the following variables:
```
LLM_MODEL=google/gemma-2b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./.chroma_db
```

## Usage

### Running the Streamlit App

```bash
streamlit run src/app.py
```

This will start the Streamlit server and open the AI assistant in your default web browser.

### Using the Application

1. **Chat Interface**: Have a conversation with the AI assistant
2. **Document Upload**: Upload documents to enhance the AI's knowledge
3. **Summarization**: Generate concise summaries of longer texts
4. **Sentiment Analysis**: Analyze the sentiment of text input
5. **Entity Extraction**: Identify named entities in text
6. **Code Generation**: Generate code based on natural language descriptions

## Technical Details

### LLM Integration

The project uses a lightweight Hugging Face model (default: Gemma 2B) optimized for CPU-only environments. The LLM abstraction layer allows easy switching between different models.

### Retrieval System

The RAG system uses ChromaDB as a vector store for document retrieval. It processes documents into chunks, generates embeddings, and retrieves relevant information based on user queries.

### Performance Optimization

- LRU caching for repetitive operations
- 4-bit quantization for model efficiency
- Document chunking with overlap for better retrieval

## Testing

Run the test suite:

```bash
python -m pytest tests/
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.