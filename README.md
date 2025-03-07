# Verba: Multi-Functional AI Assistant

## Overview
Verba is a multi-functional AI assistant that leverages Google's Gemini API for natural language processing tasks. It provides functionalities such as text generation, sentiment analysis, named entity recognition, and more.

## Features
- Text Generation
- Sentiment Analysis
- Named Entity Recognition
- Question Answering
- Retrieval-Augmented Generation (RAG)

## Setup

### Prerequisites
- Python 3.8+
- [Google API Key](https://cloud.google.com/docs/authentication/api-keys)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/verba.git
    cd verba
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    ```bash
    cp .env.example .env
    # Edit .env to add your Google API Key
    ```

### Running the Application
1. Start the Streamlit application:
    ```bash
    streamlit run src/app.py
    ```

2. Open your browser and navigate to `http://localhost:8501` to access the application.

### Running the tests
1. Run the tests with:
   ```bash
   python -m pytest tests/ -v --cov=src
   ```

## Usage
- Enter your queries in the text input box and get responses from the AI assistant powered by Google's Gemini API.

## Contributing
Contributions are welcome! Please read the contributing guidelines first.

## License
This project is licensed under the MIT License. See the LICENSE file for details.