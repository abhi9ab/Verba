"""Streamlit application for the AI assistant."""
import os
import logging
import streamlit as st
import nltk
from typing import Dict, List, Any, Optional

from .llm_abstraction import LLMFactory, LLMInterface
from .data_processing import DataProcessor
from .nlp_tasks import NLPProcessor
from .retrieval import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Verba AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        return True
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        return False

@st.cache_resource
def initialize_components():
    """Initialize and cache the core components"""
    # Download NLTK data first
    if not download_nltk_data():
        st.error("Failed to download required NLTK data")
        st.stop()
    llm_config = {
        'model_name': 'gemini-1.5-flash-8b',
        'embedding_model': 'models/embedding-001',
        'temperature': 0.7,
        'top_p': 0.9,
        'top_k': 40
    }
    
    try:
        # Check for required environment variables
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
            st.stop()
            
        persist_dir = os.environ.get('CHROMA_PERSIST_DIR', os.path.join(os.getcwd(), ".chroma"))
        
        # Initialize components
        llm = LLMFactory.create_llm("gemini", llm_config)
        data_processor = DataProcessor()
        nlp_processor = NLPProcessor(llm)
        rag_system = RAGSystem(llm, persist_directory=persist_dir)
        
        # Verify RAG system health
        health = rag_system.check_collection_health()
        if health["status"] != "healthy":
            logger.warning(f"RAG system health check failed: {health.get('error', 'unknown error')}")
        
        return {
            'llm': llm,
            'data_processor': data_processor,
            'nlp_processor': nlp_processor,
            'rag_system': rag_system
        }
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}")
        st.error(f"Error initializing components: {str(e)}")
        st.stop()

def chat_interface(components: Dict[str, Any]):
    """Simple chat interface without RAG"""
    st.subheader("Chat with AI")

    # Add clear history button
    if st.sidebar.button("Clear Chat History"):
        st.session_state.messages = []
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                response = components['llm'].generate(prompt)
                st.markdown(response)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })

def summarize_interface(components: Dict[str, Any]):
    """Text summarization interface with RAG support"""
    st.subheader("Text Summarization")
    
    # Add document upload for summarization
    uploaded_file = st.file_uploader(
        "Upload a document to summarize", 
        type=['txt', 'pdf', 'csv', 'json']
    )
    
    text = ""
    if uploaded_file:
        try:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            with open(temp_path, 'r') as f:
                text = f.read()
            
            os.remove(temp_path)
            st.success("Document loaded successfully!")
        except Exception as e:
            st.error(f"Error loading document: {str(e)}")
    
    text = st.text_area("Or enter text to summarize:", value=text, height=200)
    max_length = st.slider("Maximum summary length:", 50, 500, 200)
    use_context = st.checkbox("Use uploaded documents for context", value=False)
    
    if st.button("Summarize") and text:
        with st.spinner("Generating summary..."):
            if use_context:
                # Use RAG for enhanced summarization
                prompt = f"Summarize this text in {max_length} words, using any relevant context: {text}"
                result = components['rag_system'].augmented_generation(prompt)
                summary = result["response"]
                
                st.markdown("### Summary")
                st.write(summary)
                
                if result["references"]:
                    with st.expander("Related Context"):
                        for i, ref in enumerate(result["references"], 1):
                            score = ref["relevance_score"]
                            st.markdown(f"**Reference {i}** (Relevance: {score:.2%})")
                            st.markdown(ref["text"])
            else:
                # Direct summarization without context
                summary = components['nlp_processor'].summarize_text(text, max_length)
                st.markdown("### Summary")
                st.write(summary)

def sentiment_interface(nlp: NLPProcessor):
    """Sentiment analysis interface"""
    st.subheader("Sentiment Analysis")
    
    text = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Analyze Sentiment") and text:
        with st.spinner("Analyzing sentiment..."):
            result = nlp.analyze_sentiment(text)
            
            st.markdown("### Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment", result["sentiment"])
            with col2:
                st.metric("Confidence", f"{result['confidence']:.2%}")
            st.write("**Explanation:**", result["explanation"])

def entity_interface(nlp: NLPProcessor):
    """Named entity recognition interface"""
    st.subheader("Named Entity Recognition")
    
    text = st.text_area("Enter text to analyze:", height=150)
    
    if st.button("Extract Entities") and text:
        with st.spinner("Extracting entities..."):
            entities = nlp.extract_entities(text)
            
            st.markdown("### Extracted Entities")
            for entity in entities:
                st.markdown(f"- **{entity['entity']}** ({entity['type']})")

def qa_interface(components: Dict[str, Any]):
    """Question answering interface with RAG support"""
    st.subheader("Question Answering")
    
    # Add document upload for context
    uploaded_file = st.file_uploader(
        "Upload a document for context", 
        type=['txt', 'pdf', 'csv', 'json']
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                components['rag_system'].add_documents_from_file(temp_path)
                st.success(f"Added document to context!")
                
                os.remove(temp_path)
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

    # Show context status
    with st.sidebar:
        try:
            health = components['rag_system'].check_collection_health()
            st.markdown(f"**Available Context:** {health.get('document_count', 0)} documents")
        except Exception as e:
            st.error(f"Error checking context: {str(e)}")
    
    question = st.text_input("Enter your question:")
    use_context = st.checkbox("Use uploaded documents as context", value=True)
    
    if st.button("Get Answer") and question:
        with st.spinner("Finding answer..."):
            if use_context:
                # Use RAG for answer generation
                result = components['rag_system'].augmented_generation(question)
                answer = result["response"]
                
                st.markdown("### Answer")
                st.write(answer)
                
                if result["references"]:
                    with st.expander("Sources"):
                        for i, ref in enumerate(result["references"], 1):
                            score = ref["relevance_score"]
                            st.markdown(f"**Source {i}** (Relevance: {score:.2%})")
                            st.markdown(ref["text"])
            else:
                # Use direct QA without context
                answer = components['nlp_processor'].answer_question("", question)
                st.markdown("### Answer")
                st.write(answer)

def code_interface(nlp: NLPProcessor):
    """Code generation interface"""
    st.subheader("Code Generation")
    
    description = st.text_area("Describe what you want the code to do:", height=100)
    language = st.selectbox("Programming Language:", 
                           ["python", "javascript", "java", "cpp", "rust"])
    
    if st.button("Generate Code") and description:
        with st.spinner("Generating code..."):
            code = nlp.generate_code(description, language)
            st.markdown("### Generated Code")
            st.code(code, language=language)

def main():
    """Main application function"""
    st.title("Verba AI Assistant 🤖")

    with st.sidebar:
        st.header("Configuration")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
        max_length = st.slider("Max Response Length", 100, 1000, 512)
    
    # Initialize components with loading state
    with st.spinner("Initializing AI components..."):
        try:
            components = initialize_components()
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")
            return

    # Sidebar for task selection
    task = st.sidebar.selectbox(
        "Select Task",
        ["Chat", "Summarize", "Sentiment Analysis", "Entity Extraction", 
         "Question Answering", "Code Generation"]
    )

    # Main content area
    if task == "Chat":
        chat_interface(components)
    elif task == "Summarize":
        summarize_interface(components)
    elif task == "Sentiment Analysis":
        sentiment_interface(components['nlp_processor'])
    elif task == "Entity Extraction":
        entity_interface(components['nlp_processor'])
    elif task == "Question Answering":
        qa_interface(components)  # Pass all components
    elif task == "Code Generation":
        code_interface(components['nlp_processor'])

if __name__ == "__main__":
    main()