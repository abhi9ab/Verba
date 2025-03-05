"""
Streamlit application for the AI assistant.
"""

import os
import logging
import time
import streamlit as st
from typing import Dict, List, Any, Optional

from llm_abstraction import LLMFactory
from data_processing import DataProcessor
from nlp_tasks import NLPProcessor
from retrieval import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Multi-Functional AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application state
@st.cache_resource
def initialize_components():
    """Initialize and cache the core components of the application"""
    # Initialize LLM
    llm_config = {
        'model_name': os.environ.get('LLM_MODEL', 'google/gemma-2b'),
        'embedding_model': os.environ.get('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    }
    llm = LLMFactory.create_llm("huggingface", llm_config)
    
    # Initialize other components
    data_processor = DataProcessor()
    nlp_processor = NLPProcessor(llm)
    
    # Initialize RAG system with persistence
    persist_dir = os.environ.get('CHROMA_PERSIST_DIR', './.chroma_db')
    rag_system = RAGSystem(llm, persist_directory=persist_dir)
    
    return {
        "llm": llm,
        "data_processor": data_processor,
        "nlp_processor": nlp_processor,
        "rag_system": rag_system
    }

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "current_persona" not in st.session_state:
        st.session_state.current_persona = "Professional"

# Main application
def main():
    """Main application function"""
    # Initialize components and session state
    components = initialize_components()
    init_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🤖 AI Assistant")
        
        # Persona selection
        st.subheader("Assistant Persona")
        persona = st.radio(
            "Select assistant persona:",
            ["Professional", "Technical", "Casual"],
            index=["Professional", "Technical", "Casual"].index(st.session_state.current_persona)
        )
        if persona != st.session_state.current_persona:
            st.session_state.current_persona = persona
        
        # Document upload
        st.subheader("Document Management")
        uploaded_file = st.file_uploader("Upload a document:", type=["txt", "csv", "json", "md"])
        
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            st.write(f"File: {file_details['FileName']}")
            
            # Save the file to a temporary location
            temp_file_path = f"./temp/{uploaded_file.name}"
            os.makedirs(os.path.dirname(temp_file_path), exist_ok=True)
            
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process the file if the user confirms
            if st.button("Process Document"):
                with st.spinner("Processing document..."):
                    try:
                        # Add the document to the RAG system
                        doc_ids = components["rag_system"].add_documents_from_file(
                            temp_file_path,
                            metadata={"filename": uploaded_file.name}
                        )
                        
                        # Update session state with new document
                        st.session_state.documents.append({
                            "name": uploaded_file.name,
                            "path": temp_file_path,
                            "ids": doc_ids
                        })
                        
                        st.success(f"Document processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
        
        # Show processed documents
        if st.session_state.documents:
            st.subheader("Processed Documents")
            for doc in st.session_state.documents:
                st.write(f"📄 {doc['name']}")
        
        # Feature selection
        st.subheader("Available Features")
        feature = st.selectbox(
            "Select feature:",
            ["Chat with AI", "Summarize Text", "Sentiment Analysis", "Extract Entities", "Generate Code"]
        )
    
    # Main content
    st.title("Multi-Functional AI Assistant")
    
    if feature == "Chat with AI":
        chat_interface(components)
    elif feature == "Summarize Text":
        summarization_interface(components)
    elif feature == "Sentiment Analysis":
        sentiment_analysis_interface(components)
    elif feature == "Extract Entities":
        entity_extraction_interface(components)
    elif feature == "Generate Code":
        code_generation_interface(components)

def chat_interface(components):
    """Chat interface for conversing with the AI assistant"""
    st.header("Chat with AI")
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").write(message["content"])
    
    # User input
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        # Add user message to history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Display user message
        st.chat_message("user").write(user_input)
        
        # Generate response with persona
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare context from conversation history
                context = "\n".join([
                    f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                    for msg in st.session_state.conversation_history[-5:]  # Last 5 messages for context
                ])
                
                # Get RAG-enhanced response
                persona_prefix = ""
                if st.session_state.current_persona == "Professional":
                    persona_prefix = "Respond in a formal, professional tone. "
                elif st.session_state.current_persona == "Technical":
                    persona_prefix = "Respond with technical details and precision. "
                elif st.session_state.current_persona == "Casual":
                    persona_prefix = "Respond in a casual, conversational tone. "
                
                # Use RAG if documents are available, otherwise just the LLM
                if st.session_state.documents:
                    enhanced_query = f"{persona_prefix}Context: {context}\n\nUser's last message: {user_input}"
                    result = components["rag_system"].augmented_generation(enhanced_query)
                    response = result["response"]
                    
                    # Show references if available
                    if result["references"]:
                        response += "\n\n*Response informed by your documents*"
                else:
                    prompt = f"{persona_prefix}You are a helpful AI assistant. Maintain conversation context.\n\nConversation history:\n{context}\n\nUser: {user_input}\nAssistant:"
                    response = components["llm"].generate(prompt)
                
                # Display the response
                st.write(response)
                
                # Add assistant response to history
                st.session_state.conversation_history.append({"role": "assistant", "content": response})

def summarization_interface(components):
    """Interface for text summarization"""
    st.header("Text Summarization")
    
    text_input = st.text_area("Enter text to summarize:", height=300)
    max_length = st.slider("Maximum summary length:", 50, 500, 200)
    
    if st.button("Summarize"):
        if text_input:
            with st.spinner("Generating summary..."):
                summary = components["nlp_processor"].summarize_text(text_input, max_length=max_length)
                st.subheader("Summary:")
                st.write(summary)
        else:
            st.warning("Please enter some text to summarize.")

def sentiment_analysis_interface(components):
    """Interface for sentiment analysis"""
    st.header("Sentiment Analysis")
    
    text_input = st.text_area("Enter text to analyze:", height=200)
    
    if st.button("Analyze Sentiment"):
        if text_input:
            with st.spinner("Analyzing sentiment..."):
                result = components["nlp_processor"].analyze_sentiment(text_input)
                
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Sentiment")
                    if result["sentiment"] == "POSITIVE":
                        st.markdown("✅ **Positive**")
                    elif result["sentiment"] == "NEGATIVE":
                        st.markdown("❌ **Negative**")
                    else:
                        st.markdown("⚖️ **Neutral**")
                
                with col2:
                    st.subheader("Confidence")
                    st.progress(result["confidence"])
                    st.text(f"{result['confidence'] * 100:.1f}%")
                
                st.subheader("Explanation")
                st.write(result["explanation"])
        else:
            st.warning("Please enter some text to analyze.")

def entity_extraction_interface(components):
    """Interface for named entity recognition"""
    st.header("Named Entity Recognition")
    
    text_input = st.text_area("Enter text to extract entities from:", height=200)
    
    if st.button("Extract Entities"):
        if text_input:
            with st.spinner("Extracting entities..."):
                entities = components["nlp_processor"].extract_entities(text_input)
                
                if entities:
                    # Create a dataframe for better display
                    import pandas as pd
                    df = pd.DataFrame(entities)
                    
                    st.subheader("Extracted Entities")
                    st.dataframe(df)
                    
                    # Highlight entities in text
                    highlighted_text = text_input
                    for entity in entities:
                        if "entity" in entity and "type" in entity:
                            # Simple highlighting using markdown
                            highlighted_text = highlighted_text.replace(
                                entity["entity"],
                                f"**[{entity['entity']} ({entity['type']})]**"
                            )
                    
                    st.subheader("Highlighted Text")
                    st.markdown(highlighted_text)
                else:
                    st.info("No entities found in the text.")
        else:
            st.warning("Please enter some text to extract entities from.")

def code_generation_interface(components):
    """Interface for code generation"""
    st.header("Code Generation")
    
    description = st.text_area("Describe what you want the code to do:", height=150)
    language = st.selectbox("Programming language:", ["python", "javascript", "java", "c++", "sql"])
    
    if st.button("Generate Code"):
        if description:
            with st.spinner("Generating code..."):
                code = components["nlp_processor"].generate_code(description, language=language)
                
                st.subheader("Generated Code")
                st.code(code, language=language)
                
                # Add copy button (using HTML/JS)
                st.markdown(
                    """
                    <script>
                    function copyCode() {
                        const codeBlock = document.querySelector('code');
                        const textArea = document.createElement('textarea');
                        textArea.value = codeBlock.textContent;
                        document.body.appendChild(textArea);
                        textArea.select();
                        document.execCommand('copy');
                        document.body.removeChild(textArea);
                        alert('Code copied to clipboard!');
                    }
                    </script>
                    <button onclick="copyCode()">Copy Code</button>
                    """,
                    unsafe_allow_html=True
                )
        else:
            st.warning("Please describe what you want the code to do.")

if __name__ == "__main__":
    main()