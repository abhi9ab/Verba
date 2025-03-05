"""
Retrieval module implementing RAG (Retrieval-Augmented Generation) functionality
using ChromaDB for vector storage and retrieval.
"""

import os
import logging
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union

import chromadb
from chromadb.config import Settings

from llm_abstraction import LLMInterface
from data_processing import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGSystem:
    """Retrieval-Augmented Generation system using ChromaDB for document storage and retrieval"""
    
    def __init__(self, llm: LLMInterface, persist_directory: Optional[str] = None):
        """
        Initialize the RAG system with an LLM and optional persistence.
        
        Args:
            llm: The language model implementation to use for embeddings and generation
            persist_directory: Directory to persist vector database (optional)
        """
        self.llm = llm
        self.data_processor = DataProcessor()
        
        # Initialize ChromaDB client
        self.chroma_settings = Settings()
        if persist_directory:
            os.makedirs(persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=persist_directory, settings=self.chroma_settings)
        else:
            self.client = chromadb.Client(settings=self.chroma_settings)
            
        # Create a default collection
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}  # Using cosine similarity
        )
    
    def add_document(self, document: str, metadata: Dict[str, Any] = None) -> str:
        """
        Add a document to the RAG system.
        
        Args:
            document: Document text to add
            metadata: Additional metadata for the document
            
        Returns:
            Document ID
        """
        if metadata is None:
            metadata = {}
            
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())
        
        # Preprocess the document
        preprocessed_doc = self.data_processor.preprocess_text(
            document,
            lowercase=True,
            remove_stopwords=False,  # Keep stopwords for context
            lemmatize=False,  # Don't lemmatize for retrieval
            remove_special_chars=False  # Keep special chars for context
        )
        
        # Split document into chunks
        chunks = self.data_processor.chunk_text(
            preprocessed_doc, 
            chunk_size=1000,
            overlap=200
        )
        
        # Add each chunk to the collection
        for i, chunk in enumerate(chunks):
            # Generate embedding using the LLM
            try:
                embedding = self.llm.get_embeddings(chunk)
                
                # Create chunk metadata
                chunk_metadata = {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
                
                # Add to ChromaDB
                self.collection.add(
                    ids=[f"{doc_id}_{i}"],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )
                
                logger.info(f"Added document chunk {i+1}/{len(chunks)} with ID {doc_id}_{i}")
                
            except Exception as e:
                logger.error(f"Error adding document chunk {i}: {str(e)}")
        
        return doc_id
    
    def add_documents_from_file(self, file_path: str, metadata: Dict[str, Any] = None) -> List[str]:
        """
        Add documents from a file to the RAG system.
        
        Args:
            file_path: Path to the file to process
            metadata: Additional metadata for the documents
            
        Returns:
            List of document IDs
        """
        if metadata is None:
            metadata = {
                "source": os.path.basename(file_path)
            }
        else:
            metadata["source"] = os.path.basename(file_path)
        
        # Load the file
        content = self.data_processor.load_file(file_path)
        
        # Process based on content type
        doc_ids = []
        
        if isinstance(content, str):
            # Text file - add as a single document
            doc_id = self.add_document(content, metadata)
            doc_ids.append(doc_id)
            
        elif isinstance(content, list):
            # CSV or JSON - add each row/item as a separate document
            for item in content:
                if isinstance(item, dict):
                    # Convert dictionary to string representation
                    item_text = "\n".join([f"{k}: {v}" for k, v in item.items()])
                    item_metadata = {**metadata, "keys": ", ".join(item.keys())}
                    doc_id = self.add_document(item_text, item_metadata)
                    doc_ids.append(doc_id)
                else:
                    # Handle other list item types
                    doc_id = self.add_document(str(item), metadata)
                    doc_ids.append(doc_id)
        
        return doc_ids
    
    def retrieve(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: Query text
            n_results: Number of results to retrieve
            
        Returns:
            List of retrieved documents with metadata
        """
        try:
            # Generate embedding for the query
            query_embedding = self.llm.get_embeddings(query)
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            
            # Format the results
            formatted_results = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    formatted_results.append({
                        "text": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "id": results["ids"][0][i] if results["ids"] else None,
                        "distance": results["distances"][0][i] if "distances" in results and results["distances"] else None
                    })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def augmented_generation(self, query: str, n_docs: int = 3) -> Dict[str, Any]:
        """
        Generate a response augmented with retrieved documents.
        
        Args:
            query: User query
            n_docs: Number of documents to retrieve
            
        Returns:
            Dictionary with response text and used references
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, n_results=n_docs)
        
        if not retrieved_docs:
            # Fallback to regular generation if no documents are retrieved
            response = self.llm.generate(query)
            return {
                "response": response,
                "references": []
            }
        
        # Construct augmented prompt
        context_text = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(retrieved_docs)])
        
        augmented_prompt = f"""Use the following retrieved documents to answer the query. If the documents don't contain relevant information, use your knowledge to provide a helpful response.

Retrieved documents:
{context_text}

Query: {query}

Response:"""
        
        # Generate response
        response = self.llm.generate(augmented_prompt, max_length=800)
        
        # Return response with references
        return {
            "response": response,
            "references": retrieved_docs
        }