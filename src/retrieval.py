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
from chromadb import PersistentClient
import chromadb.utils.embedding_functions as embedding_functions

from .llm_abstraction import LLMInterface
from .data_processing import DataProcessor

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

        # Set default persistence path if none provided
        if persist_directory is None:
            persist_directory = os.path.join(os.getcwd(), ".chroma")
        
        # Initialize ChromaDB client with persistence
        try:
            os.makedirs(persist_directory, exist_ok=True)
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=settings
            )
            logger.info(f"Initialized ChromaDB with persistence at: {persist_directory}")
            
            # Verify client connection
            self.client.heartbeat()
            
            # Create Gemini embedding function
            embedding_function = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=os.getenv('GOOGLE_API_KEY')
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="documents",
                metadata={
                    "hnsw:space": "cosine",
                    "persist_directory": persist_directory
                },
                embedding_function=embedding_function
            )
            
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def add_document(self, document: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the RAG system."""
        if metadata is None:
            metadata = {}
            
        doc_id = str(uuid.uuid4())
        
        try:
            # Preprocess document
            preprocessed_doc = self.data_processor.preprocess_text(
                document,
                lowercase=True,
                remove_stopwords=False,
                lemmatize=False,
                remove_special_chars=False
            )
            
            # Split into chunks
            chunks = self.data_processor.chunk_text(
                preprocessed_doc, 
                chunk_size=1000,
                overlap=200
            )
            
            # Prepare batch data
            chunk_ids = []
            chunk_texts = []
            chunk_metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_ids.append(f"{doc_id}_{i}")
                chunk_texts.append(chunk)
                chunk_metadatas.append({
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                })
            
            # Add to ChromaDB in single batch
            self.collection.add(
                ids=chunk_ids,
                documents=chunk_texts,
                metadatas=chunk_metadatas
            )
            
            logger.info(f"Added document with ID {doc_id} in {len(chunks)} chunks")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error adding document: {str(e)}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks from the collection"""
        try:
            # Delete all chunks with matching doc_id
            self.collection.delete(
                where={"doc_id": doc_id}
            )
            logger.info(f"Deleted document {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False

    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Use correct where clause format
            self.collection.delete(
                where={"doc_id": {"$exists": True}}
            )
            logger.info("Cleared collection")
            return True
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
            return False

    def check_collection_health(self) -> Dict[str, Any]:
        """Check the health and status of the collection"""
        try:
            # Verify client connection
            heartbeat = self.client.heartbeat()
            
            # Get collection stats
            count = self.collection.count()
            persist_dir = self.collection.metadata.get("persist_directory")
            
            return {
                "status": "healthy",
                "document_count": count,
                "collection_name": self.collection.name,
                "persistence": {
                    "enabled": bool(persist_dir),
                    "path": persist_dir,
                    "heartbeat": heartbeat
                }
            }
        except Exception as e:
            logger.error(f"Error checking collection health: {str(e)}")
            return {
                "status": "unhealthy",
                "document_count": 0,
                "error": str(e),
                "persistence": {
                    "enabled": False,
                    "path": None,
                    "heartbeat": None
                }
            }
    
    def reset_collection(self) -> bool:
        """Reset the collection (WARNING: destructive operation)"""
        try:
            # Get persistence info before reset
            health = self.check_collection_health()
            persist_dir = health["persistence"]["path"]
            
            # Delete all documents first
            self.collection.delete(where={})
            logger.info("All documents deleted from collection")

            # Reset the client if needed
            if hasattr(self.client, "reset"):
                self.client.reset()
                logger.warning("ChromaDB reset completed")
            
            # Reinitialize collection
            if persist_dir:
                self.__init__(self.llm, persist_dir)
                logger.info("Collection reinitialized with persistence")
            
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {str(e)}")
            return False
    
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
    
    def augmented_generation(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Generate a response using RAG"""
        try:
            # Get relevant documents
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format context from retrieved documents
            context = ""
            references = []
            
            if results and results['documents']:
                for doc, metadata, distance in zip(
                    results['documents'][0], 
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    context += f"\n{doc}"
                    references.append({
                        "text": doc,
                        "metadata": metadata,
                        "relevance_score": 1 - distance  # Convert distance to similarity
                    })
            
            # Generate response with context
            prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

    Context:
    {context}

    Question: {query}

    Answer:"""
            
            response = self.llm.generate(prompt)
            
            return {
                "response": response,
                "references": references if references else []
            }
            
        except Exception as e:
            logger.error(f"Error in augmented generation: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "references": []
            }