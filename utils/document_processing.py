"""Document processing functions for the RAG Chatbot."""

import os
import tempfile
import shutil
import logging
import base64
from typing import Dict, List, Any, Optional
from enum import Enum

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import (CharacterTextSplitter, RecursiveCharacterTextSplitter)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from utils.database import register_document, register_knowledge_base, get_knowledge_bases

BASE_DIR = "/app/data"
DB_DIR = os.path.join(BASE_DIR, "db")
FAISS_DIR = os.path.join(BASE_DIR, "FAISS_Index")
DB_PATH = os.path.join(DB_DIR, "chat_history.db")


# Setup logging
logger = logging.getLogger("rag-chatbot.document_processing")

# Define enums for embedding models and chunking strategies
class EmbeddingModel(Enum):
    OPEN_AI = "text-embedding-3-small"

class ChunkingStrategy(Enum):
    SEMANTIC_PERCENTILE = "semantic_percentile"
    SEMANTIC_INTERQUARTILE = "semantic_interquartile"
    SEMANTIC_STD_DEV = "semantic_std_dev"
    TEXT_SPLIT = "text_splitter"
    RECURSIVE = "recursive"

def load_pdf_with_pages(file):
    """Load a PDF file and extract content with page numbers."""
    logger.info(f"Loading PDF file: {file.name}")
    
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        try:
            # Save the uploaded file to a temporary file
            file_content = file.read()
            file.seek(0)  # Reset the file pointer for future use
            
            # Validate file is not corrupted
            if len(file_content) == 0:
                logger.error(f"File {file.name} is empty")
                raise ValueError(f"Uploaded file {file.name} is empty")
                
            temp_file.write(file_content)
            temp_file.flush()
            logger.debug(f"Saved file to temporary location: {temp_file.name}")

            # Use PyPDFLoader to load PDF content
            loader = PyPDFLoader(temp_file.name)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} pages from PDF")

            # Extract content and page numbers
            pages = []
            for doc in documents:
                page_number = doc.metadata.get('page', 0) + 1  # Convert to 1-based indexing
                pages.append({
                    "page_content": doc.page_content,
                    "page_number": page_number
                })

            return {
                "filename": file.name,
                "pages": pages
            }
        except Exception as e:
            logger.exception(f"Error loading PDF file {file.name}: {str(e)}")
            raise
        finally:
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Removed temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file.name}: {str(e)}")

def load_docx_with_pages(file):
    """Load a DOCX file and extract content with page numbers."""
    logger.info(f"Loading DOCX file: {file.name}")
    
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        try:
            # Save the uploaded file to a temporary file
            file_content = file.read()
            file.seek(0)  # Reset the file pointer for future use
            
            # Validate file is not corrupted
            if len(file_content) == 0:
                logger.error(f"File {file.name} is empty")
                raise ValueError(f"Uploaded file {file.name} is empty")
                
            temp_file.write(file_content)
            temp_file.flush()
            logger.debug(f"Saved file to temporary location: {temp_file.name}")

            # Use Docx2txtLoader to load DOCX content
            loader = Docx2txtLoader(temp_file.name)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} sections from DOCX")

            # Extract content and page numbers
            pages = []
            for i, doc in enumerate(documents):
                pages.append({
                    "page_content": doc.page_content,
                    "page_number": i + 1  # 1-based page numbering
                })

            return {
                "filename": file.name,
                "pages": pages
            }
        except Exception as e:
            logger.exception(f"Error loading DOCX file {file.name}: {str(e)}")
            raise
        finally:
            try:
                os.unlink(temp_file.name)
                logger.debug(f"Removed temporary file: {temp_file.name}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file.name}: {str(e)}")

def initialize_embedding_model(embedding_model):
    """Initialize the embedding model based on the model name."""
    logger.info(f"Initializing embedding model: {embedding_model}")
    
    try:
        if embedding_model.startswith("text"):  # OpenAI models
            embeddings = OpenAIEmbeddings(model=embedding_model)
            logger.debug(f"Initialized OpenAI embedding model: {embedding_model}")
            return embeddings
        else:  # Default to Hugging Face models
            # embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            # logger.debug(f"Initialized Hugging Face embedding model: {embedding_model}")
            # return embeddings
            print("Only Open ai is accepted")
    except Exception as e:
        logger.exception(f"Error initializing embedding model {embedding_model}: {str(e)}")
        raise

def create_chunking(
        chunking_type,
        documents,
        chunk_size=1000,
        chunk_overlap=100,
        percentile=90,
        interquartile_range_factor=1.5,
        standard_deviation_factor=3,
):
    """Create document chunks based on the specified chunking strategy."""
    logger.info(f"Creating chunks with strategy: {chunking_type}")
    logger.debug(f"Documents to chunk: {len(documents)}")
    
    try:
        if chunking_type == "text_splitter":
            logger.debug(f"Using CharacterTextSplitter with size={chunk_size}, overlap={chunk_overlap}")
            text_splitter = CharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunked_documents = text_splitter.split_documents(documents)
        elif chunking_type == "recursive":
            logger.debug(f"Using RecursiveCharacterTextSplitter with size={chunk_size}, overlap={chunk_overlap}")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            chunked_documents = text_splitter.split_documents(documents)
        elif chunking_type == "semantic_percentile":
            logger.debug(f"Using SemanticChunker with percentile={percentile}")
            embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
            semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile",
                                            breakpoint_threshold_amount=percentile)
            chunked_documents = semantic_chunker.split_documents(documents)
        elif chunking_type == "semantic_interquartile":
            logger.debug(f"Using SemanticChunker with interquartile factor={interquartile_range_factor}")
            embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
            semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="interquartile",
                                            breakpoint_threshold_amount=interquartile_range_factor)
            chunked_documents = semantic_chunker.split_documents(documents)
        elif chunking_type == "semantic_std_dev":
            logger.debug(f"Using SemanticChunker with std_dev factor={standard_deviation_factor}")
            embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
            semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="standard_deviation",
                                            breakpoint_threshold_amount=standard_deviation_factor)
            chunked_documents = semantic_chunker.split_documents(documents)
        else:
            logger.error(f"Invalid chunking type: {chunking_type}")
            raise ValueError(
                "Invalid chunking type. Choose from 'text_splitter', 'recursive', 'semantic_percentile', 'semantic_interquartile', or 'semantic_std_dev'."
            )

        logger.info(f"Created {len(chunked_documents)} chunks")
        return chunked_documents
    except Exception as e:
        logger.exception(f"Error creating chunks with strategy {chunking_type}: {str(e)}")
        raise

def process_and_chunk_file(
        file,
        kb_name,
        embedding_model_name,
        chunking_strategy_name,
        chunk_size=1000,
        chunk_overlap=200,
):
    """Process a file and chunk it into documents for indexing."""
    logger.info(f"Processing file {file.name} for knowledge base {kb_name}")
    
    try:
        # Ensure the FAISS_Index directory exists
        os.makedirs(FAISS_DIR, exist_ok=True)
        logger.debug("Ensured FAISS_Index directory exists")

        file_name = file.name.lower()

        # Register knowledge base if doesn't exist
        kb_id = register_knowledge_base(
            name=kb_name,
            embedding_model=embedding_model_name, 
            chunking_strategy=chunking_strategy_name
        )
        logger.debug(f"Registered knowledge base with ID: {kb_id}")

        # Load the file with page metadata
        if file_name.endswith('.pdf'):
            file_data = load_pdf_with_pages(file)
            document_type = "pdf"
        elif file_name.endswith('.docx'):
            file_data = load_docx_with_pages(file)
            document_type = "docx"
        else:
            logger.error(f"Unsupported file format: {file_name}")
            raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

        # Convert the loaded data with page metadata into LangChain document format
        total_pages = len(file_data["pages"])
        documents = []
        for page in file_data["pages"]:
            documents.append(
                Document(
                    page_content=page["page_content"],
                    metadata={
                        "page": page["page_number"] - 1,
                        "source": file_data["filename"],
                        "total_pages": total_pages
                    }
                )
            )
        logger.debug(f"Created {len(documents)} document objects")

        # Initialize embedding model
        embedding_model = initialize_embedding_model(embedding_model_name)

        # Create chunks
        chunks = create_chunking(
            chunking_type=chunking_strategy_name, 
            documents=documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Register document in database
        register_document(
            knowledge_base_id=kb_id,
            filename=file_data["filename"],
            document_type=document_type,
            page_count=total_pages,
            chunk_count=len(chunks)
        )
        logger.debug(f"Registered document in database")

        # Create or update FAISS index
        index_path = f"{FAISS_DIR}/{kb_name}"
        logger.debug(f"Preparing FAISS index at: {index_path}")

        try:
            if os.path.exists(index_path) and os.path.isdir(index_path):
                # Try to update existing index
                logger.info(f"Updating existing FAISS index at {index_path}")
                existing_vectorstore = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
                existing_vectorstore.add_documents(chunks)
                existing_vectorstore.save_local(index_path)
                logger.debug(f"Updated existing FAISS index with {len(chunks)} chunks")
            else:
                # Create new index
                logger.info(f"Creating new FAISS index at {index_path}")
                vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
                vectorstore.save_local(index_path)
                logger.debug(f"Created new FAISS index with {len(chunks)} chunks")
        except Exception as e:
            # Handle specific FAISS errors
            logger.exception(f"FAISS error: {str(e)}")
            logger.info("Trying to create a new index due to error")
            
            # Remove problematic directory if it exists
            if os.path.exists(index_path):
                try:
                    shutil.rmtree(index_path)
                    logger.debug(f"Removed problematic FAISS index directory: {index_path}")
                except Exception as rm_error:
                    logger.error(f"Failed to remove directory {index_path}: {str(rm_error)}")
            
            # Create a new clean index
            os.makedirs(index_path, exist_ok=True)
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embedding_model)
            vectorstore.save_local(index_path)
            logger.debug(f"Created new FAISS index with {len(chunks)} chunks after error recovery")

        logger.info(f"Successfully processed file {file.name} with {len(chunks)} chunks")
        return {
            "status": "success",
            "filename": file_data["filename"],
            "page_count": total_pages,
            "chunk_count": len(chunks),
            "kb_name": kb_name
        }
    except Exception as e:
        logger.exception(f"Error processing file {file.name}: {str(e)}")
        import traceback
        return {
            "status": "error", 
            "message": str(e), 
            "traceback": traceback.format_exc()
        }

def retrieve_documents(kb_name: str, embedding_model_name: str, query: str, k: int = 4) -> List[Document]:
    """Retrieve relevant documents for a query."""
    logger.info(f"Retrieving documents for query from KB: {kb_name}")
    
    try:
        path = f"{FAISS_DIR}/{kb_name}"
        if not os.path.exists(path):
            logger.error(f"Knowledge base directory does not exist: {path}")
            raise ValueError(f"Knowledge base '{kb_name}' does not exist at path: {path}")
            
        embeddings = initialize_embedding_model(embedding_model_name)
        vectorstore = FAISS.load_local(
            folder_path=path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.debug(f"Loaded FAISS index from: {path}")
        
        docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)
        logger.debug(f"Retrieved {len(docs_and_scores)} documents with scores")

        for doc, score in docs_and_scores:
            normalized_score = 1 / (1 + float(score))
            doc.metadata['score'] = normalized_score

        return [doc for doc, _ in docs_and_scores]
    except Exception as e:
        logger.exception(f"Error retrieving documents: {str(e)}")
        raise Exception(f"Error retrieving documents: {str(e)}")

def get_retriever(kb_name: str, embedding_model_name: str, k: int = 16, search_type: str = "mmr"):
    """Get a retriever for the specified index."""
    logger.info(f"Creating retriever for KB: {kb_name}")
    
    try:
        path = f"{FAISS_DIR}/{kb_name}"
        
        # Detailed validation of the knowledge base directory
        if not os.path.exists(path):
            logger.error(f"Knowledge base path does not exist: {path}")
            raise ValueError(f"Knowledge base '{kb_name}' does not exist at path: {path}")
            
        if not os.path.isdir(path):
            logger.error(f"Knowledge base path is not a directory: {path}")
            raise ValueError(f"Knowledge base path '{path}' exists but is not a directory")
        
        # Check if directory is empty
        if not os.listdir(path):
            logger.error(f"Knowledge base directory is empty: {path}")
            raise ValueError(f"Knowledge base directory '{path}' exists but is empty")
            
        # Initialize embedding model
        embeddings = initialize_embedding_model(embedding_model_name)
        
        # Load the vectorstore
        vectorstore = FAISS.load_local(
            folder_path=path, 
            embeddings=embeddings, 
            allow_dangerous_deserialization=True
        )
        logger.debug(f"Loaded FAISS index from: {path}")
        
        # Get retriever directly from vectorstore without modifying methods
        retriever = vectorstore.as_retriever(
            search_type=search_type, 
            k=k,
            fetch_k=k*2
        )
        
        # We won't modify the retriever's methods anymore - this was causing the error
        logger.debug(f"Created retriever with search_type={search_type}, k={k}")
        
        return retriever
    except Exception as e:
        logger.exception(f"Failed to get retriever for {kb_name}: {str(e)}")
        import traceback
        logger.debug(f"Retriever error traceback: {traceback.format_exc()}")
        raise Exception(f"Error getting retriever: {str(e)}")

def auto_create_knowledge_base_if_needed(embedding_model_name: str = "text-embedding-3-small") -> str:
    """Auto-create a knowledge base if none exists."""
    logger.info("Checking if knowledge base needs to be auto-created")
    
    # Check if any knowledge bases exist
    kbs = get_knowledge_bases()
    
    if not kbs:
        # Create a default knowledge base
        kb_name = "default_knowledge_base"
        logger.info(f"No knowledge bases found, creating default: {kb_name}")
        
        kb_id = register_knowledge_base(
            name=kb_name,
            embedding_model=embedding_model_name,
            chunking_strategy="semantic_percentile",  # Changed to match main processing
            description="Default knowledge base"
        )
        
        # Create an empty FAISS index directory for this knowledge base
        index_path = f"{FAISS_DIR}/{kb_name}"
        os.makedirs(index_path, exist_ok=True)
        logger.debug(f"Created empty index directory: {index_path}")
        
        return kb_name
    
    # Return the first knowledge base name
    logger.info(f"Using existing knowledge base: {kbs[0]['name']}")
    return kbs[0]["name"]

def get_all_knowledge_base_names() -> List[str]:
    """Get names of all knowledge bases."""
    logger.debug("Getting all knowledge base names")
    kbs = get_knowledge_bases()
    names = [kb["name"] for kb in kbs]
    logger.debug(f"Found {len(names)} knowledge bases")
    return names

def kb_exists(kb_name: str) -> bool:
    """Check if a knowledge base exists."""
    exists = os.path.exists(f"{FAISS_DIR}/{kb_name}")
    logger.debug(f"Checking if knowledge base {kb_name} exists: {exists}")
    return exists