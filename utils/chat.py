"""Chat functionality for the RAG Chatbot."""
# Add or update these imports at the top of chat.py
import logging
from typing import List, Dict, Any, Optional
import yaml
import os

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS

# Import from document_processing (this was missing)
from utils.document_processing import initialize_embedding_model, get_retriever
from utils.database import add_message, add_sources, get_messages, update_conversation_title


BASE_DIR = "/app/data"
DB_DIR = os.path.join(BASE_DIR, "db")
FAISS_DIR = os.path.join(BASE_DIR, "FAISS_Index")
DB_PATH = os.path.join(DB_DIR, "chat_history.db")

# Setup logging
logger = logging.getLogger("rag-chatbot.chat")


def load_settings(settings_path="settings.yml"):
    """Load settings from a YAML file and override with environment variables."""
    try:
        with open(settings_path, 'r') as file:
            settings = yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Error loading settings file: {e}")
        # Return default settings
        settings = {
            "chat_model": "gpt-4o-mini",
            "embeddings_model": "text-embedding-3-small",
            "retrieval": {
                "top_k": 4,
                "search_type": "mmr",
                "fetch_k": 8
            }
        }

    # Override with environment variables if they exist
    if os.getenv('OPENAI_API_KEY'):
        settings['openai_key'] = os.getenv('OPENAI_API_KEY')

    return settings

def get_conversation_chain(kb_name: str, embedding_model: str, chat_model: str, retrieval_k: int = 4):
    """Create a conversational chain for the RAG system."""
    logger.info(f"Creating conversation chain with KB: {kb_name}, model: {chat_model}")
    
    try:
        # Get the retriever for the knowledge base
        retriever = get_retriever(
            kb_name=kb_name,
            embedding_model_name=embedding_model,
            k=retrieval_k,
            search_type="mmr"
        )
        
        # Initialize the LLM
        llm = ChatOpenAI(model_name=chat_model, temperature=0.7)
        logger.debug(f"Initialized ChatOpenAI with model: {chat_model}")
        
        # Create system prompt template with better context handling
        template = """
        You are an Enterprise RAG (Retrieval-Augmented Generation) Chatbot providing helpful information based on the documents in the knowledge base.
        
        Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. Don't try to make up an answer.
        The answer should be comprehensive and detailed, drawing specifically from the provided context.
        
        If the question relates to the documents, provide page numbers and document names as source citations.
        
        Context:
        {context}
        
        Chat History:
        {chat_history}
        
        Question: {question}
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template,
        )
        
        # Initialize memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create the chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=False  # Set to True for debugging
        )
        
        logger.info("Conversation chain created successfully")
        return chain
    except Exception as e:
        logger.exception(f"Error creating conversation chain: {str(e)}")
        raise

def process_query(conversation_id: int, query: str, kb_name: str = None, kb_names: list = None, embedding_model: str = "text-embedding-3-small", chat_model: str = "gpt-4o-mini", retrieval_k: int = 4):
    """Process a user query and generate a response with sources from multiple knowledge bases."""
    logger.info(f"Processing query for conversation: {conversation_id}")
    
    # If kb_names is provided, use all selected KBs instead of just the active one
    if kb_names and len(kb_names) > 0:
        using_multiple_kbs = True
        logger.info(f"Using multiple knowledge bases: {', '.join(kb_names)}")
    elif kb_name:
        kb_names = [kb_name]
        using_multiple_kbs = False
        logger.info(f"Using single knowledge base: {kb_name}")
    else:
        logger.error("No knowledge base specified")
        raise ValueError("No knowledge base specified")
    
    try:
        # We'll collect all source documents across knowledge bases
        all_source_docs = []
        all_kb_docs = {}  # Store documents by KB
        all_answers = []
        queried_kb_count = 0
        successful_kb_count = 0
        
        # Query each knowledge base
        for current_kb in kb_names:
            try:
                queried_kb_count += 1
                logger.info(f"Processing KB: {current_kb}")
                
                # Check if the KB exists before trying to access it
                kb_path = f"{FAISS_DIR}/{current_kb}"
                if not os.path.exists(kb_path) or not os.path.isdir(kb_path):
                    logger.warning(f"Knowledge base {current_kb} does not exist or is not accessible at {kb_path}")
                    continue
                
                # Initialize embedding model from document_processing module
                embeddings = initialize_embedding_model(embedding_model)
                
                # Load vectorstore directly
                vectorstore = FAISS.load_local(
                    folder_path=kb_path, 
                    embeddings=embeddings, 
                    allow_dangerous_deserialization=True
                )
                
                # Get relevant documents directly with similarity scores
                docs_and_scores = vectorstore.similarity_search_with_score(query, k=retrieval_k)
                
                # Only add documents if we got results
                if docs_and_scores:
                    # Process the retrieved documents
                    kb_docs = []
                    for doc, score in docs_and_scores:
                        # Calculate a true relevance score (convert distance to similarity)
                        # FAISS returns distance, smaller is better, so we invert
                        relevance = 1.0 / (1.0 + float(score))
                        
                        # Add metadata
                        doc.metadata['kb_name'] = current_kb
                        doc.metadata['score'] = relevance
                        doc.metadata['raw_score'] = float(score)
                        
                        # Add document to KB-specific list and all documents list
                        kb_docs.append(doc)
                    
                    # Store documents for this KB
                    all_kb_docs[current_kb] = kb_docs
                    all_source_docs.extend(kb_docs)
                    
                    # Only try to get an answer if we have documents
                    if kb_docs:
                        # Initialize the LLM
                        llm = ChatOpenAI(model_name=chat_model, temperature=0.7)
                        
                        # Create context from the top documents
                        context = "\n\n".join([
                            f"Document: {doc.metadata.get('source', 'Unknown')}, "
                            f"Page: {doc.metadata.get('page', 0) + 1}, "
                            f"Knowledge Base: {current_kb}\n{doc.page_content}"
                            for doc in kb_docs[:retrieval_k]
                        ])
                        
                        # Create system prompt with better support for multilingual content
                        template = f"""
                        You are an Enterprise RAG (Retrieval-Augmented Generation) Chatbot providing helpful information based on the documents in the knowledge base.
                        
                        Use the following context to answer the question. If you don't know the answer, just say you don't know. Don't try to make up an answer.
                        
                        If the content is in Arabic or another language, preserve that language in your response. DO NOT translate non-English content unless specifically asked.
                        
                        Always provide page numbers and document names as source citations.
                        
                        Context:
                        {context}
                        
                        Question: {query}
                        
                        Answer (preserving any original language in the content):
                        """
                        
                        # Get response from LLM
                        messages = [{"role": "system", "content": template}]
                        response = llm.invoke(messages)
                        answer = response.content
                        
                        all_answers.append((current_kb, answer))
                        successful_kb_count += 1
                
            except Exception as kb_error:
                logger.error(f"Error querying KB {current_kb}: {str(kb_error)}")
                continue
        
        # Sort documents by relevance score
        all_source_docs.sort(key=lambda doc: doc.metadata.get('score', 0), reverse=True)
        
        # Deduplicate sources while preserving order
        seen_sources = set()
        unique_source_docs = []
        for doc in all_source_docs:
            source_key = (doc.metadata.get('source', ''), doc.metadata.get('page', 0))
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_source_docs.append(doc)
        
        # Determine the final answer
        if len(all_answers) == 0:
            answer = f"I couldn't find any relevant information in the {queried_kb_count} selected knowledge bases. Please check if these knowledge bases contain the information you're looking for, or try rephrasing your question."
        elif len(all_answers) == 1:
            answer = all_answers[0][1]
        else:
            # With multiple KBs that returned results, create a prompt that highlights the most relevant content
            # Sort all documents by relevance score to prioritize the most relevant content
            top_docs = sorted(all_source_docs, key=lambda d: d.metadata.get('score', 0), reverse=True)[:retrieval_k * 2]
            
            # Create context with the most relevant documents from all KBs
            context_from_kbs = "\n\n".join([
                f"Information from {doc.metadata.get('kb_name', 'Unknown KB')} - {doc.metadata.get('source', 'Unknown source')}, "
                f"Page {doc.metadata.get('page', 0) + 1} (Relevance: {int(doc.metadata.get('score', 0.5) * 100)}%):\n{doc.page_content}" 
                for doc in top_docs
            ])
            
            synthesis_prompt = f"""
            Based on information from multiple knowledge bases, provide a comprehensive answer to the query: "{query}"
            
            Remember:
            - Preserve the original language of any content (e.g., if content is in Arabic, maintain it in Arabic).
            - Clearly cite sources by specifying the knowledge base name, document name, and page number.
            - Prioritize information from more relevant sources (higher relevance percentage).
            - Make sure to directly address the query.
            
            Here is the relevant information from the knowledge bases:
            
            {context_from_kbs}
            
            Provide a well-structured answer that directly addresses the query.
            """
            
            # Generate a synthesized answer
            llm = ChatOpenAI(model_name=chat_model, temperature=0.7)
            messages = [{"role": "system", "content": synthesis_prompt}]
            response = llm.invoke(messages)
            answer = response.content
        
        # Format sources with detailed information - create two distinct sets
        query_relevant_sources = []
        kb_sources = {}
        
        # Process unique source documents for query-relevant sources
        for doc in unique_source_docs:
            source = {
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 0) + 1,  # Convert to 1-based indexing
                'score': doc.metadata.get('score', 0.5),
                'kb_name': doc.metadata.get('kb_name', 'Unknown KB')
            }
            query_relevant_sources.append(source)
        
        # Process documents by KB for the KB-specific tab
        for kb_name, docs in all_kb_docs.items():
            kb_specific_sources = []
            seen = set()
            
            for doc in docs:
                key = (doc.metadata.get('source', 'Unknown'), doc.metadata.get('page', 0))
                if key not in seen:
                    seen.add(key)
                    kb_specific_sources.append({
                        'source': doc.metadata.get('source', 'Unknown'),
                        'page': doc.metadata.get('page', 0) + 1,
                        'score': doc.metadata.get('score', 0.5),
                        'kb_name': kb_name
                    })
            
            kb_sources[kb_name] = kb_specific_sources
        
        # Add assistant message to database
        assistant_message_id = add_message(conversation_id, "assistant", answer)
        
        # Combine all sources for database storage
        all_sources = query_relevant_sources
        
        # Add sources to database
        if all_sources:
            add_sources(assistant_message_id, all_sources)
            logger.debug(f"Added {len(all_sources)} sources to message {assistant_message_id}")
        
        return {
            "content": answer,
            "sources": {
                "query_relevant": query_relevant_sources,
                "kb_specific": kb_sources
            },
            "message_id": assistant_message_id
        }
    except Exception as e:
        logger.exception(f"Error processing query: {str(e)}")
        error_message = f"I encountered an error while processing your query: {str(e)}"
        message_id = add_message(conversation_id, "assistant", error_message)
        return {
            "content": error_message,
            "sources": {"query_relevant": [], "kb_specific": {}},
            "message_id": message_id,
            "error": str(e)
        }

def direct_openai_query(conversation_id: int, query: str, model_name: str = "gpt-4o-mini"):
    """Process a direct query to OpenAI without using a knowledge base."""
    logger.info(f"Processing direct OpenAI query with model: {model_name}")
    
    try:
        # Initialize the LLM
        llm = ChatOpenAI(model_name=model_name, temperature=0.7)
        
        # Simple system prompt for direct conversation
        system_message = """
        You are a helpful AI assistant. Provide informative, accurate, and helpful responses to the user's questions.
        If you don't know the answer to something, be honest about it rather than making up information.
        """
        
        # Get message history
        messages = get_messages(conversation_id)
        formatted_messages = []
        
        # Add system message
        formatted_messages.append({"role": "system", "content": system_message})
        
        # Add conversation history (excluding system messages)
        for msg_id, role, content, timestamp in messages:
            if role != "system":
                formatted_messages.append({"role": role, "content": content})
        
        # Process the query
        logger.debug(f"Sending direct query to OpenAI: {query[:50]}...")
        response = llm.invoke(formatted_messages)
        answer = response.content
        
        logger.debug(f"Received direct answer of length: {len(answer)}")
        
        # Add assistant message to database
        assistant_message_id = add_message(conversation_id, "assistant", answer)
        
        return {
            "content": answer,
            "message_id": assistant_message_id
        }
    except Exception as e:
        logger.exception(f"Error in direct OpenAI query: {str(e)}")
        error_message = f"I encountered an error while processing your query: {str(e)}"
        message_id = add_message(conversation_id, "assistant", error_message)
        return {
            "content": error_message,
            "message_id": message_id,
            "error": str(e)
        }


def format_sources(source_docs):
    """Format source documents for display with enhanced information."""
    sources = []
    for doc in source_docs:
        # Safely handle score conversion
        try:
            score_value = doc.metadata.get('score', 0.5)
            score = float(score_value) if score_value is not None else 0.5
        except (ValueError, TypeError):
            score = 0.5  # Default if conversion fails
            
        source = {
            'source': doc.metadata.get('source', 'Unknown'),
            'page': doc.metadata.get('page', 0) + 1,  # Convert to 1-based indexing
            'score': score,
            'kb_name': doc.metadata.get('kb_name', 'Unknown KB')
        }
        sources.append(source)
    
    # Remove duplicates while preserving order
    unique_sources = []
    seen = set()
    for source in sources:
        key = (source['source'], source['page'], source['kb_name'])
        if key not in seen:
            seen.add(key)
            unique_sources.append(source)
    
    # Sort by relevance score
    return sorted(unique_sources, key=lambda x: x['score'], reverse=True)

def get_suggested_prompts() -> List[str]:
    """Get suggested prompts for the user."""
    return [
        "Can you summarize the key points in these documents?",
        "What are the main topics covered in the knowledge base?",
        "Extract the most important information from these files.",
        "What insights can you provide based on the documents?",
        "Compare and contrast the information across these documents.",
        "Generate a concise summary of the uploaded content."
    ]