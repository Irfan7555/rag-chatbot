"""Database operations for the RAG Chatbot."""

import os
import sqlite3
from typing import List, Dict, Any, Optional, Tuple

# Base directory for persistent storage (mounted as Docker volume)
BASE_DIR = "/app/data"
DB_DIR = os.path.join(BASE_DIR, "db")
FAISS_DIR = os.path.join(BASE_DIR, "FAISS_Index")
DB_PATH = os.path.join(DB_DIR, "chat_history.db")

def init_database():
    """Initialize the SQLite database with required tables."""


    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(FAISS_DIR, exist_ok=True)
    
   
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create conversations table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        conversation_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (conversation_id) REFERENCES conversations (id)
    )
    ''')
    
    # Create sources table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL,
        source_document TEXT NOT NULL,
        page_number INTEGER,
        score REAL,
        FOREIGN KEY (message_id) REFERENCES messages (id)
    )
    ''')
    
    # Create settings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT UNIQUE NOT NULL,
        value TEXT NOT NULL
    )
    ''')
    
    # Create knowledge_bases table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS knowledge_bases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        document_count INTEGER DEFAULT 0,
        embedding_model TEXT NOT NULL,
        chunking_strategy TEXT NOT NULL
    )
    ''')
    
    # Create documents table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        knowledge_base_id INTEGER NOT NULL,
        filename TEXT NOT NULL,
        document_type TEXT NOT NULL,
        page_count INTEGER NOT NULL,
        chunk_count INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    return DB_PATH

def get_conversations() -> List[Tuple[int, str, str]]:
    """Get all conversations from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM conversations ORDER BY last_updated DESC")
    conversations = cursor.fetchall()
    conn.close()
    return conversations

def create_conversation(title="New Chat") -> int:
    """Create a new conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO conversations (title) VALUES (?)", (title,))
    conversation_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return conversation_id

def get_messages(conversation_id: int) -> List[Tuple[int, str, str, str]]:
    """Get all messages for a specific conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, role, content, created_at 
    FROM messages 
    WHERE conversation_id = ? 
    ORDER BY created_at
    """, (conversation_id,))
    messages = cursor.fetchall()
    conn.close()
    return messages

def add_message(conversation_id: int, role: str, content: str) -> int:
    """Add a new message to a conversation and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Add message
    cursor.execute("""
    INSERT INTO messages (conversation_id, role, content) 
    VALUES (?, ?, ?)
    """, (conversation_id, role, content))
    message_id = cursor.lastrowid
    
    # Update conversation last_updated timestamp
    cursor.execute("""
    UPDATE conversations 
    SET last_updated = CURRENT_TIMESTAMP 
    WHERE id = ?
    """, (conversation_id,))
    
    conn.commit()
    conn.close()
    return message_id

# In database.py, update the add_sources function

def add_sources(message_id: int, sources: List[Dict[str, Any]]) -> None:
    """Add source documents for a message with enhanced metadata."""
    if not sources:
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # First, make sure the sources table has the necessary columns
    try:
        # Check if kb_name column exists
        cursor.execute("PRAGMA table_info(sources)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'kb_name' not in columns:
            # Add kb_name column if it doesn't exist
            cursor.execute("ALTER TABLE sources ADD COLUMN kb_name TEXT")
            conn.commit()
    except Exception as e:
        logger.warning(f"Error checking/updating sources table schema: {str(e)}")
    
    for source in sources:
        # Ensure score is a float
        try:
            score = float(source.get('score', 0.5))
        except (ValueError, TypeError):
            score = 0.5
            
        cursor.execute("""
        INSERT INTO sources (message_id, source_document, page_number, score, kb_name) 
        VALUES (?, ?, ?, ?, ?)
        """, (
            message_id, 
            source.get('source', ''), 
            source.get('page', 0), 
            score,
            source.get('kb_name', 'Unknown KB')
        ))
    
    conn.commit()
    conn.close()

def get_sources(message_id: int) -> List[Dict[str, Any]]:
    """Get sources for a specific message with KB information."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if kb_name column exists
        cursor.execute("PRAGMA table_info(sources)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'kb_name' in columns:
            cursor.execute("""
            SELECT source_document, page_number, score, kb_name
            FROM sources 
            WHERE message_id = ? 
            ORDER BY score DESC
            """, (message_id,))
            sources = cursor.fetchall()
            conn.close()
            
            return [{"source": src, "page": page, "score": score, "kb_name": kb_name} 
                   for src, page, score, kb_name in sources]
        else:
            # Fallback if kb_name column doesn't exist
            cursor.execute("""
            SELECT source_document, page_number, score
            FROM sources 
            WHERE message_id = ? 
            ORDER BY score DESC
            """, (message_id,))
            sources = cursor.fetchall()
            conn.close()
            
            return [{"source": src, "page": page, "score": score, "kb_name": "Unknown KB"} 
                   for src, page, score in sources]
    except Exception as e:
        logger.warning(f"Error getting sources: {str(e)}")
        conn.close()
        return []
def update_conversation_title(conversation_id: int, new_title: str) -> None:
    """Update the title of a conversation."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE conversations SET title = ? WHERE id = ?", (new_title, conversation_id))
    conn.commit()
    conn.close()

def delete_conversation(conversation_id: int) -> None:
    """Delete a conversation and all its messages."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Delete sources first (foreign key constraint)
    cursor.execute("""
    DELETE FROM sources 
    WHERE message_id IN (SELECT id FROM messages WHERE conversation_id = ?)
    """, (conversation_id,))
    
    # Delete messages
    cursor.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    
    # Delete conversation
    cursor.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    
    conn.commit()
    conn.close()

def register_knowledge_base(name: str, embedding_model: str, chunking_strategy: str, description: str = "") -> int:
    """Register a new knowledge base and return its ID."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("""
        INSERT INTO knowledge_bases (name, description, embedding_model, chunking_strategy)
        VALUES (?, ?, ?, ?)
        """, (name, description, embedding_model, chunking_strategy))
        kb_id = cursor.lastrowid
        conn.commit()
        return kb_id
    except sqlite3.IntegrityError:
        # If the knowledge base already exists, get its ID
        cursor.execute("SELECT id FROM knowledge_bases WHERE name = ?", (name,))
        result = cursor.fetchone()
        return result[0] if result else None
    finally:
        conn.close()

def get_knowledge_bases() -> List[Dict[str, Any]]:
    """Get all knowledge bases."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, name, description, created_at, document_count, embedding_model, chunking_strategy
    FROM knowledge_bases
    ORDER BY created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": row[3],
            "document_count": row[4],
            "embedding_model": row[5],
            "chunking_strategy": row[6]
        }
        for row in rows
    ]

def register_document(knowledge_base_id: int, filename: str, document_type: str, page_count: int, chunk_count: int) -> int:
    """Register a document in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Add document
    cursor.execute("""
    INSERT INTO documents (knowledge_base_id, filename, document_type, page_count, chunk_count)
    VALUES (?, ?, ?, ?, ?)
    """, (knowledge_base_id, filename, document_type, page_count, chunk_count))
    document_id = cursor.lastrowid
    
    # Update document count in knowledge base
    cursor.execute("""
    UPDATE knowledge_bases
    SET document_count = document_count + 1
    WHERE id = ?
    """, (knowledge_base_id,))
    
    conn.commit()
    conn.close()
    return document_id

def get_documents(knowledge_base_id: int) -> List[Dict[str, Any]]:
    """Get all documents for a knowledge base."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, filename, document_type, page_count, chunk_count, created_at
    FROM documents
    WHERE knowledge_base_id = ?
    ORDER BY created_at DESC
    """, (knowledge_base_id,))
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": row[0],
            "filename": row[1],
            "document_type": row[2],
            "page_count": row[3],
            "chunk_count": row[4],
            "created_at": row[5]
        }
        for row in rows
    ]

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return result[0]
    return default

def set_setting(key: str, value: Any) -> None:
    """Set a setting in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO settings (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = ?
    """, (key, str(value), str(value)))
    conn.commit()
    conn.close()

def get_active_knowledge_base() -> Optional[Dict[str, Any]]:
    """Get the currently active knowledge base."""
    active_kb_name = get_setting("active_knowledge_base")
    if not active_kb_name:
        return None
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, name, description, created_at, document_count, embedding_model, chunking_strategy
    FROM knowledge_bases
    WHERE name = ?
    """, (active_kb_name,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "id": row[0],
        "name": row[1],
        "description": row[2],
        "created_at": row[3],
        "document_count": row[4],
        "embedding_model": row[5],
        "chunking_strategy": row[6]
    }

def set_active_knowledge_base(kb_name: str) -> None:
    """Set the active knowledge base."""
    set_setting("active_knowledge_base", kb_name)