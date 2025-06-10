"""Database operations for the RAG Chatbot using PostgreSQL."""

import os
import psycopg2
import psycopg2.extras
from typing import List, Dict, Any, Optional, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Base directory for persistent storage (mounted as Docker volume)
BASE_DIR = "data"
FAISS_DIR = os.path.join(BASE_DIR, "FAISS_Index")

# PostgreSQL connection string
DATABASE_URL = "postgresql://postgres:789456123@localhost/rag_chatbot"

def get_connection():
    """Get a database connection."""
    return psycopg2.connect(DATABASE_URL)

def init_database():
    """Initialize the PostgreSQL database with required tables."""
    os.makedirs(FAISS_DIR, exist_ok=True)

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Create conversations table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')

        # Create messages table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
        )
        ''')

        # Create sources table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id SERIAL PRIMARY KEY,
            message_id INTEGER NOT NULL,
            source_document TEXT NOT NULL,
            page_number INTEGER,
            score REAL,
            kb_name TEXT,
            FOREIGN KEY (message_id) REFERENCES messages (id) ON DELETE CASCADE
        )
        ''')

        # Create settings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            id SERIAL PRIMARY KEY,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL
        )
        ''')

        # Create knowledge_bases table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_bases (
            id SERIAL PRIMARY KEY,
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
            id SERIAL PRIMARY KEY,
            knowledge_base_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            document_type TEXT NOT NULL,
            page_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_bases (id) ON DELETE CASCADE
        )
        ''')

        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sources_message_id ON sources(message_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_documents_kb_id ON documents(knowledge_base_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_last_updated ON conversations(last_updated)')

        conn.commit()
        logger.info("Database initialized successfully")

    except Exception as e:
        conn.rollback()
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

    return DATABASE_URL

def get_conversations() -> List[Tuple[int, str, str]]:
    """Get all conversations from the database."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT id, title, created_at FROM conversations ORDER BY last_updated DESC")
        conversations = cursor.fetchall()
        return conversations
    except Exception as e:
        logger.error(f"Error getting conversations: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def create_conversation(title="New Chat") -> int:
    """Create a new conversation and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO conversations (title) VALUES (%s) RETURNING id", (title,))
        conversation_id = cursor.fetchone()[0]
        conn.commit()
        return conversation_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating conversation: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_messages(conversation_id: int) -> List[Tuple[int, str, str, str]]:
    """Get all messages for a specific conversation."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT id, role, content, created_at 
        FROM messages 
        WHERE conversation_id = %s 
        ORDER BY created_at
        """, (conversation_id,))
        messages = cursor.fetchall()
        return messages
    except Exception as e:
        logger.error(f"Error getting messages: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def add_message(conversation_id: int, role: str, content: str) -> int:
    """Add a new message to a conversation and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Add message
        cursor.execute("""
        INSERT INTO messages (conversation_id, role, content) 
        VALUES (%s, %s, %s) RETURNING id
        """, (conversation_id, role, content))
        message_id = cursor.fetchone()[0]

        # Update conversation last_updated timestamp
        cursor.execute("""
        UPDATE conversations 
        SET last_updated = CURRENT_TIMESTAMP 
        WHERE id = %s
        """, (conversation_id,))

        conn.commit()
        return message_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error adding message: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def add_sources(message_id: int, sources: List[Dict[str, Any]]) -> None:
    """Add source documents for a message with enhanced metadata."""
    if not sources:
        return

    conn = get_connection()
    cursor = conn.cursor()

    try:
        for source in sources:
            # Ensure score is a float
            try:
                score = float(source.get('score', 0.5))
            except (ValueError, TypeError):
                score = 0.5

            cursor.execute("""
            INSERT INTO sources (message_id, source_document, page_number, score, kb_name) 
            VALUES (%s, %s, %s, %s, %s)
            """, (
                message_id,
                source.get('source', ''),
                source.get('page', 0),
                score,
                source.get('kb_name', 'Unknown KB')
            ))

        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error adding sources: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_sources(message_id: int) -> List[Dict[str, Any]]:
    """Get sources for a specific message with KB information."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        SELECT source_document, page_number, score, kb_name
        FROM sources 
        WHERE message_id = %s 
        ORDER BY score DESC
        """, (message_id,))
        sources = cursor.fetchall()

        return [{"source": src, "page": page, "score": score, "kb_name": kb_name}
               for src, page, score, kb_name in sources]
    except Exception as e:
        logger.error(f"Error getting sources: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def update_conversation_title(conversation_id: int, new_title: str) -> None:
    """Update the title of a conversation."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("UPDATE conversations SET title = %s WHERE id = %s", (new_title, conversation_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error updating conversation title: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def delete_conversation(conversation_id: int) -> None:
    """Delete a conversation and all its messages (CASCADE will handle related records)."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Delete conversation (CASCADE will handle messages and sources)
        cursor.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting conversation: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def register_knowledge_base(name: str, embedding_model: str, chunking_strategy: str, description: str = "") -> int:
    """Register a new knowledge base and return its ID."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO knowledge_bases (name, description, embedding_model, chunking_strategy)
        VALUES (%s, %s, %s, %s) RETURNING id
        """, (name, description, embedding_model, chunking_strategy))
        kb_id = cursor.fetchone()[0]
        conn.commit()
        return kb_id
    except psycopg2.IntegrityError:
        # If the knowledge base already exists, get its ID
        conn.rollback()
        cursor.execute("SELECT id FROM knowledge_bases WHERE name = %s", (name,))
        result = cursor.fetchone()
        return result[0] if result else None
    except Exception as e:
        conn.rollback()
        logger.error(f"Error registering knowledge base: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_knowledge_bases() -> List[Dict[str, Any]]:
    """Get all knowledge bases."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cursor.execute("""
        SELECT id, name, description, created_at, document_count, embedding_model, chunking_strategy
        FROM knowledge_bases
        ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting knowledge bases: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def register_document(knowledge_base_id: int, filename: str, document_type: str, page_count: int, chunk_count: int) -> int:
    """Register a document in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Add document
        cursor.execute("""
        INSERT INTO documents (knowledge_base_id, filename, document_type, page_count, chunk_count)
        VALUES (%s, %s, %s, %s, %s) RETURNING id
        """, (knowledge_base_id, filename, document_type, page_count, chunk_count))
        document_id = cursor.fetchone()[0]

        # Update document count in knowledge base
        cursor.execute("""
        UPDATE knowledge_bases
        SET document_count = document_count + 1
        WHERE id = %s
        """, (knowledge_base_id,))

        conn.commit()
        return document_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error registering document: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_documents(knowledge_base_id: int) -> List[Dict[str, Any]]:
    """Get all documents for a knowledge base."""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cursor.execute("""
        SELECT id, filename, document_type, page_count, chunk_count, created_at
        FROM documents
        WHERE knowledge_base_id = %s
        ORDER BY created_at DESC
        """, (knowledge_base_id,))
        rows = cursor.fetchall()

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error getting documents: {str(e)}")
        return []
    finally:
        cursor.close()
        conn.close()

def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting from the database."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT value FROM settings WHERE key = %s", (key,))
        result = cursor.fetchone()

        if result:
            return result[0]
        return default
    except Exception as e:
        logger.error(f"Error getting setting: {str(e)}")
        return default
    finally:
        cursor.close()
        conn.close()

def set_setting(key: str, value: Any) -> None:
    """Set a setting in the database."""
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
        INSERT INTO settings (key, value) VALUES (%s, %s)
        ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value
        """, (key, str(value)))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Error setting value: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()

def get_active_knowledge_base() -> Optional[Dict[str, Any]]:
    """Get the currently active knowledge base."""
    active_kb_name = get_setting("active_knowledge_base")
    if not active_kb_name:
        return None

    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    try:
        cursor.execute("""
        SELECT id, name, description, created_at, document_count, embedding_model, chunking_strategy
        FROM knowledge_bases
        WHERE name = %s
        """, (active_kb_name,))
        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)
    except Exception as e:
        logger.error(f"Error getting active knowledge base: {str(e)}")
        return None
    finally:
        cursor.close()
        conn.close()

def set_active_knowledge_base(kb_name: str) -> None:
    """Set the active knowledge base."""
    set_setting("active_knowledge_base", kb_name)

# Connection pool management (optional but recommended for production)
def create_connection_pool():
    """Create a connection pool for better performance."""
    try:
        from psycopg2 import pool
        return pool.SimpleConnectionPool(
            1, 20,  # min and max connections
            DATABASE_URL
        )
    except ImportError:
        logger.warning("psycopg2.pool not available, using direct connections")
        return None