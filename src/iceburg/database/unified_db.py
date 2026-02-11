"""
Unified Database Abstraction Layer
==================================
Provides dual-write capability for SQLite and PostgreSQL migration

Feature Flag: ICEBURG_USE_POSTGRES (default: false)
- When false: Use SQLite only (current behavior)
- When true: Use PostgreSQL, also write to SQLite (dual-write for safety)
- Read from PostgreSQL when enabled, SQLite when disabled
"""

import os
import logging
import sqlite3
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Feature flag
USE_POSTGRES = os.getenv("ICEBURG_USE_POSTGRES", "0") == "1"
DUAL_WRITE = USE_POSTGRES  # When using PostgreSQL, also write to SQLite

# PostgreSQL connection (lazy-loaded)
_postgres_conn = None


def get_postgres_connection():
    """Get PostgreSQL connection (lazy-loaded)"""
    global _postgres_conn
    
    if not USE_POSTGRES:
        return None
    
    if _postgres_conn is None:
        try:
            import psycopg2
            from psycopg2.extras import RealDictCursor
            
            database_url = os.getenv("DATABASE_URL")
            if not database_url:
                logger.warning("DATABASE_URL not set, PostgreSQL disabled")
                return None
            
            _postgres_conn = psycopg2.connect(
                database_url,
                cursor_factory=RealDictCursor
            )
            logger.info("✅ PostgreSQL connection established")
        except ImportError:
            logger.warning("psycopg2 not installed, PostgreSQL disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            return None
    
    return _postgres_conn


class UnifiedDatabase:
    """
    Unified database abstraction with dual-write support
    """
    
    def __init__(self, sqlite_path: Optional[Path] = None):
        """
        Initialize unified database
        
        Args:
            sqlite_path: Path to SQLite database (defaults to conversations.db)
        """
        self.use_postgres = USE_POSTGRES
        self.dual_write = DUAL_WRITE
        
        # SQLite connection
        if sqlite_path is None:
            sqlite_path = Path.home() / "Documents" / "iceburg_data" / "conversations.db"
        self.sqlite_path = Path(sqlite_path)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite
        self._init_sqlite()
        
        # Initialize PostgreSQL if enabled
        if self.use_postgres:
            self._init_postgres()
        
        logger.info(f"UnifiedDatabase initialized: postgres={self.use_postgres}, dual_write={self.dual_write}")
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(str(self.sqlite_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_message TEXT NOT NULL,
                agent_used TEXT,
                mode TEXT,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_conversation_id 
            ON conversations(conversation_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_postgres(self):
        """Initialize PostgreSQL database schema"""
        conn = get_postgres_connection()
        if not conn:
            return
        
        try:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    conversation_id TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_message TEXT NOT NULL,
                    assistant_message TEXT,
                    agent_used VARCHAR(50),
                    mode VARCHAR(50),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_conversation_id 
                ON conversations(conversation_id)
            ''')
            
            conn.commit()
            logger.info("✅ PostgreSQL schema initialized")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL schema: {e}")
            conn.rollback()
    
    def save_conversation(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        agent_used: Optional[str] = None,
        mode: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Save conversation to database(s)"""
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Write to SQLite (always, for dual-write or fallback)
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO conversations 
                (conversation_id, timestamp, user_message, assistant_message, agent_used, mode, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (conversation_id, timestamp, user_message, assistant_message, agent_used, mode, metadata_json))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to write to SQLite: {e}")
        
        # Write to PostgreSQL if enabled
        if self.use_postgres:
            try:
                pg_conn = get_postgres_connection()
                if pg_conn:
                    cursor = pg_conn.cursor()
                    cursor.execute('''
                        INSERT INTO conversations 
                        (conversation_id, timestamp, user_message, assistant_message, agent_used, mode, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ''', (conversation_id, timestamp, user_message, assistant_message, agent_used, mode, json.dumps(metadata) if metadata else None))
                    pg_conn.commit()
            except Exception as e:
                logger.error(f"Failed to write to PostgreSQL: {e}")
                # Don't fail if PostgreSQL write fails - SQLite write succeeded
    
    def get_conversations(
        self,
        conversation_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get conversations from database"""
        if self.use_postgres:
            # Read from PostgreSQL
            try:
                pg_conn = get_postgres_connection()
                if pg_conn:
                    cursor = pg_conn.cursor()
                    if conversation_id:
                        cursor.execute('''
                            SELECT * FROM conversations 
                            WHERE conversation_id = %s 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        ''', (conversation_id, limit))
                    else:
                        cursor.execute('''
                            SELECT * FROM conversations 
                            ORDER BY timestamp DESC 
                            LIMIT %s
                        ''', (limit,))
                    
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
            except Exception as e:
                logger.error(f"Failed to read from PostgreSQL: {e}, falling back to SQLite")
        
        # Read from SQLite (fallback or when PostgreSQL disabled)
        try:
            conn = sqlite3.connect(str(self.sqlite_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if conversation_id:
                cursor.execute('''
                    SELECT * FROM conversations 
                    WHERE conversation_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (conversation_id, limit))
            else:
                cursor.execute('''
                    SELECT * FROM conversations 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to read from SQLite: {e}")
            return []


# Global instance (singleton pattern)
_unified_db_instance = None


def get_unified_db() -> UnifiedDatabase:
    """Get global unified database instance"""
    global _unified_db_instance
    if _unified_db_instance is None:
        _unified_db_instance = UnifiedDatabase()
    return _unified_db_instance
