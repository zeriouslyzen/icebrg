"""
Local Persistence System
Comprehensive local storage for ICEBURG personality, information, and all interactions
Similar to browser storage but for long-term persistence on M4 Mac
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class PersonalityState:
    """ICEBURG's personality state"""
    identity: str = "ICEBURG"
    personality_traits: Dict[str, Any] = None
    preferences: Dict[str, Any] = None
    knowledge_base: Dict[str, Any] = None
    memory_context: Dict[str, Any] = None
    last_updated: str = None
    
    def __post_init__(self):
        if self.personality_traits is None:
            self.personality_traits = {}
        if self.preferences is None:
            self.preferences = {}
        if self.knowledge_base is None:
            self.knowledge_base = {}
        if self.memory_context is None:
            self.memory_context = {}
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()


@dataclass
class ConversationEntry:
    """Conversation entry for persistence"""
    conversation_id: str
    timestamp: str
    user_message: str
    assistant_message: str
    agent_used: str
    mode: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResearchEntry:
    """Research output entry for persistence"""
    research_id: str
    timestamp: str
    query: str
    result: str
    agents_used: List[str] = None
    sources: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.agents_used is None:
            self.agents_used = []
        if self.sources is None:
            self.sources = []
        if self.metadata is None:
            self.metadata = {}


class LocalPersistence:
    """
    Comprehensive local persistence system for ICEBURG
    Similar to browser storage but for long-term persistence on M4 Mac
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize local persistence system
        
        Args:
            data_dir: Data directory path (defaults to ~/Documents/iceburg_data/)
        """
        if data_dir is None:
            # Use local Documents directory (not synced to iCloud by default)
            home = Path.home()
            data_dir = home / "Documents" / "iceburg_data"
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.personality_file = self.data_dir / "personality.json"
        self.conversations_db = self.data_dir / "conversations.db"
        self.research_db = self.data_dir / "research.db"
        self.knowledge_file = self.data_dir / "knowledge.json"
        self.memory_file = self.data_dir / "memory.json"
        self.agents_file = self.data_dir / "agents.json"
        
        # Thread lock for concurrent access
        self.lock = Lock()
        
        # Initialize storage
        self._initialize_storage()
        
        logger.info(f"✅ Local Persistence initialized at: {self.data_dir}")
    
    def _initialize_storage(self):
        """Initialize storage files and databases"""
        # Initialize personality file
        if not self.personality_file.exists():
            personality = PersonalityState()
            self._save_personality(personality)
        
        # Initialize conversations database
        self._init_conversations_db()
        
        # Initialize research database
        self._init_research_db()
        
        # Initialize knowledge file
        if not self.knowledge_file.exists():
            self._save_json(self.knowledge_file, {})
        
        # Initialize memory file
        if not self.memory_file.exists():
            self._save_json(self.memory_file, {})
        
        # Initialize agents file
        if not self.agents_file.exists():
            self._save_json(self.agents_file, {})
    
    def _init_conversations_db(self):
        """Initialize conversations SQLite database"""
        with self.lock:
            conn = sqlite3.connect(self.conversations_db)
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
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON conversations(timestamp)
            ''')
            
            conn.commit()
            conn.close()
    
    def _init_research_db(self):
        """Initialize research SQLite database"""
        with self.lock:
            conn = sqlite3.connect(self.research_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    research_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    result TEXT NOT NULL,
                    agents_used TEXT,
                    sources TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_research_id 
                ON research(research_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_research_timestamp 
                ON research(timestamp)
            ''')
            
            conn.commit()
            conn.close()
    
    # ========== Personality Persistence ==========
    
    def save_personality(self, personality: PersonalityState):
        """Save ICEBURG's personality state"""
        personality.last_updated = datetime.now().isoformat()
        self._save_personality(personality)
        logger.info("✅ Personality state saved")
    
    def load_personality(self) -> PersonalityState:
        """Load ICEBURG's personality state"""
        if not self.personality_file.exists():
            return PersonalityState()
        
        try:
            data = self._load_json(self.personality_file)
            return PersonalityState(**data)
        except Exception as e:
            logger.error(f"Error loading personality: {e}")
            return PersonalityState()
    
    def update_personality_trait(self, trait: str, value: Any):
        """Update a personality trait"""
        personality = self.load_personality()
        personality.personality_traits[trait] = value
        self.save_personality(personality)
    
    def update_preference(self, key: str, value: Any):
        """Update a preference"""
        personality = self.load_personality()
        personality.preferences[key] = value
        self.save_personality(personality)
    
    def update_knowledge(self, key: str, value: Any):
        """Update knowledge base"""
        personality = self.load_personality()
        personality.knowledge_base[key] = value
        self.save_personality(personality)
    
    def update_memory_context(self, key: str, value: Any):
        """Update memory context"""
        personality = self.load_personality()
        personality.memory_context[key] = value
        self.save_personality(personality)
    
    # ========== Conversation Persistence ==========
    
    def save_conversation(self, entry: ConversationEntry):
        """Save conversation entry"""
        with self.lock:
            conn = sqlite3.connect(self.conversations_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO conversations 
                (conversation_id, timestamp, user_message, assistant_message, agent_used, mode, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.conversation_id,
                entry.timestamp,
                entry.user_message,
                entry.assistant_message,
                entry.agent_used,
                entry.mode,
                json.dumps(entry.metadata or {})
            ))
            
            conn.commit()
            conn.close()
        
        logger.debug(f"✅ Conversation saved: {entry.conversation_id}")
    
    def get_conversations(self, conversation_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get conversations"""
        with self.lock:
            conn = sqlite3.connect(self.conversations_db)
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
            conn.close()
            
            # Convert to dictionaries
            conversations = []
            for row in rows:
                conversations.append({
                    "id": row[0],
                    "conversation_id": row[1],
                    "timestamp": row[2],
                    "user_message": row[3],
                    "assistant_message": row[4],
                    "agent_used": row[5],
                    "mode": row[6],
                    "metadata": json.loads(row[7]) if row[7] else {}
                })
            
            return conversations
    
    # ========== Research Persistence ==========
    
    def save_research(self, entry: ResearchEntry):
        """Save research output"""
        with self.lock:
            conn = sqlite3.connect(self.research_db)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO research 
                (research_id, timestamp, query, result, agents_used, sources, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                entry.research_id,
                entry.timestamp,
                entry.query,
                entry.result,
                json.dumps(entry.agents_used or []),
                json.dumps(entry.sources or []),
                json.dumps(entry.metadata or {})
            ))
            
            conn.commit()
            conn.close()
        
        logger.debug(f"✅ Research saved: {entry.research_id}")
    
    def get_research(self, research_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get research outputs"""
        with self.lock:
            conn = sqlite3.connect(self.research_db)
            cursor = conn.cursor()
            
            if research_id:
                cursor.execute('''
                    SELECT * FROM research 
                    WHERE research_id = ? 
                    ORDER BY timestamp DESC
                ''', (research_id,))
            else:
                cursor.execute('''
                    SELECT * FROM research 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            # Convert to dictionaries
            research_list = []
            for row in rows:
                research_list.append({
                    "id": row[0],
                    "research_id": row[1],
                    "timestamp": row[2],
                    "query": row[3],
                    "result": row[4],
                    "agents_used": json.loads(row[5]) if row[5] else [],
                    "sources": json.loads(row[6]) if row[6] else [],
                    "metadata": json.loads(row[7]) if row[7] else {}
                })
            
            return research_list
    
    # ========== Knowledge Persistence ==========
    
    def save_knowledge(self, key: str, value: Any):
        """Save knowledge entry"""
        knowledge = self._load_json(self.knowledge_file)
        knowledge[key] = value
        self._save_json(self.knowledge_file, knowledge)
        logger.debug(f"✅ Knowledge saved: {key}")
    
    def get_knowledge(self, key: Optional[str] = None) -> Any:
        """Get knowledge entry"""
        knowledge = self._load_json(self.knowledge_file)
        if key:
            return knowledge.get(key)
        return knowledge
    
    # ========== Memory Persistence ==========
    
    def save_memory(self, key: str, value: Any):
        """Save memory entry"""
        memory = self._load_json(self.memory_file)
        memory[key] = value
        self._save_json(self.memory_file, memory)
        logger.debug(f"✅ Memory saved: {key}")
    
    def get_memory(self, key: Optional[str] = None) -> Any:
        """Get memory entry"""
        memory = self._load_json(self.memory_file)
        if key:
            return memory.get(key)
        return memory
    
    # ========== Agent Persistence ==========
    
    def save_agent_state(self, agent_id: str, state: Dict[str, Any]):
        """Save agent state"""
        agents = self._load_json(self.agents_file)
        agents[agent_id] = {
            **state,
            "last_updated": datetime.now().isoformat()
        }
        self._save_json(self.agents_file, agents)
        logger.debug(f"✅ Agent state saved: {agent_id}")
    
    def get_agent_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent state"""
        agents = self._load_json(self.agents_file)
        return agents.get(agent_id)
    
    def get_all_agent_states(self) -> Dict[str, Any]:
        """Get all agent states"""
        return self._load_json(self.agents_file)
    
    # ========== Helper Methods ==========
    
    def _save_personality(self, personality: PersonalityState):
        """Save personality to file"""
        self._save_json(self.personality_file, asdict(personality))
    
    def _save_json(self, file_path: Path, data: Any):
        """Save JSON data to file"""
        with self.lock:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _load_json(self, file_path: Path) -> Any:
        """Load JSON data from file"""
        if not file_path.exists():
            return {}
        
        with self.lock:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading JSON from {file_path}: {e}")
                return {}
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        stats = {
            "data_dir": str(self.data_dir),
            "personality_exists": self.personality_file.exists(),
            "conversations_count": 0,
            "research_count": 0,
            "knowledge_entries": 0,
            "memory_entries": 0,
            "agent_states": 0
        }
        
        # Count conversations
        try:
            conn = sqlite3.connect(self.conversations_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM conversations")
            stats["conversations_count"] = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
        
        # Count research
        try:
            conn = sqlite3.connect(self.research_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM research")
            stats["research_count"] = cursor.fetchone()[0]
            conn.close()
        except Exception:
            pass
        
        # Count knowledge and memory
        knowledge = self._load_json(self.knowledge_file)
        memory = self._load_json(self.memory_file)
        agents = self._load_json(self.agents_file)
        
        stats["knowledge_entries"] = len(knowledge)
        stats["memory_entries"] = len(memory)
        stats["agent_states"] = len(agents)
        
        return stats

