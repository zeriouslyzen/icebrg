#!/usr/bin/env python3
"""
Emotional Intelligence & Contextual Memory System for 2025
Persistent memory, emotional awareness, and contextual understanding
"""

import asyncio
import json
import sqlite3
import numpy as np
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import queue
import hashlib
import pickle
import os

# Audio analysis for emotional detection
import librosa
from scipy import signal

# Text analysis for emotional understanding
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class EmotionalState:
    """Emotional state data"""
    timestamp: datetime
    emotion: str  # happy, sad, angry, excited, calm, etc.
    intensity: float  # 0.0 to 1.0
    confidence: float
    context: str
    source: str  # voice, text, behavior
    features: Dict[str, Any]


@dataclass
class MemoryEntry:
    """Memory entry data"""
    id: str
    timestamp: datetime
    content: str
    emotional_context: EmotionalState
    importance: float  # 0.0 to 1.0
    category: str  # conversation, task, preference, etc.
    tags: List[str]
    relationships: List[str]  # IDs of related memories
    access_count: int
    last_accessed: datetime


@dataclass
class UserProfile:
    """User profile data"""
    user_id: str
    name: str
    preferences: Dict[str, Any]
    emotional_patterns: Dict[str, List[EmotionalState]]
    communication_style: Dict[str, Any]
    relationship_history: List[MemoryEntry]
    created_at: datetime
    updated_at: datetime


class EmotionalMemorySystem:
    """Advanced emotional intelligence and memory system"""
    
    def __init__(self, db_path: str = "data/emotional_memory.db"):
        self.db_path = db_path
        self.is_active = False
        
        # Database
        self.db_connection = None
        
        # Current state
        self.current_user = None
        self.current_emotional_state = None
        self.memory_cache = {}
        
        # Emotional analysis
        self.emotion_analyzer = None
        self.voice_emotion_analyzer = None
        
        # Memory management
        self.memory_queue = queue.Queue()
        self.importance_threshold = 0.3
        self.max_memories = 10000
        
        # Threads
        self.memory_thread = None
        
        # Callbacks
        self.on_emotion_detected: Optional[Callable[[EmotionalState], None]] = None
        self.on_memory_created: Optional[Callable[[MemoryEntry], None]] = None
        self.on_memory_retrieved: Optional[Callable[[MemoryEntry], None]] = None
        self.on_user_profile_updated: Optional[Callable[[UserProfile], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        
    
    def initialize(self) -> bool:
        """Initialize the memory system"""
        try:
            # Initialize database
            self._initialize_database()
            
            # Initialize emotional analyzers
            self._initialize_emotional_analyzers()
            
            # Start memory processing thread
            self.memory_thread = threading.Thread(target=self._memory_worker)
            self.memory_thread.daemon = True
            self.memory_thread.start()
            
            self.is_active = True
            
            return True
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
            return False
    
    def _initialize_database(self):
        """Initialize SQLite database"""
        try:
            # Ensure data directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    content TEXT,
                    emotional_context TEXT,
                    importance REAL,
                    category TEXT,
                    tags TEXT,
                    relationships TEXT,
                    access_count INTEGER,
                    last_accessed TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT,
                    preferences TEXT,
                    emotional_patterns TEXT,
                    communication_style TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS emotional_states (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    emotion TEXT,
                    intensity REAL,
                    confidence REAL,
                    context TEXT,
                    source TEXT,
                    features TEXT
                )
            ''')
            
            self.db_connection.commit()
            
        except Exception as e:
            raise
    
    def _initialize_emotional_analyzers(self):
        """Initialize emotional analysis models"""
        try:
            if TRANSFORMERS_AVAILABLE:
                # Text emotion analysis
                self.emotion_analyzer = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    return_all_scores=True
                )
            
            # Voice emotion analysis (simplified)
            self.voice_emotion_analyzer = VoiceEmotionAnalyzer()
            
        except Exception as e:
    
    def _memory_worker(self):
        """Memory processing worker thread"""
        while self.is_active:
            try:
                # Process memory queue
                memory_data = self.memory_queue.get(timeout=1)
                
                if memory_data:
                    self._process_memory(memory_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                if self.on_error:
                    self.on_error(e)
    
    def _process_memory(self, memory_data: Dict[str, Any]):
        """Process and store memory"""
        try:
            # Analyze emotional context
            emotional_state = self._analyze_emotional_context(memory_data)
            
            # Calculate importance
            importance = self._calculate_importance(memory_data, emotional_state)
            
            # Create memory entry
            memory_entry = MemoryEntry(
                id=self._generate_memory_id(memory_data),
                timestamp=datetime.now(),
                content=memory_data.get("content", ""),
                emotional_context=emotional_state,
                importance=importance,
                category=memory_data.get("category", "general"),
                tags=memory_data.get("tags", []),
                relationships=[],
                access_count=0,
                last_accessed=datetime.now()
            )
            
            # Store in database
            self._store_memory(memory_entry)
            
            # Update cache
            self.memory_cache[memory_entry.id] = memory_entry
            
            # Callback
            if self.on_memory_created:
                self.on_memory_created(memory_entry)
            
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def _analyze_emotional_context(self, data: Dict[str, Any]) -> EmotionalState:
        """Analyze emotional context from data"""
        try:
            emotion = "neutral"
            intensity = 0.5
            confidence = 0.7
            source = "text"
            features = {}
            
            # Text emotion analysis
            if "text" in data and self.emotion_analyzer:
                text = data["text"]
                results = self.emotion_analyzer(text)
                
                if results:
                    # Get highest scoring emotion
                    best_emotion = max(results[0], key=lambda x: x['score'])
                    emotion = best_emotion['label']
                    confidence = best_emotion['score']
                    intensity = confidence
            
            # Voice emotion analysis
            elif "audio_features" in data and self.voice_emotion_analyzer:
                audio_features = data["audio_features"]
                voice_emotion = self.voice_emotion_analyzer.analyze(audio_features)
                emotion = voice_emotion["emotion"]
                intensity = voice_emotion["intensity"]
                confidence = voice_emotion["confidence"]
                source = "voice"
                features = voice_emotion["features"]
            
            return EmotionalState(
                timestamp=datetime.now(),
                emotion=emotion,
                intensity=intensity,
                confidence=confidence,
                context=data.get("context", ""),
                source=source,
                features=features
            )
            
        except Exception as e:
            return EmotionalState(
                timestamp=datetime.now(),
                emotion="neutral",
                intensity=0.5,
                confidence=0.5,
                context="",
                source="unknown",
                features={}
            )
    
    def _calculate_importance(self, data: Dict[str, Any], emotional_state: EmotionalState) -> float:
        """Calculate memory importance"""
        try:
            importance = 0.5  # Base importance
            
            # Emotional intensity factor
            importance += emotional_state.intensity * 0.3
            
            # Content length factor
            content_length = len(data.get("content", ""))
            if content_length > 100:
                importance += 0.1
            
            # Category factor
            category = data.get("category", "")
            if category in ["preference", "important", "task"]:
                importance += 0.2
            
            # User interaction factor
            if data.get("user_interaction", False):
                importance += 0.1
            
            return min(1.0, importance)
            
        except Exception as e:
            return 0.5
    
    def _generate_memory_id(self, data: Dict[str, Any]) -> str:
        """Generate unique memory ID"""
        content = data.get("content", "")
        timestamp = datetime.now().isoformat()
        combined = f"{content}_{timestamp}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _store_memory(self, memory_entry: MemoryEntry):
        """Store memory in database"""
        try:
            cursor = self.db_connection.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO memories 
                (id, timestamp, content, emotional_context, importance, category, tags, relationships, access_count, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                memory_entry.id,
                memory_entry.timestamp.isoformat(),
                memory_entry.content,
                json.dumps(asdict(memory_entry.emotional_context)),
                memory_entry.importance,
                memory_entry.category,
                json.dumps(memory_entry.tags),
                json.dumps(memory_entry.relationships),
                memory_entry.access_count,
                memory_entry.last_accessed.isoformat()
            ))
            
            self.db_connection.commit()
            
        except Exception as e:
            raise
    
    # Public API methods
    def store_memory(self, content: str, category: str = "general", tags: List[str] = None, context: str = ""):
        """Store a new memory"""
        try:
            memory_data = {
                "content": content,
                "category": category,
                "tags": tags or [],
                "context": context,
                "timestamp": datetime.now()
            }
            
            self.memory_queue.put(memory_data)
            
        except Exception as e:
            if self.on_error:
                self.on_error(e)
    
    def retrieve_memories(self, query: str = "", category: str = "", limit: int = 10) -> List[MemoryEntry]:
        """Retrieve memories based on query"""
        try:
            cursor = self.db_connection.cursor()
            
            # Build query
            sql = "SELECT * FROM memories WHERE 1=1"
            params = []
            
            if query:
                sql += " AND content LIKE ?"
                params.append(f"%{query}%")
            
            if category:
                sql += " AND category = ?"
                params.append(category)
            
            sql += " ORDER BY importance DESC, timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            memories = []
            for row in rows:
                memory = self._row_to_memory(row)
                memories.append(memory)
                
                # Update access count
                self._update_access_count(memory.id)
                
                # Callback
                if self.on_memory_retrieved:
                    self.on_memory_retrieved(memory)
            
            return memories
            
        except Exception as e:
            return []
    
    def _row_to_memory(self, row) -> MemoryEntry:
        """Convert database row to MemoryEntry"""
        try:
            emotional_context_data = json.loads(row[3])
            emotional_context = EmotionalState(**emotional_context_data)
            
            return MemoryEntry(
                id=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                content=row[2],
                emotional_context=emotional_context,
                importance=row[4],
                category=row[5],
                tags=json.loads(row[6]),
                relationships=json.loads(row[7]),
                access_count=row[8],
                last_accessed=datetime.fromisoformat(row[9])
            )
            
        except Exception as e:
            return None
    
    def _update_access_count(self, memory_id: str):
        """Update memory access count"""
        try:
            cursor = self.db_connection.cursor()
            cursor.execute('''
                UPDATE memories 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), memory_id))
            self.db_connection.commit()
            
        except Exception as e:
    
    def get_emotional_patterns(self, days: int = 30) -> Dict[str, List[EmotionalState]]:
        """Get emotional patterns over time"""
        try:
            cursor = self.db_connection.cursor()
            since_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute('''
                SELECT emotional_context FROM memories 
                WHERE timestamp > ?
                ORDER BY timestamp
            ''', (since_date,))
            
            rows = cursor.fetchall()
            patterns = {}
            
            for row in rows:
                emotional_data = json.loads(row[0])
                emotion = emotional_data['emotion']
                
                if emotion not in patterns:
                    patterns[emotion] = []
                
                patterns[emotion].append(EmotionalState(**emotional_data))
            
            return patterns
            
        except Exception as e:
            return {}
    
    def get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from memory"""
        try:
            preferences = self.retrieve_memories(category="preference", limit=50)
            
            user_prefs = {}
            for memory in preferences:
                # Extract preferences from memory content
                content = memory.content.lower()
                
                if "like" in content or "prefer" in content:
                    # Simple preference extraction
                    user_prefs[memory.content] = memory.importance
            
            return user_prefs
            
        except Exception as e:
            return {}
    
    def set_callbacks(self,
        on_emotion_detected: Optional[Callable[[EmotionalState], None]] = None,
                     on_memory_created: Optional[Callable[[MemoryEntry], None]] = None,
                     on_memory_retrieved: Optional[Callable[[MemoryEntry], None]] = None,
                     on_user_profile_updated: Optional[Callable[[UserProfile], None]] = None,
                     on_error: Optional[Callable[[Exception], None]] = None):
        """Set callback functions"""
        self.on_emotion_detected = on_emotion_detected
        self.on_memory_created = on_memory_created
        self.on_memory_retrieved = on_memory_retrieved
        self.on_user_profile_updated = on_user_profile_updated
        self.on_error = on_error
    
    def get_status(self) -> Dict[str, Any]:
        """Get memory system status"""
        try:
            if self.db_connection:
                cursor = self.db_connection.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories")
                memory_count = cursor.fetchone()[0]
            else:
                memory_count = 0
            
            return {
                "is_active": self.is_active,
                "memory_count": memory_count,
                "cache_size": len(self.memory_cache),
                "emotion_analyzer_available": self.emotion_analyzer is not None,
                "voice_emotion_analyzer_available": self.voice_emotion_analyzer is not None
            }
            
        except Exception as e:
            return {"is_active": False}
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_active = False
        
        if self.db_connection:
            self.db_connection.close()
        


class VoiceEmotionAnalyzer:
    """Voice emotion analysis using audio features"""
    
    def __init__(self):
        self.sample_rate = 16000
        
    def analyze(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze voice emotion from audio features"""
        try:
            # Extract features
            pitch = audio_features.get("pitch", 0)
            energy = audio_features.get("energy", 0)
            spectral_centroid = audio_features.get("spectral_centroid", 0)
            mfcc = audio_features.get("mfcc", [])
            
            # Simple emotion classification based on features
            emotion = "neutral"
            intensity = 0.5
            confidence = 0.6
            
            # High pitch + high energy = excited
            if pitch > 200 and energy > 0.1:
                emotion = "excited"
                intensity = min(1.0, (pitch - 200) / 100 + energy)
                confidence = 0.8
            
            # Low pitch + low energy = sad/calm
            elif pitch < 150 and energy < 0.05:
                emotion = "calm"
                intensity = 0.3
                confidence = 0.7
            
            # High energy + medium pitch = happy
            elif energy > 0.08 and 150 <= pitch <= 200:
                emotion = "happy"
                intensity = energy * 2
                confidence = 0.7
            
            return {
                "emotion": emotion,
                "intensity": intensity,
                "confidence": confidence,
                "features": {
                    "pitch": pitch,
                    "energy": energy,
                    "spectral_centroid": spectral_centroid
                }
            }
            
        except Exception as e:
            return {
                "emotion": "neutral",
                "intensity": 0.5,
                "confidence": 0.5,
                "features": {}
            }
