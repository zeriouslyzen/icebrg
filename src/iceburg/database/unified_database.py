"""
ICEBURG Unified Database System
Centralized database for all ICEBURG data with optimized performance

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
import sqlite3
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from contextlib import asynccontextmanager
import threading
from concurrent.futures import ThreadPoolExecutor

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    database_path: str = "data/iceburg_unified.db"
    max_connections: int = 10
    connection_timeout: float = 30.0
    query_timeout: float = 60.0
    enable_wal_mode: bool = True
    enable_foreign_keys: bool = True
    cache_size: int = 10000
    journal_mode: str = "WAL"
    synchronous: str = "NORMAL"
    temp_store: str = "MEMORY"

@dataclass
class QueryResult:
    """Result of a database query"""
    success: bool
    data: List[Dict[str, Any]]
    error: Optional[str] = None
    execution_time: float = 0.0
    rows_affected: int = 0

class UnifiedDatabase:
    """
    Unified database system for all ICEBURG data
    Consolidates all data storage into a single, optimized database
    """
    
    def __init__(self, cfg: IceburgConfig, db_config: DatabaseConfig = None):
        self.cfg = cfg
        self.db_config = db_config or DatabaseConfig()
        
        # Ensure data directory exists
        self.db_path = Path(self.db_config.database_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection pool
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.max_connections = self.db_config.max_connections
        
        # Performance tracking
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_execution_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Query cache
        self.query_cache = {}
        self.cache_size = self.db_config.cache_size
        
        # Initialize database
        self._initialize_database()
        self._create_tables()
        self._create_indexes()
        
        logger.info(f"🗄️ Unified Database initialized: {self.db_path}")
    
    def _initialize_database(self):
        """Initialize database with optimal settings"""
        
        # Create initial connection to set up database
        conn = sqlite3.connect(str(self.db_path), timeout=self.db_config.connection_timeout)
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA optimize")
        conn.close()
        
        # Initialize connection pool
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize connection pool for concurrent access"""
        
        with self.pool_lock:
            for _ in range(self.max_connections):
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.db_config.connection_timeout,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = -10000")
                conn.execute("PRAGMA temp_store = MEMORY")
                conn.execute("PRAGMA foreign_keys = ON")
                
                self.connection_pool.append(conn)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        
        conn = None
        try:
            with self.pool_lock:
                if self.connection_pool:
                    conn = self.connection_pool.pop()
                else:
                    # Create new connection if pool is empty
                    conn = sqlite3.connect(
                        str(self.db_path),
                        timeout=self.db_config.connection_timeout,
                        check_same_thread=False
                    )
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA journal_mode = WAL")
                    conn.execute("PRAGMA synchronous = NORMAL")
                    conn.execute("PRAGMA cache_size = -10000")
                    conn.execute("PRAGMA temp_store = MEMORY")
                    conn.execute("PRAGMA foreign_keys = ON")
            
            yield conn
            
        finally:
            if conn:
                with self.pool_lock:
                    if len(self.connection_pool) < self.max_connections:
                        self.connection_pool.append(conn)
                    else:
                        conn.close()
    
    def _create_tables(self):
        """Create all database tables"""
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Core system tables
            self._create_core_tables(cursor)
            
            # Agent and performance tables
            self._create_agent_tables(cursor)
            
            # Knowledge and research tables
            self._create_knowledge_tables(cursor)
            
            # Optimization and analytics tables
            self._create_optimization_tables(cursor)
            
            # Memory and emergence tables
            self._create_memory_tables(cursor)
            
            # Blockchain and verification tables
            self._create_blockchain_tables(cursor)
            
            # Astro-physiology tables
            self._create_astro_physiology_tables(cursor)
            
            # V2: Multi-user workspace tables
            self._create_user_workspace_tables(cursor)
            
            conn.commit()
    
    def _create_core_tables(self, cursor):
        """Create core system tables"""
        
        # System configuration
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_config (
                config_id TEXT PRIMARY KEY,
                config_key TEXT UNIQUE NOT NULL,
                config_value TEXT NOT NULL,
                config_type TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # System metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_unit TEXT,
                timestamp REAL NOT NULL,
                source TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # User sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_data TEXT NOT NULL,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                expires_at REAL,
                metadata TEXT
            )
        ''')
    
    def _create_agent_tables(self, cursor):
        """Create agent and performance tables"""
        
        # Agent performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_performance (
                performance_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                agent_version TEXT,
                task_type TEXT NOT NULL,
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                quality_score REAL NOT NULL,
                resource_usage TEXT NOT NULL,
                input_size INTEGER NOT NULL,
                output_size INTEGER NOT NULL,
                error_message TEXT,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Agent coordination
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_coordination (
                coordination_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agents_involved TEXT NOT NULL,
                coordination_pattern TEXT NOT NULL,
                total_execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                quality_score REAL NOT NULL,
                resource_efficiency REAL NOT NULL,
                coordination_overhead REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Multi-agent sessions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS multi_agent_sessions (
                session_id TEXT PRIMARY KEY,
                session_type TEXT NOT NULL,
                agents_used TEXT NOT NULL,
                total_agents INTEGER NOT NULL,
                session_duration REAL NOT NULL,
                success_rate REAL NOT NULL,
                quality_score REAL NOT NULL,
                created_at REAL NOT NULL,
                completed_at REAL,
                metadata TEXT
            )
        ''')
    
    def _create_knowledge_tables(self, cursor):
        """Create knowledge and research tables"""
        
        # Knowledge concepts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_concepts (
                concept_id TEXT PRIMARY KEY,
                concept_name TEXT NOT NULL,
                concept_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                definition TEXT NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Knowledge relationships
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_relationships (
                relationship_id TEXT PRIMARY KEY,
                source_concept_id TEXT NOT NULL,
                target_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL NOT NULL,
                confidence REAL NOT NULL,
                evidence TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (source_concept_id) REFERENCES knowledge_concepts(concept_id),
                FOREIGN KEY (target_concept_id) REFERENCES knowledge_concepts(concept_id)
            )
        ''')
        
        # Research outputs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_outputs (
                output_id TEXT PRIMARY KEY,
                research_type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                domains TEXT NOT NULL,
                quality_score REAL NOT NULL,
                validation_status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Cross-domain synthesis
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cross_domain_synthesis (
                synthesis_id TEXT PRIMARY KEY,
                domains TEXT NOT NULL,
                synthesis_type TEXT NOT NULL,
                result TEXT NOT NULL,
                quality_score REAL NOT NULL,
                processing_time REAL NOT NULL,
                success BOOLEAN NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
    
    def _create_optimization_tables(self, cursor):
        """Create optimization and analytics tables"""
        
        # Model evolution tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_evolution (
                evolution_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                capability_name TEXT NOT NULL,
                current_value REAL NOT NULL,
                predicted_value REAL NOT NULL,
                improvement_factor REAL NOT NULL,
                confidence REAL NOT NULL,
                timeframe TEXT NOT NULL,
                optimization_potential TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Technology trends
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS technology_trends (
                trend_id TEXT PRIMARY KEY,
                technology TEXT NOT NULL,
                trend_type TEXT NOT NULL,
                emergence_level TEXT NOT NULL,
                confidence REAL NOT NULL,
                growth_rate REAL NOT NULL,
                signal_count INTEGER NOT NULL,
                first_detected REAL NOT NULL,
                last_updated REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Performance optimization
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_optimization (
                optimization_id TEXT PRIMARY KEY,
                optimization_type TEXT NOT NULL,
                system_component TEXT NOT NULL,
                current_performance REAL NOT NULL,
                optimized_performance REAL NOT NULL,
                improvement_percentage REAL NOT NULL,
                implementation_effort TEXT NOT NULL,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                implemented_at REAL,
                metadata TEXT
            )
        ''')
        
        # Latent space reasoning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS latent_space_reasoning (
                reasoning_id TEXT PRIMARY KEY,
                reasoning_type TEXT NOT NULL,
                processing_time REAL NOT NULL,
                convergence_iterations INTEGER NOT NULL,
                confidence_score REAL NOT NULL,
                emergence_signals INTEGER NOT NULL,
                vector_dimensions INTEGER NOT NULL,
                attention_weights TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
    
    def _create_memory_tables(self, cursor):
        """Create memory and emergence tables"""
        
        # Memory entries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_entries (
                memory_id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                domain TEXT NOT NULL,
                importance REAL NOT NULL,
                associations TEXT,
                created_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                memory_strength REAL NOT NULL,
                cross_references TEXT,
                metadata TEXT
            )
        ''')
        
        # Emergence patterns
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                pattern_data TEXT NOT NULL,
                confidence REAL NOT NULL,
                first_detected REAL NOT NULL,
                last_updated REAL NOT NULL,
                pattern_strength REAL NOT NULL,
                cross_domain_connections TEXT,
                metadata TEXT
            )
        ''')
        
        # Breakthrough discoveries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS breakthrough_discoveries (
                discovery_id TEXT PRIMARY KEY,
                discovery_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                domains TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                validation_status TEXT NOT NULL,
                impact_level TEXT NOT NULL,
                created_at REAL NOT NULL,
                validated_at REAL,
                metadata TEXT
            )
        ''')
        
        # Curiosity queries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS curiosity_queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                uncertainty_score REAL NOT NULL,
                novelty_score REAL NOT NULL,
                exploration_type TEXT NOT NULL,
                domains TEXT NOT NULL,
                priority REAL NOT NULL,
                timestamp REAL NOT NULL,
                answered BOOLEAN DEFAULT 0,
                answer_quality REAL,
                metadata TEXT
            )
        ''')
        
        # Knowledge gaps
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS knowledge_gaps (
                gap_id TEXT PRIMARY KEY,
                gap_description TEXT NOT NULL,
                uncertainty_level REAL NOT NULL,
                related_domains TEXT NOT NULL,
                exploration_priority REAL NOT NULL,
                suggested_queries TEXT NOT NULL,
                timestamp REAL NOT NULL,
                resolved BOOLEAN DEFAULT 0,
                resolution_timestamp REAL,
                resolution_quality REAL,
                metadata TEXT
            )
        ''')
        
        # Questions storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                question_type TEXT NOT NULL,
                domain TEXT NOT NULL,
                timestamp REAL NOT NULL,
                answered BOOLEAN DEFAULT 0,
                answer_id TEXT,
                answer_quality REAL,
                principle_ids TEXT,
                related_queries TEXT,
                metadata TEXT
            )
        ''')
        
        # Principles with versioning
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS principles (
                principle_id TEXT PRIMARY KEY,
                principle_name TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                summary TEXT NOT NULL,
                framing TEXT,
                domains TEXT NOT NULL,
                concepts TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                confidence REAL NOT NULL,
                validation_status TEXT NOT NULL,
                evidence_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                is_current BOOLEAN DEFAULT 1,
                previous_version_id TEXT,
                metadata TEXT
            )
        ''')
        
        # Principle evolution history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS principle_evolution (
                evolution_id TEXT PRIMARY KEY,
                principle_id TEXT NOT NULL,
                from_version INTEGER NOT NULL,
                to_version INTEGER NOT NULL,
                evolution_type TEXT NOT NULL,
                changes_description TEXT NOT NULL,
                confidence_delta REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
    
    def _create_blockchain_tables(self, cursor):
        """Create blockchain and verification tables"""
        
        # Blockchain entries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blockchain_entries (
                entry_id TEXT PRIMARY KEY,
                block_hash TEXT UNIQUE NOT NULL,
                previous_hash TEXT,
                data TEXT NOT NULL,
                nonce INTEGER NOT NULL,
                difficulty INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                verified BOOLEAN NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Verification records
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_records (
                verification_id TEXT PRIMARY KEY,
                record_type TEXT NOT NULL,
                record_data TEXT NOT NULL,
                verification_hash TEXT NOT NULL,
                verification_status TEXT NOT NULL,
                verified_at REAL NOT NULL,
                verifier TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Mining performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS mining_performance (
                mining_id TEXT PRIMARY KEY,
                operation_type TEXT NOT NULL,
                processing_time REAL NOT NULL,
                block_size INTEGER NOT NULL,
                difficulty INTEGER NOT NULL,
                success BOOLEAN NOT NULL,
                resource_usage TEXT NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
    
    def _create_astro_physiology_tables(self, cursor):
        """Create astro-physiology context tables"""
        
        # Astro-physiology analyses
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS astro_physiology_analyses (
                analysis_id TEXT PRIMARY KEY,
                user_id TEXT,
                session_id TEXT,
                birth_datetime TEXT NOT NULL,
                location TEXT,
                timestamp REAL NOT NULL,
                algorithmic_data TEXT NOT NULL,
                llm_response TEXT,
                query_text TEXT,
                metadata TEXT
            )
        ''')
        
        # Astro-physiology interventions
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS astro_physiology_interventions (
                intervention_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                user_id TEXT,
                intervention_data TEXT NOT NULL,
                started_at REAL NOT NULL,
                completed_at REAL,
                outcome TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                metadata TEXT,
                FOREIGN KEY (analysis_id) REFERENCES astro_physiology_analyses(analysis_id)
            )
        ''')
        
        # Astro-physiology feedback
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS astro_physiology_feedback (
                feedback_id TEXT PRIMARY KEY,
                intervention_id TEXT,
                analysis_id TEXT,
                user_id TEXT,
                rating REAL,
                comment TEXT,
                outcome TEXT,
                timestamp REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (intervention_id) REFERENCES astro_physiology_interventions(intervention_id),
                FOREIGN KEY (analysis_id) REFERENCES astro_physiology_analyses(analysis_id)
            )
        ''')
        
        # Astro-physiology health tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS astro_physiology_health_tracking (
                tracking_id TEXT PRIMARY KEY,
                user_id TEXT,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Astro-physiology monitoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS astro_physiology_monitoring (
                monitoring_id TEXT PRIMARY KEY,
                intervention_id TEXT,
                user_id TEXT,
                check_in_time REAL NOT NULL,
                adherence_score REAL,
                health_metrics TEXT,
                adaptation_recommendations TEXT,
                alerts TEXT,
                metadata TEXT,
                FOREIGN KEY (intervention_id) REFERENCES astro_physiology_interventions(intervention_id)
            )
        ''')
        
        # Astro-physiology model updates
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS astro_physiology_model_updates (
                update_id TEXT PRIMARY KEY,
                model_type TEXT NOT NULL,
                update_data TEXT NOT NULL,
                performance_improvement REAL,
                timestamp REAL NOT NULL,
                metadata TEXT
            )
        ''')
    
    def _create_user_workspace_tables(self, cursor):
        """V2: Create multi-user workspace tables"""
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                name TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Teams table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Team members table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS team_members (
                team_member_id TEXT PRIMARY KEY,
                team_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'member',
                joined_at REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (team_id) REFERENCES teams(team_id),
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                UNIQUE(team_id, user_id)
            )
        ''')
        
        # Shared analyses table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shared_analyses (
                share_id TEXT PRIMARY KEY,
                analysis_id TEXT NOT NULL,
                shared_by TEXT NOT NULL,
                shared_with TEXT,
                visibility_level TEXT NOT NULL DEFAULT 'private',
                shared_at REAL NOT NULL,
                metadata TEXT,
                FOREIGN KEY (analysis_id) REFERENCES astro_physiology_analyses(analysis_id),
                FOREIGN KEY (shared_by) REFERENCES users(user_id),
                FOREIGN KEY (shared_with) REFERENCES users(user_id)
            )
        ''')
    
    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Core system indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(config_key)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_sessions_user ON user_sessions(user_id)')
            
            # Agent performance indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_performance_agent_time ON agent_performance(agent_name, timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_performance_task ON agent_performance(task_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_agent_coordination_session ON agent_coordination(session_id)')
            
            # Knowledge indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_concepts_domain ON knowledge_concepts(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_concepts_type ON knowledge_concepts(concept_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_source ON knowledge_relationships(source_concept_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_relationships_target ON knowledge_relationships(target_concept_id)')
            
            # Optimization indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_model_evolution_model ON model_evolution(model_name, model_version)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_technology_trends_tech ON technology_trends(technology)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_optimization_type ON performance_optimization(optimization_type)')
            
            # Memory indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_type ON memory_entries(memory_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_domain ON memory_entries(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_entries_accessed ON memory_entries(last_accessed)')
            
            # Curiosity and question indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_curiosity_queries_timestamp ON curiosity_queries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_curiosity_queries_type ON curiosity_queries(exploration_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_curiosity_queries_answered ON curiosity_queries(answered)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_timestamp ON knowledge_gaps(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_resolved ON knowledge_gaps(resolved)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_knowledge_gaps_priority ON knowledge_gaps(exploration_priority)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_timestamp ON questions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(question_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_domain ON questions(domain)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_questions_answered ON questions(answered)')
            
            # Principles indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_principles_name ON principles(principle_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_principles_version ON principles(principle_id, version)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_principles_current ON principles(is_current)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_principles_timestamp ON principles(created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_principle_evolution_principle ON principle_evolution(principle_id)')
            
            # Blockchain indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_blockchain_entries_hash ON blockchain_entries(block_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_blockchain_entries_time ON blockchain_entries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_verification_records_type ON verification_records(record_type)')
            
            # Astro-physiology indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_analyses_user ON astro_physiology_analyses(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_analyses_session ON astro_physiology_analyses(session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_analyses_timestamp ON astro_physiology_analyses(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_interventions_analysis ON astro_physiology_interventions(analysis_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_interventions_user ON astro_physiology_interventions(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_interventions_status ON astro_physiology_interventions(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_feedback_intervention ON astro_physiology_feedback(intervention_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_feedback_analysis ON astro_physiology_feedback(analysis_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_feedback_user ON astro_physiology_feedback(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_tracking_user ON astro_physiology_health_tracking(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_tracking_metric ON astro_physiology_health_tracking(metric_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_tracking_timestamp ON astro_physiology_health_tracking(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_monitoring_intervention ON astro_physiology_monitoring(intervention_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_astro_monitoring_user ON astro_physiology_monitoring(user_id)')
            
            # V2: Multi-user workspace indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_members_team ON team_members(team_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_members_user ON team_members(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_analyses_shared_by ON shared_analyses(shared_by)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_shared_analyses_shared_with ON shared_analyses(shared_with)')
            
            conn.commit()
    
    async def execute_query(
        self,
        query: str,
        params: Tuple = (),
        fetch: bool = True,
        cache: bool = True
    ) -> QueryResult:
        """Execute database query with performance tracking"""
        
        start_time = time.time()
        
        # Check cache first
        cache_key = f"{query}:{params}" if cache else None
        if cache_key and cache_key in self.query_cache:
            self.query_stats["cache_hits"] += 1
            cached_result = self.query_cache[cache_key]
            return QueryResult(
                success=True,
                data=cached_result["data"],
                execution_time=time.time() - start_time,
                rows_affected=cached_result.get("rows_affected", 0)
            )
        
        self.query_stats["cache_misses"] += 1
        
        try:
            async with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                
                if fetch:
                    data = [dict(row) for row in cursor.fetchall()]
                else:
                    data = []
                    conn.commit()
                
                execution_time = time.time() - start_time
                rows_affected = cursor.rowcount
                
                # Update query statistics
                self.query_stats["total_queries"] += 1
                self.query_stats["successful_queries"] += 1
                self.query_stats["avg_execution_time"] = (
                    (self.query_stats["avg_execution_time"] * (self.query_stats["total_queries"] - 1) + execution_time) /
                    self.query_stats["total_queries"]
                )
                
                # Cache result if requested
                if cache_key and len(data) < 1000:  # Only cache small results
                    self.query_cache[cache_key] = {
                        "data": data,
                        "rows_affected": rows_affected,
                        "timestamp": time.time()
                    }
                    
                    # Manage cache size
                    if len(self.query_cache) > self.cache_size:
                        # Remove oldest entries
                        oldest_keys = sorted(
                            self.query_cache.keys(),
                            key=lambda k: self.query_cache[k]["timestamp"]
                        )[:len(self.query_cache) - self.cache_size]
                        
                        for key in oldest_keys:
                            del self.query_cache[key]
                
                return QueryResult(
                    success=True,
                    data=data,
                    execution_time=execution_time,
                    rows_affected=rows_affected
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.query_stats["total_queries"] += 1
            self.query_stats["failed_queries"] += 1
            
            logger.error(f"Database query failed: {e}")
            
            return QueryResult(
                success=False,
                data=[],
                error=str(e),
                execution_time=execution_time
            )
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and performance metrics"""
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Get table sizes
            cursor.execute('''
                SELECT name, 
                       (SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=m.name) as table_count
                FROM sqlite_master m 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ''')
            
            tables = cursor.fetchall()
            
            # Get database size
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            
            # Get connection pool status
            pool_status = {
                "available_connections": len(self.connection_pool),
                "max_connections": self.max_connections,
                "cache_size": len(self.query_cache),
                "max_cache_size": self.cache_size
            }
            
            return {
                "database_path": str(self.db_path),
                "database_size_bytes": db_size,
                "database_size_mb": db_size / (1024 * 1024),
                "tables": len(tables),
                "connection_pool": pool_status,
                "query_statistics": self.query_stats,
                "cache_hit_rate": (
                    self.query_stats["cache_hits"] / 
                    max(1, self.query_stats["cache_hits"] + self.query_stats["cache_misses"])
                )
            }
    
    def optimize_database(self):
        """Optimize database performance"""
        
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Run VACUUM to reclaim space
            cursor.execute("VACUUM")
            
            # Update statistics
            cursor.execute("ANALYZE")
            
            # Optimize database
            cursor.execute("PRAGMA optimize")
            
            conn.commit()
        
        logger.info("🗄️ Database optimization completed")
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        
        backup_file = Path(backup_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(self.db_path)) as source:
            with sqlite3.connect(str(backup_file)) as backup:
                source.backup(backup)
        
        logger.info(f"🗄️ Database backup created: {backup_path}")
    
    def close(self):
        """Close all database connections"""
        
        with self.pool_lock:
            for conn in self.connection_pool:
                conn.close()
            self.connection_pool.clear()
        
        logger.info("🗄️ Database connections closed")
    
    # ========== Astro-Physiology Methods ==========
    
    async def store_astro_analysis(
        self,
        analysis_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        birth_datetime: str,
        location: Optional[str],
        algorithmic_data: Dict[str, Any],
        llm_response: Optional[str] = None,
        query_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store astro-physiology analysis"""
        try:
            query = '''
                INSERT INTO astro_physiology_analyses (
                    analysis_id, user_id, session_id, birth_datetime, location,
                    timestamp, algorithmic_data, llm_response, query_text, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                analysis_id,
                user_id,
                session_id,
                birth_datetime,
                location,
                time.time(),
                json.dumps(algorithmic_data),
                llm_response,
                query_text,
                json.dumps(metadata or {})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to store astro analysis: {e}")
            return False
    
    async def get_user_astro_history(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get user's astro-physiology analysis history"""
        try:
            if user_id:
                query = '''
                    SELECT * FROM astro_physiology_analyses
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                params = (user_id, limit)
            elif session_id:
                query = '''
                    SELECT * FROM astro_physiology_analyses
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                params = (session_id, limit)
            else:
                return []
            
            result = await self.execute_query(query, params, fetch=True)
            if result.success:
                for row in result.data:
                    if row.get('algorithmic_data'):
                        row['algorithmic_data'] = json.loads(row['algorithmic_data'])
                    if row.get('metadata'):
                        row['metadata'] = json.loads(row['metadata'])
            return result.data if result.success else []
        except Exception as e:
            logger.error(f"Failed to get user astro history: {e}")
            return []
    
    async def store_intervention(
        self,
        intervention_id: str,
        analysis_id: str,
        user_id: Optional[str],
        intervention_data: Dict[str, Any],
        status: str = 'active',
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store astro-physiology intervention"""
        try:
            query = '''
                INSERT INTO astro_physiology_interventions (
                    intervention_id, analysis_id, user_id, intervention_data,
                    started_at, status, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                intervention_id,
                analysis_id,
                user_id,
                json.dumps(intervention_data),
                time.time(),
                status,
                json.dumps(metadata or {})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to store intervention: {e}")
            return False
    
    async def store_feedback(
        self,
        feedback_id: str,
        user_id: Optional[str],
        intervention_id: Optional[str] = None,
        analysis_id: Optional[str] = None,
        rating: Optional[float] = None,
        comment: Optional[str] = None,
        outcome: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store user feedback"""
        try:
            query = '''
                INSERT INTO astro_physiology_feedback (
                    feedback_id, intervention_id, analysis_id, user_id,
                    rating, comment, outcome, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                feedback_id,
                intervention_id,
                analysis_id,
                user_id,
                rating,
                comment,
                json.dumps(outcome) if outcome else None,
                time.time(),
                json.dumps(metadata or {})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to store feedback: {e}")
            return False
    
    async def store_health_metric(
        self,
        tracking_id: str,
        user_id: Optional[str],
        metric_name: str,
        metric_value: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store health tracking metric"""
        try:
            query = '''
                INSERT INTO astro_physiology_health_tracking (
                    tracking_id, user_id, metric_name, metric_value, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?)
            '''
            params = (
                tracking_id,
                user_id,
                metric_name,
                metric_value,
                time.time(),
                json.dumps(metadata or {})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to store health metric: {e}")
            return False
    
    async def get_user_interventions(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get user's interventions"""
        try:
            if user_id:
                if status:
                    query = '''
                        SELECT * FROM astro_physiology_interventions
                        WHERE user_id = ? AND status = ?
                        ORDER BY started_at DESC
                        LIMIT ?
                    '''
                    params = (user_id, status, limit)
                else:
                    query = '''
                        SELECT * FROM astro_physiology_interventions
                        WHERE user_id = ?
                        ORDER BY started_at DESC
                        LIMIT ?
                    '''
                    params = (user_id, limit)
            elif session_id:
                # Get user_id from analyses first
                analysis_query = '''
                    SELECT user_id FROM astro_physiology_analyses
                    WHERE session_id = ?
                    LIMIT 1
                '''
                analysis_result = await self.execute_query(analysis_query, (session_id,), fetch=True)
                if analysis_result.success and analysis_result.data:
                    user_id = analysis_result.data[0].get('user_id')
                    if user_id:
                        return await self.get_user_interventions(user_id=user_id, status=status, limit=limit)
                return []
            else:
                return []
            
            result = await self.execute_query(query, params, fetch=True)
            if result.success:
                for row in result.data:
                    if row.get('intervention_data'):
                        row['intervention_data'] = json.loads(row['intervention_data'])
                    if row.get('outcome'):
                        row['outcome'] = json.loads(row['outcome'])
                    if row.get('metadata'):
                        row['metadata'] = json.loads(row['metadata'])
            return result.data if result.success else []
        except Exception as e:
            logger.error(f"Failed to get user interventions: {e}")
            return []
    
    async def get_user_feedback(
        self,
        user_id: Optional[str] = None,
        intervention_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get user feedback"""
        try:
            if intervention_id:
                query = '''
                    SELECT * FROM astro_physiology_feedback
                    WHERE intervention_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                params = (intervention_id, limit)
            elif user_id:
                query = '''
                    SELECT * FROM astro_physiology_feedback
                    WHERE user_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                params = (user_id, limit)
            else:
                return []
            
            result = await self.execute_query(query, params, fetch=True)
            if result.success:
                for row in result.data:
                    if row.get('outcome'):
                        row['outcome'] = json.loads(row['outcome'])
                    if row.get('metadata'):
                        row['metadata'] = json.loads(row['metadata'])
            return result.data if result.success else []
        except Exception as e:
            logger.error(f"Failed to get user feedback: {e}")
            return []
    
    async def get_user_health_tracking(
        self,
        user_id: Optional[str],
        metric_name: Optional[str] = None,
        days: int = 30,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get user health tracking data"""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            if metric_name:
                query = '''
                    SELECT * FROM astro_physiology_health_tracking
                    WHERE user_id = ? AND metric_name = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                params = (user_id, metric_name, cutoff_time, limit)
            else:
                query = '''
                    SELECT * FROM astro_physiology_health_tracking
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                '''
                params = (user_id, cutoff_time, limit)
            
            result = await self.execute_query(query, params, fetch=True)
            if result.success:
                for row in result.data:
                    if row.get('metadata'):
                        row['metadata'] = json.loads(row['metadata'])
            return result.data if result.success else []
        except Exception as e:
            logger.error(f"Failed to get user health tracking: {e}")
            return []
    
    # ========== V2: Multi-User Workspace Methods ==========
    
    async def create_user(
        self,
        user_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new user"""
        try:
            query = '''
                INSERT INTO users (user_id, email, name, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            params = (
                user_id,
                email,
                name,
                time.time(),
                time.time(),
                json.dumps(metadata or {})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    async def get_user(
        self,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            query = '''
                SELECT * FROM users WHERE user_id = ?
            '''
            result = await self.execute_query(query, (user_id,), fetch=True)
            if result.success and result.data:
                user = result.data[0]
                if user.get('metadata'):
                    user['metadata'] = json.loads(user['metadata'])
                return user
            return None
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    async def create_team(
        self,
        team_id: str,
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a new team"""
        try:
            query = '''
                INSERT INTO teams (team_id, name, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?)
            '''
            params = (
                team_id,
                name,
                time.time(),
                time.time(),
                json.dumps(metadata or {})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to create team: {e}")
            return False
    
    async def add_team_member(
        self,
        team_id: str,
        user_id: str,
        role: str = "member"
    ) -> bool:
        """Add user to team"""
        try:
            team_member_id = f"{team_id}_{user_id}"
            query = '''
                INSERT OR REPLACE INTO team_members (team_member_id, team_id, user_id, role, joined_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            '''
            params = (
                team_member_id,
                team_id,
                user_id,
                role,
                time.time(),
                json.dumps({})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to add team member: {e}")
            return False
    
    async def share_analysis(
        self,
        analysis_id: str,
        shared_by: str,
        shared_with: Optional[str] = None,
        visibility_level: str = "private"
    ) -> bool:
        """Share analysis with user or team"""
        try:
            share_id = f"share_{analysis_id}_{shared_by}_{time.time()}"
            query = '''
                INSERT INTO shared_analyses (share_id, analysis_id, shared_by, shared_with, visibility_level, shared_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            '''
            params = (
                share_id,
                analysis_id,
                shared_by,
                shared_with,
                visibility_level,
                time.time(),
                json.dumps({})
            )
            result = await self.execute_query(query, params, fetch=False)
            return result.success
        except Exception as e:
            logger.error(f"Failed to share analysis: {e}")
            return False
    
    async def get_shared_analyses(
        self,
        user_id: str,
        visibility_level: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get analyses shared with user"""
        try:
            if visibility_level:
                query = '''
                    SELECT * FROM shared_analyses
                    WHERE (shared_with = ? OR visibility_level = 'public')
                    AND visibility_level = ?
                    ORDER BY shared_at DESC
                '''
                params = (user_id, visibility_level)
            else:
                query = '''
                    SELECT * FROM shared_analyses
                    WHERE shared_with = ? OR visibility_level = 'public'
                    ORDER BY shared_at DESC
                '''
                params = (user_id,)
            
            result = await self.execute_query(query, params, fetch=True)
            if result.success:
                for row in result.data:
                    if row.get('metadata'):
                        row['metadata'] = json.loads(row['metadata'])
            return result.data if result.success else []
        except Exception as e:
            logger.error(f"Failed to get shared analyses: {e}")
            return []


# Helper functions for integration
def create_unified_database(cfg: IceburgConfig, db_config: DatabaseConfig = None) -> UnifiedDatabase:
    """Create unified database instance"""
    return UnifiedDatabase(cfg, db_config)

async def execute_query(
    db: UnifiedDatabase,
    query: str,
    params: Tuple = (),
    fetch: bool = True,
    cache: bool = True
) -> QueryResult:
    """Execute database query"""
    return await db.execute_query(query, params, fetch, cache)
