"""
ICEBURG Memory Persistence Layer
Provides methods for persisting curiosity queries, knowledge gaps, questions, and principles
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MemoryPersistenceLayer:
    """Persistence layer for ICEBURG memory systems"""
    
    def __init__(self, unified_db):
        """Initialize persistence layer with UnifiedDatabase instance"""
        self.db = unified_db
    
    # ========== Curiosity Query Persistence ==========
    
    async def store_curiosity_query(
        self,
        query_id: str,
        query_text: str,
        uncertainty_score: float,
        novelty_score: float,
        exploration_type: str,
        domains: List[str],
        priority: float,
        timestamp: float = None,
        answered: bool = False,
        answer_quality: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a curiosity query to database"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            query = '''
                INSERT OR REPLACE INTO curiosity_queries (
                    query_id, query_text, uncertainty_score, novelty_score,
                    exploration_type, domains, priority, timestamp,
                    answered, answer_quality, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                query_id,
                query_text,
                uncertainty_score,
                novelty_score,
                exploration_type,
                json.dumps(domains),
                priority,
                timestamp,
                1 if answered else 0,
                answer_quality,
                json.dumps(metadata or {})
            )
            
            result = await self.db.execute_query(query, params, fetch=False)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to store curiosity query: {e}")
            return False
    
    async def get_curiosity_queries(
        self,
        exploration_type: str = None,
        answered: bool = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve curiosity queries from database"""
        try:
            conditions = []
            params = []
            
            if exploration_type:
                conditions.append("exploration_type = ?")
                params.append(exploration_type)
            
            if answered is not None:
                conditions.append("answered = ?")
                params.append(1 if answered else 0)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f'''
                SELECT * FROM curiosity_queries
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            result = await self.db.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for row in result.data:
                if 'domains' in row and isinstance(row['domains'], str):
                    row['domains'] = json.loads(row['domains'])
                if 'metadata' in row and isinstance(row['metadata'], str):
                    row['metadata'] = json.loads(row['metadata'])
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to retrieve curiosity queries: {e}")
            return []
    
    # ========== Knowledge Gap Persistence ==========
    
    async def store_knowledge_gap(
        self,
        gap_id: str,
        gap_description: str,
        uncertainty_level: float,
        related_domains: List[str],
        exploration_priority: float,
        suggested_queries: List[str],
        timestamp: float = None,
        resolved: bool = False,
        resolution_timestamp: float = None,
        resolution_quality: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a knowledge gap to database"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            query = '''
                INSERT OR REPLACE INTO knowledge_gaps (
                    gap_id, gap_description, uncertainty_level, related_domains,
                    exploration_priority, suggested_queries, timestamp,
                    resolved, resolution_timestamp, resolution_quality, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                gap_id,
                gap_description,
                uncertainty_level,
                json.dumps(related_domains),
                exploration_priority,
                json.dumps(suggested_queries),
                timestamp,
                1 if resolved else 0,
                resolution_timestamp,
                resolution_quality,
                json.dumps(metadata or {})
            )
            
            result = await self.db.execute_query(query, params, fetch=False)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to store knowledge gap: {e}")
            return False
    
    async def get_knowledge_gaps(
        self,
        resolved: bool = None,
        min_priority: float = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve knowledge gaps from database"""
        try:
            conditions = []
            params = []
            
            if resolved is not None:
                conditions.append("resolved = ?")
                params.append(1 if resolved else 0)
            
            if min_priority is not None:
                conditions.append("exploration_priority >= ?")
                params.append(min_priority)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f'''
                SELECT * FROM knowledge_gaps
                WHERE {where_clause}
                ORDER BY exploration_priority DESC, timestamp DESC
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            result = await self.db.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for row in result.data:
                for field in ['related_domains', 'suggested_queries', 'metadata']:
                    if field in row and isinstance(row[field], str):
                        row[field] = json.loads(row[field])
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge gaps: {e}")
            return []
    
    # ========== Question Persistence ==========
    
    async def store_question(
        self,
        question_id: str,
        question_text: str,
        question_type: str,
        domain: str,
        timestamp: float = None,
        answered: bool = False,
        answer_id: str = None,
        answer_quality: float = None,
        principle_ids: List[str] = None,
        related_queries: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a question to database"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            query = '''
                INSERT OR REPLACE INTO questions (
                    question_id, question_text, question_type, domain,
                    timestamp, answered, answer_id, answer_quality,
                    principle_ids, related_queries, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                question_id,
                question_text,
                question_type,
                domain,
                timestamp,
                1 if answered else 0,
                answer_id,
                answer_quality,
                json.dumps(principle_ids or []),
                json.dumps(related_queries or []),
                json.dumps(metadata or {})
            )
            
            result = await self.db.execute_query(query, params, fetch=False)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to store question: {e}")
            return False
    
    async def get_questions(
        self,
        question_type: str = None,
        domain: str = None,
        answered: bool = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve questions from database"""
        try:
            conditions = []
            params = []
            
            if question_type:
                conditions.append("question_type = ?")
                params.append(question_type)
            
            if domain:
                conditions.append("domain = ?")
                params.append(domain)
            
            if answered is not None:
                conditions.append("answered = ?")
                params.append(1 if answered else 0)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f'''
                SELECT * FROM questions
                WHERE {where_clause}
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            result = await self.db.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for row in result.data:
                for field in ['principle_ids', 'related_queries', 'metadata']:
                    if field in row and isinstance(row[field], str):
                        row[field] = json.loads(row[field])
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to retrieve questions: {e}")
            return []
    
    # ========== Principle Persistence ==========
    
    async def store_principle(
        self,
        principle_id: str,
        principle_name: str,
        summary: str,
        domains: List[str],
        confidence: float,
        validation_status: str,
        framing: str = None,
        concepts: List[str] = None,
        created_at: float = None,
        updated_at: float = None,
        version: int = 1,
        evidence_count: int = 0,
        usage_count: int = 0,
        is_current: bool = True,
        previous_version_id: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a principle to database"""
        try:
            if created_at is None:
                created_at = time.time()
            if updated_at is None:
                updated_at = time.time()
            
            query = '''
                INSERT OR REPLACE INTO principles (
                    principle_id, principle_name, version, summary, framing,
                    domains, concepts, created_at, updated_at, confidence,
                    validation_status, evidence_count, usage_count,
                    is_current, previous_version_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                principle_id,
                principle_name,
                version,
                summary,
                framing,
                json.dumps(domains),
                json.dumps(concepts or []),
                created_at,
                updated_at,
                confidence,
                validation_status,
                evidence_count,
                usage_count,
                1 if is_current else 0,
                previous_version_id,
                json.dumps(metadata or {})
            )
            
            result = await self.db.execute_query(query, params, fetch=False)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to store principle: {e}")
            return False
    
    async def get_principles(
        self,
        principle_name: str = None,
        domain: str = None,
        is_current: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Retrieve principles from database"""
        try:
            conditions = []
            params = []
            
            if principle_name:
                conditions.append("principle_name = ?")
                params.append(principle_name)
            
            if domain:
                conditions.append("domains LIKE ?")
                params.append(f'%"{domain}"%')
            
            if is_current:
                conditions.append("is_current = 1")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            query = f'''
                SELECT * FROM principles
                WHERE {where_clause}
                ORDER BY updated_at DESC
                LIMIT ? OFFSET ?
            '''
            
            params.extend([limit, offset])
            result = await self.db.execute_query(query, tuple(params))
            
            # Parse JSON fields
            for row in result.data:
                for field in ['domains', 'concepts', 'metadata']:
                    if field in row and isinstance(row[field], str):
                        row[field] = json.loads(row[field])
            
            return result.data
            
        except Exception as e:
            logger.error(f"Failed to retrieve principles: {e}")
            return []
    
    async def record_principle_evolution(
        self,
        evolution_id: str,
        principle_id: str,
        from_version: int,
        to_version: int,
        evolution_type: str,
        changes_description: str,
        confidence_delta: float,
        timestamp: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Record principle evolution to database"""
        try:
            if timestamp is None:
                timestamp = time.time()
            
            query = '''
                INSERT INTO principle_evolution (
                    evolution_id, principle_id, from_version, to_version,
                    evolution_type, changes_description, confidence_delta,
                    timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                evolution_id,
                principle_id,
                from_version,
                to_version,
                evolution_type,
                changes_description,
                confidence_delta,
                timestamp,
                json.dumps(metadata or {})
            )
            
            result = await self.db.execute_query(query, params, fetch=False)
            return result.success
            
        except Exception as e:
            logger.error(f"Failed to record principle evolution: {e}")
            return False

