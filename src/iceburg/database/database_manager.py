"""
ICEBURG Database Manager
High-level interface for unified database operations

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from ..config import IceburgConfig
from .unified_database import UnifiedDatabase, DatabaseConfig, QueryResult
from .data_migration import DataMigrationSystem

logger = logging.getLogger(__name__)

@dataclass
class DatabaseOperation:
    """Database operation record"""
    operation_id: str
    operation_type: str  # "insert", "update", "delete", "select"
    table_name: str
    records_affected: int
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

class DatabaseManager:
    """
    High-level interface for unified database operations
    Provides convenient methods for common database operations
    """
    
    def __init__(self, cfg: IceburgConfig, db_config: DatabaseConfig = None):
        self.cfg = cfg
        self.unified_db = UnifiedDatabase(cfg, db_config)
        self.migration_system = DataMigrationSystem(cfg, self.unified_db)
        
        # Operation tracking
        self.operation_log: List[DatabaseOperation] = []
        
        logger.info("🗄️ Database Manager initialized")
    
    async def initialize_database(self, migrate_existing_data: bool = True) -> Dict[str, Any]:
        """Initialize database and optionally migrate existing data"""
        
        logger.info("🗄️ Initializing unified database...")
        
        initialization_result = {
            "database_initialized": True,
            "migration_completed": False,
            "migration_summary": None,
            "database_stats": None
        }
        
        try:
            # Get initial database stats
            initialization_result["database_stats"] = self.unified_db.get_database_stats()
            
            # Migrate existing data if requested
            if migrate_existing_data:
                logger.info("🔄 Migrating existing data...")
                migration_summary = await self.migration_system.migrate_all_data()
                initialization_result["migration_completed"] = True
                initialization_result["migration_summary"] = migration_summary
            
            # Optimize database after migration
            self.unified_db.optimize_database()
            
            # Get final database stats
            initialization_result["database_stats"] = self.unified_db.get_database_stats()
            
            logger.info("✅ Database initialization completed")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            initialization_result["database_initialized"] = False
            initialization_result["error"] = str(e)
        
        return initialization_result
    
    # Agent Performance Operations
    async def record_agent_performance(
        self,
        agent_name: str,
        agent_version: str,
        task_type: str,
        execution_time: float,
        success: bool,
        quality_score: float,
        resource_usage: Dict[str, float],
        input_size: int,
        output_size: int,
        error_message: str = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Record agent performance metrics"""
        
        performance_id = f"perf_{agent_name}_{task_type}_{int(time.time())}"
        
        query = '''
            INSERT INTO agent_performance (
                performance_id, agent_name, agent_version, task_type,
                execution_time, success, quality_score, resource_usage,
                input_size, output_size, error_message, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            performance_id, agent_name, agent_version, task_type,
            execution_time, success, quality_score, json.dumps(resource_usage),
            input_size, output_size, error_message, time.time(),
            json.dumps(metadata or {})
        )
        
        result = await self.unified_db.execute_query(query, params, fetch=False)
        await self._log_operation("insert", "agent_performance", 1, result.execution_time, result.success)
        
        return performance_id
    
    async def get_agent_performance(
        self,
        agent_name: str = None,
        task_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get agent performance data"""
        
        query = "SELECT * FROM agent_performance WHERE 1=1"
        params = []
        
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = await self.unified_db.execute_query(query, tuple(params))
        await self._log_operation("select", "agent_performance", len(result.data), result.execution_time, result.success)
        
        return result.data
    
    # Knowledge Operations
    async def store_knowledge_concept(
        self,
        concept_name: str,
        concept_type: str,
        domain: str,
        definition: str,
        confidence: float,
        source: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store knowledge concept"""
        
        concept_id = f"concept_{concept_name}_{domain}_{int(time.time())}"
        
        query = '''
            INSERT INTO knowledge_concepts (
                concept_id, concept_name, concept_type, domain,
                definition, confidence, source, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            concept_id, concept_name, concept_type, domain,
            definition, confidence, source, time.time(), time.time(),
            json.dumps(metadata or {})
        )
        
        result = await self.unified_db.execute_query(query, params, fetch=False)
        await self._log_operation("insert", "knowledge_concepts", 1, result.execution_time, result.success)
        
        return concept_id
    
    async def get_knowledge_concepts(
        self,
        domain: str = None,
        concept_type: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get knowledge concepts"""
        
        query = "SELECT * FROM knowledge_concepts WHERE 1=1"
        params = []
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        if concept_type:
            query += " AND concept_type = ?"
            params.append(concept_type)
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = await self.unified_db.execute_query(query, tuple(params))
        await self._log_operation("select", "knowledge_concepts", len(result.data), result.execution_time, result.success)
        
        return result.data
    
    # Memory Operations
    async def store_memory(
        self,
        memory_type: str,
        content: str,
        domain: str,
        importance: float,
        associations: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store memory entry"""
        
        memory_id = f"mem_{memory_type}_{domain}_{int(time.time())}"
        
        query = '''
            INSERT INTO memory_entries (
                memory_id, memory_type, content, domain, importance,
                associations, created_at, last_accessed, memory_strength,
                cross_references, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            memory_id, memory_type, content, domain, importance,
            json.dumps(associations or []), time.time(), time.time(), importance,
            json.dumps([]), json.dumps(metadata or {})
        )
        
        result = await self.unified_db.execute_query(query, params, fetch=False)
        await self._log_operation("insert", "memory_entries", 1, result.execution_time, result.success)
        
        return memory_id
    
    async def get_memories(
        self,
        memory_type: str = None,
        domain: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get memory entries"""
        
        query = "SELECT * FROM memory_entries WHERE 1=1"
        params = []
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        
        if domain:
            query += " AND domain = ?"
            params.append(domain)
        
        query += " ORDER BY last_accessed DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = await self.unified_db.execute_query(query, tuple(params))
        await self._log_operation("select", "memory_entries", len(result.data), result.execution_time, result.success)
        
        return result.data
    
    # Research Operations
    async def store_research_output(
        self,
        research_type: str,
        title: str,
        content: str,
        domains: List[str],
        quality_score: float,
        validation_status: str,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store research output"""
        
        output_id = f"research_{research_type}_{int(time.time())}"
        
        query = '''
            INSERT INTO research_outputs (
                output_id, research_type, title, content, domains,
                quality_score, validation_status, created_at, updated_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            output_id, research_type, title, content, json.dumps(domains),
            quality_score, validation_status, time.time(), time.time(),
            json.dumps(metadata or {})
        )
        
        result = await self.unified_db.execute_query(query, params, fetch=False)
        await self._log_operation("insert", "research_outputs", 1, result.execution_time, result.success)
        
        return output_id
    
    async def get_research_outputs(
        self,
        research_type: str = None,
        domains: List[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get research outputs"""
        
        query = "SELECT * FROM research_outputs WHERE 1=1"
        params = []
        
        if research_type:
            query += " AND research_type = ?"
            params.append(research_type)
        
        if domains:
            # Check if any of the domains are in the domains JSON array
            domain_conditions = " OR ".join(["domains LIKE ?" for _ in domains])
            query += f" AND ({domain_conditions})"
            params.extend([f"%{domain}%" for domain in domains])
        
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = await self.unified_db.execute_query(query, tuple(params))
        await self._log_operation("select", "research_outputs", len(result.data), result.execution_time, result.success)
        
        return result.data
    
    # System Metrics Operations
    async def record_system_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: str = None,
        source: str = "system",
        metadata: Dict[str, Any] = None
    ) -> str:
        """Record system metric"""
        
        metric_id = f"metric_{metric_name}_{int(time.time())}"
        
        query = '''
            INSERT INTO system_metrics (
                metric_id, metric_name, metric_value, metric_unit,
                timestamp, source, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            metric_id, metric_name, metric_value, metric_unit,
            time.time(), source, json.dumps(metadata or {})
        )
        
        result = await self.unified_db.execute_query(query, params, fetch=False)
        await self._log_operation("insert", "system_metrics", 1, result.execution_time, result.success)
        
        return metric_id
    
    async def get_system_metrics(
        self,
        metric_name: str = None,
        source: str = None,
        start_time: float = None,
        end_time: float = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get system metrics"""
        
        query = "SELECT * FROM system_metrics WHERE 1=1"
        params = []
        
        if metric_name:
            query += " AND metric_name = ?"
            params.append(metric_name)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = await self.unified_db.execute_query(query, tuple(params))
        await self._log_operation("select", "system_metrics", len(result.data), result.execution_time, result.success)
        
        return result.data
    
    # Cross-Domain Synthesis Operations
    async def record_synthesis(
        self,
        domains: List[str],
        synthesis_type: str,
        result: str,
        quality_score: float,
        processing_time: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Record cross-domain synthesis"""
        
        synthesis_id = f"synthesis_{synthesis_type}_{int(time.time())}"
        
        query = '''
            INSERT INTO cross_domain_synthesis (
                synthesis_id, domains, synthesis_type, result,
                quality_score, processing_time, success, timestamp, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        params = (
            synthesis_id, json.dumps(domains), synthesis_type, result,
            quality_score, processing_time, success, time.time(),
            json.dumps(metadata or {})
        )
        
        result = await self.unified_db.execute_query(query, params, fetch=False)
        await self._log_operation("insert", "cross_domain_synthesis", 1, result.execution_time, result.success)
        
        return synthesis_id
    
    async def get_synthesis_history(
        self,
        synthesis_type: str = None,
        domains: List[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get synthesis history"""
        
        query = "SELECT * FROM cross_domain_synthesis WHERE 1=1"
        params = []
        
        if synthesis_type:
            query += " AND synthesis_type = ?"
            params.append(synthesis_type)
        
        if domains:
            # Check if any of the domains are in the domains JSON array
            domain_conditions = " OR ".join(["domains LIKE ?" for _ in domains])
            query += f" AND ({domain_conditions})"
            params.extend([f"%{domain}%" for domain in domains])
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        result = await self.unified_db.execute_query(query, tuple(params))
        await self._log_operation("select", "cross_domain_synthesis", len(result.data), result.execution_time, result.success)
        
        return result.data
    
    # Utility Operations
    async def _log_operation(
        self,
        operation_type: str,
        table_name: str,
        records_affected: int,
        execution_time: float,
        success: bool,
        error_message: str = None
    ):
        """Log database operation"""
        
        operation_id = f"op_{operation_type}_{table_name}_{int(time.time())}"
        
        operation = DatabaseOperation(
            operation_id=operation_id,
            operation_type=operation_type,
            table_name=table_name,
            records_affected=records_affected,
            execution_time=execution_time,
            success=success,
            error_message=error_message
        )
        
        self.operation_log.append(operation)
        
        # Keep only last 1000 operations
        if len(self.operation_log) > 1000:
            self.operation_log = self.operation_log[-1000:]
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        
        db_stats = self.unified_db.get_database_stats()
        migration_stats = self.migration_system.get_migration_status()
        
        # Calculate operation statistics
        total_operations = len(self.operation_log)
        successful_operations = len([op for op in self.operation_log if op.success])
        failed_operations = total_operations - successful_operations
        
        avg_execution_time = (
            sum(op.execution_time for op in self.operation_log) / max(1, total_operations)
        )
        
        return {
            "database": db_stats,
            "migration": migration_stats,
            "operations": {
                "total_operations": total_operations,
                "successful_operations": successful_operations,
                "failed_operations": failed_operations,
                "success_rate": successful_operations / max(1, total_operations),
                "avg_execution_time": avg_execution_time
            }
        }
    
    def optimize_database(self):
        """Optimize database performance"""
        self.unified_db.optimize_database()
    
    def backup_database(self, backup_path: str):
        """Create database backup"""
        self.unified_db.backup_database(backup_path)
    
    def close(self):
        """Close database connections"""
        self.unified_db.close()


# Helper functions for integration
def create_database_manager(cfg: IceburgConfig, db_config: DatabaseConfig = None) -> DatabaseManager:
    """Create database manager instance"""
    return DatabaseManager(cfg, db_config)

async def initialize_database(
    db_manager: DatabaseManager,
    migrate_existing_data: bool = True
) -> Dict[str, Any]:
    """Initialize database and migrate existing data"""
    return await db_manager.initialize_database(migrate_existing_data)
