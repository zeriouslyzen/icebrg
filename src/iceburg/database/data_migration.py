"""
ICEBURG Data Migration System
Migrates existing fragmented data to unified database

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import shutil

from ..config import IceburgConfig
from .unified_database import UnifiedDatabase, DatabaseConfig

logger = logging.getLogger(__name__)

@dataclass
class MigrationStatus:
    """Status of data migration"""
    migration_id: str
    source_system: str
    target_table: str
    records_migrated: int
    records_failed: int
    migration_time: float
    status: str  # "pending", "in_progress", "completed", "failed"
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DataMigrationSystem:
    """
    Migrates existing fragmented data to unified database
    """
    
    def __init__(self, cfg: IceburgConfig, unified_db: UnifiedDatabase):
        self.cfg = cfg
        self.unified_db = unified_db
        self.migration_log: List[MigrationStatus] = []
        
        # Source database paths
        self.source_databases = {
            "api_usage": "data/metrics/api_usage.db",
            "benchmark_results": "data/metrics/benchmark_results.db",
            "emergence_memory": "data/emergence_memory.db",
            "chroma_vector": "data/chroma/simple_vector_db.sqlite",
            "sovereign_library": "data/sovereign_library.db",
            "demo_sovereign_library": "data/demo_sovereign_library.db"
        }
        
        # JSON data paths
        self.json_data_paths = [
            "data/memory_system/memories.json",
            "data/memory_system/patterns.json",
            "data/universal_knowledge_base/",
            "data/research_outputs/",
            "data/conversation_logs/",
            "data/optimization/"
        ]
        
        logger.info("🔄 Data Migration System initialized")
    
    async def migrate_all_data(self) -> Dict[str, Any]:
        """Migrate all existing data to unified database"""
        
        start_time = time.time()
        migration_summary = {
            "total_migrations": 0,
            "successful_migrations": 0,
            "failed_migrations": 0,
            "total_records_migrated": 0,
            "total_records_failed": 0,
            "migration_time": 0.0,
            "migration_details": []
        }
        
        logger.info("🔄 Starting comprehensive data migration...")
        
        # Migrate SQLite databases
        for db_name, db_path in self.source_databases.items():
            if Path(db_path).exists():
                result = await self._migrate_sqlite_database(db_name, db_path)
                migration_summary["migration_details"].append(result)
                migration_summary["total_migrations"] += 1
                
                if result["status"] == "completed":
                    migration_summary["successful_migrations"] += 1
                    migration_summary["total_records_migrated"] += result["records_migrated"]
                else:
                    migration_summary["failed_migrations"] += 1
                    migration_summary["total_records_failed"] += result["records_failed"]
        
        # Migrate JSON data
        for json_path in self.json_data_paths:
            if Path(json_path).exists():
                result = await self._migrate_json_data(json_path)
                migration_summary["migration_details"].append(result)
                migration_summary["total_migrations"] += 1
                
                if result["status"] == "completed":
                    migration_summary["successful_migrations"] += 1
                    migration_summary["total_records_migrated"] += result["records_migrated"]
                else:
                    migration_summary["failed_migrations"] += 1
                    migration_summary["total_records_failed"] += result["records_failed"]
        
        migration_summary["migration_time"] = time.time() - start_time
        
        logger.info(f"✅ Data migration completed: {migration_summary['successful_migrations']}/{migration_summary['total_migrations']} successful")
        
        return migration_summary
    
    async def _migrate_sqlite_database(self, db_name: str, db_path: str) -> Dict[str, Any]:
        """Migrate SQLite database to unified database"""
        
        migration_id = f"sqlite_{db_name}_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"🔄 Migrating SQLite database: {db_name}")
        
        try:
            # Connect to source database
            source_conn = sqlite3.connect(db_path)
            source_conn.row_factory = sqlite3.Row
            source_cursor = source_conn.cursor()
            
            # Get table information
            source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in source_cursor.fetchall()]
            
            records_migrated = 0
            records_failed = 0
            
            # Migrate each table
            for table_name in tables:
                try:
                    # Get table schema
                    source_cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = source_cursor.fetchall()
                    
                    # Get all data from table
                    source_cursor.execute(f"SELECT * FROM {table_name}")
                    rows = source_cursor.fetchall()
                    
                    # Map to unified database table
                    target_table = self._map_table_name(db_name, table_name)
                    
                    if target_table:
                        # Migrate data
                        for row in rows:
                            try:
                                await self._migrate_table_row(
                                    db_name, table_name, target_table, dict(row)
                                )
                                records_migrated += 1
                            except Exception as e:
                                logger.warning(f"Failed to migrate row from {table_name}: {e}")
                                records_failed += 1
                    
                except Exception as e:
                    logger.error(f"Failed to migrate table {table_name}: {e}")
                    records_failed += 1
            
            source_conn.close()
            
            migration_time = time.time() - start_time
            
            result = {
                "migration_id": migration_id,
                "source_system": f"sqlite_{db_name}",
                "target_table": "multiple",
                "records_migrated": records_migrated,
                "records_failed": records_failed,
                "migration_time": migration_time,
                "status": "completed" if records_failed == 0 else "completed_with_errors"
            }
            
            self.migration_log.append(MigrationStatus(**result))
            
            return result
            
        except Exception as e:
            migration_time = time.time() - start_time
            
            result = {
                "migration_id": migration_id,
                "source_system": f"sqlite_{db_name}",
                "target_table": "multiple",
                "records_migrated": 0,
                "records_failed": 1,
                "migration_time": migration_time,
                "status": "failed",
                "error_message": str(e)
            }
            
            self.migration_log.append(MigrationStatus(**result))
            
            return result
    
    async def _migrate_json_data(self, json_path: str) -> Dict[str, Any]:
        """Migrate JSON data to unified database"""
        
        migration_id = f"json_{Path(json_path).name}_{int(time.time())}"
        start_time = time.time()
        
        logger.info(f"🔄 Migrating JSON data: {json_path}")
        
        try:
            records_migrated = 0
            records_failed = 0
            
            json_path_obj = Path(json_path)
            
            if json_path_obj.is_file():
                # Single JSON file
                await self._migrate_json_file(json_path_obj, records_migrated, records_failed)
            elif json_path_obj.is_dir():
                # Directory of JSON files
                for json_file in json_path_obj.rglob("*.json"):
                    await self._migrate_json_file(json_file, records_migrated, records_failed)
            
            migration_time = time.time() - start_time
            
            result = {
                "migration_id": migration_id,
                "source_system": f"json_{json_path_obj.name}",
                "target_table": "multiple",
                "records_migrated": records_migrated,
                "records_failed": records_failed,
                "migration_time": migration_time,
                "status": "completed" if records_failed == 0 else "completed_with_errors"
            }
            
            self.migration_log.append(MigrationStatus(**result))
            
            return result
            
        except Exception as e:
            migration_time = time.time() - start_time
            
            result = {
                "migration_id": migration_id,
                "source_system": f"json_{Path(json_path).name}",
                "target_table": "multiple",
                "records_migrated": 0,
                "records_failed": 1,
                "migration_time": migration_time,
                "status": "failed",
                "error_message": str(e)
            }
            
            self.migration_log.append(MigrationStatus(**result))
            
            return result
    
    async def _migrate_json_file(self, json_file: Path, records_migrated: int, records_failed: int):
        """Migrate single JSON file"""
        
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Determine target table based on file path
            target_table = self._map_json_file_to_table(json_file)
            
            if target_table:
                if isinstance(data, list):
                    for item in data:
                        try:
                            await self._migrate_json_record(item, target_table)
                            records_migrated += 1
                        except Exception as e:
                            logger.warning(f"Failed to migrate JSON record: {e}")
                            records_failed += 1
                elif isinstance(data, dict):
                    try:
                        await self._migrate_json_record(data, target_table)
                        records_migrated += 1
                    except Exception as e:
                        logger.warning(f"Failed to migrate JSON record: {e}")
                        records_failed += 1
                        
        except Exception as e:
            logger.error(f"Failed to migrate JSON file {json_file}: {e}")
            records_failed += 1
    
    async def _migrate_json_record(self, record: Dict[str, Any], target_table: str):
        """Migrate single JSON record to target table"""
        
        # Map JSON fields to database columns
        mapped_data = self._map_json_fields(record, target_table)
        
        if mapped_data:
            # Insert into unified database
            columns = list(mapped_data.keys())
            values = list(mapped_data.values())
            placeholders = ", ".join(["?" for _ in values])
            
            query = f"INSERT INTO {target_table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            await self.unified_db.execute_query(query, tuple(values), fetch=False, cache=False)
    
    def _map_table_name(self, db_name: str, table_name: str) -> Optional[str]:
        """Map source table name to unified database table"""
        
        mapping = {
            "api_usage": {
                "api_usage": "system_metrics"
            },
            "benchmark_results": {
                "benchmark_results": "system_metrics"
            },
            "emergence_memory": {
                "memory_entries": "memory_entries",
                "memory_patterns": "emergence_patterns"
            },
            "sovereign_library": {
                "knowledge_entries": "knowledge_concepts",
                "research_outputs": "research_outputs"
            }
        }
        
        return mapping.get(db_name, {}).get(table_name)
    
    def _map_json_file_to_table(self, json_file: Path) -> Optional[str]:
        """Map JSON file to unified database table"""
        
        file_name = json_file.name.lower()
        file_path = str(json_file).lower()
        
        if "memory" in file_name:
            return "memory_entries"
        elif "pattern" in file_name:
            return "emergence_patterns"
        elif "research" in file_path:
            return "research_outputs"
        elif "conversation" in file_path:
            return "user_sessions"
        elif "optimization" in file_path:
            return "performance_optimization"
        elif "knowledge" in file_path:
            return "knowledge_concepts"
        else:
            return "system_metrics"  # Default fallback
    
    def _map_json_fields(self, record: Dict[str, Any], target_table: str) -> Dict[str, Any]:
        """Map JSON fields to database columns"""
        
        # Common field mappings
        field_mappings = {
            "id": "id",
            "timestamp": "timestamp",
            "created_at": "created_at",
            "updated_at": "updated_at",
            "metadata": "metadata",
            "content": "content",
            "data": "data"
        }
        
        mapped_data = {}
        
        for key, value in record.items():
            # Map field name
            db_field = field_mappings.get(key, key)
            
            # Convert value to appropriate type
            if isinstance(value, (dict, list)):
                mapped_data[db_field] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                mapped_data[db_field] = value
            else:
                mapped_data[db_field] = str(value)
        
        # Add required fields if missing
        if "timestamp" not in mapped_data:
            mapped_data["timestamp"] = time.time()
        
        if "created_at" not in mapped_data:
            mapped_data["created_at"] = time.time()
        
        return mapped_data
    
    async def _migrate_table_row(
        self,
        db_name: str,
        source_table: str,
        target_table: str,
        row_data: Dict[str, Any]
    ):
        """Migrate single table row"""
        
        # Map row data to target table schema
        mapped_data = self._map_table_row_fields(row_data, source_table, target_table)
        
        if mapped_data:
            # Insert into unified database
            columns = list(mapped_data.keys())
            values = list(mapped_data.values())
            placeholders = ", ".join(["?" for _ in values])
            
            query = f"INSERT INTO {target_table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            await self.unified_db.execute_query(query, tuple(values), fetch=False, cache=False)
    
    def _map_table_row_fields(
        self,
        row_data: Dict[str, Any],
        source_table: str,
        target_table: str
    ) -> Dict[str, Any]:
        """Map table row fields to target table schema"""
        
        # Define field mappings for each table
        field_mappings = {
            "memory_entries": {
                "memory_id": "memory_id",
                "memory_type": "memory_type",
                "data": "content",
                "timestamp": "created_at",
                "source": "domain",
                "tags": "associations"
            },
            "system_metrics": {
                "metric_name": "metric_name",
                "value": "metric_value",
                "timestamp": "timestamp",
                "source": "source"
            },
            "knowledge_concepts": {
                "concept_id": "concept_id",
                "concept_name": "concept_name",
                "definition": "definition",
                "domain": "domain",
                "confidence": "confidence"
            }
        }
        
        mapping = field_mappings.get(target_table, {})
        mapped_data = {}
        
        for source_field, value in row_data.items():
            target_field = mapping.get(source_field, source_field)
            
            # Convert value to appropriate type
            if isinstance(value, (dict, list)):
                mapped_data[target_field] = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                mapped_data[target_field] = value
            else:
                mapped_data[target_field] = str(value)
        
        # Add required fields if missing
        if "timestamp" not in mapped_data:
            mapped_data["timestamp"] = time.time()
        
        if "created_at" not in mapped_data:
            mapped_data["created_at"] = time.time()
        
        return mapped_data
    
    def get_migration_status(self) -> Dict[str, Any]:
        """Get migration status and statistics"""
        
        total_migrations = len(self.migration_log)
        successful_migrations = len([m for m in self.migration_log if m.status == "completed"])
        failed_migrations = len([m for m in self.migration_log if m.status == "failed"])
        
        total_records_migrated = sum(m.records_migrated for m in self.migration_log)
        total_records_failed = sum(m.records_failed for m in self.migration_log)
        
        return {
            "total_migrations": total_migrations,
            "successful_migrations": successful_migrations,
            "failed_migrations": failed_migrations,
            "success_rate": successful_migrations / max(1, total_migrations),
            "total_records_migrated": total_records_migrated,
            "total_records_failed": total_records_failed,
            "migration_details": [
                {
                    "migration_id": m.migration_id,
                    "source_system": m.source_system,
                    "target_table": m.target_table,
                    "records_migrated": m.records_migrated,
                    "records_failed": m.records_failed,
                    "status": m.status,
                    "error_message": m.error_message
                }
                for m in self.migration_log
            ]
        }
    
    def create_backup(self, backup_dir: str):
        """Create backup of source data before migration"""
        
        backup_path = Path(backup_dir)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Backup SQLite databases
        for db_name, db_path in self.source_databases.items():
            if Path(db_path).exists():
                backup_file = backup_path / f"{db_name}.db"
                shutil.copy2(db_path, backup_file)
        
        # Backup JSON data
        for json_path in self.json_data_paths:
            if Path(json_path).exists():
                if Path(json_path).is_file():
                    backup_file = backup_path / Path(json_path).name
                    shutil.copy2(json_path, backup_file)
                elif Path(json_path).is_dir():
                    backup_dir = backup_path / Path(json_path).name
                    shutil.copytree(json_path, backup_dir, dirs_exist_ok=True)
        
        logger.info(f"📦 Backup created: {backup_dir}")


# Helper functions for integration
def create_data_migration_system(cfg: IceburgConfig, unified_db: UnifiedDatabase) -> DataMigrationSystem:
    """Create data migration system instance"""
    return DataMigrationSystem(cfg, unified_db)

async def migrate_all_data(migration_system: DataMigrationSystem) -> Dict[str, Any]:
    """Migrate all existing data to unified database"""
    return await migration_system.migrate_all_data()
