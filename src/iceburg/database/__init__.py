"""
ICEBURG Unified Database System
Centralized database for all ICEBURG data with migration capabilities

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

from .unified_database import (
    UnifiedDatabase,
    DatabaseConfig,
    QueryResult,
    create_unified_database,
    execute_query
)

from .data_migration import (
    DataMigrationSystem,
    MigrationStatus,
    create_data_migration_system,
    migrate_all_data
)

__all__ = [
    # Unified Database
    "UnifiedDatabase",
    "DatabaseConfig", 
    "QueryResult",
    "create_unified_database",
    "execute_query",
    
    # Data Migration
    "DataMigrationSystem",
    "MigrationStatus",
    "create_data_migration_system",
    "migrate_all_data"
]
