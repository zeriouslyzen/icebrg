#!/usr/bin/env python3
"""
Clean relationships table to only keep valid relationships.
Removes relationships where source or target entities don't exist.

This fixes the data integrity issue where 97.9% of relationships
reference non-existent entities.
"""

import sqlite3
from pathlib import Path
from datetime import datetime

DB_PATH = Path.home() / "Documents" / "iceburg_matrix" / "matrix.db"
BACKUP_PATH = Path.home() / "Documents" / "iceburg_matrix" / "matrix.db.backup"

def backup_database():
    """Create backup before cleanup."""
    import shutil
    if DB_PATH.exists():
        print(f"Creating backup: {BACKUP_PATH}")
        shutil.copy2(DB_PATH, BACKUP_PATH)
        print("✅ Backup created")
    else:
        print(f"⚠️  Database not found: {DB_PATH}")

def clean_relationships(dry_run=False):
    """Clean relationships table to only keep valid relationships."""
    
    if not DB_PATH.exists():
        print(f"❌ Database not found: {DB_PATH}")
        return
    
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    
    print("=" * 60)
    print("Matrix Relationships Cleanup")
    print("=" * 60)
    
    # Get counts
    cursor.execute("SELECT COUNT(*) FROM relationships")
    total = cursor.fetchone()[0]
    
    print(f"\nTotal relationships: {total:,}")
    
    # Count invalid
    cursor.execute("""
        SELECT COUNT(*) 
        FROM relationships r
        WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = r.source_id)
           OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = r.target_id)
    """)
    invalid = cursor.fetchone()[0]
    valid = total - invalid
    
    print(f"Valid relationships: {valid:,} ({valid/total*100:.1f}%)")
    print(f"Invalid relationships: {invalid:,} ({invalid/total*100:.1f}%)")
    
    if dry_run:
        print("\n🔍 DRY RUN - No changes will be made")
        print(f"Would delete {invalid:,} invalid relationships")
        print(f"Would keep {valid:,} valid relationships")
        conn.close()
        return
    
    # Create backup
    backup_database()
    
    # Delete invalid relationships
    print(f"\n🗑️  Deleting {invalid:,} invalid relationships...")
    start_time = datetime.now()
    
    cursor.execute("""
        DELETE FROM relationships
        WHERE NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = relationships.source_id)
           OR NOT EXISTS (SELECT 1 FROM entities e WHERE e.entity_id = relationships.target_id)
    """)
    conn.commit()
    
    deleted = cursor.rowcount
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Deleted {deleted:,} invalid relationships in {elapsed:.1f}s")
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM relationships")
    remaining = cursor.fetchone()[0]
    print(f"✅ Remaining valid relationships: {remaining:,}")
    
    # Create indexes for performance
    print("\n📊 Creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source_valid ON relationships(source_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target_valid ON relationships(target_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON relationships(relationship_type)")
    conn.commit()
    print("✅ Indexes created")
    
    # Show breakdown by type
    cursor.execute("""
        SELECT relationship_type, COUNT(*) 
        FROM relationships 
        GROUP BY relationship_type 
        ORDER BY COUNT(*) DESC
        LIMIT 10
    """)
    breakdown = cursor.fetchall()
    
    print("\n📈 Top relationship types:")
    for rel_type, count in breakdown:
        print(f"   {rel_type}: {count:,}")
    
    conn.close()
    print("\n✅ Data cleanup complete!")
    print(f"💾 Backup saved to: {BACKUP_PATH}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean invalid relationships from matrix.db")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without making changes")
    args = parser.parse_args()
    
    clean_relationships(dry_run=args.dry_run)
