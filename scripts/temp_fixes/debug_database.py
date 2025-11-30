#!/usr/bin/env python3
"""
Debug database storage and retrieval
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iceburg.monitoring.unified_performance_tracker import UnifiedPerformanceTracker
import time

def debug_database():
    print("🔍 Debugging database storage and retrieval...")
    
    # Create tracker
    tracker = UnifiedPerformanceTracker()
    
    # Check database directly
    import sqlite3
    with sqlite3.connect(tracker.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM performance_metrics")
        count = cursor.fetchone()[0]
        print(f"Total records in database: {count}")
        
        # Get recent records
        cutoff_time = time.time() - 3600  # 1 hour ago
        cursor.execute("""
            SELECT query_id, response_time, accuracy, timestamp 
            FROM performance_metrics 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC 
            LIMIT 5
        """, (cutoff_time,))
        
        recent_records = cursor.fetchall()
        print(f"Recent records (last hour): {len(recent_records)}")
        for record in recent_records:
            print(f"  {record}")
    
    # Test get_performance_summary
    print("\nTesting get_performance_summary...")
    summary = tracker.get_performance_summary(hours=1)
    print(f"Summary: {summary}")
    
    # Test _get_recent_metrics directly
    print("\nTesting _get_recent_metrics...")
    recent_metrics = tracker._get_recent_metrics(hours=1)
    print(f"Recent metrics count: {len(recent_metrics)}")
    for metric in recent_metrics:
        print(f"  {metric.query_id}: {metric.response_time}s, {metric.accuracy}")

if __name__ == "__main__":
    debug_database()
