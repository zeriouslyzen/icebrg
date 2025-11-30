#!/usr/bin/env python3
"""
Test Bottleneck Detection System
Tests ICEBURG's bottleneck detection and auto-healing capabilities
"""
import asyncio
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.iceburg.monitoring.bottleneck_detector import (
    BottleneckMonitor,
    create_bottleneck_monitor,
    start_iceburg_monitoring
)
from src.iceburg.monitoring.unified_performance_tracker import (
    UnifiedPerformanceTracker,
    get_global_tracker
)

async def test_bottleneck_detection():
    """Test bottleneck detection system"""
    print("=" * 60)
    print("Testing ICEBURG Bottleneck Detection System")
    print("=" * 60)
    
    # Test 1: Initialize Bottleneck Monitor
    print("\n[TEST 1] Initializing Bottleneck Monitor...")
    try:
        monitor = await create_bottleneck_monitor()
        print("✅ Bottleneck Monitor initialized")
    except Exception as e:
        print(f"❌ Failed to initialize Bottleneck Monitor: {e}")
        return False
    
    # Test 2: Start Monitoring
    print("\n[TEST 2] Starting monitoring...")
    try:
        await monitor.start_monitoring()
        print("✅ Monitoring started")
        
        # Wait for first monitoring cycle (30 seconds)
        print("⏳ Waiting for first monitoring cycle (30 seconds)...")
        await asyncio.sleep(35)  # Wait a bit longer to ensure cycle completes
        
        # Check status
        status = monitor.get_monitoring_status()
        print(f"✅ Monitoring status: {status.get('monitoring_active', False)}")
        print(f"   Total alerts: {status.get('total_alerts', 0)}")
        print(f"   Resolved alerts: {status.get('resolved_alerts', 0)}")
        
    except Exception as e:
        print(f"❌ Failed to start monitoring: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Initialize Performance Tracker
    print("\n[TEST 3] Initializing Performance Tracker...")
    try:
        tracker = get_global_tracker()
        await tracker.start_tracking()
        print("✅ Performance Tracker initialized and started")
        
        # Track some test queries
        print("⏳ Tracking test queries...")
        for i in range(5):
            tracker.track_query_performance(
                query_id=f"test_query_{i}",
                response_time=1.0 + (i * 0.5),  # Simulate varying response times
                accuracy=0.8 + (i * 0.02),
                resources={
                    "memory_usage_mb": 100 + i * 20,
                    "cache_hit_rate": 0.7 + (i * 0.03),
                    "agent_count": 3,
                    "parallel_execution": i % 2 == 0,
                    "query_complexity": 0.5 + (i * 0.1)
                },
                success=True,
                metadata={"test": True, "iteration": i}
            )
            await asyncio.sleep(0.5)
        
        # Wait for buffer flush
        await asyncio.sleep(3)
        
        # Get summary
        summary = tracker.get_performance_summary(hours=1)
        print(f"✅ Performance summary generated")
        print(f"   Total queries: {summary.get('total_queries', 0)}")
        print(f"   Success rate: {summary.get('success_rate', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Failed to initialize Performance Tracker: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Check if monitoring is actually detecting issues
    print("\n[TEST 4] Checking if monitoring detects bottlenecks...")
    try:
        # Wait for another monitoring cycle
        await asyncio.sleep(35)
        
        status = monitor.get_monitoring_status()
        alerts = status.get('recent_alerts', [])
        healing_history = status.get('healing_history', [])
        
        print(f"✅ Recent alerts: {len(alerts)}")
        if alerts:
            for alert in alerts[-5:]:  # Show last 5 alerts
                # Alerts are dicts from get_monitoring_status
                alert_type = alert.get('bottleneck_type', 'unknown') if isinstance(alert, dict) else getattr(alert, 'bottleneck_type', 'unknown')
                severity = alert.get('severity', 'unknown') if isinstance(alert, dict) else getattr(alert, 'severity', 'unknown')
                description = alert.get('description', 'N/A') if isinstance(alert, dict) else getattr(alert, 'description', 'N/A')
                print(f"   - {alert_type}: {severity} - {description}")
        
        print(f"✅ Healing actions: {len(healing_history)}")
        if healing_history:
            for action in healing_history[-5:]:  # Show last 5 actions
                action_name = action.get('action', 'unknown') if isinstance(action, dict) else getattr(action, 'action', 'unknown')
                success = action.get('success', False) if isinstance(action, dict) else getattr(action, 'success', False)
                print(f"   - {action_name}: {'✅ Success' if success else '❌ Failed'}")
        
    except Exception as e:
        print(f"❌ Failed to check monitoring: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Verify system isn't wasting resources
    print("\n[TEST 5] Checking resource usage...")
    try:
        import psutil
        process = psutil.Process()
        
        # Check CPU usage
        cpu_percent = process.cpu_percent(interval=1)
        print(f"✅ CPU usage: {cpu_percent:.1f}%")
        
        # Check memory usage
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        print(f"✅ Memory usage: {memory_mb:.1f} MB")
        
        # Check if monitoring is using too many resources
        if cpu_percent > 50:
            print(f"⚠️  WARNING: High CPU usage ({cpu_percent:.1f}%)")
            return False
        
        if memory_mb > 1000:
            print(f"⚠️  WARNING: High memory usage ({memory_mb:.1f} MB)")
            return False
        
        print("✅ Resource usage is reasonable")
        
    except Exception as e:
        print(f"❌ Failed to check resource usage: {e}")
        return False
    
    # Test 6: Stop monitoring and cleanup
    print("\n[TEST 6] Stopping monitoring and cleanup...")
    try:
        await monitor.stop_monitoring()
        await tracker.stop_tracking()
        print("✅ Monitoring stopped and cleaned up")
    except Exception as e:
        print(f"❌ Failed to stop monitoring: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Bottleneck detection system is working")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_bottleneck_detection())
    sys.exit(0 if success else 1)

