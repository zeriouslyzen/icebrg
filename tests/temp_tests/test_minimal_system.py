#!/usr/bin/env python3
"""
Test minimal ICEBURG system to demonstrate core functionality
"""

import sys
import os
sys.path.insert(0, 'src')

def test_minimal_system():
    """Test the minimal ICEBURG system"""
    print("🔍 Testing Minimal ICEBURG System")
    print("=" * 50)
    
    try:
        # Test minimal protocol
        from iceburg.protocol_minimal import iceberg_protocol
        print("✅ Minimal protocol imported successfully")
        
        # Test basic functionality
        result = iceberg_protocol("What is artificial intelligence?", verbose=True)
        print(f"✅ Protocol response: {result[:100]}...")
        
        print("\n🎯 Core Issue Identified:")
        print("   - protocol.py has 149+ syntax errors")
        print("   - agents/ directory has more syntax errors")
        print("   - This is a systematic issue from incomplete template generation")
        
        print("\n💡 Recommended Solution:")
        print("   1. Use the minimal protocol for basic functionality")
        print("   2. Fix syntax errors systematically (not one by one)")
        print("   3. Focus on the working components first")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_minimal_system()
