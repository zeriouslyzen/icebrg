#!/usr/bin/env python3
"""
Simple test script for the modular ICEBURG protocol refactoring.
Tests core functionality without importing problematic agent modules.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_core_modules():
    """Test core modular protocol modules work."""
    
    print("🧪 Testing Core Modular Protocol Modules")
    print("=" * 50)
    
    try:
        from src.iceburg.protocol.config import ProtocolConfig
        print("✅ ProtocolConfig import successful")
        
        from src.iceburg.protocol.models import Query, Mode, AgentTask, AgentResult
        print("✅ Protocol models import successful")
        
        from src.iceburg.protocol.triage import classify_query
        print("✅ Triage module import successful")
        
        from src.iceburg.protocol.planner import plan
        print("✅ Planner module import successful")
        
        from src.iceburg.protocol.synthesis.evidence import synthesize
        print("✅ Synthesis module import successful")
        
        from src.iceburg.protocol.reporting.formatter import format_report
        print("✅ Reporting module import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Core modules test failed: {e}")
        return False


def test_config_creation():
    """Test configuration creation."""
    
    print("\n📋 Test: Configuration Creation")
    print("=" * 40)
    
    try:
        from src.iceburg.protocol.config import ProtocolConfig
        
        # Test basic config
        config = ProtocolConfig()
        print(f"✅ Basic config created: fast={config.fast}, verbose={config.verbose}")
        
        # Test CIM Stack config
        config_cim = ProtocolConfig(
            force_molecular=True,
            force_bioelectric=True,
            force_hypothesis_testing=True
        )
        print(f"✅ CIM Stack config created: molecular={config_cim.force_molecular}")
        
        # Test AGI config
        config_agi = ProtocolConfig(force_agi=True)
        print(f"✅ AGI config created: force_agi={config_agi.force_agi}")
        
        # Test blockchain config
        config_blockchain = ProtocolConfig(enable_blockchain_verification=True)
        print(f"✅ Blockchain config created: blockchain={config_blockchain.enable_blockchain_verification}")
        
        # Test multimodal config
        config_multimodal = ProtocolConfig(
            enable_multimodal_processing=True,
            enable_visual_generation=True
        )
        print(f"✅ Multimodal config created: multimodal={config_multimodal.enable_multimodal_processing}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation test failed: {e}")
        return False


def test_query_creation():
    """Test query object creation."""
    
    print("\n🔍 Test: Query Object Creation")
    print("=" * 40)
    
    try:
        from src.iceburg.protocol.models import Query
        
        # Test basic query
        query = Query(text="What is artificial intelligence?")
        print(f"✅ Basic query created: {query.text[:30]}...")
        
        # Test multimodal query
        multimodal_query = Query(
            text="Analyze this image",
            multimodal_input="test_image.jpg",
            documents=["research.pdf"],
            multimodal_evidence=["evidence1", "evidence2"]
        )
        print(f"✅ Multimodal query created: {multimodal_query.text}")
        print(f"   Documents: {len(multimodal_query.documents or [])}")
        print(f"   Evidence: {len(multimodal_query.multimodal_evidence or [])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Query creation test failed: {e}")
        return False


def test_triage():
    """Test query triage functionality."""
    
    print("\n🎯 Test: Query Triage")
    print("=" * 30)
    
    try:
        from src.iceburg.protocol.triage import classify_query
        from src.iceburg.protocol.models import Query
        from src.iceburg.protocol.config import ProtocolConfig
        
        config = ProtocolConfig(verbose=True)
        
        # Test simple query
        simple_query = Query(text="hello")
        mode = classify_query(simple_query, config)
        print(f"✅ Simple query triaged: {mode.name} (confidence: {mode.confidence})")
        
        # Test complex query
        complex_query = Query(text="research breakthrough in quantum computing")
        mode = classify_query(complex_query, config)
        print(f"✅ Complex query triaged: {mode.name} (confidence: {mode.confidence})")
        
        # Test experimental query
        experimental_query = Query(text="molecular synthesis and bioelectric integration")
        mode = classify_query(experimental_query, config)
        print(f"✅ Experimental query triaged: {mode.name} (confidence: {mode.confidence})")
        
        return True
        
    except Exception as e:
        print(f"❌ Triage test failed: {e}")
        return False


def test_planning():
    """Test task planning functionality."""
    
    print("\n📋 Test: Task Planning")
    print("=" * 30)
    
    try:
        from src.iceburg.protocol.planner import plan
        from src.iceburg.protocol.models import Query, Mode
        from src.iceburg.protocol.config import ProtocolConfig
        
        config = ProtocolConfig(verbose=True)
        query = Query(text="test query")
        mode = Mode(name="standard", confidence=1.0)
        
        tasks = plan(query, mode, config)
        print(f"✅ Task planning successful: {len(tasks)} tasks created")
        
        # Test with CIM Stack
        config_cim = ProtocolConfig(force_molecular=True, force_bioelectric=True)
        tasks_cim = plan(query, mode, config_cim)
        print(f"✅ CIM Stack planning: {len(tasks_cim)} tasks created")
        
        # Test with AGI capabilities
        config_agi = ProtocolConfig(force_agi=True)
        tasks_agi = plan(query, mode, config_agi)
        print(f"✅ AGI planning: {len(tasks_agi)} tasks created")
        
        # Test with blockchain verification
        config_blockchain = ProtocolConfig(enable_blockchain_verification=True)
        tasks_blockchain = plan(query, mode, config_blockchain)
        print(f"✅ Blockchain planning: {len(tasks_blockchain)} tasks created")
        
        # Test with multimodal processing
        config_multimodal = ProtocolConfig(enable_multimodal_processing=True, enable_visual_generation=True)
        tasks_multimodal = plan(query, mode, config_multimodal)
        print(f"✅ Multimodal planning: {len(tasks_multimodal)} tasks created")
        
        return True
        
    except Exception as e:
        print(f"❌ Planning test failed: {e}")
        return False


def test_legacy_compatibility():
    """Test legacy protocol compatibility."""
    
    print("\n🔄 Test: Legacy Compatibility")
    print("=" * 40)
    
    try:
        # Test the new protocol facade
        from src.iceburg.protocol import iceberg_protocol
        
        # Test legacy function call
        result = iceberg_protocol(
            initial_query="Test legacy compatibility",
            verbose=True,
            fast=True
        )
        
        print(f"✅ Legacy compatibility test passed: {len(result)} characters")
        return True
        
    except Exception as e:
        print(f"❌ Legacy compatibility test failed: {e}")
        return False


def main():
    """Run all tests."""
    
    print("🧪 ICEBURG Modular Protocol Test Suite (Core)")
    print("=" * 60)
    
    tests = [
        test_core_modules,
        test_config_creation,
        test_query_creation,
        test_triage,
        test_planning,
        test_legacy_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n📊 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core tests passed! The modular protocol structure is working.")
        print("\n✅ MODULAR PROTOCOL REFACTORING COMPLETE!")
        print("   - Core modules: ✅ Working")
        print("   - Configuration: ✅ Working") 
        print("   - Query handling: ✅ Working")
        print("   - Triage system: ✅ Working")
        print("   - Task planning: ✅ Working")
        print("   - Legacy compatibility: ✅ Working")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
