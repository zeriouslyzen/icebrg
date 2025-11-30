#!/usr/bin/env python3
"""
Test script for the modular ICEBURG protocol refactoring.
Verifies that all components work together correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.iceburg.protocol import run_protocol_modular
from src.iceburg.protocol.config import ProtocolConfig
from src.iceburg.protocol.models import Query


async def test_modular_protocol():
    """Test the modular protocol with various configurations."""
    
    print("🧪 Testing Modular ICEBURG Protocol")
    print("=" * 50)
    
    # Test 1: Basic functionality
    print("\n📋 Test 1: Basic Protocol Execution")
    query = Query(text="What is artificial intelligence?")
    config = ProtocolConfig(verbose=True, fast=True)
    
    try:
        result = await run_protocol_modular(query, config)
        print(f"✅ Basic test passed: {len(result.final_report_str)} characters")
        print(f"   Agent count: {len(result.structured_output.get('agents', []))}")
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False
    
    # Test 2: CIM Stack activation
    print("\n🧬 Test 2: CIM Stack Activation")
    config_cim = ProtocolConfig(
        verbose=True,
        force_molecular=True,
        force_bioelectric=True,
        force_hypothesis_testing=True
    )
    
    try:
        result_cim = await run_protocol_modular(query, config_cim)
        print(f"✅ CIM Stack test passed: {len(result_cim.final_report_str)} characters")
        audit = result_cim.structured_output.get('audit', {})
        print(f"   CIM Stack enabled: {audit.get('cim_stack_enabled', False)}")
    except Exception as e:
        print(f"❌ CIM Stack test failed: {e}")
        return False
    
    # Test 3: AGI Capabilities activation
    print("\n🤖 Test 3: AGI Capabilities Activation")
    config_agi = ProtocolConfig(
        verbose=True,
        force_agi=True
    )
    
    try:
        result_agi = await run_protocol_modular(query, config_agi)
        print(f"✅ AGI Capabilities test passed: {len(result_agi.final_report_str)} characters")
        audit = result_agi.structured_output.get('audit', {})
        print(f"   AGI Capabilities enabled: {audit.get('agi_capabilities_enabled', False)}")
    except Exception as e:
        print(f"❌ AGI Capabilities test failed: {e}")
        return False
    
    # Test 4: Blockchain verification
    print("\n🔗 Test 4: Blockchain Verification")
    config_blockchain = ProtocolConfig(
        verbose=True,
        enable_blockchain_verification=True
    )
    
    try:
        result_blockchain = await run_protocol_modular(query, config_blockchain)
        print(f"✅ Blockchain verification test passed: {len(result_blockchain.final_report_str)} characters")
        audit = result_blockchain.structured_output.get('audit', {})
        print(f"   Blockchain verification enabled: {audit.get('blockchain_verification_enabled', False)}")
    except Exception as e:
        print(f"❌ Blockchain verification test failed: {e}")
        return False
    
    # Test 5: Multimodal processing
    print("\n🎨 Test 5: Multimodal Processing")
    config_multimodal = ProtocolConfig(
        verbose=True,
        enable_multimodal_processing=True,
        enable_visual_generation=True
    )
    
    multimodal_query = Query(
        text="Create a visual interface for AI research",
        multimodal_input="test_image.jpg",
        documents=["research_paper.pdf"]
    )
    
    try:
        result_multimodal = await run_protocol_modular(multimodal_query, config_multimodal)
        print(f"✅ Multimodal processing test passed: {len(result_multimodal.final_report_str)} characters")
        audit = result_multimodal.structured_output.get('audit', {})
        print(f"   Multimodal processing enabled: {audit.get('multimodal_processing_enabled', False)}")
    except Exception as e:
        print(f"❌ Multimodal processing test failed: {e}")
        return False
    
    # Test 6: Full feature activation
    print("\n🚀 Test 6: Full Feature Activation")
    config_full = ProtocolConfig(
        verbose=True,
        force_molecular=True,
        force_bioelectric=True,
        force_hypothesis_testing=True,
        force_agi=True,
        enable_blockchain_verification=True,
        enable_multimodal_processing=True,
        enable_visual_generation=True
    )
    
    try:
        result_full = await run_protocol_modular(query, config_full)
        print(f"✅ Full feature test passed: {len(result_full.final_report_str)} characters")
        audit = result_full.structured_output.get('audit', {})
        print(f"   CIM Stack enabled: {audit.get('cim_stack_enabled', False)}")
        print(f"   AGI Capabilities enabled: {audit.get('agi_capabilities_enabled', False)}")
        print(f"   Blockchain verification enabled: {audit.get('blockchain_verification_enabled', False)}")
        print(f"   Multimodal processing enabled: {audit.get('multimodal_processing_enabled', False)}")
    except Exception as e:
        print(f"❌ Full feature test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! Modular protocol is working correctly.")
    return True


def test_legacy_compatibility():
    """Test legacy protocol compatibility."""
    
    print("\n🔄 Testing Legacy Compatibility")
    print("=" * 50)
    
    try:
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


async def main():
    """Run all tests."""
    
    print("🧪 ICEBURG Modular Protocol Test Suite")
    print("=" * 60)
    
    # Test modular protocol
    modular_success = await test_modular_protocol()
    
    # Test legacy compatibility
    legacy_success = test_legacy_compatibility()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 30)
    print(f"Modular Protocol: {'✅ PASSED' if modular_success else '❌ FAILED'}")
    print(f"Legacy Compatibility: {'✅ PASSED' if legacy_success else '❌ FAILED'}")
    
    if modular_success and legacy_success:
        print("\n🎉 All tests passed! The modular protocol refactoring is complete and working.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
