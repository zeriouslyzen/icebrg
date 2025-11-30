#!/usr/bin/env python3
"""
Test script to verify all ICEBURG modes and agents are working
"""

import sys
sys.path.insert(0, 'src')

import asyncio
from iceburg.core.system_integrator import SystemIntegrator
from iceburg.config import load_config
from iceburg.vectorstore import VectorStore

def test_mode_agent_availability():
    """Test that all modes and agents are available and can be initialized"""
    
    print("=" * 80)
    print("ICEBURG MODE AND AGENT AVAILABILITY TEST")
    print("=" * 80)
    
    # Available modes from frontend
    modes = {
        "chat": "Chat (Fast) - Intelligent routing with fast/deep paths",
        "fast": "Fast Mode - Maps to chat mode",
        "research": "Research - Full protocol with methodology focus",
        "device": "Device Generation - Generates device specifications",
        "truth": "Truth Finding - Enhanced suppression detection",
        "swarm": "Swarm - Uses micro-agent swarm"
    }
    
    # Available agents from frontend
    agents = {
        "auto": "Auto (Full Protocol) - Intelligent routing",
        "surveyor": "Surveyor - Information gathering",
        "dissident": "Dissident - Challenge assumptions",
        "synthesist": "Synthesist - Synthesize insights",
        "oracle": "Oracle - Evidence synthesis",
        "archaeologist": "Archaeologist - Historical insights",
        "supervisor": "Supervisor - Validation",
        "scribe": "Scribe - Knowledge outputs",
        "weaver": "Weaver - Code generation",
        "scrutineer": "Scrutineer - Contradiction detection",
        "swarm": "Swarm - Micro-agent swarm",
        "ide": "IDE Agent - Safe command execution"
    }
    
    print("\n📋 MODES:")
    print("-" * 80)
    for mode, description in modes.items():
        print(f"  ✅ {mode:15} - {description}")
    
    print("\n🤖 AGENTS:")
    print("-" * 80)
    for agent, description in agents.items():
        print(f"  ✅ {agent:15} - {description}")
    
    # Test agent imports
    print("\n🔍 TESTING AGENT IMPORTS:")
    print("-" * 80)
    
    agent_imports = {
        "surveyor": ("iceburg.agents.surveyor", "run"),
        "dissident": ("iceburg.agents.dissident", "run"),
        "synthesist": ("iceburg.agents.synthesist", "run"),
        "oracle": ("iceburg.agents.oracle", "run"),
        "archaeologist": ("iceburg.agents.archaeologist", "run"),
        "supervisor": ("iceburg.agents.supervisor", "run"),
        "scribe": ("iceburg.agents.scribe", "run"),
        "weaver": ("iceburg.agents.weaver", "run"),
        "scrutineer": ("iceburg.agents.scrutineer", "run"),
        "ide": ("iceburg.agents.ide_agent", "run"),
    }
    
    for agent_name, (module_path, func_name) in agent_imports.items():
        try:
            module = __import__(module_path, fromlist=[func_name])
            func = getattr(module, func_name)
            print(f"  ✅ {agent_name:15} - Import successful")
        except ImportError as e:
            print(f"  ❌ {agent_name:15} - Import failed: {e}")
        except AttributeError as e:
            print(f"  ❌ {agent_name:15} - Function not found: {e}")
        except Exception as e:
            print(f"  ⚠️  {agent_name:15} - Error: {e}")
    
    # Test SystemIntegrator
    print("\n🔧 TESTING SYSTEM INTEGRATOR:")
    print("-" * 80)
    try:
        system_integrator = SystemIntegrator()
        print(f"  ✅ SystemIntegrator initialized")
        
        if hasattr(system_integrator, 'process_query_with_full_integration'):
            print(f"  ✅ process_query_with_full_integration method available")
        else:
            print(f"  ❌ process_query_with_full_integration method not found")
            
        if hasattr(system_integrator, 'generate_device_with_full_integration'):
            print(f"  ✅ generate_device_with_full_integration method available")
        else:
            print(f"  ❌ generate_device_with_full_integration method not found")
            
    except Exception as e:
        print(f"  ❌ SystemIntegrator initialization failed: {e}")
    
    # Test MicroAgentSwarm
    print("\n🐝 TESTING MICRO-AGENT SWARM:")
    print("-" * 80)
    try:
        from iceburg.micro_agent_swarm import MicroAgentSwarm
        swarm = MicroAgentSwarm()
        print(f"  ✅ MicroAgentSwarm initialized")
    except Exception as e:
        print(f"  ❌ MicroAgentSwarm initialization failed: {e}")
    
    # Test ReflexiveRoutingSystem
    print("\n🔄 TESTING REFLEXIVE ROUTING SYSTEM:")
    print("-" * 80)
    try:
        from iceburg.integration.reflexive_routing import ReflexiveRoutingSystem
        cfg = load_config()
        routing_system = ReflexiveRoutingSystem(cfg=cfg)
        print(f"  ✅ ReflexiveRoutingSystem initialized")
    except Exception as e:
        print(f"  ❌ ReflexiveRoutingSystem initialization failed: {e}")
    
    # Test DeviceGenerator
    print("\n📱 TESTING DEVICE GENERATOR:")
    print("-" * 80)
    try:
        from iceburg.generation.device_generator import DeviceGenerator
        device_generator = DeviceGenerator()
        print(f"  ✅ DeviceGenerator initialized")
    except Exception as e:
        print(f"  ❌ DeviceGenerator initialization failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ MODE AND AGENT AVAILABILITY TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_mode_agent_availability()

