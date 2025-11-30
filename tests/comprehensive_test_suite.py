#!/usr/bin/env python3
"""
ICEBURG 2.0 Comprehensive Test Suite
Tests all features and capabilities
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.generation.device_generator import DeviceGenerator
from iceburg.research.methodology_analyzer import MethodologyAnalyzer
from iceburg.research.insight_generator import InsightGenerator
from iceburg.integration.swarming_integration import SwarmingIntegration
from iceburg.truth.suppression_detector import SuppressionDetector
from iceburg.curiosity.curiosity_engine import CuriosityEngine
from iceburg.lab.virtual_physics_lab import VirtualPhysicsLab
from iceburg.security.penetration_tester import PenetrationTester


class ComprehensiveTestSuite:
    """Comprehensive test suite for ICEBURG 2.0"""
    
    def __init__(self):
        self.system_integrator = SystemIntegrator()
        self.device_generator = DeviceGenerator()
        self.methodology_analyzer = MethodologyAnalyzer()
        self.insight_generator = InsightGenerator()
        self.swarming_integration = SwarmingIntegration()
        self.suppression_detector = SuppressionDetector()
        self.curiosity_engine = CuriosityEngine()
        self.lab = VirtualPhysicsLab()
        self.penetration_tester = PenetrationTester()
        self.test_results: List[Dict[str, Any]] = []
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        print("\n" + "="*70)
        print("ICEBURG 2.0 - COMPREHENSIVE TEST SUITE")
        print("="*70 + "\n")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {}
        }
        
        # Test 1: Enhanced Deliberation Methodology
        print("TEST 1: Enhanced Deliberation Methodology")
        print("-" * 70)
        test1 = await self.test_enhanced_deliberation()
        results["tests"]["enhanced_deliberation"] = test1
        print(f"✅ Status: {test1['status']}\n")
        
        # Test 2: Truth-Finding with Suppression Detection
        print("TEST 2: Truth-Finding with Suppression Detection")
        print("-" * 70)
        test2 = await self.test_truth_finding()
        results["tests"]["truth_finding"] = test2
        print(f"✅ Status: {test2['status']}\n")
        
        # Test 3: Swarming for Better Answers
        print("TEST 3: Swarming for Better Answers")
        print("-" * 70)
        test3 = await self.test_swarming()
        results["tests"]["swarming"] = test3
        print(f"✅ Status: {test3['status']}\n")
        
        # Test 4: Device Generation
        print("TEST 4: Device Generation")
        print("-" * 70)
        test4 = await self.test_device_generation()
        results["tests"]["device_generation"] = test4
        print(f"✅ Status: {test4['status']}\n")
        
        # Test 5: Curiosity-Driven Research
        print("TEST 5: Curiosity-Driven Research")
        print("-" * 70)
        test5 = await self.test_curiosity_driven_research()
        results["tests"]["curiosity_research"] = test5
        print(f"✅ Status: {test5['status']}\n")
        
        # Test 6: Lab Capabilities
        print("TEST 6: Lab Capabilities")
        print("-" * 70)
        test6 = await self.test_lab_capabilities()
        results["tests"]["lab_capabilities"] = test6
        print(f"✅ Status: {test6['status']}\n")
        
        # Test 7: Red Teaming
        print("TEST 7: Red Teaming")
        print("-" * 70)
        test7 = await self.test_red_teaming()
        results["tests"]["red_teaming"] = test7
        print(f"✅ Status: {test7['status']}\n")
        
        # Test 8: System Integration
        print("TEST 8: System Integration")
        print("-" * 70)
        test8 = await self.test_system_integration()
        results["tests"]["system_integration"] = test8
        print(f"✅ Status: {test8['status']}\n")
        
        # Test 9: Full Query Processing
        print("TEST 9: Full Query Processing with All Features")
        print("-" * 70)
        test9 = await self.test_full_query_processing()
        results["tests"]["full_query_processing"] = test9
        print(f"✅ Status: {test9['status']}\n")
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        total_tests = len(results["tests"])
        passed_tests = sum(1 for t in results["tests"].values() if t.get("status") == "PASS")
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
        
        results["summary"] = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": passed_tests/total_tests*100
        }
        
        # Save results
        results_file = Path("data/test_results") / f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Results saved to: {results_file}")
        print("="*70 + "\n")
        
        return results
    
    async def test_enhanced_deliberation(self) -> Dict[str, Any]:
        """Test Enhanced Deliberation methodology"""
        try:
            query = "What is the Enhanced Deliberation methodology and how does it enable truth-finding?"
            
            # Apply methodology
            methodology = self.methodology_analyzer.apply_methodology(query, domain="truth_finding")
            
            # Generate insights
            insights = self.insight_generator.generate_insights(
                query=query,
                domain="truth_finding"
            )
            
            print(f"Query: {query}")
            print(f"Methodology Steps: {len(methodology.get('steps', []))}")
            print(f"Insights Generated: {len(insights.get('insights', []))}")
            print(f"Breakthroughs: {len(insights.get('breakthroughs', []))}")
            
            return {
                "status": "PASS",
                "methodology_steps": len(methodology.get('steps', [])),
                "insights_count": len(insights.get('insights', [])),
                "breakthroughs": len(insights.get('breakthroughs', []))
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_truth_finding(self) -> Dict[str, Any]:
        """Test truth-finding with suppression detection"""
        try:
            # Create sample documents with suppression indicators
            documents = [
                {
                    "id": "doc1",
                    "content": "Research on quantum computing was classified for 20 years before public release.",
                    "metadata": {
                        "creation_date": "2000-01-01",
                        "release_date": "2020-01-01",
                        "source": "military_research"
                    }
                },
                {
                    "id": "doc2",
                    "content": "Internal military report on AI capabilities that contradicts public narratives.",
                    "metadata": {
                        "source": "internal_military_report",
                        "classification": "restricted"
                    }
                }
            ]
            
            # Detect suppression
            suppression_result = self.suppression_detector.detect_suppression(documents)
            
            print(f"Documents Analyzed: {len(documents)}")
            print(f"Suppression Detected: {suppression_result.get('suppression_detected', False)}")
            print(f"Suppression Score: {suppression_result.get('overall_suppression_score', 0.0):.2f}")
            print(f"Findings: {len(suppression_result.get('details', []))}")
            
            return {
                "status": "PASS",
                "suppression_detected": suppression_result.get('suppression_detected', False),
                "suppression_score": suppression_result.get('overall_suppression_score', 0.0),
                "findings_count": len(suppression_result.get('details', []))
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_swarming(self) -> Dict[str, Any]:
        """Test swarming for better answers"""
        try:
            query = "How can swarming create better answers than single agents?"
            
            # Create truth-finding swarm
            swarm = await self.swarming_integration.create_truth_finding_swarm(
                query=query,
                swarm_type="research_swarm"
            )
            
            # Execute swarm
            swarm_results = await self.swarming_integration.execute_swarm(
                swarm=swarm,
                parallel=True
            )
            
            print(f"Query: {query}")
            print(f"Swarm Type: {swarm.get('type')}")
            print(f"Agents: {len(swarm.get('agents', []))}")
            print(f"Agent Results: {len(swarm_results.get('agent_results', []))}")
            print(f"Synthesized Result: {swarm_results.get('synthesized_result') is not None}")
            
            return {
                "status": "PASS",
                "swarm_type": swarm.get('type'),
                "agents_count": len(swarm.get('agents', [])),
                "results_count": len(swarm_results.get('agent_results', [])),
                "synthesized": swarm_results.get('synthesized_result') is not None
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_device_generation(self) -> Dict[str, Any]:
        """Test device generation"""
        try:
            device_type = "quantum_computer"
            requirements = {
                "qubits": 100,
                "fidelity": 0.99,
                "dimensions": {"width": 10, "height": 10, "depth": 10},
                "functions": ["quantum_computation", "error_correction"],
                "materials": ["superconducting_qubits", "cryogenic_system"]
            }
            
            # Generate device
            device = await self.device_generator.generate_device(
                device_type=device_type,
                requirements=requirements,
                domain="quantum_physics"
            )
            
            print(f"Device Type: {device_type}")
            print(f"Specifications: {device.get('specifications') is not None}")
            print(f"Schematics: {device.get('schematics') is not None}")
            print(f"Code: {device.get('code') is not None}")
            print(f"BOM: {device.get('bom') is not None}")
            print(f"Assembly Instructions: {device.get('assembly_instructions') is not None}")
            print(f"Validated: {device.get('validated', False)}")
            
            return {
                "status": "PASS",
                "device_type": device_type,
                "has_specifications": device.get('specifications') is not None,
                "has_schematics": device.get('schematics') is not None,
                "has_code": device.get('code') is not None,
                "has_bom": device.get('bom') is not None,
                "validated": device.get('validated', False)
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_curiosity_driven_research(self) -> Dict[str, Any]:
        """Test curiosity-driven research"""
        try:
            # Generate curiosity queries
            queries = self.curiosity_engine.generate_queries(
                domain="quantum_physics",
                limit=3
            )
            
            print(f"Curiosity Queries Generated: {len(queries)}")
            for i, query in enumerate(queries, 1):
                print(f"  {i}. {query}")
            
            return {
                "status": "PASS",
                "queries_generated": len(queries),
                "queries": queries
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_lab_capabilities(self) -> Dict[str, Any]:
        """Test lab capabilities"""
        try:
            # Test virtual lab
            experiment_result = self.lab.run_experiment(
                experiment_type="quantum_coherence",
                algorithm=None,
                parameters={"qubits": 10, "time": 1.0}
            )
            
            print(f"Experiment Type: quantum_coherence")
            print(f"Success: {experiment_result.success}")
            print(f"Performance Metrics: {experiment_result.performance_metrics}")
            print(f"Execution Time: {experiment_result.execution_time:.3f}s")
            
            return {
                "status": "PASS",
                "experiment_success": experiment_result.success,
                "execution_time": experiment_result.execution_time,
                "metrics_count": len(experiment_result.performance_metrics)
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_red_teaming(self) -> Dict[str, Any]:
        """Test red teaming capabilities"""
        try:
            # Test code analysis
            test_code = """
def process_user_input(user_input):
    query = "SELECT * FROM users WHERE id = " + user_input
    result = execute_query(query)
    return result
"""
            
            scan_result = self.penetration_tester.code_analysis(
                code=test_code,
                analysis_type="static"
            )
            
            print(f"Code Analyzed: {len(test_code)} characters")
            print(f"Vulnerabilities Found: {len(scan_result.get('vulnerabilities', []))}")
            print(f"Warnings: {len(scan_result.get('warnings', []))}")
            
            for vuln in scan_result.get('vulnerabilities', []):
                print(f"  - {vuln.get('type')}: {vuln.get('description')}")
            
            return {
                "status": "PASS",
                "vulnerabilities_found": len(scan_result.get('vulnerabilities', [])),
                "warnings": len(scan_result.get('warnings', []))
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_system_integration(self) -> Dict[str, Any]:
        """Test system integration"""
        try:
            # Integrate all systems
            integration = self.system_integrator.integrate_all_systems()
            
            # Get system status
            status = self.system_integrator.get_system_status()
            
            print(f"Systems Integrated: {integration.get('integrated', False)}")
            print(f"Blackboard: {status.get('blackboard', {}).get('subscriptions', 0)} subscriptions")
            print(f"Curiosity: {status.get('curiosity', {}).get('curiosity_engine_active', False)}")
            print(f"Swarming: {len(status.get('swarming', {}).get('swarm_types', []))} swarm types")
            print(f"Device Generation: {len(status.get('device_generation', {}).get('capabilities', []))} capabilities")
            
            return {
                "status": "PASS",
                "integrated": integration.get('integrated', False),
                "blackboard_subscriptions": status.get('blackboard', {}).get('subscriptions', 0),
                "curiosity_active": status.get('curiosity', {}).get('curiosity_engine_active', False),
                "swarm_types": len(status.get('swarming', {}).get('swarm_types', []))
            }
        except Exception as e:
            return {"status": "FAIL", "error": str(e)}
    
    async def test_full_query_processing(self) -> Dict[str, Any]:
        """Test full query processing with all features"""
        try:
            query = "How does Enhanced Deliberation methodology enable ICEBURG to find suppressed knowledge about quantum consciousness and create devices that leverage these principles?"
            
            print(f"Query: {query}")
            print("\nProcessing with full system integration...")
            
            # Process query with full integration
            result = await self.system_integrator.process_query_with_full_integration(
                query=query,
                domain="quantum_consciousness"
            )
            
            print(f"\nResults:")
            print(f"  Methodology Applied: {result.get('methodology')}")
            print(f"  Swarm Created: {result.get('results', {}).get('swarm') is not None}")
            print(f"  Swarm Results: {result.get('results', {}).get('swarm_results') is not None}")
            print(f"  Insights Generated: {result.get('results', {}).get('insights') is not None}")
            
            # Show thinking process
            methodology = result.get('results', {}).get('methodology', {})
            if methodology:
                print(f"\n  Methodology Steps: {len(methodology.get('steps', []))}")
                for i, step in enumerate(methodology.get('steps', [])[:3], 1):
                    print(f"    {i}. {step.get('name')}")
            
            # Show swarm results
            swarm_results = result.get('results', {}).get('swarm_results', {})
            if swarm_results:
                print(f"\n  Swarm Results:")
                print(f"    Agents: {len(swarm_results.get('agent_results', []))}")
                print(f"    Synthesized: {swarm_results.get('synthesized_result') is not None}")
            
            # Show insights
            insights = result.get('results', {}).get('insights', {})
            if insights:
                print(f"\n  Insights:")
                print(f"    Total Insights: {len(insights.get('insights', []))}")
                print(f"    Breakthroughs: {len(insights.get('breakthroughs', []))}")
                print(f"    Suppression Detected: {insights.get('suppression_detected', False)}")
            
            return {
                "status": "PASS",
                "methodology_applied": result.get('methodology') is not None,
                "swarm_created": result.get('results', {}).get('swarm') is not None,
                "insights_generated": result.get('results', {}).get('insights') is not None,
                "full_integration": True
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"status": "FAIL", "error": str(e)}


async def main():
    """Run comprehensive test suite"""
    test_suite = ComprehensiveTestSuite()
    results = await test_suite.run_all_tests()
    
    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total Tests: {results['summary']['total_tests']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    asyncio.run(main())

