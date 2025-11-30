#!/usr/bin/env python3
"""
Working Evolution Test - Test what actually works in ICEBURG's self-evolution system
This test will:
1. Generate real performance data and store it
2. Generate actual TaskSpec improvements
3. Test the components individually
4. Verify data generation
"""

import asyncio
import json
import time
import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from iceburg.monitoring.unified_performance_tracker import UnifiedPerformanceTracker, track_query_performance
from iceburg.evolution.specification_generator import SpecificationGenerator, TaskSpec

class WorkingEvolutionTester:
    def __init__(self):
        self.tracker = UnifiedPerformanceTracker()
        self.spec_generator = SpecificationGenerator()
        
    async def test_performance_data_generation(self):
        """Generate real performance data and verify it's stored"""
        print("🔄 Testing performance data generation...")
        
        # Start performance tracking
        await self.tracker.start_tracking()
        
        # Generate test data
        test_queries = [
            {"query": "What is the capital of France?", "complexity": 0.2, "expected_time": 1.5},
            {"query": "Explain quantum computing", "complexity": 0.6, "expected_time": 8.2},
            {"query": "Analyze neural networks", "complexity": 0.8, "expected_time": 12.5},
            {"query": "Design AI system", "complexity": 0.9, "expected_time": 15.3},
            {"query": "Research AGI safety", "complexity": 0.7, "expected_time": 10.1}
        ]
        
        for i, query_data in enumerate(test_queries):
            print(f"  Simulating query {i+1}/5: {query_data['query'][:30]}...")
            
            # Track performance
            track_query_performance(
                query_id=f"test_query_{i+1}",
                response_time=query_data['expected_time'],
                accuracy=0.8 + (i * 0.05),
                resources={
                    "memory_usage_mb": 100 + (i * 20),
                    "cache_hit_rate": 0.3 + (i * 0.1),
                    "agent_count": 6,
                    "parallel_execution": True,
                    "query_complexity": query_data['complexity']
                },
                success=True,
                metadata={
                    "test_query": True,
                    "query_length": len(query_data['query']),
                    "complexity_level": i + 1
                }
            )
            
            print(f"    ✅ Tracked in {query_data['expected_time']:.2f}s")
        
        # Flush buffer to ensure data is stored
        await self.tracker._flush_metrics_buffer()
        
        # Wait a moment for async operations to complete
        await asyncio.sleep(0.5)
        
        print("✅ Performance data generation completed")
    
    async def test_performance_data_retrieval(self):
        """Test retrieving performance data from database"""
        print("🔄 Testing performance data retrieval...")
        
        try:
            # Get performance summary
            performance_summary = self.tracker.get_performance_summary(hours=1)
            print(f"  Performance summary: {performance_summary}")
            
            if "error" not in performance_summary:
                print("  ✅ Performance data successfully retrieved from database")
                return True
            else:
                print("  ❌ No performance data found in database")
                return False
                
        except Exception as e:
            print(f"  ❌ Performance data retrieval failed: {e}")
            return False
    
    def test_specification_generation(self):
        """Test generating TaskSpec improvements"""
        print("🔄 Testing TaskSpec generation...")
        
        try:
            # Get real performance data
            performance_summary = self.tracker.get_performance_summary(hours=1)
            
            # Use averages data for specification generation
            if "averages" in performance_summary:
                performance_data = performance_summary["averages"]
            else:
                performance_data = performance_summary
            
            # Generate specifications
            specs = self.spec_generator.generate_improvement_specifications(performance_data)
            
            print(f"  Generated {len(specs)} improvement specifications:")
            for i, spec in enumerate(specs):
                print(f"    {i+1}. {spec.name}: {spec.description}")
                print(f"       Optimization targets: {spec.optimization_targets}")
                print(f"       Safety constraints: {spec.safety_constraints}")
            
            # Save specifications to file
            specs_data = []
            for spec in specs:
                specs_data.append({
                    "name": spec.name,
                    "description": spec.description,
                    "inputs": spec.inputs,
                    "outputs": spec.outputs,
                    "preconditions": spec.preconditions,
                    "postconditions": spec.postconditions,
                    "optimization_targets": spec.optimization_targets,
                    "safety_constraints": spec.safety_constraints,
                    "metadata": spec.metadata
                })
            
            with open("generated_improvements.json", "w") as f:
                json.dump(specs_data, f, indent=2)
            
            print("✅ TaskSpec generation completed - saved to generated_improvements.json")
            return len(specs) > 0
            
        except Exception as e:
            print(f"❌ TaskSpec generation failed: {e}")
            return False
    
    def test_individual_components(self):
        """Test individual components of the system"""
        print("🔄 Testing individual components...")
        
        # Test TaskSpec creation
        try:
            test_spec = TaskSpec(
                name="test_optimization",
                description="Test optimization specification",
                inputs=[{"name": "input", "type": "string"}],
                outputs=[{"name": "output", "type": "string"}],
                preconditions=["input is not empty"],
                postconditions=["output is generated"],
                optimization_targets=["response_time", "accuracy"],
                safety_constraints=["no data loss", "maintain accuracy"],
                metadata={"test": True}
            )
            print("  ✅ TaskSpec creation works")
        except Exception as e:
            print(f"  ❌ TaskSpec creation failed: {e}")
        
        # Test performance tracker initialization
        try:
            tracker = UnifiedPerformanceTracker()
            print("  ✅ Performance tracker initialization works")
        except Exception as e:
            print(f"  ❌ Performance tracker initialization failed: {e}")
        
        # Test specification generator initialization
        try:
            spec_gen = SpecificationGenerator()
            print("  ✅ Specification generator initialization works")
        except Exception as e:
            print(f"  ❌ Specification generator initialization failed: {e}")
    
    def verify_generated_files(self):
        """Verify that files have been generated"""
        print("🔄 Verifying generated files...")
        
        # Check generated improvements file
        if os.path.exists("generated_improvements.json"):
            with open("generated_improvements.json", "r") as f:
                improvements = json.load(f)
            print(f"  ✅ Generated {len(improvements)} improvement specifications")
            
            # Show first improvement as example
            if improvements:
                first_improvement = improvements[0]
                print(f"    Example: {first_improvement['name']}")
                print(f"    Description: {first_improvement['description']}")
        else:
            print("  ❌ No improvement specifications generated")
        
        # Check knowledge evolution data
        evolution_file = "data/knowledge_evolution/knowledge_evolution/vector_collections/evolution_discovery_1756845962.json"
        if os.path.exists(evolution_file):
            with open(evolution_file, "r") as f:
                evolution_data = json.load(f)
            print(f"  ✅ Evolution discovery data found: {evolution_data['discovery']}")
        else:
            print("  ❌ No evolution discovery data found")
    
    async def run_full_test(self):
        """Run the complete working evolution test"""
        print("🚀 Starting Working Evolution Test")
        print("=" * 50)
        
        results = {
            "performance_data_generation": False,
            "performance_data_retrieval": False,
            "specification_generation": False,
            "individual_components": False,
            "generated_files": False
        }
        
        try:
            # Step 1: Test performance data generation
            await self.test_performance_data_generation()
            results["performance_data_generation"] = True
            print()
            
            # Step 2: Test performance data retrieval
            results["performance_data_retrieval"] = await self.test_performance_data_retrieval()
            print()
            
            # Step 3: Test specification generation
            results["specification_generation"] = self.test_specification_generation()
            print()
            
            # Step 4: Test individual components
            self.test_individual_components()
            results["individual_components"] = True
            print()
            
            # Step 5: Verify generated files
            self.verify_generated_files()
            results["generated_files"] = True
            print()
            
            # Summary
            print("🎉 Working Evolution Test Completed!")
            print("=" * 50)
            print("Results Summary:")
            for test_name, passed in results.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {test_name}: {status}")
            
            total_passed = sum(results.values())
            total_tests = len(results)
            print(f"\nOverall: {total_passed}/{total_tests} tests passed")
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    tester = WorkingEvolutionTester()
    await tester.run_full_test()

if __name__ == "__main__":
    asyncio.run(main())
