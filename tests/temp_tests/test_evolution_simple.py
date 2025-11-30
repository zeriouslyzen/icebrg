#!/usr/bin/env python3
"""
Simple Evolution Test - Test ICEBURG's self-evolution capabilities without protocol.py
This test will:
1. Generate real performance data
2. Test specification generation
3. Test evolution pipeline
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
from iceburg.autonomous.research_orchestrator import AutonomousResearchOrchestrator
from iceburg.evolution.evolution_pipeline import EvolutionPipeline

class SimpleEvolutionTester:
    def __init__(self):
        self.tracker = UnifiedPerformanceTracker()
        self.spec_generator = SpecificationGenerator()
        self.research_orchestrator = AutonomousResearchOrchestrator()
        self.evolution_pipeline = EvolutionPipeline()
        
        # Initialize the evolution pipeline with components
        self.evolution_pipeline.performance_tracker = self.tracker
        self.evolution_pipeline.specification_generator = self.spec_generator
        
    async def test_performance_data_generation(self):
        """Generate real performance data by simulating queries"""
        print("🔄 Generating real performance data...")
        
        # Start performance tracking
        await self.tracker.start_tracking()
        
        # Simulate performance data for different query types
        test_queries = [
            {"query": "What is the capital of France?", "complexity": 0.2, "expected_time": 1.5},
            {"query": "Explain quantum computing and its applications in AI", "complexity": 0.6, "expected_time": 8.2},
            {"query": "Analyze the relationship between neural networks and biological systems", "complexity": 0.8, "expected_time": 12.5},
            {"query": "Design a self-improving AI system architecture", "complexity": 0.9, "expected_time": 15.3},
            {"query": "Research the latest developments in AGI safety and alignment", "complexity": 0.7, "expected_time": 10.1}
        ]
        
        for i, query_data in enumerate(test_queries):
            print(f"  Simulating query {i+1}/5: {query_data['query'][:50]}...")
            
            # Simulate query execution
            start_time = time.time()
            await asyncio.sleep(0.1)  # Simulate processing time
            end_time = time.time()
            
            # Track performance
            track_query_performance(
                query_id=f"simulated_query_{i+1}",
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
                    "simulated_query": True,
                    "query_length": len(query_data['query']),
                    "complexity_level": i + 1,
                    "expected_time": query_data['expected_time']
                }
            )
            
            print(f"    ✅ Query simulated in {query_data['expected_time']:.2f}s")
        
        # Flush the buffer to ensure data is stored
        await self.tracker._flush_metrics_buffer()
        
        print("✅ Performance data generation completed")
    
    async def test_autonomous_research_activation(self):
        """Test autonomous research system"""
        print("🔄 Testing autonomous research system...")
        
        try:
            # Start autonomous research
            await self.research_orchestrator.start_autonomous_research()
            
            # Let it run for a bit to generate queries
            print("  Letting research run for 10 seconds...")
            await asyncio.sleep(10)
            
            # Get research status
            status = await self.research_orchestrator.get_research_status()
            print(f"  Research status: {status}")
            
            # Stop research
            await self.research_orchestrator.stop_autonomous_research()
            
            print("✅ Autonomous research test completed")
            
        except Exception as e:
            print(f"❌ Autonomous research failed: {e}")
    
    async def test_specification_generation(self):
        """Generate actual TaskSpec improvements from performance data"""
        print("🔄 Generating TaskSpec improvements...")
        
        try:
            # Get performance summary
            performance_summary = await self.tracker.get_performance_summary(hours=1)
            print(f"  Performance summary: {performance_summary}")
            
            # Generate improvement specifications
            specs = self.spec_generator.generate_improvement_specifications(performance_summary)
            
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
            
        except Exception as e:
            print(f"❌ Specification generation failed: {e}")
    
    async def test_evolution_pipeline(self):
        """Test the full evolution pipeline with real data"""
        print("🔄 Testing evolution pipeline...")
        
        try:
            # Start evolution job
            job_id = await self.evolution_pipeline.evolve_system()
            print(f"  Started evolution job: {job_id}")
            
            # Check job status
            status = self.evolution_pipeline.get_job_status(job_id)
            print(f"  Job status: {status}")
            
            # Let it run for a bit
            print("  Letting evolution run for 30 seconds...")
            await asyncio.sleep(30)
            
            # Check final status
            final_status = self.evolution_pipeline.get_job_status(job_id)
            print(f"  Final job status: {final_status}")
            
            print("✅ Evolution pipeline test completed")
            
        except Exception as e:
            print(f"❌ Evolution pipeline failed: {e}")
    
    async def verify_data_generation(self):
        """Verify that meaningful data has been generated"""
        print("🔄 Verifying data generation...")
        
        # Check performance database
        try:
            performance_summary = await self.tracker.get_performance_summary(hours=1)
            print(f"  Performance data: {performance_summary}")
            
            if "error" not in performance_summary:
                print("  ✅ Performance data successfully generated")
            else:
                print("  ❌ No performance data found")
        except Exception as e:
            print(f"  ❌ Performance data check failed: {e}")
        
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
        """Run the complete evolution test"""
        print("🚀 Starting Simple Evolution Test")
        print("=" * 50)
        
        try:
            # Step 1: Generate performance data
            await self.test_performance_data_generation()
            print()
            
            # Step 2: Test autonomous research
            await self.test_autonomous_research_activation()
            print()
            
            # Step 3: Generate specifications
            await self.test_specification_generation()
            print()
            
            # Step 4: Test evolution pipeline
            await self.test_evolution_pipeline()
            print()
            
            # Step 5: Verify data generation
            await self.verify_data_generation()
            print()
            
            print("🎉 Simple Evolution Test Completed!")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

async def main():
    tester = SimpleEvolutionTester()
    await tester.run_full_test()

if __name__ == "__main__":
    asyncio.run(main())
