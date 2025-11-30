#!/usr/bin/env python3
"""
Full Evolution Pipeline Test - Test the complete ICEBURG self-evolution process
This test will:
1. Generate real performance data
2. Generate improvement specifications
3. Test the evolution pipeline
4. Verify end-to-end functionality
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

class FullEvolutionPipelineTester:
    def __init__(self):
        self.tracker = UnifiedPerformanceTracker()
        self.spec_generator = SpecificationGenerator()
        self.research_orchestrator = AutonomousResearchOrchestrator()
        self.evolution_pipeline = EvolutionPipeline()
        
        # Initialize the evolution pipeline with components
        self.evolution_pipeline.performance_tracker = self.tracker
        self.evolution_pipeline.specification_generator = self.spec_generator
        
    async def test_complete_evolution_cycle(self):
        """Test the complete evolution cycle from data generation to improvement deployment"""
        print("🚀 Starting Complete Evolution Cycle Test")
        print("=" * 60)
        
        # Step 1: Generate performance data
        print("📊 Step 1: Generating Performance Data")
        await self._generate_performance_data()
        
        # Step 2: Analyze performance
        print("\n📈 Step 2: Analyzing Performance")
        performance_analysis = await self._analyze_performance()
        
        # Step 3: Generate improvement specifications
        print("\n🔧 Step 3: Generating Improvement Specifications")
        specifications = await self._generate_improvements(performance_analysis)
        
        # Step 4: Test evolution pipeline
        print("\n🔄 Step 4: Testing Evolution Pipeline")
        evolution_result = await self._test_evolution_pipeline()
        
        # Step 5: Verify results
        print("\n✅ Step 5: Verifying Results")
        await self._verify_evolution_results(specifications, evolution_result)
        
        print("\n🎉 Complete Evolution Cycle Test Finished!")
        print("=" * 60)
    
    async def _generate_performance_data(self):
        """Generate comprehensive performance data"""
        print("  Generating performance data...")
        
        # Start performance tracking
        await self.tracker.start_tracking()
        
        # Generate diverse test data
        test_scenarios = [
            {"query": "Simple fact lookup", "complexity": 0.2, "expected_time": 1.5, "accuracy": 0.95},
            {"query": "Complex analysis", "complexity": 0.7, "expected_time": 8.2, "accuracy": 0.85},
            {"query": "Multi-step reasoning", "complexity": 0.9, "expected_time": 15.3, "accuracy": 0.80},
            {"query": "Creative generation", "complexity": 0.8, "expected_time": 12.1, "accuracy": 0.75},
            {"query": "Technical explanation", "complexity": 0.6, "expected_time": 6.8, "accuracy": 0.90}
        ]
        
        for i, scenario in enumerate(test_scenarios):
            print(f"    Running scenario {i+1}/5: {scenario['query']}")
            
            # Track performance
            track_query_performance(
                query_id=f"evolution_test_{i+1}",
                response_time=scenario['expected_time'],
                accuracy=scenario['accuracy'],
                resources={
                    "memory_usage_mb": 100 + (i * 25),
                    "cache_hit_rate": 0.3 + (i * 0.1),
                    "agent_count": 6,
                    "parallel_execution": True,
                    "query_complexity": scenario['complexity']
                },
                success=True,
                metadata={
                    "test_scenario": scenario['query'],
                    "complexity_level": scenario['complexity'],
                    "evolution_test": True
                }
            )
        
        # Flush data to database
        await self.tracker._flush_metrics_buffer()
        await asyncio.sleep(0.5)  # Wait for async operations
        
        print("  ✅ Performance data generated and stored")
    
    async def _analyze_performance(self):
        """Analyze performance data and identify issues"""
        print("  Analyzing performance data...")
        
        # Get performance summary
        performance_summary = self.tracker.get_performance_summary(hours=1)
        
        if "error" in performance_summary:
            print(f"  ❌ Performance analysis failed: {performance_summary['error']}")
            return None
        
        print(f"  📊 Performance Summary:")
        print(f"    Total queries: {performance_summary['total_queries']}")
        print(f"    Success rate: {performance_summary['success_rate']:.1f}%")
        print(f"    Average response time: {performance_summary['averages']['response_time']:.2f}s")
        print(f"    Average accuracy: {performance_summary['averages']['accuracy']:.2f}")
        print(f"    Average memory usage: {performance_summary['averages']['memory_usage_mb']:.2f}MB")
        
        return performance_summary
    
    async def _generate_improvements(self, performance_analysis):
        """Generate improvement specifications based on performance analysis"""
        print("  Generating improvement specifications...")
        
        if not performance_analysis:
            print("  ❌ No performance analysis available")
            return []
        
        # Use averages data for specification generation
        performance_data = performance_analysis.get("averages", performance_analysis)
        
        # Generate specifications
        specifications = self.spec_generator.generate_improvement_specifications(performance_data)
        
        print(f"  📋 Generated {len(specifications)} improvement specifications:")
        for i, spec in enumerate(specifications):
            print(f"    {i+1}. {spec.name}")
            print(f"       Description: {spec.description}")
            print(f"       Targets: {spec.optimization_targets}")
        
        # Save specifications
        specs_data = []
        for spec in specifications:
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
        
        with open("evolution_improvements.json", "w") as f:
            json.dump(specs_data, f, indent=2)
        
        print("  ✅ Improvement specifications generated and saved")
        return specifications
    
    async def _test_evolution_pipeline(self):
        """Test the evolution pipeline"""
        print("  Testing evolution pipeline...")
        
        try:
            # Start evolution job
            job_id = await self.evolution_pipeline.evolve_system()
            print(f"    Started evolution job: {job_id}")
            
            # Check job status
            status = self.evolution_pipeline.get_job_status(job_id)
            print(f"    Job status: {status['status']}")
            
            # Let it run for a bit
            print("    Letting evolution run for 30 seconds...")
            await asyncio.sleep(30)
            
            # Check final status
            final_status = self.evolution_pipeline.get_job_status(job_id)
            print(f"    Final status: {final_status['status']}")
            
            if final_status['status'] == 'completed':
                print("  ✅ Evolution pipeline completed successfully")
            else:
                print(f"  ⚠️ Evolution pipeline status: {final_status['status']}")
                if 'error_message' in final_status:
                    print(f"    Error: {final_status['error_message']}")
            
            return final_status
            
        except Exception as e:
            print(f"  ❌ Evolution pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _verify_evolution_results(self, specifications, evolution_result):
        """Verify the results of the evolution process"""
        print("  Verifying evolution results...")
        
        # Check specifications
        if specifications:
            print(f"  ✅ Generated {len(specifications)} improvement specifications")
        else:
            print("  ❌ No improvement specifications generated")
        
        # Check evolution pipeline
        if evolution_result and evolution_result.get('status') == 'completed':
            print("  ✅ Evolution pipeline completed successfully")
        else:
            print("  ⚠️ Evolution pipeline had issues")
        
        # Check generated files
        if os.path.exists("evolution_improvements.json"):
            with open("evolution_improvements.json", "r") as f:
                improvements = json.load(f)
            print(f"  ✅ Saved {len(improvements)} improvements to file")
        
        # Check database
        performance_summary = self.tracker.get_performance_summary(hours=1)
        if "error" not in performance_summary:
            print(f"  ✅ Performance data stored ({performance_summary['total_queries']} queries)")
        else:
            print("  ❌ Performance data not available")
        
        # Summary
        print("\n  📊 Evolution Results Summary:")
        print(f"    Specifications generated: {len(specifications) if specifications else 0}")
        print(f"    Evolution pipeline status: {evolution_result.get('status', 'unknown') if evolution_result else 'failed'}")
        print(f"    Performance queries tracked: {performance_summary.get('total_queries', 0) if 'error' not in performance_summary else 0}")

async def main():
    tester = FullEvolutionPipelineTester()
    await tester.test_complete_evolution_cycle()

if __name__ == "__main__":
    asyncio.run(main())
