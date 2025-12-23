#!/usr/bin/env python3
"""
ICEBURG Elite Financial AI Benchmark Runner

Comprehensive benchmark runner for quantum, RL, and hybrid systems.
"""

import sys
import os
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any

# Add benchmarks directory to path
benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmarks_dir)

from quantum_performance import QuantumPerformanceBenchmark
from rl_performance import RLPerformanceBenchmark
from hybrid_performance import HybridPerformanceBenchmark


class BenchmarkRunner:
    """
    Comprehensive benchmark runner for ICEBURG Elite Financial AI.
    
    Executes quantum, RL, and hybrid performance benchmarks
    and generates comprehensive reports.
    """
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize benchmark runner."""
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        self.results = {
            "quantum": [],
            "rl": [],
            "hybrid": []
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def run_quantum_benchmarks(self) -> List[Dict[str, Any]]:
        """Run quantum performance benchmarks."""
        print("🔬 Running Quantum Performance Benchmarks")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Create quantum benchmark instance
            quantum_benchmark = QuantumPerformanceBenchmark(self.output_dir)
            
            # Run all quantum benchmarks
            results = quantum_benchmark.run_all_benchmarks()
            
            # Store results
            self.results["quantum"] = results
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Quantum benchmarks completed in {duration:.2f}s")
            print(f"📊 Results: {len(results)} benchmarks")
            
            return results
        
        except Exception as e:
            print(f"❌ Error running quantum benchmarks: {e}")
            return []
    
    def run_rl_benchmarks(self) -> List[Dict[str, Any]]:
        """Run RL performance benchmarks."""
        print("\n🤖 Running RL Performance Benchmarks")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Create RL benchmark instance
            rl_benchmark = RLPerformanceBenchmark(self.output_dir)
            
            # Run all RL benchmarks
            results = rl_benchmark.run_all_benchmarks()
            
            # Store results
            self.results["rl"] = results
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ RL benchmarks completed in {duration:.2f}s")
            print(f"📊 Results: {len(results)} benchmarks")
            
            return results
        
        except Exception as e:
            print(f"❌ Error running RL benchmarks: {e}")
            return []
    
    def run_hybrid_benchmarks(self) -> List[Dict[str, Any]]:
        """Run hybrid performance benchmarks."""
        print("\n🔗 Running Hybrid Performance Benchmarks")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Create hybrid benchmark instance
            hybrid_benchmark = HybridPerformanceBenchmark(self.output_dir)
            
            # Run all hybrid benchmarks
            results = hybrid_benchmark.run_all_benchmarks()
            
            # Store results
            self.results["hybrid"] = results
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Hybrid benchmarks completed in {duration:.2f}s")
            print(f"📊 Results: {len(results)} benchmarks")
            
            return results
        
        except Exception as e:
            print(f"❌ Error running hybrid benchmarks: {e}")
            return []
    
    def run_all_benchmarks(self) -> Dict[str, List[Dict[str, Any]]]:
        """Run all performance benchmarks."""
        print("🚀 ICEBURG Elite Financial AI Performance Benchmarking")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Run quantum benchmarks
        quantum_results = self.run_quantum_benchmarks()
        
        # Run RL benchmarks
        rl_results = self.run_rl_benchmarks()
        
        # Run hybrid benchmarks
        hybrid_results = self.run_hybrid_benchmarks()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        self._generate_comprehensive_report()
        
        return self.results
    
    def run_specific_benchmarks(self, benchmark_types: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Run specific benchmark types."""
        print(f"🚀 Running Specific Benchmarks: {', '.join(benchmark_types)}")
        print("=" * 60)
        
        self.start_time = time.time()
        
        results = {}
        
        if "quantum" in benchmark_types:
            results["quantum"] = self.run_quantum_benchmarks()
        
        if "rl" in benchmark_types:
            results["rl"] = self.run_rl_benchmarks()
        
        if "hybrid" in benchmark_types:
            results["hybrid"] = self.run_hybrid_benchmarks()
        
        self.end_time = time.time()
        
        # Generate report for specific benchmarks
        self._generate_specific_report(benchmark_types, results)
        
        return results
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive benchmark report."""
        print("\n📊 Generating Comprehensive Benchmark Report")
        print("=" * 60)
        
        try:
            # Calculate total duration
            total_duration = self.end_time - self.start_time
            
            # Calculate summary statistics
            total_benchmarks = sum(len(results) for results in self.results.values())
            
            # Create comprehensive report
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "total_benchmarks": total_benchmarks,
                "benchmark_types": list(self.results.keys()),
                "quantum_benchmarks": len(self.results["quantum"]),
                "rl_benchmarks": len(self.results["rl"]),
                "hybrid_benchmarks": len(self.results["hybrid"]),
                "results": self.results
            }
            
            # Save comprehensive report
            report_file = os.path.join(self.output_dir, "comprehensive_benchmark_report.json")
            with open(report_file, 'w') as f:
                import json
                json.dump(report, f, indent=2)
            
            print(f"✅ Comprehensive report saved to {report_file}")
            
            # Print summary
            print(f"\n📈 Benchmark Summary:")
            print(f"  Total Duration: {total_duration:.2f}s")
            print(f"  Total Benchmarks: {total_benchmarks}")
            print(f"  Quantum Benchmarks: {len(self.results['quantum'])}")
            print(f"  RL Benchmarks: {len(self.results['rl'])}")
            print(f"  Hybrid Benchmarks: {len(self.results['hybrid'])}")
            
        except Exception as e:
            print(f"⚠️  Error generating comprehensive report: {e}")
    
    def _generate_specific_report(self, benchmark_types: List[str], results: Dict[str, List[Dict[str, Any]]]):
        """Generate specific benchmark report."""
        print(f"\n📊 Generating Specific Benchmark Report for {', '.join(benchmark_types)}")
        print("=" * 60)
        
        try:
            # Calculate total duration
            total_duration = self.end_time - self.start_time
            
            # Calculate summary statistics
            total_benchmarks = sum(len(results) for results in results.values())
            
            # Create specific report
            report = {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "total_benchmarks": total_benchmarks,
                "benchmark_types": benchmark_types,
                "results": results
            }
            
            # Save specific report
            report_file = os.path.join(self.output_dir, f"specific_benchmark_report_{'_'.join(benchmark_types)}.json")
            with open(report_file, 'w') as f:
                import json
                json.dump(report, f, indent=2)
            
            print(f"✅ Specific report saved to {report_file}")
            
            # Print summary
            print(f"\n📈 Benchmark Summary:")
            print(f"  Total Duration: {total_duration:.2f}s")
            print(f"  Total Benchmarks: {total_benchmarks}")
            for benchmark_type in benchmark_types:
                print(f"  {benchmark_type.upper()} Benchmarks: {len(results.get(benchmark_type, []))}")
            
        except Exception as e:
            print(f"⚠️  Error generating specific report: {e}")
    
    def get_benchmark_status(self) -> Dict[str, Any]:
        """Get current benchmark status."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.end_time - self.start_time if self.end_time else None,
            "quantum_benchmarks": len(self.results["quantum"]),
            "rl_benchmarks": len(self.results["rl"]),
            "hybrid_benchmarks": len(self.results["hybrid"]),
            "total_benchmarks": sum(len(results) for results in self.results.values())
        }


def main():
    """Main benchmark runner entry point."""
    parser = argparse.ArgumentParser(description="ICEBURG Elite Financial AI Benchmark Runner")
    parser.add_argument("--type", choices=["quantum", "rl", "hybrid", "all"], 
                       default="all", help="Benchmark type to run")
    parser.add_argument("--output", default="benchmarks/results", 
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Create benchmark runner
    runner = BenchmarkRunner(args.output)
    
    try:
        if args.type == "all":
            # Run all benchmarks
            results = runner.run_all_benchmarks()
        else:
            # Run specific benchmark type
            results = runner.run_specific_benchmarks([args.type])
        
        # Print final status
        status = runner.get_benchmark_status()
        print(f"\n✅ Benchmarking Complete!")
        print(f"📊 Total Benchmarks: {status['total_benchmarks']}")
        print(f"⏱️  Total Duration: {status['duration']:.2f}s")
        print(f"📁 Results saved to: {args.output}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Benchmark runner error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
