#!/usr/bin/env python3
"""
Comprehensive Benchmark Runner
Runs all ICEBURG benchmarks including ARC-AGI
"""

import sys
import os
import asyncio
import argparse
from datetime import datetime
from pathlib import Path

# Add benchmarks directory to path
benchmarks_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, benchmarks_dir)

from run_benchmarks import BenchmarkRunner
from arc_agi_benchmark import ARCAGIBenchmark


class ComprehensiveBenchmarkRunner:
    """
    Runs all ICEBURG benchmarks:
    - Quantum Performance
    - RL Performance
    - Hybrid Performance
    - ARC-AGI Benchmark
    """
    
    def __init__(self, output_dir: str = "benchmarks/results"):
        """Initialize comprehensive benchmark runner."""
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        
    async def run_all(self, benchmark_types: list = None, verbose: bool = True):
        """Run all benchmarks."""
        
        if benchmark_types is None:
            benchmark_types = ["quantum", "rl", "hybrid", "arc_agi"]
        
        print("="*80)
        print("ICEBURG COMPREHENSIVE BENCHMARK SUITE")
        print("="*80)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Benchmark Types: {', '.join(benchmark_types)}")
        print()
        
        self.start_time = datetime.now()
        results = {}
        
        # Run standard benchmarks (quantum, RL, hybrid)
        if any(t in benchmark_types for t in ["quantum", "rl", "hybrid"]):
            print("\n" + "="*80)
            print("STANDARD PERFORMANCE BENCHMARKS")
            print("="*80)
            
            runner = BenchmarkRunner(output_dir=self.output_dir)
            
            standard_types = [t for t in benchmark_types if t in ["quantum", "rl", "hybrid"]]
            
            if standard_types:
                standard_results = runner.run_specific_benchmarks(standard_types)
                results.update(standard_results)
        
        # Run ARC-AGI benchmark
        if "arc_agi" in benchmark_types:
            print("\n" + "="*80)
            print("ARC-AGI BENCHMARK")
            print("="*80)
            
            arc_benchmark = ARCAGIBenchmark(output_dir=self.output_dir)
            arc_results = await arc_benchmark.run_all_tasks(verbose=verbose)
            results["arc_agi"] = arc_results
        
        self.end_time = datetime.now()
        
        # Generate comprehensive report
        self._generate_comprehensive_report(results, benchmark_types)
        
        return results
    
    def _generate_comprehensive_report(self, results: dict, benchmark_types: list):
        """Generate comprehensive benchmark report."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE BENCHMARK REPORT")
        print("="*80)
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        print(f"\nTotal Duration: {total_duration:.2f}s")
        print(f"Benchmark Types Run: {len(benchmark_types)}")
        
        # Standard benchmarks summary
        if any(t in benchmark_types for t in ["quantum", "rl", "hybrid"]):
            print("\nStandard Benchmarks:")
            for bench_type in ["quantum", "rl", "hybrid"]:
                if bench_type in results:
                    count = len(results[bench_type]) if isinstance(results[bench_type], list) else 0
                    print(f"  {bench_type.upper()}: {count} benchmarks")
        
        # ARC-AGI summary
        if "arc_agi" in results:
            arc_summary = results["arc_agi"].get("summary", {})
            print("\nARC-AGI Benchmark:")
            print(f"  Total Tasks: {arc_summary.get('total_tasks', 0)}")
            print(f"  Completed: {arc_summary.get('completed_tasks', 0)}")
            print(f"  Success Rate: {arc_summary.get('success_rate', 0):.1%}")
            print(f"  Average Score: {arc_summary.get('average_score', 0):.2%}")
            
            # Performance comparison
            avg_score = arc_summary.get('average_score', 0)
            print(f"\n  Performance Comparison:")
            print(f"    Human (ARC): 73-77%")
            print(f"    OpenAI o3: 87.5% (ARC-AGI-1)")
            print(f"    Grok 4: ~68% (ARC-AGI-1)")
            print(f"    GPT-4o: 50% (with prompt engineering)")
            print(f"    ICEBURG: {avg_score:.1%}")
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE BENCHMARKING COMPLETE")
        print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ICEBURG Comprehensive Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python benchmarks/run_all_benchmarks.py --all

  # Run only ARC-AGI
  python benchmarks/run_all_benchmarks.py --arc-agi

  # Run quantum and ARC-AGI
  python benchmarks/run_all_benchmarks.py --quantum --arc-agi

  # Run with verbose output
  python benchmarks/run_all_benchmarks.py --all --verbose
        """
    )
    
    parser.add_argument("--all", action="store_true",
                       help="Run all benchmarks")
    parser.add_argument("--quantum", action="store_true",
                       help="Run quantum benchmarks")
    parser.add_argument("--rl", action="store_true",
                       help="Run RL benchmarks")
    parser.add_argument("--hybrid", action="store_true",
                       help="Run hybrid benchmarks")
    parser.add_argument("--arc-agi", action="store_true",
                       help="Run ARC-AGI benchmark")
    parser.add_argument("--output", default="benchmarks/results",
                       help="Output directory for results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Determine benchmark types
    if args.all:
        benchmark_types = ["quantum", "rl", "hybrid", "arc_agi"]
    else:
        benchmark_types = []
        if args.quantum:
            benchmark_types.append("quantum")
        if args.rl:
            benchmark_types.append("rl")
        if args.hybrid:
            benchmark_types.append("hybrid")
        if args.arc_agi:
            benchmark_types.append("arc_agi")
        
        if not benchmark_types:
            # Default: run ARC-AGI if nothing specified
            print("No specific benchmarks specified. Running ARC-AGI by default.")
            benchmark_types = ["arc_agi"]
    
    runner = ComprehensiveBenchmarkRunner(output_dir=args.output)
    
    try:
        results = asyncio.run(runner.run_all(
            benchmark_types=benchmark_types,
            verbose=args.verbose
        ))
        
        print(f"\n📊 All benchmarks completed successfully!")
        print(f"📁 Results saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

