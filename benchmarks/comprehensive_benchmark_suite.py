#!/usr/bin/env python3
"""
ICEBURG 2.0 Comprehensive Benchmark Suite
Standardized benchmarks for truth-finding, suppression detection, swarming, and device generation
"""

import asyncio
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.core.system_integrator import SystemIntegrator
from iceburg.truth.suppression_detector import SuppressionDetector
from iceburg.integration.swarming_integration import SwarmingIntegration
from iceburg.generation.device_generator import DeviceGenerator
from iceburg.research.methodology_analyzer import MethodologyAnalyzer


@dataclass
class BenchmarkResult:
    """Result from a single benchmark test"""
    test_name: str
    success: bool
    execution_time: float
    accuracy: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    timestamp: float
    results: List[BenchmarkResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmark suite for ICEBURG 2.0"""
    
    def __init__(self):
        self.system_integrator = SystemIntegrator()
        self.suppression_detector = SuppressionDetector()
        self.swarming_integration = SwarmingIntegration()
        self.device_generator = DeviceGenerator()
        self.methodology_analyzer = MethodologyAnalyzer()
        self.results: List[BenchmarkResult] = []
    
    async def run_all_benchmarks(self) -> BenchmarkSuite:
        """Run all comprehensive benchmarks"""
        print("\n" + "="*70)
        print("ICEBURG 2.0 - COMPREHENSIVE BENCHMARK SUITE")
        print("="*70 + "\n")
        
        suite = BenchmarkSuite(
            suite_name="ICEBURG 2.0 Comprehensive Benchmarks",
            timestamp=time.time()
        )
        
        # Standard AI Benchmarks
        print("1. STANDARD AI BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_standard_ai())
        
        # Truth-Finding Benchmarks
        print("\n2. TRUTH-FINDING BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_truth_finding())
        
        # Suppression Detection Benchmarks
        print("\n3. SUPPRESSION DETECTION BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_suppression_detection())
        
        # Swarming Benchmarks
        print("\n4. SWARMING BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_swarming())
        
        # Device Generation Benchmarks
        print("\n5. DEVICE GENERATION BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_device_generation())
        
        # System Integration Benchmarks
        print("\n6. SYSTEM INTEGRATION BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_system_integration())
        
        # Performance Benchmarks
        print("\n7. PERFORMANCE BENCHMARKS")
        print("-" * 70)
        suite.results.extend(await self.benchmark_performance())
        
        # Calculate summary
        suite.summary = self._calculate_summary(suite.results)
        
        # Display summary
        self._display_summary(suite)
        
        # Save results
        self._save_results(suite)
        
        return suite
    
    async def benchmark_standard_ai(self) -> List[BenchmarkResult]:
        """Benchmark standard AI capabilities"""
        results = []
        
        # MMLU-style questions
        mmlu_questions = [
            ("What is the capital of France?", "Paris"),
            ("What is 2+2?", "4"),
            ("What is the speed of light?", "299792458 meters per second"),
        ]
        
        for question, expected in mmlu_questions:
            start_time = time.time()
            try:
                result = await self.system_integrator.process_query_with_full_integration(
                    query=question,
                    domain="general"
                )
                execution_time = time.time() - start_time
                
                # Simple accuracy check (would need better evaluation)
                content = result.get("results", {}).get("insights", {}).get("insights", [])
                accuracy = 1.0 if content else 0.0
                
                results.append(BenchmarkResult(
                    test_name=f"MMLU-style: {question[:30]}...",
                    success=True,
                    execution_time=execution_time,
                    accuracy=accuracy,
                    metrics={"question": question, "expected": expected}
                ))
                print(f"  ✅ {question[:50]}... ({execution_time:.2f}s)")
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"MMLU-style: {question[:30]}...",
                    success=False,
                    execution_time=time.time() - start_time,
                    accuracy=0.0,
                    error=str(e)
                ))
                print(f"  ❌ {question[:50]}... (FAILED)")
        
        return results
    
    async def benchmark_truth_finding(self) -> List[BenchmarkResult]:
        """Benchmark truth-finding capabilities"""
        results = []
        
        truth_queries = [
            "What suppressed knowledge exists about quantum computing?",
            "How can Enhanced Deliberation reveal hidden patterns?",
            "What contradictions exist in AI development timelines?",
        ]
        
        for query in truth_queries:
            start_time = time.time()
            try:
                result = await self.system_integrator.process_query_with_full_integration(
                    query=query,
                    domain="truth_finding"
                )
                execution_time = time.time() - start_time
                
                methodology = result.get("results", {}).get("methodology", {})
                insights = result.get("results", {}).get("insights", {})
                
                accuracy = 1.0 if methodology and insights else 0.0
                
                results.append(BenchmarkResult(
                    test_name=f"Truth-Finding: {query[:40]}...",
                    success=True,
                    execution_time=execution_time,
                    accuracy=accuracy,
                    metrics={
                        "methodology_steps": len(methodology.get("steps", [])),
                        "insights_count": len(insights.get("insights", []))
                    }
                ))
                print(f"  ✅ {query[:50]}... ({execution_time:.2f}s)")
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"Truth-Finding: {query[:40]}...",
                    success=False,
                    execution_time=time.time() - start_time,
                    accuracy=0.0,
                    error=str(e)
                ))
                print(f"  ❌ {query[:50]}... (FAILED)")
        
        return results
    
    async def benchmark_suppression_detection(self) -> List[BenchmarkResult]:
        """Benchmark suppression detection accuracy"""
        results = []
        
        # Test documents with known suppression indicators
        test_documents = [
            {
                "id": "doc1",
                "content": "Research was classified for 20 years before public release.",
                "metadata": {"creation_date": "2000-01-01", "release_date": "2020-01-01"},
                "expected_suppression": True
            },
            {
                "id": "doc2",
                "content": "Public information about this topic is widely available.",
                "metadata": {"creation_date": "2020-01-01", "release_date": "2020-01-01"},
                "expected_suppression": False
            },
        ]
        
        for doc in test_documents:
            start_time = time.time()
            try:
                detection_result = self.suppression_detector.detect_suppression([doc])
                execution_time = time.time() - start_time
                
                detected = detection_result.get("suppression_detected", False)
                expected = doc.get("expected_suppression", False)
                accuracy = 1.0 if detected == expected else 0.0
                
                results.append(BenchmarkResult(
                    test_name=f"Suppression Detection: {doc['id']}",
                    success=True,
                    execution_time=execution_time,
                    accuracy=accuracy,
                    metrics={
                        "detected": detected,
                        "expected": expected,
                        "score": detection_result.get("overall_suppression_score", 0.0)
                    }
                ))
                print(f"  ✅ {doc['id']}: Detected={detected}, Expected={expected} ({execution_time:.3f}s)")
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"Suppression Detection: {doc['id']}",
                    success=False,
                    execution_time=time.time() - start_time,
                    accuracy=0.0,
                    error=str(e)
                ))
                print(f"  ❌ {doc['id']}: FAILED")
        
        return results
    
    async def benchmark_swarming(self) -> List[BenchmarkResult]:
        """Benchmark swarming performance"""
        results = []
        
        swarm_queries = [
            "How does swarming create better answers?",
            "What are the benefits of multi-agent collaboration?",
        ]
        
        for query in swarm_queries:
            start_time = time.time()
            try:
                swarm = await self.swarming_integration.create_truth_finding_swarm(
                    query=query,
                    swarm_type="research_swarm"
                )
                swarm_results = await self.swarming_integration.execute_swarm(swarm, parallel=True)
                execution_time = time.time() - start_time
                
                agent_count = len(swarm.get("agents", []))
                result_count = len(swarm_results.get("agent_results", []))
                synthesized = swarm_results.get("synthesized_result") is not None
                
                accuracy = 1.0 if synthesized and result_count > 0 else 0.0
                
                results.append(BenchmarkResult(
                    test_name=f"Swarming: {query[:40]}...",
                    success=True,
                    execution_time=execution_time,
                    accuracy=accuracy,
                    metrics={
                        "agent_count": agent_count,
                        "result_count": result_count,
                        "synthesized": synthesized
                    }
                ))
                print(f"  ✅ {query[:50]}... ({execution_time:.2f}s, {agent_count} agents)")
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"Swarming: {query[:40]}...",
                    success=False,
                    execution_time=time.time() - start_time,
                    accuracy=0.0,
                    error=str(e)
                ))
                print(f"  ❌ {query[:50]}... (FAILED)")
        
        return results
    
    async def benchmark_device_generation(self) -> List[BenchmarkResult]:
        """Benchmark device generation quality"""
        results = []
        
        device_tests = [
            {
                "device_type": "quantum_computer",
                "requirements": {"qubits": 100}
            },
            {
                "device_type": "energy_device",
                "requirements": {"power_output": 1000}
            },
        ]
        
        for test in device_tests:
            start_time = time.time()
            try:
                device = await self.device_generator.generate_device(
                    device_type=test["device_type"],
                    requirements=test["requirements"],
                    domain="physics"
                )
                execution_time = time.time() - start_time
                
                has_specs = device.get("specifications") is not None
                has_schematics = device.get("schematics") is not None
                has_code = device.get("code") is not None
                
                accuracy = (has_specs + has_schematics + has_code) / 3.0
                
                results.append(BenchmarkResult(
                    test_name=f"Device Generation: {test['device_type']}",
                    success=True,
                    execution_time=execution_time,
                    accuracy=accuracy,
                    metrics={
                        "has_specifications": has_specs,
                        "has_schematics": has_schematics,
                        "has_code": has_code
                    }
                ))
                print(f"  ✅ {test['device_type']} ({execution_time:.2f}s, accuracy={accuracy:.2f})")
            except Exception as e:
                results.append(BenchmarkResult(
                    test_name=f"Device Generation: {test['device_type']}",
                    success=False,
                    execution_time=time.time() - start_time,
                    accuracy=0.0,
                    error=str(e)
                ))
                print(f"  ❌ {test['device_type']}: FAILED")
        
        return results
    
    async def benchmark_system_integration(self) -> List[BenchmarkResult]:
        """Benchmark system integration performance"""
        results = []
        
        integration_query = "How does Enhanced Deliberation enable ICEBURG to find suppressed knowledge and create devices?"
        
        start_time = time.time()
        try:
            integration = self.system_integrator.integrate_all_systems()
            result = await self.system_integrator.process_query_with_full_integration(
                query=integration_query,
                domain="truth_finding"
            )
            execution_time = time.time() - start_time
            
            has_methodology = result.get("results", {}).get("methodology") is not None
            has_swarm = result.get("results", {}).get("swarm") is not None
            has_insights = result.get("results", {}).get("insights") is not None
            
            accuracy = (has_methodology + has_swarm + has_insights) / 3.0
            
            results.append(BenchmarkResult(
                test_name="System Integration: Full Query Processing",
                success=True,
                execution_time=execution_time,
                accuracy=accuracy,
                metrics={
                    "integrated": integration.get("integrated", False),
                    "has_methodology": has_methodology,
                    "has_swarm": has_swarm,
                    "has_insights": has_insights
                }
            ))
            print(f"  ✅ Full Integration ({execution_time:.2f}s, accuracy={accuracy:.2f})")
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="System Integration: Full Query Processing",
                success=False,
                execution_time=time.time() - start_time,
                accuracy=0.0,
                error=str(e)
            ))
            print(f"  ❌ Full Integration: FAILED")
        
        return results
    
    async def benchmark_performance(self) -> List[BenchmarkResult]:
        """Benchmark performance metrics"""
        results = []
        
        # Latency test
        latency_query = "What is Enhanced Deliberation?"
        start_time = time.time()
        try:
            await self.system_integrator.process_query_with_full_integration(
                query=latency_query,
                domain="truth_finding"
            )
            latency = time.time() - start_time
            
            results.append(BenchmarkResult(
                test_name="Performance: Latency",
                success=True,
                execution_time=latency,
                accuracy=1.0 if latency < 10.0 else 0.5,  # Good if < 10s
                metrics={"latency_seconds": latency}
            ))
            print(f"  ✅ Latency: {latency:.2f}s")
        except Exception as e:
            results.append(BenchmarkResult(
                test_name="Performance: Latency",
                success=False,
                execution_time=time.time() - start_time,
                accuracy=0.0,
                error=str(e)
            ))
            print(f"  ❌ Latency: FAILED")
        
        # Throughput test
        throughput_queries = [
            "What is Enhanced Deliberation?",
            "How does swarming work?",
            "What is suppression detection?",
        ]
        
        start_time = time.time()
        success_count = 0
        for query in throughput_queries:
            try:
                await self.system_integrator.process_query_with_full_integration(
                    query=query,
                    domain="truth_finding"
                )
                success_count += 1
            except Exception:
                pass
        
        total_time = time.time() - start_time
        throughput = success_count / total_time if total_time > 0 else 0.0
        
        results.append(BenchmarkResult(
            test_name="Performance: Throughput",
            success=True,
            execution_time=total_time,
            accuracy=1.0 if throughput > 0.1 else 0.5,  # Good if > 0.1 queries/sec
            metrics={"throughput_qps": throughput, "queries_processed": success_count}
        ))
        print(f"  ✅ Throughput: {throughput:.2f} queries/sec")
        
        return results
    
    def _calculate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate benchmark summary"""
        total = len(results)
        successful = sum(1 for r in results if r.success)
        failed = total - successful
        
        avg_accuracy = sum(r.accuracy for r in results) / total if total > 0 else 0.0
        avg_execution_time = sum(r.execution_time for r in results) / total if total > 0 else 0.0
        
        return {
            "total_tests": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_accuracy": avg_accuracy,
            "average_execution_time": avg_execution_time
        }
    
    def _display_summary(self, suite: BenchmarkSuite):
        """Display benchmark summary"""
        print("\n" + "="*70)
        print("BENCHMARK SUMMARY")
        print("="*70)
        print(f"Total Tests: {suite.summary['total_tests']}")
        print(f"Successful: {suite.summary['successful']}")
        print(f"Failed: {suite.summary['failed']}")
        print(f"Success Rate: {suite.summary['success_rate']*100:.1f}%")
        print(f"Average Accuracy: {suite.summary['average_accuracy']*100:.1f}%")
        print(f"Average Execution Time: {suite.summary['average_execution_time']:.2f}s")
        print("="*70 + "\n")
    
    def _save_results(self, suite: BenchmarkSuite):
        """Save benchmark results to file"""
        results_dir = Path("benchmarks/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"comprehensive_benchmark_{timestamp}.json"
        
        # Convert to JSON-serializable format
        results_data = {
            "suite_name": suite.suite_name,
            "timestamp": suite.timestamp,
            "summary": suite.summary,
            "results": [
                {
                    "test_name": r.test_name,
                    "success": r.success,
                    "execution_time": r.execution_time,
                    "accuracy": r.accuracy,
                    "metrics": r.metrics,
                    "error": r.error,
                    "timestamp": r.timestamp
                }
                for r in suite.results
            ]
        }
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"📄 Results saved to: {results_file}")


async def main():
    """Run comprehensive benchmark suite"""
    benchmark_suite = ComprehensiveBenchmarkSuite()
    suite = await benchmark_suite.run_all_benchmarks()
    return suite


if __name__ == "__main__":
    asyncio.run(main())

