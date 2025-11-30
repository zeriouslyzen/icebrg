"""
Test runner for ICEBURG Elite Financial AI test suite.

Comprehensive test runner that executes all tests and provides detailed reporting.
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from tests.quantum.test_circuits import TestQuantumCircuits, TestQuantumKernels, TestQuantumSampling, TestQuantumIntegration
from tests.quantum.test_qgan import TestQuantumGenerator, TestDiscriminator, TestQGANIntegration, TestQGANFinancialData
from tests.rl.test_agents import (
    TestBaseAgent, TestPPOTrader, TestSACTrader, TestTradingEnvironment, 
    TestOrderBook, TestMarketSimulator, TestRLEmergenceDetection
)
from tests.hybrid.test_quantum_rl import (
    TestQuantumOracle, TestHybridPolicy, TestQuantumRLIntegration, 
    TestQuantumEnhancedAgent, TestQuantumRLConfig, TestQuantumRLWorkflow
)
from tests.integration.test_icberg_integration import (
    TestICEBURGQuantumRLIntegration, TestICEBURGFinancialAIIntegration, 
    TestICEBURGEliteTradingIntegration, TestICEBURGIntegrationWorkflow
)
from tests.pipeline.test_financial_pipeline import (
    TestFinancialAnalysisPipeline, TestPipelineMonitor, TestPipelineOrchestrator, TestPipelineIntegration
)


class TestRunner:
    """Comprehensive test runner for ICEBURG Elite Financial AI."""
    
    def __init__(self):
        """Initialize test runner."""
        self.test_suites = {
            "quantum": [
                TestQuantumCircuits,
                TestQuantumKernels,
                TestQuantumSampling,
                TestQuantumIntegration,
                TestQuantumGenerator,
                TestDiscriminator,
                TestQGANIntegration,
                TestQGANFinancialData
            ],
            "rl": [
                TestBaseAgent,
                TestPPOTrader,
                TestSACTrader,
                TestTradingEnvironment,
                TestOrderBook,
                TestMarketSimulator,
                TestRLEmergenceDetection
            ],
            "hybrid": [
                TestQuantumOracle,
                TestHybridPolicy,
                TestQuantumRLIntegration,
                TestQuantumEnhancedAgent,
                TestQuantumRLConfig,
                TestQuantumRLWorkflow
            ],
            "integration": [
                TestICEBURGQuantumRLIntegration,
                TestICEBURGFinancialAIIntegration,
                TestICEBURGEliteTradingIntegration,
                TestICEBURGIntegrationWorkflow
            ],
            "pipeline": [
                TestFinancialAnalysisPipeline,
                TestPipelineMonitor,
                TestPipelineOrchestrator,
                TestPipelineIntegration
            ]
        }
        
        self.results = {
            "start_time": None,
            "end_time": None,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "errors": [],
            "suites": {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and return comprehensive results."""
        print("🚀 Starting ICEBURG Elite Financial AI Test Suite")
        print("=" * 60)
        
        self.results["start_time"] = datetime.now().isoformat()
        
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        errors = []
        
        for suite_name, test_classes in self.test_suites.items():
            print(f"\n📋 Running {suite_name.upper()} Tests")
            print("-" * 40)
            
            suite_results = self._run_test_suite(suite_name, test_classes)
            
            self.results["suites"][suite_name] = suite_results
            total_tests += suite_results["total_tests"]
            passed_tests += suite_results["passed_tests"]
            failed_tests += suite_results["failed_tests"]
            skipped_tests += suite_results["skipped_tests"]
            errors.extend(suite_results["errors"])
            
            print(f"✅ {suite_name.upper()} Tests Complete: {suite_results['passed_tests']}/{suite_results['total_tests']} passed")
        
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_tests"] = total_tests
        self.results["passed_tests"] = passed_tests
        self.results["failed_tests"] = failed_tests
        self.results["skipped_tests"] = skipped_tests
        self.results["errors"] = errors
        
        self._print_summary()
        self._save_results()
        
        return self.results
    
    def _run_test_suite(self, suite_name: str, test_classes: List) -> Dict[str, Any]:
        """Run a specific test suite."""
        suite_results = {
            "suite_name": suite_name,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "errors": [],
            "test_results": []
        }
        
        for test_class in test_classes:
            try:
                # Create test suite
                suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
                
                # Run tests
                runner = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, 'w'))
                result = runner.run(suite)
                
                # Collect results
                test_count = result.testsRun
                failure_count = len(result.failures)
                error_count = len(result.errors)
                skip_count = len(result.skipped) if hasattr(result, 'skipped') else 0
                
                suite_results["total_tests"] += test_count
                suite_results["passed_tests"] += test_count - failure_count - error_count - skip_count
                suite_results["failed_tests"] += failure_count + error_count
                suite_results["skipped_tests"] += skip_count
                
                # Collect errors
                for failure in result.failures:
                    suite_results["errors"].append({
                        "type": "failure",
                        "test": failure[0],
                        "error": failure[1]
                    })
                
                for error in result.errors:
                    suite_results["errors"].append({
                        "type": "error",
                        "test": error[0],
                        "error": error[1]
                    })
                
                # Store test results
                suite_results["test_results"].append({
                    "test_class": test_class.__name__,
                    "tests_run": test_count,
                    "failures": failure_count,
                    "errors": error_count,
                    "skipped": skip_count
                })
                
            except Exception as e:
                suite_results["errors"].append({
                    "type": "suite_error",
                    "test_class": test_class.__name__,
                    "error": str(e)
                })
        
        return suite_results
    
    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total = self.results["total_tests"]
        passed = self.results["passed_tests"]
        failed = self.results["failed_tests"]
        skipped = self.results["skipped_tests"]
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        print(f"⏱️  Duration: {self._calculate_duration()}")
        
        if failed > 0:
            print(f"\n❌ FAILED TESTS:")
            for suite_name, suite_results in self.results["suites"].items():
                if suite_results["failed_tests"] > 0:
                    print(f"  {suite_name.upper()}: {suite_results['failed_tests']} failures")
        
        if skipped > 0:
            print(f"\n⏭️  SKIPPED TESTS:")
            for suite_name, suite_results in self.results["suites"].items():
                if suite_results["skipped_tests"] > 0:
                    print(f"  {suite_name.upper()}: {suite_results['skipped_tests']} skipped")
    
    def _calculate_duration(self) -> str:
        """Calculate test duration."""
        if self.results["start_time"] and self.results["end_time"]:
            start = datetime.fromisoformat(self.results["start_time"])
            end = datetime.fromisoformat(self.results["end_time"])
            duration = end - start
            return str(duration).split('.')[0]  # Remove microseconds
        return "Unknown"
    
    def _save_results(self):
        """Save test results to file."""
        try:
            results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"\n💾 Test results saved to: {results_file}")
        except Exception as e:
            print(f"\n⚠️  Could not save test results: {e}")
    
    def run_specific_suite(self, suite_name: str) -> Dict[str, Any]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        print(f"🚀 Running {suite_name.upper()} Test Suite")
        print("=" * 60)
        
        suite_results = self._run_test_suite(suite_name, self.test_suites[suite_name])
        
        print(f"\n✅ {suite_name.upper()} Tests Complete: {suite_results['passed_tests']}/{suite_results['total_tests']} passed")
        
        return suite_results
    
    def run_specific_test(self, suite_name: str, test_class_name: str) -> Dict[str, Any]:
        """Run a specific test class."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        test_classes = self.test_suites[suite_name]
        test_class = None
        
        for tc in test_classes:
            if tc.__name__ == test_class_name:
                test_class = tc
                break
        
        if test_class is None:
            raise ValueError(f"Unknown test class: {test_class_name}")
        
        print(f"🚀 Running {test_class_name}")
        print("=" * 60)
        
        try:
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            return {
                "test_class": test_class_name,
                "tests_run": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
                "success": result.wasSuccessful()
            }
        
        except Exception as e:
            return {
                "test_class": test_class_name,
                "error": str(e),
                "success": False
            }


def main():
    """Main test runner entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ICEBURG Elite Financial AI Test Runner")
    parser.add_argument("--suite", help="Run specific test suite (quantum, rl, hybrid, integration, pipeline)")
    parser.add_argument("--test", help="Run specific test class")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    if args.suite and args.test:
        # Run specific test class
        result = runner.run_specific_test(args.suite, args.test)
        print(f"\nResult: {result}")
    
    elif args.suite:
        # Run specific test suite
        result = runner.run_specific_suite(args.suite)
        print(f"\nResult: {result}")
    
    else:
        # Run all tests
        result = runner.run_all_tests()
        
        # Exit with appropriate code
        if result["failed_tests"] > 0:
            sys.exit(1)
        else:
            sys.exit(0)


if __name__ == "__main__":
    main()
