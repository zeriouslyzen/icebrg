#!/usr/bin/env python3
"""
ICEBURG Elite Financial AI Test Runner

Simple test runner script for executing the comprehensive test suite.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tests'))

def run_tests():
    """Run the comprehensive test suite."""
    print("🚀 ICEBURG Elite Financial AI Test Suite")
    print("=" * 50)
    
    try:
        # Import and run test runner
        from tests.test_runner import TestRunner
        
        runner = TestRunner()
        results = runner.run_all_tests()
        
        # Print final results
        print("\n" + "=" * 50)
        print("📊 FINAL RESULTS")
        print("=" * 50)
        
        total = results["total_tests"]
        passed = results["passed_tests"]
        failed = results["failed_tests"]
        skipped = results["skipped_tests"]
        
        print(f"Total Tests: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        
        if total > 0:
            success_rate = (passed / total) * 100
            print(f"📈 Success Rate: {success_rate:.1f}%")
        
        # Exit with appropriate code
        if failed > 0:
            print(f"\n❌ {failed} tests failed!")
            return 1
        else:
            print(f"\n✅ All tests passed!")
            return 0
    
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory.")
        return 1
    
    except Exception as e:
        print(f"❌ Test runner error: {e}")
        return 1

def run_specific_suite(suite_name):
    """Run a specific test suite."""
    print(f"🚀 Running {suite_name.upper()} Test Suite")
    print("=" * 50)
    
    try:
        from tests.test_runner import TestRunner
        
        runner = TestRunner()
        results = runner.run_specific_suite(suite_name)
        
        print(f"\n✅ {suite_name.upper()} Tests Complete: {results['passed_tests']}/{results['total_tests']} passed")
        
        if results["failed_tests"] > 0:
            return 1
        else:
            return 0
    
    except Exception as e:
        print(f"❌ Error running {suite_name} tests: {e}")
        return 1

def run_quick_tests():
    """Run quick tests (quantum and RL only)."""
    print("🚀 Running Quick Tests (Quantum + RL)")
    print("=" * 50)
    
    try:
        from tests.test_runner import TestRunner
        
        runner = TestRunner()
        
        # Run quantum tests
        quantum_results = runner.run_specific_suite("quantum")
        print(f"✅ Quantum Tests: {quantum_results['passed_tests']}/{quantum_results['total_tests']} passed")
        
        # Run RL tests
        rl_results = runner.run_specific_suite("rl")
        print(f"✅ RL Tests: {rl_results['passed_tests']}/{rl_results['total_tests']} passed")
        
        total_passed = quantum_results['passed_tests'] + rl_results['passed_tests']
        total_tests = quantum_results['total_tests'] + rl_results['total_tests']
        total_failed = quantum_results['failed_tests'] + rl_results['failed_tests']
        
        print(f"\n📊 Quick Test Results: {total_passed}/{total_tests} passed")
        
        if total_failed > 0:
            return 1
        else:
            return 0
    
    except Exception as e:
        print(f"❌ Error running quick tests: {e}")
        return 1

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ICEBURG Elite Financial AI Test Runner")
    parser.add_argument("--suite", choices=["quantum", "rl", "hybrid", "integration", "pipeline"], 
                       help="Run specific test suite")
    parser.add_argument("--quick", action="store_true", help="Run quick tests (quantum + RL only)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick:
        return run_quick_tests()
    elif args.suite:
        return run_specific_suite(args.suite)
    else:
        return run_tests()

if __name__ == "__main__":
    sys.exit(main())
