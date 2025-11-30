#!/bin/bash
# Run Comprehensive Test Suite for ICEBURG 2.0

cd "$(dirname "$0")/.."

echo "ICEBURG 2.0 - Comprehensive Test Suite"
echo "======================================"
echo ""

# Run comprehensive test suite
echo "Running comprehensive test suite..."
python3 tests/comprehensive_test_suite.py

echo ""
echo "Running individual feature tests..."
echo ""

# Run truth-finding test
echo "1. Truth-Finding Test..."
python3 tests/test_truth_finding.py
echo ""

# Run device generation test
echo "2. Device Generation Test..."
python3 tests/test_device_generation.py
echo ""

# Run swarming test
echo "3. Swarming Test..."
python3 tests/test_swarming.py
echo ""

# Run full system test
echo "4. Full System Integration Test..."
python3 tests/test_full_system.py
echo ""

echo "======================================"
echo "All tests complete!"
echo "Check data/test_results/ for detailed results"
echo "======================================"

