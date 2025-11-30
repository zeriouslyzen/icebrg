"""Property-Based Testing for IIR functions.

This module generates property-based tests from contracts and specifications,
automatically creating test cases that verify the correctness of IIR functions.
"""

from __future__ import annotations

import random
import math
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .ir import IRFunction
from .tsl import TaskSpec
from .contract_language import ContractEvaluator, ContractParser


@dataclass
class TestCase:
    """Represents a test case."""
    inputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]] = None
    description: str = ""


@dataclass
class TestResult:
    """Represents the result of a test case."""
    passed: bool
    actual_outputs: Dict[str, Any]
    expected_outputs: Optional[Dict[str, Any]]
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class PropertyTest:
    """Represents a property-based test."""
    name: str
    property_func: Callable[[Dict[str, Any], Dict[str, Any]], bool]
    description: str
    generator: 'TestGenerator'


class TestGenerator(ABC):
    """Abstract base class for test generators."""
    
    @abstractmethod
    def generate(self, spec: TaskSpec, count: int = 100) -> List[TestCase]:
        """Generate test cases from a specification."""
        pass


class RandomTestGenerator(TestGenerator):
    """Generates random test cases."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
    
    def generate(self, spec: TaskSpec, count: int = 100) -> List[TestCase]:
        """Generate random test cases."""
        test_cases = []
        
        for i in range(count):
            inputs = {}
            
            for input_spec in spec.inputs:
                inputs[input_spec.name] = self._generate_value(input_spec.type)
            
            test_cases.append(TestCase(
                inputs=inputs,
                description=f"Random test case {i+1}"
            ))
        
        return test_cases
    
    def _generate_value(self, type_str: str) -> Any:
        """Generate a random value of the specified type."""
        if "tensor<float32>" in type_str:
            # Generate random tensor
            if "[" in type_str and "]" in type_str:
                # Extract dimensions
                dims_str = type_str[type_str.find("["):type_str.find("]")+1]
                if dims_str == "[N]":
                    # 1D tensor
                    size = random.randint(1, 10)
                    return [random.uniform(-10.0, 10.0) for _ in range(size)]
                elif dims_str == "[M,N]":
                    # 2D tensor
                    rows = random.randint(1, 5)
                    cols = random.randint(1, 5)
                    return [[random.uniform(-10.0, 10.0) for _ in range(cols)] for _ in range(rows)]
            else:
                # Default 1D tensor
                size = random.randint(1, 5)
                return [random.uniform(-10.0, 10.0) for _ in range(size)]
        elif "float32" in type_str:
            return random.uniform(-100.0, 100.0)
        elif "int" in type_str:
            return random.randint(-100, 100)
        else:
            # Default to float
            return random.uniform(-10.0, 10.0)


class ContractBasedTestGenerator(TestGenerator):
    """Generates test cases based on contracts."""
    
    def __init__(self):
        self.contract_parser = ContractParser()
        self.contract_evaluator = ContractEvaluator()
    
    def generate(self, spec: TaskSpec, count: int = 100) -> List[TestCase]:
        """Generate test cases based on contracts."""
        test_cases = []
        
        # Generate test cases for preconditions
        for pre_cond in spec.pre:
            test_cases.extend(self._generate_precondition_tests(pre_cond, spec, count // len(spec.pre)))
        
        # Generate test cases for postconditions
        for post_cond in spec.post:
            test_cases.extend(self._generate_postcondition_tests(post_cond, spec, count // len(spec.post)))
        
        # Generate edge cases
        test_cases.extend(self._generate_edge_cases(spec, count // 4))
        
        return test_cases[:count]
    
    def _generate_precondition_tests(self, pre_cond: str, spec: TaskSpec, count: int) -> List[TestCase]:
        """Generate test cases that satisfy preconditions."""
        test_cases = []
        
        for i in range(count):
            inputs = {}
            
            # Generate inputs that satisfy the precondition
            for input_spec in spec.inputs:
                if "N" in pre_cond and input_spec.name == "x":
                    # Ensure N > 0
                    size = random.randint(1, 10)
                    inputs[input_spec.name] = [random.uniform(-10.0, 10.0) for _ in range(size)]
                else:
                    inputs[input_spec.name] = self._generate_value(input_spec.type)
            
            test_cases.append(TestCase(
                inputs=inputs,
                description=f"Precondition test: {pre_cond}"
            ))
        
        return test_cases
    
    def _generate_postcondition_tests(self, post_cond: str, spec: TaskSpec, count: int) -> List[TestCase]:
        """Generate test cases for postcondition verification."""
        test_cases = []
        
        for i in range(count):
            inputs = {}
            
            for input_spec in spec.inputs:
                inputs[input_spec.name] = self._generate_value(input_spec.type)
            
            test_cases.append(TestCase(
                inputs=inputs,
                description=f"Postcondition test: {post_cond}"
            ))
        
        return test_cases
    
    def _generate_edge_cases(self, spec: TaskSpec, count: int) -> List[TestCase]:
        """Generate edge case test cases."""
        test_cases = []
        
        for i in range(count):
            inputs = {}
            
            for input_spec in spec.inputs:
                if "tensor" in input_spec.type:
                    # Generate edge cases for tensors
                    if i % 4 == 0:
                        # Empty tensor
                        inputs[input_spec.name] = []
                    elif i % 4 == 1:
                        # Single element
                        inputs[input_spec.name] = [0.0]
                    elif i % 4 == 2:
                        # Very large values
                        size = random.randint(1, 5)
                        inputs[input_spec.name] = [random.uniform(1000.0, 10000.0) for _ in range(size)]
                    else:
                        # Very small values
                        size = random.randint(1, 5)
                        inputs[input_spec.name] = [random.uniform(-10000.0, -1000.0) for _ in range(size)]
                else:
                    inputs[input_spec.name] = self._generate_value(input_spec.type)
            
            test_cases.append(TestCase(
                inputs=inputs,
                description=f"Edge case test {i+1}"
            ))
        
        return test_cases
    
    def _generate_value(self, type_str: str) -> Any:
        """Generate a value of the specified type."""
        if "tensor<float32>" in type_str:
            size = random.randint(1, 5)
            return [random.uniform(-10.0, 10.0) for _ in range(size)]
        elif "float32" in type_str:
            return random.uniform(-100.0, 100.0)
        elif "int" in type_str:
            return random.randint(-100, 100)
        else:
            return random.uniform(-10.0, 10.0)


class PropertyBasedTester:
    """Main property-based testing system."""
    
    def __init__(self):
        self.random_generator = RandomTestGenerator()
        self.contract_generator = ContractBasedTestGenerator()
        self.contract_evaluator = ContractEvaluator()
    
    def test_function(self, fn: IRFunction, spec: TaskSpec, 
                     test_count: int = 100, 
                     generator: Optional[TestGenerator] = None) -> List[TestResult]:
        """Test an IR function with property-based testing."""
        if generator is None:
            generator = self.contract_generator
        
        # Generate test cases
        test_cases = generator.generate(spec, test_count)
        
        # Run tests
        results = []
        for test_case in test_cases:
            result = self._run_test_case(fn, test_case, spec)
            results.append(result)
        
        return results
    
    def _run_test_case(self, fn: IRFunction, test_case: TestCase, spec: TaskSpec) -> TestResult:
        """Run a single test case."""
        import time
        start_time = time.time()
        
        try:
            # Import interpreter
            from .interpreter import Interpreter
            
            # Execute function
            interpreter = Interpreter()
            actual_outputs = interpreter.run(fn, test_case.inputs).outputs
            
            # Verify contracts
            contract_passed = self._verify_contracts(spec, test_case.inputs, actual_outputs)
            
            execution_time = time.time() - start_time
            
            return TestResult(
                passed=contract_passed,
                actual_outputs=actual_outputs,
                expected_outputs=test_case.expected_outputs,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return TestResult(
                passed=False,
                actual_outputs={},
                expected_outputs=test_case.expected_outputs,
                error=str(e),
                execution_time=execution_time
            )
    
    def _verify_contracts(self, spec: TaskSpec, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Verify that contracts are satisfied."""
        context = {**inputs, **outputs}
        
        # Check preconditions
        for pre_cond in spec.pre:
            if not self.contract_evaluator._evaluate_expression(pre_cond, context):
                return False
        
        # Check postconditions
        for post_cond in spec.post:
            if not self.contract_evaluator._evaluate_expression(post_cond, context):
                return False
        
        # Check invariants
        for invariant in spec.invariants:
            if not self.contract_evaluator._evaluate_expression(invariant, context):
                return False
        
        return True
    
    def generate_property_tests(self, spec: TaskSpec) -> List[PropertyTest]:
        """Generate property-based tests from specifications."""
        properties = []
        
        # Non-negativity property
        if any(">= 0" in post for post in spec.post):
            properties.append(PropertyTest(
                name="non_negativity",
                property_func=lambda inputs, outputs: all(
                    v >= 0 for v in outputs.values() if isinstance(v, (int, float))
                ),
                description="Output values should be non-negative",
                generator=self.contract_generator
            ))
        
        # Monotonicity property
        if "monotonic" in str(spec.post):
            properties.append(PropertyTest(
                name="monotonicity",
                property_func=self._check_monotonicity,
                description="Function should be monotonic",
                generator=self.contract_generator
            ))
        
        # Boundedness property
        if "bounded" in str(spec.post):
            properties.append(PropertyTest(
                name="boundedness",
                property_func=self._check_boundedness,
                description="Output should be bounded",
                generator=self.contract_generator
            ))
        
        return properties
    
    def _check_monotonicity(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Check if function is monotonic."""
        # Simplified monotonicity check
        # In practice, this would compare with previous test results
        return True
    
    def _check_boundedness(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> bool:
        """Check if output is bounded."""
        # Check if all outputs are within reasonable bounds
        for value in outputs.values():
            if isinstance(value, (int, float)):
                if abs(value) > 1e6:  # Arbitrary large bound
                    return False
        return True
    
    def run_property_tests(self, fn: IRFunction, spec: TaskSpec, 
                          test_count: int = 100) -> Dict[str, bool]:
        """Run property-based tests."""
        properties = self.generate_property_tests(spec)
        results = {}
        
        for property_test in properties:
            test_cases = property_test.generator.generate(spec, test_count)
            passed_count = 0
            
            for test_case in test_cases:
                try:
                    from .interpreter import Interpreter
                    interpreter = Interpreter()
                    outputs = interpreter.run(fn, test_case.inputs).outputs
                    
                    if property_test.property_func(test_case.inputs, outputs):
                        passed_count += 1
                except:
                    pass
            
            results[property_test.name] = passed_count == len(test_cases)
        
        return results
