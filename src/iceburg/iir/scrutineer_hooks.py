"""Scrutineer-style contract validation for IIR functions.

This module provides contract checking capabilities similar to the Scrutineer agent,
but focused on validating TSL specifications and IR function contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re

from .tsl import TaskSpec
from .ir import IRFunction
from .interpreter import Interpreter, InterpreterResult


@dataclass
class ContractValidationResult:
    """Result of contract validation."""
    is_valid: bool
    confidence_score: float  # 0.0 to 1.0
    violations: List[str]
    evidence_level: str  # A, B, C, S, X (matching Scrutineer scale)
    validation_notes: str


class ContractValidator:
    """Validates TSL and IR contracts using Scrutineer-style analysis."""
    
    def __init__(self) -> None:
        self.interpreter = Interpreter()
    
    def validate_tsl_contracts(self, spec: TaskSpec, test_inputs: List[Dict[str, Any]]) -> ContractValidationResult:
        """Validate TSL pre/post conditions and invariants."""
        violations = []
        confidence_score = 1.0
        
        # Check pre-conditions
        for pre_cond in spec.pre:
            if not self._evaluate_condition(pre_cond, test_inputs[0] if test_inputs else {}):
                violations.append(f"Pre-condition failed: {pre_cond}")
                confidence_score -= 0.2
        
        # Check post-conditions (would need actual execution)
        # For v0, we'll do basic syntax validation
        for post_cond in spec.post:
            if not self._validate_condition_syntax(post_cond):
                violations.append(f"Invalid post-condition syntax: {post_cond}")
                confidence_score -= 0.1
        
        # Check invariants
        for invariant in spec.invariants:
            if not self._validate_condition_syntax(invariant):
                violations.append(f"Invalid invariant syntax: {invariant}")
                confidence_score -= 0.1
        
        # Determine evidence level
        evidence_level = self._determine_evidence_level(violations, confidence_score)
        
        return ContractValidationResult(
            is_valid=len(violations) == 0,
            confidence_score=max(0.0, confidence_score),
            violations=violations,
            evidence_level=evidence_level,
            validation_notes=f"Validated {len(spec.pre + spec.post + spec.invariants)} contracts"
        )
    
    def validate_ir_contracts(self, fn: IRFunction, test_inputs: List[Dict[str, Any]]) -> ContractValidationResult:
        """Validate IR function contracts by execution."""
        violations = []
        confidence_score = 1.0
        
        for inputs in test_inputs:
            try:
                result = self.interpreter.run(fn, inputs)
                
                # Check post-conditions
                for post_cond in fn.contracts.get("post", []):
                    if not self._evaluate_condition(post_cond, result.outputs):
                        violations.append(f"Post-condition failed: {post_cond}")
                        confidence_score -= 0.3
                        
            except Exception as e:
                violations.append(f"Execution failed: {str(e)}")
                confidence_score -= 0.5
        
        evidence_level = self._determine_evidence_level(violations, confidence_score)
        
        return ContractValidationResult(
            is_valid=len(violations) == 0,
            confidence_score=max(0.0, confidence_score),
            violations=violations,
            evidence_level=evidence_level,
            validation_notes=f"Validated {len(test_inputs)} test cases"
        )
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a condition string in the given context."""
        # Simple condition evaluation for v0
        # Supports basic comparisons like "y >= 0.0", "x > 5"
        
        # Replace variable names with values from context
        expr = condition
        for var_name, value in context.items():
            expr = expr.replace(var_name, str(value))
        
        # Handle common operators
        if ">=" in expr:
            lhs, rhs = expr.split(">=")
            try:
                return float(lhs.strip()) >= float(rhs.strip())
            except (ValueError, TypeError):
                return False
        elif ">" in expr:
            lhs, rhs = expr.split(">")
            try:
                return float(lhs.strip()) > float(rhs.strip())
            except (ValueError, TypeError):
                return False
        elif "<=" in expr:
            lhs, rhs = expr.split("<=")
            try:
                return float(lhs.strip()) <= float(rhs.strip())
            except (ValueError, TypeError):
                return False
        elif "<" in expr:
            lhs, rhs = expr.split("<")
            try:
                return float(lhs.strip()) < float(rhs.strip())
            except (ValueError, TypeError):
                return False
        elif "==" in expr:
            lhs, rhs = expr.split("==")
            try:
                return float(lhs.strip()) == float(rhs.strip())
            except (ValueError, TypeError):
                return lhs.strip() == rhs.strip()
        
        return True  # Default to true for unrecognized conditions
    
    def _validate_condition_syntax(self, condition: str) -> bool:
        """Validate that a condition has reasonable syntax."""
        # Basic syntax validation
        if not condition.strip():
            return False
        
        # Check for basic operators
        operators = [">=", "<=", ">", "<", "==", "!="]
        has_operator = any(op in condition for op in operators)
        
        # Check for variable names (basic pattern)
        has_variable = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', condition)
        
        return has_operator and has_variable
    
    def _determine_evidence_level(self, violations: List[str], confidence_score: float) -> str:
        """Determine evidence level based on violations and confidence."""
        if confidence_score >= 0.9 and len(violations) == 0:
            return "A"  # Well-Established
        elif confidence_score >= 0.7 and len(violations) <= 1:
            return "B"  # Plausible
        elif confidence_score >= 0.5:
            return "C"  # Highly Speculative
        elif confidence_score >= 0.3:
            return "S"  # Suppressed but Valid
        else:
            return "X"  # Actively Censored
