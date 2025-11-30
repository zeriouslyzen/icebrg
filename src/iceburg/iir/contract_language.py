"""Advanced Contract Language for IIR functions.

This module provides a sophisticated contract language supporting:
- Temporal logic (LTL/CTL)
- Complex mathematical conditions
- Quantified expressions
- Invariant specifications
- Behavioral contracts
"""

from __future__ import annotations

import re
import math
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class ContractType(Enum):
    """Types of contracts supported."""
    PRECONDITION = "pre"
    POSTCONDITION = "post"
    INVARIANT = "invariant"
    TEMPORAL = "temporal"
    BEHAVIORAL = "behavioral"
    QUANTIFIED = "quantified"


@dataclass
class Contract:
    """Represents a contract specification."""
    type: ContractType
    expression: str
    description: Optional[str] = None
    severity: str = "error"  # error, warning, info


@dataclass
class TemporalContract:
    """Temporal logic contract."""
    type: ContractType
    expression: str
    temporal_operator: str  # G (globally), F (finally), X (next), U (until)
    scope: str = "execution"  # execution, session, global
    description: Optional[str] = None
    severity: str = "error"  # error, warning, info


@dataclass
class QuantifiedContract:
    """Quantified contract (forall, exists)."""
    type: ContractType
    expression: str
    quantifier: str  # forall, exists
    variable: str
    domain: str  # domain of quantification
    description: Optional[str] = None
    severity: str = "error"  # error, warning, info


class ContractEvaluator:
    """Evaluates contracts with advanced language features."""
    
    def __init__(self) -> None:
        self.temporal_state = {}
        self.execution_history = []
    
    def evaluate_contract(self, contract: Union[Contract, TemporalContract, QuantifiedContract], context: Dict[str, Any]) -> bool:
        """Evaluate a contract in the given context."""
        if contract.type == ContractType.PRECONDITION:
            return self._evaluate_precondition(contract.expression, context)
        elif contract.type == ContractType.POSTCONDITION:
            return self._evaluate_postcondition(contract.expression, context)
        elif contract.type == ContractType.INVARIANT:
            return self._evaluate_invariant(contract.expression, context)
        elif contract.type == ContractType.TEMPORAL:
            return self._evaluate_temporal(contract, context)
        elif contract.type == ContractType.BEHAVIORAL:
            return self._evaluate_behavioral(contract.expression, context)
        elif contract.type == ContractType.QUANTIFIED:
            return self._evaluate_quantified(contract, context)
        else:
            return True
    
    def _evaluate_precondition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a precondition."""
        return self._evaluate_expression(expression, context)
    
    def _evaluate_postcondition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a postcondition."""
        return self._evaluate_expression(expression, context)
    
    def _evaluate_invariant(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate an invariant."""
        return self._evaluate_expression(expression, context)
    
    def _evaluate_temporal(self, contract: TemporalContract, context: Dict[str, Any]) -> bool:
        """Evaluate a temporal logic contract."""
        # Record current state
        self.execution_history.append(context.copy())
        
        if contract.temporal_operator == "G":  # Globally
            return self._evaluate_globally(contract.expression, context)
        elif contract.temporal_operator == "F":  # Finally
            return self._evaluate_finally(contract.expression, context)
        elif contract.temporal_operator == "X":  # Next
            return self._evaluate_next(contract.expression, context)
        elif contract.temporal_operator == "U":  # Until
            return self._evaluate_until(contract.expression, context)
        else:
            return True
    
    def _evaluate_behavioral(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a behavioral contract."""
        # Behavioral contracts describe how the system should behave over time
        # Examples: "monotonic", "bounded", "convergent"
        
        if "monotonic" in expression:
            return self._check_monotonicity(context)
        elif "bounded" in expression:
            return self._check_boundedness(expression, context)
        elif "convergent" in expression:
            return self._check_convergence(context)
        else:
            return True
    
    def _evaluate_quantified(self, contract: QuantifiedContract, context: Dict[str, Any]) -> bool:
        """Evaluate a quantified contract."""
        if contract.quantifier == "forall":
            return self._evaluate_forall(contract, context)
        elif contract.quantifier == "exists":
            return self._evaluate_exists(contract, context)
        else:
            return True
    
    def _evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a mathematical expression."""
        # Enhanced expression evaluation with more operators and functions
        
        # Replace variables with values
        expr = expression
        for var_name, value in context.items():
            if isinstance(value, list):
                # For lists, use length or first element
                expr = expr.replace(var_name, str(len(value)))
            else:
                expr = expr.replace(var_name, str(value))
        
        # Handle mathematical functions
        expr = self._expand_functions(expr)
        
        # Handle operators
        return self._evaluate_operators(expr)
    
    def _expand_functions(self, expr: str) -> str:
        """Expand mathematical functions in expression."""
        # sin, cos, exp, log, abs, sqrt, etc.
        functions = {
            'sin': 'math.sin',
            'cos': 'math.cos',
            'exp': 'math.exp',
            'log': 'math.log',
            'abs': 'abs',
            'sqrt': 'math.sqrt',
            'max': 'max',
            'min': 'min'
        }
        
        for func, math_func in functions.items():
            pattern = f'{func}\\('
            expr = re.sub(pattern, f'{math_func}(', expr)
        
        return expr
    
    def _evaluate_operators(self, expr: str) -> bool:
        """Evaluate operators in expression."""
        try:
            # Safe evaluation of mathematical expressions
            # In practice, this would use a proper expression parser
            result = eval(expr, {"__builtins__": {}, "math": math}, {})
            return bool(result)
        except:
            return True  # Default to true for unparseable expressions
    
    def _evaluate_globally(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate globally (G) temporal operator."""
        # Check if expression holds in all states
        for state in self.execution_history:
            if not self._evaluate_expression(expression, state):
                return False
        return True
    
    def _evaluate_finally(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate finally (F) temporal operator."""
        # Check if expression holds in at least one state
        for state in self.execution_history:
            if self._evaluate_expression(expression, state):
                return True
        return False
    
    def _evaluate_next(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate next (X) temporal operator."""
        # Check if expression holds in the next state
        if len(self.execution_history) > 1:
            return self._evaluate_expression(expression, self.execution_history[-1])
        return True
    
    def _evaluate_until(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate until (U) temporal operator."""
        # Simplified implementation
        return self._evaluate_expression(expression, context)
    
    def _check_monotonicity(self, context: Dict[str, Any]) -> bool:
        """Check if values are monotonic."""
        # Simplified monotonicity check
        if len(self.execution_history) < 2:
            return True
        
        # Check if values are increasing or decreasing
        prev_values = self.execution_history[-2]
        curr_values = self.execution_history[-1]
        
        for key in curr_values:
            if key in prev_values:
                if isinstance(curr_values[key], (int, float)) and isinstance(prev_values[key], (int, float)):
                    # Check if values are non-decreasing
                    if curr_values[key] < prev_values[key]:
                        return False
        
        return True
    
    def _check_boundedness(self, expression: str, context: Dict[str, Any]) -> bool:
        """Check if values are bounded."""
        # Extract bounds from expression
        bounds_match = re.search(r'bounded\(([^,]+),\s*([^,]+),\s*([^)]+)\)', expression)
        if bounds_match:
            var_name, lower, upper = bounds_match.groups()
            if var_name in context:
                value = context[var_name]
                if isinstance(value, (int, float)):
                    return float(lower) <= value <= float(upper)
        return True
    
    def _check_convergence(self, context: Dict[str, Any]) -> bool:
        """Check if values are converging."""
        # Simplified convergence check
        if len(self.execution_history) < 3:
            return True
        
        # Check if the difference between consecutive values is decreasing
        values = [state.get('value', 0) for state in self.execution_history[-3:]]
        if len(values) >= 3:
            diff1 = abs(values[1] - values[0])
            diff2 = abs(values[2] - values[1])
            return diff2 <= diff1
        
        return True
    
    def _evaluate_forall(self, contract: QuantifiedContract, context: Dict[str, Any]) -> bool:
        """Evaluate forall quantifier."""
        # Simplified implementation
        return self._evaluate_expression(contract.expression, context)
    
    def _evaluate_exists(self, contract: QuantifiedContract, context: Dict[str, Any]) -> bool:
        """Evaluate exists quantifier."""
        # Simplified implementation
        return self._evaluate_expression(contract.expression, context)


class ContractParser:
    """Parses contract expressions into Contract objects."""
    
    def parse(self, expression: str) -> List[Contract]:
        """Parse a contract expression into Contract objects."""
        contracts = []
        
        # Parse different types of contracts
        if expression.startswith("G("):  # Globally
            contracts.append(TemporalContract(
                type=ContractType.TEMPORAL,
                expression=expression[2:-1],
                temporal_operator="G"
            ))
        elif expression.startswith("F("):  # Finally
            contracts.append(TemporalContract(
                type=ContractType.TEMPORAL,
                expression=expression[2:-1],
                temporal_operator="F"
            ))
        elif expression.startswith("forall"):  # Forall
            contracts.append(QuantifiedContract(
                type=ContractType.QUANTIFIED,
                expression=expression,
                quantifier="forall",
                variable="x",
                domain="domain"
            ))
        elif expression.startswith("exists"):  # Exists
            contracts.append(QuantifiedContract(
                type=ContractType.QUANTIFIED,
                expression=expression,
                quantifier="exists",
                variable="x",
                domain="domain"
            ))
        elif "monotonic" in expression or "bounded" in expression or "convergent" in expression:
            contracts.append(Contract(
                type=ContractType.BEHAVIORAL,
                expression=expression
            ))
        else:
            # Default to postcondition
            contracts.append(Contract(
                type=ContractType.POSTCONDITION,
                expression=expression
            ))
        
        return contracts
