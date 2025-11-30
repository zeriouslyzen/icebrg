from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import math
import hashlib

from .ir import IRFunction, MapOp, ReduceOp, CallOp, IfOp, WhileOp, MatMulOp, ConvOp


@dataclass
class InterpreterResult:
    outputs: Dict[str, Any]
    trace_hash: str


class Interpreter:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def run(self, fn: IRFunction, inputs: Dict[str, Any]) -> InterpreterResult:
        env: Dict[str, Any] = dict(inputs)
        hasher = hashlib.sha256()

        for block in fn.blocks:
            for let in block.get("let", []):
                value = self._eval_expr(let["expr"], env)
                env[let["name"]] = value
                hasher.update(str(value).encode())

            # return values
            ret_names: List[str] = block.get("ret", [])
            if ret_names:
                outputs = {name: env[name] for name in ret_names}
                trace_hash = hasher.hexdigest()
                self._check_post(fn, outputs)
                return InterpreterResult(outputs=outputs, trace_hash=trace_hash)

        # If no explicit return encountered
        return InterpreterResult(outputs={}, trace_hash=hasher.hexdigest())

    def _eval_expr(self, expr: Any, env: Dict[str, Any]) -> Any:
        if isinstance(expr, MapOp):
            args = [env[a] for a in expr.args]
            if expr.op == "mul":
                return self._elementwise_mul(*args)
            elif expr.op == "add":
                return self._elementwise_add(*args)
            elif expr.op == "sin":
                return self._elementwise_sin(args[0])
            elif expr.op == "cos":
                return self._elementwise_cos(args[0])
            elif expr.op == "exp":
                return self._elementwise_exp(args[0])
            elif expr.op == "log":
                return self._elementwise_log(args[0])
            elif expr.op == "abs":
                return self._elementwise_abs(args[0])
            else:
                raise ValueError(f"Unsupported map op: {expr.op}")

        elif isinstance(expr, ReduceOp):
            data = env[expr.arg]
            if expr.op == "add":
                acc = expr.init
                for v in data:
                    acc = acc + v
                return acc
            elif expr.op == "mul":
                acc = expr.init
                for v in data:
                    acc = acc * v
                return acc
            elif expr.op == "max":
                return max(data) if data else expr.init
            elif expr.op == "min":
                return min(data) if data else expr.init
            elif expr.op == "mean":
                return sum(data) / len(data) if data else 0.0
            else:
                raise ValueError(f"Unsupported reduce op: {expr.op}")

        elif isinstance(expr, CallOp):
            args = [env[a] for a in expr.args]
            if expr.fn == "sqrt":
                return self._apply_function(math.sqrt, args[0])
            elif expr.fn == "sin":
                return self._apply_function(math.sin, args[0])
            elif expr.fn == "cos":
                return self._apply_function(math.cos, args[0])
            elif expr.fn == "exp":
                return self._apply_function(math.exp, args[0])
            elif expr.fn == "log":
                return self._apply_function(math.log, args[0])
            elif expr.fn == "abs":
                return self._apply_function(abs, args[0])
            else:
                raise ValueError(f"Unsupported call fn: {expr.fn}")

        elif isinstance(expr, IfOp):
            condition_result = self._evaluate_condition(expr.condition, env)
            if condition_result:
                return self._eval_expr(expr.then_expr, env)
            else:
                return self._eval_expr(expr.else_expr, env)

        elif isinstance(expr, WhileOp):
            loop_env = env.copy()
            while self._evaluate_condition(expr.condition, loop_env):
                result = self._eval_expr(expr.body, loop_env)
                # Update loop variable if it exists
                if expr.loop_var in loop_env:
                    loop_env[expr.loop_var] = result
            return result if 'result' in locals() else None

        elif isinstance(expr, MatMulOp):
            left = env[expr.left]
            right = env[expr.right]
            return self._matrix_multiply(left, right)

        elif isinstance(expr, ConvOp):
            input_tensor = env[expr.input_tensor]
            kernel = env[expr.kernel]
            return self._convolution(input_tensor, kernel, expr.stride, expr.padding)

        # dict-form (JSON) support for minimal ergonomics
        elif isinstance(expr, dict):
            if "map" in expr:
                return self._eval_expr(MapOp(**expr["map"]), env)
            elif "reduce" in expr:
                return self._eval_expr(ReduceOp(**expr["reduce"]), env)
            elif "call" in expr:
                return self._eval_expr(CallOp(**expr["call"]), env)
            elif "if" in expr:
                return self._eval_expr(IfOp(**expr["if"]), env)
            elif "while" in expr:
                return self._eval_expr(WhileOp(**expr["while"]), env)
            elif "matmul" in expr:
                return self._eval_expr(MatMulOp(**expr["matmul"]), env)
            elif "conv" in expr:
                return self._eval_expr(ConvOp(**expr["conv"]), env)
            elif "constant" in expr:
                return expr["constant"]
            else:
                raise ValueError(f"Unknown expr dict: {expr}")

        else:
            raise ValueError(f"Unknown expr type: {type(expr)}")

    @staticmethod
    def _elementwise_mul(a: Union[List[float], float], b: Union[List[float], float]) -> Union[List[float], float]:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("shape mismatch for mul")
            return [x * y for x, y in zip(a, b)]
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a * b
        else:
            raise ValueError("type mismatch for mul")

    @staticmethod
    def _elementwise_add(a: Union[List[float], float], b: Union[List[float], float]) -> Union[List[float], float]:
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                raise ValueError("shape mismatch for add")
            return [x + y for x, y in zip(a, b)]
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return a + b
        else:
            raise ValueError("type mismatch for add")

    @staticmethod
    def _elementwise_sin(a: List[float]) -> List[float]:
        return [math.sin(x) for x in a]

    @staticmethod
    def _elementwise_cos(a: List[float]) -> List[float]:
        return [math.cos(x) for x in a]

    @staticmethod
    def _elementwise_exp(a: List[float]) -> List[float]:
        return [math.exp(x) for x in a]

    @staticmethod
    def _elementwise_log(a: List[float]) -> List[float]:
        return [math.log(x) for x in a]

    @staticmethod
    def _elementwise_abs(a: List[float]) -> List[float]:
        return [abs(x) for x in a]

    @staticmethod
    def _apply_function(func, arg):
        """Apply function to scalar or list."""
        if isinstance(arg, list):
            return [func(x) for x in arg]
        else:
            return func(arg)

    def _evaluate_condition(self, condition: str, env: Dict[str, Any]) -> bool:
        """Evaluate a condition string in the given environment."""
        # Simple condition evaluation for v0
        # Supports basic comparisons like "x > 0", "y >= 5"
        
        # Replace variable names with values from environment
        expr = condition
        for var_name, value in env.items():
            if isinstance(value, list):
                # For lists, use length or first element
                expr = expr.replace(var_name, str(len(value)))
            else:
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

    @staticmethod
    def _matrix_multiply(a: List[List[float]], b: List[List[float]]) -> List[List[float]]:
        """Simple matrix multiplication for 2D lists."""
        if not a or not b:
            return []
        
        rows_a, cols_a = len(a), len(a[0])
        rows_b, cols_b = len(b), len(b[0])
        
        if cols_a != rows_b:
            raise ValueError(f"Matrix dimension mismatch: {cols_a} != {rows_b}")
        
        result = [[0.0 for _ in range(cols_b)] for _ in range(rows_a)]
        
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        
        return result

    @staticmethod
    def _convolution(input_tensor: List[List[float]], kernel: List[List[float]], 
                    stride: List[int], padding: List[int]) -> List[List[float]]:
        """Simple 2D convolution operation."""
        if not input_tensor or not kernel:
            return []
        
        # Simple implementation for demonstration
        # In practice, this would be much more sophisticated
        input_h, input_w = len(input_tensor), len(input_tensor[0])
        kernel_h, kernel_w = len(kernel), len(kernel[0])
        
        # Calculate output dimensions
        output_h = (input_h + 2 * padding[0] - kernel_h) // stride[0] + 1
        output_w = (input_w + 2 * padding[1] - kernel_w) // stride[1] + 1
        
        result = [[0.0 for _ in range(output_w)] for _ in range(output_h)]
        
        for i in range(output_h):
            for j in range(output_w):
                for ki in range(kernel_h):
                    for kj in range(kernel_w):
                        input_i = i * stride[0] + ki - padding[0]
                        input_j = j * stride[1] + kj - padding[1]
                        
                        if 0 <= input_i < input_h and 0 <= input_j < input_w:
                            result[i][j] += input_tensor[input_i][input_j] * kernel[ki][kj]
        
        return result

    @staticmethod
    def _check_post(fn: IRFunction, outputs: Dict[str, Any]) -> None:
        # Minimal: handle simple non-negativity checks like "y >= 0.0"
        for cond in fn.contracts.get("post", []):
            if ">=" in cond:
                lhs, rhs = [s.strip() for s in cond.split(">=")]
                if lhs in outputs:
                    value = outputs[lhs]
                    # Handle both scalar and list values
                    if isinstance(value, list):
                        # For lists, check all elements
                        for v in value:
                            if not (v >= float(rhs)):
                                raise AssertionError(f"Post-condition failed: {cond} (element {v})")
                    else:
                        # For scalars, check directly
                        if not (value >= float(rhs)):
                            raise AssertionError(f"Post-condition failed: {cond}")


