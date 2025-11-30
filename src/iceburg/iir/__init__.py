"""ICEBURG Intermediate IR (IIR) minimal v0.

Exports:
- TSL dataclasses and JSON helpers
- IR core nodes and types
- Deterministic interpreter
- WASM backend stub (no-op placeholder)
"""

from .tsl import TaskSpec, IOType, Budget, tsl_from_json, tsl_to_json
from .ir import (
    ScalarType,
    TensorType,
    TupleType,
    RecordType,
    IRType,
    IRValue,
    IRFunction,
    MapOp,
    ReduceOp,
    CallOp,
    IfOp,
    WhileOp,
    MatMulOp,
    ConvOp,
    LetBinding,
)
from .interpreter import Interpreter, InterpreterResult
from .wasm_backend import WasmBackend, WasmCompileResult, CrossCheckResult
from .scrutineer_hooks import ContractValidator, ContractValidationResult
from .contract_language import ContractEvaluator, ContractParser, Contract, TemporalContract, QuantifiedContract, ContractType
from .optimizer import Optimizer, EGraph, PartialEvaluator
from .property_testing import PropertyBasedTester, TestGenerator, RandomTestGenerator, ContractBasedTestGenerator
from .multi_backend import MultiBackendCompiler, BackendType, NativeCCompiler, CudaCompiler

__all__ = [
    # TSL
    "TaskSpec",
    "IOType",
    "Budget",
    "tsl_from_json",
    "tsl_to_json",
    # IR
    "ScalarType",
    "TensorType",
    "TupleType",
    "RecordType",
    "IRType",
    "IRValue",
    "IRFunction",
    "MapOp",
    "ReduceOp",
    "CallOp",
    "IfOp",
    "WhileOp",
    "MatMulOp",
    "ConvOp",
    "LetBinding",
    # Runtime
    "Interpreter",
    "InterpreterResult",
    # WASM
    "WasmBackend",
    "WasmCompileResult",
    "CrossCheckResult",
    # Contract Validation
    "ContractValidator",
    "ContractValidationResult",
    # Advanced Contracts
    "ContractEvaluator",
    "ContractParser",
    "Contract",
    "TemporalContract",
    "QuantifiedContract",
    "ContractType",
    # Optimization
    "Optimizer",
    "EGraph",
    "PartialEvaluator",
    # Property Testing
    "PropertyBasedTester",
    "TestGenerator",
    "RandomTestGenerator",
    "ContractBasedTestGenerator",
    # Multi-Backend
    "MultiBackendCompiler",
    "BackendType",
    "NativeCCompiler",
    "CudaCompiler",
]


