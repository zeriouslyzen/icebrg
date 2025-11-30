from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


# Types
@dataclass(frozen=True)
class ScalarType:
    dtype: str  # e.g., "float32", "int64"


@dataclass(frozen=True)
class TensorType:
    dtype: str
    shape: List[Union[int, str]]  # ints or symbolic dims like "N"


@dataclass(frozen=True)
class TupleType:
    elements: List["IRType"]


@dataclass(frozen=True)
class RecordType:
    fields: Dict[str, "IRType"]


IRType = Union[ScalarType, TensorType, TupleType, RecordType]


# Values and Ops
@dataclass
class IRValue:
    name: str
    ty: IRType


@dataclass
class MapOp:
    op: str  # e.g., "mul", "add", "sin", "cos", "exp", "log"
    args: List[str]  # value names


@dataclass
class ReduceOp:
    op: str  # e.g., "add", "mul", "max", "min", "mean"
    init: Any
    arg: str  # value name


@dataclass
class CallOp:
    fn: str
    args: List[str]


@dataclass
class IfOp:
    condition: str  # condition expression
    then_expr: Union[MapOp, ReduceOp, CallOp, "IfOp", "WhileOp"]
    else_expr: Union[MapOp, ReduceOp, CallOp, "IfOp", "WhileOp"]


@dataclass
class WhileOp:
    condition: str  # condition expression
    body: Union[MapOp, ReduceOp, CallOp, "IfOp", "WhileOp"]
    loop_var: str  # loop variable name


@dataclass
class MatMulOp:
    left: str  # left matrix name
    right: str  # right matrix name


@dataclass
class ConvOp:
    input_tensor: str
    kernel: str
    stride: List[int]
    padding: List[int]


@dataclass
class LetBinding:
    name: str
    expr: Union[MapOp, ReduceOp, CallOp, IfOp, WhileOp, MatMulOp, ConvOp]


@dataclass
class IRFunction:
    fn: str
    params: Dict[str, IRType]
    blocks: List[Dict[str, Any]]  # minimal v0: list with {"let": [LetBinding...], "ret": [names...]}
    contracts: Dict[str, List[str]] = field(default_factory=dict)  # e.g., {"post": ["y >= 0"]}


