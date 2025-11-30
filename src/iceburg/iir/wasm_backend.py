from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .ir import IRFunction
from .interpreter import Interpreter, InterpreterResult
from .wasm_compiler import WasmCompiler, WasmModule, WasmRuntime


@dataclass
class WasmCompileResult:
    ok: bool
    module_bytes: bytes | None
    diagnostics: str


@dataclass
class CrossCheckResult:
    interpreter_result: InterpreterResult
    wasm_result: Any  # placeholder for now
    match: bool
    tolerance: float


class WasmBackend:
    """Real WASM backend that compiles IR to actual WebAssembly bytecode."""

    def __init__(self) -> None:
        self.compiler = WasmCompiler()
        self.runtime = WasmRuntime()

    def compile(self, fn: IRFunction) -> WasmCompileResult:
        try:
            # Compile to real WASM
            module = self.compiler.compile(fn)
            
            # Validate the generated module
            if not self.runtime.validate_module(module):
                return WasmCompileResult(ok=False, module_bytes=None, diagnostics="Invalid WASM module generated")
            
            return WasmCompileResult(ok=True, module_bytes=module.bytes, diagnostics="WASM compilation successful")
        except Exception as e:
            return WasmCompileResult(ok=False, module_bytes=None, diagnostics=f"Compilation error: {str(e)}")

    def cross_check(self, fn: IRFunction, inputs: Dict[str, Any], tolerance: float = 1e-6) -> CrossCheckResult:
        """Cross-check interpreter vs WASM backend results."""
        # Run interpreter
        interpreter = Interpreter()
        interp_result = interpreter.run(fn, inputs)
        
        # Compile and run WASM
        compile_result = self.compile(fn)
        if not compile_result.ok:
            return CrossCheckResult(
                interpreter_result=interp_result,
                wasm_result={"error": compile_result.diagnostics},
                match=False,
                tolerance=tolerance
            )
        
        # Execute WASM (simplified for now)
        try:
            # Convert inputs to the format expected by WASM
            wasm_args = []
            for param_name in fn.params.keys():
                if param_name in inputs:
                    value = inputs[param_name]
                    if isinstance(value, list):
                        wasm_args.append(float(value[0]) if value else 0.0)
                    else:
                        wasm_args.append(float(value))
                else:
                    wasm_args.append(0.0)
            
            # Execute WASM function
            wasm_outputs = self.runtime.execute(
                self.compiler.compile(fn), 
                fn.fn, 
                wasm_args
            )
            
            wasm_result = {"outputs": wasm_outputs, "compiled": True}
            
            # Compare results
            match = self._compare_results(interp_result.outputs, wasm_outputs, tolerance)
            
        except Exception as e:
            wasm_result = {"error": str(e), "compiled": True}
            match = False
            
        return CrossCheckResult(
            interpreter_result=interp_result,
            wasm_result=wasm_result,
            match=match,
            tolerance=tolerance
        )
    
    def _compare_results(self, interp_outputs: Dict[str, Any], wasm_outputs: List[float], tolerance: float) -> bool:
        """Compare interpreter and WASM results within tolerance."""
        if not interp_outputs or not wasm_outputs:
            return False
        
        # Simple comparison - in practice this would be more sophisticated
        for key, value in interp_outputs.items():
            if isinstance(value, list) and value:
                if abs(value[0] - wasm_outputs[0]) > tolerance:
                    return False
            elif isinstance(value, (int, float)):
                if abs(value - wasm_outputs[0]) > tolerance:
                    return False
        
        return True


