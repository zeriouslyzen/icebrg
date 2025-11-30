"""
Backend Registry
Manages multiple backend compilation targets for Visual IR
"""

from __future__ import annotations
from typing import Dict, List, Any, Union
from pathlib import Path

from . import BackendType
from .html5_backend import HTML5Backend, GeneratedArtifacts
from .react_backend import ReactBackend, ReactComponents
from .swiftui_backend import SwiftUIBackend, SwiftUICode
from ..visual_ir import VisualIRFunction


# Type alias for all artifact types
ArtifactTypes = Union[GeneratedArtifacts, ReactComponents, SwiftUICode]


class VisualBackendRegistry:
    """Registry for visual generation backends"""
    
    def __init__(self):
        self.backends: Dict[BackendType, Any] = {
            BackendType.HTML5: HTML5Backend(),
            BackendType.REACT: ReactBackend(),
            BackendType.SWIFTUI: SwiftUIBackend(),
            # BackendType.FLUTTER: FlutterBackend(),  # Future implementation
        }
        
        self.available_backends = list(self.backends.keys())
    
    def compile(
        self,
        visual_ir: VisualIRFunction,
        backend_type: BackendType
    ) -> ArtifactTypes:
        """Compile Visual IR to a specific backend"""
        if backend_type not in self.backends:
            raise ValueError(f"Backend {backend_type} not available")
        
        backend = self.backends[backend_type]
        return backend.compile(visual_ir)
    
    def compile_multi_target(
        self,
        visual_ir: VisualIRFunction,
        targets: List[BackendType]
    ) -> Dict[BackendType, ArtifactTypes]:
        """Compile Visual IR to multiple backends"""
        results = {}
        
        for target in targets:
            if target in self.backends:
                try:
                    results[target] = self.compile(visual_ir, target)
                except Exception as e:
                    print(f"Error compiling to {target}: {e}")
                    # Continue with other backends
        
        return results
    
    def compile_all(
        self,
        visual_ir: VisualIRFunction
    ) -> Dict[BackendType, ArtifactTypes]:
        """Compile Visual IR to all available backends"""
        return self.compile_multi_target(visual_ir, self.available_backends)
    
    def get_backend(self, backend_type: BackendType) -> Any:
        """Get a specific backend instance"""
        return self.backends.get(backend_type)
    
    def register_backend(self, backend_type: BackendType, backend_instance: Any) -> None:
        """Register a new backend"""
        self.backends[backend_type] = backend_instance
        if backend_type not in self.available_backends:
            self.available_backends.append(backend_type)
    
    def save_artifacts(
        self,
        artifacts: Dict[BackendType, ArtifactTypes],
        output_dir: Path
    ) -> None:
        """Save all artifacts to output directory"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for backend_type, artifact in artifacts.items():
            backend_dir = output_dir / backend_type.value
            backend_dir.mkdir(exist_ok=True)
            artifact.save_to_directory(backend_dir)


if __name__ == "__main__":
    # Example usage
    from ..visual_ir import create_component_node, create_style_node, VisualIRFunction
    from ..visual_tsl import ComponentType
    from ..visual_ir import StyleProperty
    
    # Create a simple button UI
    visual_ir = VisualIRFunction(fn="multi_backend_example")
    
    button = create_component_node(
        component_id="btn_1",
        component_type=ComponentType.BUTTON,
        properties={"text": "Click Me", "type": "button"}
    )
    
    button_style = create_style_node(
        style_id="btn_style_1",
        properties={
            StyleProperty.BACKGROUND: "#007bff",
            StyleProperty.COLOR: "#ffffff",
            StyleProperty.PADDING: "10px 20px",
            StyleProperty.BORDER_RADIUS: "4px",
            StyleProperty.BORDER: "none"
        }
    )
    
    button.style_ref = "btn_style_1"
    button.accessibility_attrs = {"aria-label": "Click Me", "role": "button"}
    
    visual_ir.add_component(button)
    visual_ir.style_graph.add_style(button_style)
    
    # Compile to all backends
    registry = VisualBackendRegistry()
    
    print("Available backends:", [b.value for b in registry.available_backends])
    print("\nCompiling to all backends...")
    
    results = registry.compile_all(visual_ir)
    
    print(f"\nSuccessfully compiled to {len(results)} backends:")
    for backend_type in results.keys():
        print(f"  - {backend_type.value}")
    
    # Save to directory
    output_dir = Path("./generated_multi_backend")
    registry.save_artifacts(results, output_dir)
    print(f"\nArtifacts saved to: {output_dir}")

