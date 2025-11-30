"""
Visual Architect Agent
Main agent for visual UI generation through IIR
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import re

from ..config import IceburgConfig
from ..iir.visual_tsl import (
    UISpec, VisualComponent, LayoutSpec, ThemeSpec,
    AccessibilitySpec, PerformanceBudget, ComponentType,
    LayoutType, Interaction, EventType, StyleSpec,
    create_button_spec, create_input_spec, create_container_spec
)
from ..iir.visual_ir import VisualIRFunction, ComponentNode, StyleNode, StyleProperty, create_component_node, create_style_node
from ..iir.visual_contracts import VisualContractValidator, ContractViolation, ALL_CONTRACTS
from ..iir.visual_optimizer import VisualOptimizer, OptimizationResult
from ..iir.backends import BackendType
from ..iir.backends.backend_registry import VisualBackendRegistry, ArtifactTypes


@dataclass
class VisualGenerationResult:
    """Complete result of visual generation"""
    spec: UISpec
    ir: VisualIRFunction
    artifacts: Dict[BackendType, ArtifactTypes]
    validation: ValidationResult
    optimization: OptimizationResult
    metadata: Dict[str, Any]


@dataclass
class ValidationResult:
    """Result of contract validation"""
    passed: bool
    violations: List[ContractViolation]
    warnings: List[ContractViolation]
    errors: List[ContractViolation]


class VisualArchitect:
    """Main visual generation agent"""
    
    def __init__(self):
        self.contract_validator = VisualContractValidator()
        self.visual_optimizer = VisualOptimizer()
        self.backend_registry = VisualBackendRegistry()
        self.component_counter = 0
        self.style_counter = 0
    
    def run(
        self,
        cfg: IceburgConfig,
        user_intent: str,
        verbose: bool = False
    ) -> VisualGenerationResult:
        """
        Generate UI from user intent
        
        Args:
            cfg: ICEBURG configuration
            user_intent: Natural language description of desired UI
            verbose: Print progress information
            
        Returns:
            Complete visual generation result
        """
        if verbose:
            print("[VISUAL_ARCHITECT] Starting visual generation...")
            print(f"[VISUAL_ARCHITECT] User intent: {user_intent}")
        
        # Step 1: Parse user intent
        intent = self._parse_visual_intent(user_intent, verbose)
        
        # Step 2: Generate Visual TSL spec
        ui_spec = self._generate_ui_spec(intent, verbose)
        
        # Step 3: Convert TSL to Visual IR
        visual_ir = self._tsl_to_ir(ui_spec, verbose)
        
        # Step 4: Optimize IR
        optimization_result = self.visual_optimizer.optimize(visual_ir)
        if verbose:
            print(f"[VISUAL_ARCHITECT] Optimization complete: {optimization_result.improvements}")
        
        # Step 5: Validate contracts
        validation_result = self._validate_ir(optimization_result.optimized_ir, verbose)
        
        # Step 6: Compile to target platforms
        targets = [BackendType.HTML5, BackendType.REACT, BackendType.SWIFTUI]
        artifacts = self.backend_registry.compile_multi_target(
            optimization_result.optimized_ir,
            targets
        )
        
        if verbose:
            print(f"[VISUAL_ARCHITECT] Compiled to {len(artifacts)} backends")
        
        # Step 7: Gather metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "user_intent": user_intent,
            "components_generated": len(visual_ir.ui_components),
            "styles_generated": len(visual_ir.style_graph.styles),
            "backends_compiled": [b.value for b in artifacts.keys()],
            "validation_passed": validation_result.passed
        }
        
        return VisualGenerationResult(
            spec=ui_spec,
            ir=optimization_result.optimized_ir,
            artifacts=artifacts,
            validation=validation_result,
            optimization=optimization_result,
            metadata=metadata
        )
    
    def _parse_visual_intent(self, user_intent: str, verbose: bool = False) -> Dict[str, Any]:
        """Parse natural language intent into structured format"""
        if verbose:
            print("[VISUAL_ARCHITECT] Parsing user intent...")
        
        intent = {
            "components": [],
            "theme": {},
            "layout": "flex",
            "interactions": []
        }
        
        # Simple keyword-based parsing (in production, would use LLM)
        user_intent_lower = user_intent.lower()
        
        # Detect components
        if "button" in user_intent_lower:
            # Extract button text
            button_text = self._extract_quoted_text(user_intent) or "Button"
            intent["components"].append({
                "type": "button",
                "text": button_text,
                "id": "btn_1"
            })
        
        if "input" in user_intent_lower or "textfield" in user_intent_lower:
            placeholder = self._extract_quoted_text(user_intent) or "Enter text"
            intent["components"].append({
                "type": "input",
                "placeholder": placeholder,
                "id": "input_1"
            })
        
        if "text" in user_intent_lower and "says" in user_intent_lower:
            text_content = self._extract_quoted_text(user_intent) or "Text"
            intent["components"].append({
                "type": "text",
                "text": text_content,
                "id": "text_1"
            })
        
        # Detect theme/colors
        colors = {
            "blue": "#007bff",
            "red": "#dc3545",
            "green": "#28a745",
            "yellow": "#ffc107",
            "gray": "#6c757d",
            "grey": "#6c757d",
            "white": "#ffffff",
            "black": "#000000"
        }
        
        for color_name, color_value in colors.items():
            if color_name in user_intent_lower:
                intent["theme"]["primary_color"] = color_value
                break
        
        # Detect layout
        if "grid" in user_intent_lower:
            intent["layout"] = "grid"
        elif "vertical" in user_intent_lower or "column" in user_intent_lower:
            intent["layout"] = "flex_column"
        elif "horizontal" in user_intent_lower or "row" in user_intent_lower:
            intent["layout"] = "flex_row"
        
        # Detect interactions
        if "click" in user_intent_lower:
            intent["interactions"].append({
                "event": "click",
                "description": "Handle click event"
            })
        
        return intent
    
    def _extract_quoted_text(self, text: str) -> Optional[str]:
        """Extract text within quotes"""
        # Match text in quotes
        matches = re.findall(r'["\']([^"\']+)["\']', text)
        return matches[0] if matches else None
    
    def _generate_ui_spec(self, intent: Dict[str, Any], verbose: bool = False) -> UISpec:
        """Generate Visual TSL specification from parsed intent"""
        if verbose:
            print("[VISUAL_ARCHITECT] Generating UI specification...")
        
        components = []
        
        # Generate components from intent
        for comp_data in intent.get("components", []):
            comp_type = comp_data.get("type")
            comp_id = comp_data.get("id")
            
            if comp_type == "button":
                component = create_button_spec(
                    text=comp_data.get("text", "Button"),
                    on_click="Handle button click",
                    id=comp_id
                )
                components.append(component)
            
            elif comp_type == "input":
                component = create_input_spec(
                    input_type="text",
                    placeholder=comp_data.get("placeholder", "Enter text"),
                    id=comp_id
                )
                components.append(component)
            
            elif comp_type == "text":
                from ..iir.visual_tsl import VisualComponent, ComponentType
                component = VisualComponent(
                    component_type=ComponentType.TEXT,
                    id=comp_id,
                    properties={"text": comp_data.get("text", "Text")},
                    accessibility={"role": "text"}
                )
                components.append(component)
        
        # Determine layout
        layout_type = LayoutType.FLEX
        direction = "column"
        
        if intent.get("layout") == "grid":
            layout_type = LayoutType.GRID
        elif intent.get("layout") == "flex_row":
            direction = "row"
        
        layout = LayoutSpec(
            layout_type=layout_type,
            direction=direction,
            gap="10px",
            padding="20px"
        )
        
        # Generate theme
        theme_data = intent.get("theme", {})
        theme = ThemeSpec(
            primary_color=theme_data.get("primary_color", "#007bff"),
            secondary_color="#6c757d",
            background_color="#ffffff",
            text_color="#212529"
        )
        
        # Create UI specification
        ui_spec = UISpec(
            name="generated_ui",
            components=components,
            layout=layout,
            theme=theme,
            accessibility_requirements=AccessibilitySpec(
                wcag_level="AA",
                min_contrast_ratio=4.5,
                keyboard_navigable=True
            ),
            performance_budget=PerformanceBudget(
                max_load_time_ms=100,
                max_bundle_size_kb=50
            ),
            security_contracts=ALL_CONTRACTS,
            created_at=datetime.now().isoformat()
        )
        
        return ui_spec
    
    def _tsl_to_ir(self, ui_spec: UISpec, verbose: bool = False) -> VisualIRFunction:
        """Convert Visual TSL specification to Visual IR"""
        if verbose:
            print("[VISUAL_ARCHITECT] Converting TSL to IR...")
        
        visual_ir = VisualIRFunction(fn=ui_spec.name)
        
        # Convert components
        for tsl_component in ui_spec.components:
            ir_component = self._tsl_component_to_ir(tsl_component, ui_spec)
            visual_ir.add_component(ir_component)
        
        # Convert theme to global styles
        if ui_spec.theme:
            visual_ir.style_graph.theme_variables = self._theme_to_variables(ui_spec.theme)
        
        # Set layout
        if ui_spec.layout:
            from ..iir.visual_ir import LayoutGraph
            visual_ir.layout_graph = LayoutGraph(layout_type=ui_spec.layout.layout_type)
        
        # Store accessibility and performance metadata
        if ui_spec.accessibility_requirements:
            visual_ir.accessibility_report = ui_spec.accessibility_requirements.to_dict()
        
        if ui_spec.performance_budget:
            visual_ir.performance_metrics = {
                "budget": ui_spec.performance_budget.to_dict(),
                "estimated_load_ms": 50,  # Placeholder
                "estimated_bundle_kb": 25  # Placeholder
            }
        
        return visual_ir
    
    def _tsl_component_to_ir(
        self,
        tsl_component: VisualComponent,
        ui_spec: UISpec
    ) -> ComponentNode:
        """Convert TSL component to IR component node"""
        from ..iir.ir import IRValue, ScalarType
        
        # Convert properties to IR values
        ir_props = {}
        for key, value in tsl_component.properties.items():
            ir_props[key] = IRValue(str(value), ScalarType("string"))
        
        # Create IR component
        ir_component = ComponentNode(
            id=tsl_component.id or f"comp_{self.component_counter}",
            type=tsl_component.component_type,
            props=ir_props,
            accessibility_attrs=tsl_component.accessibility
        )
        self.component_counter += 1
        
        # Convert style
        if tsl_component.style:
            style_id = f"style_{self.style_counter}"
            self.style_counter += 1
            
            style_node = self._tsl_style_to_ir(tsl_component.style, style_id)
            ir_component.style_ref = style_id
            
            # Note: Style will be added to visual_ir.style_graph by caller
        
        # Convert event handlers
        for interaction in tsl_component.interactions:
            ir_component.event_handlers[interaction.event_type] = f"handler_{len(ir_component.event_handlers)}"
        
        # Convert children recursively
        for child in tsl_component.children:
            child_ir = self._tsl_component_to_ir(child, ui_spec)
            ir_component.children_refs.append(child_ir.id)
        
        return ir_component
    
    def _tsl_style_to_ir(self, tsl_style: StyleSpec, style_id: str) -> StyleNode:
        """Convert TSL style to IR style node"""
        from ..iir.ir import IRValue, ScalarType
        
        properties = {}
        
        # Convert colors
        if tsl_style.colors:
            for key, value in tsl_style.colors.items():
                if key == "primary":
                    properties[StyleProperty.COLOR] = IRValue(value, ScalarType("string"))
                elif key == "background":
                    properties[StyleProperty.BACKGROUND] = IRValue(value, ScalarType("string"))
        
        # Convert spacing
        if tsl_style.spacing:
            if "padding" in tsl_style.spacing:
                properties[StyleProperty.PADDING] = IRValue(tsl_style.spacing["padding"], ScalarType("string"))
            if "margin" in tsl_style.spacing:
                properties[StyleProperty.MARGIN] = IRValue(tsl_style.spacing["margin"], ScalarType("string"))
        
        # Convert borders
        if tsl_style.borders:
            if "border" in tsl_style.borders:
                properties[StyleProperty.BORDER] = IRValue(tsl_style.borders["border"], ScalarType("string"))
            if "border-radius" in tsl_style.borders:
                properties[StyleProperty.BORDER_RADIUS] = IRValue(tsl_style.borders["border-radius"], ScalarType("string"))
        
        # Convert custom properties
        for key, value in tsl_style.custom.items():
            # Try to map to known style properties
            try:
                prop = StyleProperty(key)
                properties[prop] = IRValue(str(value), ScalarType("string"))
            except ValueError:
                # Unknown property, skip
                pass
        
        return StyleNode(id=style_id, properties=properties)
    
    def _theme_to_variables(self, theme: ThemeSpec) -> Dict[str, Any]:
        """Convert theme to CSS variables"""
        from ..iir.ir import IRValue, ScalarType
        
        return {
            "primary-color": IRValue(theme.primary_color, ScalarType("string")),
            "secondary-color": IRValue(theme.secondary_color, ScalarType("string")),
            "background-color": IRValue(theme.background_color, ScalarType("string")),
            "text-color": IRValue(theme.text_color, ScalarType("string")),
            "font-family": IRValue(theme.font_family, ScalarType("string")),
            "font-size-base": IRValue(theme.font_size_base, ScalarType("string"))
        }
    
    def _validate_ir(self, visual_ir: VisualIRFunction, verbose: bool = False) -> ValidationResult:
        """Validate Visual IR against contracts"""
        if verbose:
            print("[VISUAL_ARCHITECT] Validating contracts...")
        
        violations = self.contract_validator.validate(visual_ir, ALL_CONTRACTS)
        
        errors = [v for v in violations if v.severity == "error"]
        warnings = [v for v in violations if v.severity == "warning"]
        
        passed = len(errors) == 0
        
        if verbose:
            print(f"[VISUAL_ARCHITECT] Validation: {len(errors)} errors, {len(warnings)} warnings")
        
        return ValidationResult(
            passed=passed,
            violations=violations,
            errors=errors,
            warnings=warnings
        )


if __name__ == "__main__":
    # Example usage
    from ..config import load_config
    
    cfg = load_config()
    architect = VisualArchitect()
    
    # Test with simple button
    result = architect.run(
        cfg,
        'Create a blue button that says "Click Me"',
        verbose=True
    )
    
    print("\nGeneration Result:")
    print(f"Components: {len(result.ir.ui_components)}")
    print(f"Backends: {list(result.artifacts.keys())}")
    print(f"Validation: {'Passed' if result.validation.passed else 'Failed'}")
    print(f"Errors: {len(result.validation.errors)}")
    print(f"Warnings: {len(result.validation.warnings)}")

