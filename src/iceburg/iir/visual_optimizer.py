"""
Visual Optimizer
Extends E-graph optimization with visual-specific rules
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .visual_ir import VisualIRFunction, ComponentNode, StyleNode, StyleProperty
from .visual_tsl import ComponentType
from .optimizer import Optimizer


@dataclass
class OptimizationResult:
    """Result of optimization"""
    optimized_ir: VisualIRFunction
    applied_rules: List[str]
    improvements: Dict[str, Any]


class OptimizationRule(ABC):
    """Base class for optimization rules"""
    
    @abstractmethod
    def name(self) -> str:
        """Rule name"""
        pass
    
    @abstractmethod
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        """Check if rule can be applied"""
        pass
    
    @abstractmethod
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        """Apply optimization rule, return True if changed"""
        pass


# Structure Optimization Rules

class RemoveUnnecessaryWrappers(OptimizationRule):
    """Remove container divs that have only one child"""
    
    def name(self) -> str:
        return "remove_unnecessary_wrappers"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        for component in visual_ir.ui_components.values():
            if component.type == ComponentType.CONTAINER and len(component.children_refs) == 1:
                return True
        return False
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        changed = False
        
        # Find components with single-child containers
        to_remove = []
        for comp_id, component in visual_ir.ui_components.items():
            if component.type == ComponentType.CONTAINER and len(component.children_refs) == 1:
                # Check if container has no styling or events
                if not component.style_ref and not component.event_handlers:
                    to_remove.append(comp_id)
        
        # Remove unnecessary wrappers
        for comp_id in to_remove:
            component = visual_ir.ui_components[comp_id]
            child_id = component.children_refs[0]
            
            # Update parent references
            for other_comp in visual_ir.ui_components.values():
                if comp_id in other_comp.children_refs:
                    idx = other_comp.children_refs.index(comp_id)
                    other_comp.children_refs[idx] = child_id
                    changed = True
        
        return changed


class FlattenNestedContainers(OptimizationRule):
    """Flatten nested containers of the same type"""
    
    def name(self) -> str:
        return "flatten_nested_containers"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        for component in visual_ir.ui_components.values():
            if component.type == ComponentType.CONTAINER:
                for child_id in component.children_refs:
                    child = visual_ir.get_component(child_id)
                    if child and child.type == ComponentType.CONTAINER:
                        return True
        return False
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        changed = False
        
        for component in list(visual_ir.ui_components.values()):
            if component.type != ComponentType.CONTAINER:
                continue
            
            new_children = []
            for child_id in component.children_refs:
                child = visual_ir.get_component(child_id)
                if child and child.type == ComponentType.CONTAINER:
                    # Flatten: add child's children directly
                    new_children.extend(child.children_refs)
                    changed = True
                else:
                    new_children.append(child_id)
            
            if changed:
                component.children_refs = new_children
        
        return changed


class CoalesceAdjacentTextNodes(OptimizationRule):
    """Merge adjacent text nodes"""
    
    def name(self) -> str:
        return "coalesce_adjacent_text_nodes"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        for component in visual_ir.ui_components.values():
            prev_child_type = None
            for child_id in component.children_refs:
                child = visual_ir.get_component(child_id)
                if child and child.type == ComponentType.TEXT:
                    if prev_child_type == ComponentType.TEXT:
                        return True
                    prev_child_type = ComponentType.TEXT
                else:
                    prev_child_type = None
        return False
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        # Simplified implementation
        return False


# Style Optimization Rules

class ExtractRepeatedStyles(OptimizationRule):
    """Extract repeated style patterns into reusable classes"""
    
    def name(self) -> str:
        return "extract_repeated_styles"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        # Check if there are duplicate style patterns
        style_patterns = {}
        for style_id, style_node in visual_ir.style_graph.styles.items():
            pattern = tuple(sorted(style_node.properties.items()))
            if pattern in style_patterns:
                return True
            style_patterns[pattern] = style_id
        return False
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        # Find duplicate styles and merge them
        style_patterns = {}
        duplicates = []
        
        for style_id, style_node in visual_ir.style_graph.styles.items():
            pattern = tuple(sorted((k.value, str(v)) for k, v in style_node.properties.items()))
            
            if pattern in style_patterns:
                duplicates.append((style_id, style_patterns[pattern]))
            else:
                style_patterns[pattern] = style_id
        
        # Update component style references
        for old_id, new_id in duplicates:
            for component in visual_ir.ui_components.values():
                if component.style_ref == old_id:
                    component.style_ref = new_id
        
        return len(duplicates) > 0


# Accessibility Optimization Rules

class AddMissingAriaLabels(OptimizationRule):
    """Add ARIA labels to interactive elements"""
    
    def name(self) -> str:
        return "add_missing_aria_labels"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        interactive_types = [ComponentType.BUTTON, ComponentType.INPUT, ComponentType.LINK]
        for component in visual_ir.ui_components.values():
            if component.type in interactive_types:
                if 'aria-label' not in component.accessibility_attrs:
                    return True
        return False
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        changed = False
        interactive_types = [ComponentType.BUTTON, ComponentType.INPUT, ComponentType.LINK]
        
        for component in visual_ir.ui_components.values():
            if component.type not in interactive_types:
                continue
            
            if 'aria-label' not in component.accessibility_attrs:
                # Generate label from text or type
                label = ""
                if 'text' in component.props:
                    label = str(component.props['text'].value if hasattr(component.props['text'], 'value') else component.props['text'])
                elif 'placeholder' in component.props:
                    label = str(component.props['placeholder'].value if hasattr(component.props['placeholder'], 'value') else component.props['placeholder'])
                else:
                    label = f"{component.type.value} element"
                
                component.accessibility_attrs['aria-label'] = label
                changed = True
        
        return changed


class EnsureKeyboardNavigation(OptimizationRule):
    """Ensure interactive elements are keyboard navigable"""
    
    def name(self) -> str:
        return "ensure_keyboard_navigation"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        interactive_types = [ComponentType.BUTTON, ComponentType.INPUT, ComponentType.LINK]
        for component in visual_ir.ui_components.values():
            if component.type in interactive_types:
                if 'tabindex' not in component.props:
                    return True
        return False
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        changed = False
        interactive_types = [ComponentType.BUTTON, ComponentType.INPUT, ComponentType.LINK]
        
        tabindex = 0
        for component in visual_ir.ui_components.values():
            if component.type in interactive_types:
                if 'tabindex' not in component.props and 'tabindex' not in component.accessibility_attrs:
                    component.accessibility_attrs['tabindex'] = str(tabindex)
                    tabindex += 1
                    changed = True
        
        return changed


# Security Optimization Rules

class SanitizeUserContent(OptimizationRule):
    """Ensure user content is sanitized"""
    
    def name(self) -> str:
        return "sanitize_user_content"
    
    def can_apply(self, visual_ir: VisualIRFunction) -> bool:
        return True  # Always applicable as a safety measure
    
    def apply(self, visual_ir: VisualIRFunction) -> bool:
        # Add sanitization contracts to components with user input
        changed = False
        
        for component in visual_ir.ui_components.values():
            if component.type == ComponentType.INPUT:
                if "sanitize_input" not in component.contracts:
                    component.contracts.append("sanitize_input")
                    changed = True
        
        return changed


class VisualOptimizer(Optimizer):
    """Optimizes Visual IR with visual-specific rules"""
    
    def __init__(self):
        super().__init__()
        
        # Register visual-specific optimization rules
        self.optimization_rules: List[OptimizationRule] = [
            # Structure optimization
            RemoveUnnecessaryWrappers(),
            FlattenNestedContainers(),
            CoalesceAdjacentTextNodes(),
            
            # Style optimization
            ExtractRepeatedStyles(),
            
            # Accessibility optimization
            AddMissingAriaLabels(),
            EnsureKeyboardNavigation(),
            
            # Security optimization
            SanitizeUserContent(),
        ]
    
    def optimize(
        self,
        visual_ir: VisualIRFunction,
        max_iterations: int = 10
    ) -> OptimizationResult:
        """Apply optimization rules to Visual IR"""
        applied_rules = []
        improvements = {
            "iterations": 0,
            "rules_applied": 0,
            "components_optimized": 0,
            "styles_optimized": 0
        }
        
        original_component_count = len(visual_ir.ui_components)
        original_style_count = len(visual_ir.style_graph.styles)
        
        # Apply rules iteratively
        for iteration in range(max_iterations):
            changed = False
            
            for rule in self.optimization_rules:
                if rule.can_apply(visual_ir):
                    if rule.apply(visual_ir):
                        applied_rules.append(rule.name())
                        improvements["rules_applied"] += 1
                        changed = True
            
            improvements["iterations"] = iteration + 1
            
            if not changed:
                break
        
        # Calculate improvements
        improvements["components_optimized"] = original_component_count - len(visual_ir.ui_components)
        improvements["styles_optimized"] = original_style_count - len(visual_ir.style_graph.styles)
        
        return OptimizationResult(
            optimized_ir=visual_ir,
            applied_rules=applied_rules,
            improvements=improvements
        )


if __name__ == "__main__":
    # Example usage
    from .visual_ir import create_component_node, create_style_node, VisualIRFunction
    from .visual_tsl import ComponentType
    
    # Create a UI with optimization opportunities
    visual_ir = VisualIRFunction(fn="optimization_example")
    
    # Create container with single child (should be optimized away)
    container = create_component_node(
        component_id="container_1",
        component_type=ComponentType.CONTAINER
    )
    
    button = create_component_node(
        component_id="btn_1",
        component_type=ComponentType.BUTTON,
        properties={"text": "Click Me"}
    )
    
    container.children_refs = ["btn_1"]
    
    visual_ir.add_component(container)
    visual_ir.add_component(button)
    visual_ir.root_component_id = "container_1"
    
    # Optimize
    optimizer = VisualOptimizer()
    result = optimizer.optimize(visual_ir)
    
    print("Optimization Results:")
    print(f"Rules applied: {result.applied_rules}")
    print(f"Improvements: {result.improvements}")

