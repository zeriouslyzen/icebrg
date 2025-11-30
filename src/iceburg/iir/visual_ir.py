"""
Visual Intermediate Representation (Visual IR)
Extends ICEBURG's IIR system with visual/UI primitives
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import json

from .ir import IRFunction, IRValue, ScalarType, TensorType
from .visual_tsl import ComponentType, EventType, LayoutType


class StyleProperty(Enum):
    """CSS-like style properties"""
    COLOR = "color"
    BACKGROUND = "background"
    FONT_SIZE = "font-size"
    FONT_FAMILY = "font-family"
    PADDING = "padding"
    MARGIN = "margin"
    BORDER = "border"
    BORDER_RADIUS = "border-radius"
    BOX_SHADOW = "box-shadow"
    WIDTH = "width"
    HEIGHT = "height"
    DISPLAY = "display"
    POSITION = "position"
    Z_INDEX = "z-index"


@dataclass
class ComponentNode:
    """IR node representing a UI component"""
    id: str
    type: ComponentType
    props: Dict[str, IRValue]
    children_refs: List[str] = field(default_factory=list)
    style_ref: Optional[str] = None
    event_handlers: Dict[EventType, str] = field(default_factory=dict)  # Maps to IR function IDs
    accessibility_attrs: Dict[str, str] = field(default_factory=dict)
    contracts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "props": {k: str(v) for k, v in self.props.items()},
            "children_refs": self.children_refs,
            "style_ref": self.style_ref,
            "event_handlers": {k.value: v for k, v in self.event_handlers.items()},
            "accessibility_attrs": self.accessibility_attrs,
            "contracts": self.contracts
        }


@dataclass
class StyleNode:
    """IR node representing styles"""
    id: str
    properties: Dict[StyleProperty, IRValue]
    computed_properties: Dict[str, Any] = field(default_factory=dict)
    media_queries: Dict[str, Dict[StyleProperty, IRValue]] = field(default_factory=dict)
    pseudo_states: Dict[str, Dict[StyleProperty, IRValue]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "properties": {k.value: str(v) for k, v in self.properties.items()},
            "computed_properties": self.computed_properties,
            "media_queries": {
                mq: {k.value: str(v) for k, v in props.items()}
                for mq, props in self.media_queries.items()
            },
            "pseudo_states": {
                state: {k.value: str(v) for k, v in props.items()}
                for state, props in self.pseudo_states.items()
            }
        }


@dataclass
class StyleGraph:
    """Graph of all styles in the UI"""
    styles: Dict[str, StyleNode] = field(default_factory=dict)
    global_styles: Dict[StyleProperty, IRValue] = field(default_factory=dict)
    theme_variables: Dict[str, IRValue] = field(default_factory=dict)
    
    def add_style(self, style_node: StyleNode) -> None:
        """Add a style node to the graph"""
        self.styles[style_node.id] = style_node
    
    def get_style(self, style_id: str) -> Optional[StyleNode]:
        """Get a style node by ID"""
        return self.styles.get(style_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "styles": {sid: s.to_dict() for sid, s in self.styles.items()},
            "global_styles": {k.value: str(v) for k, v in self.global_styles.items()},
            "theme_variables": {k: str(v) for k, v in self.theme_variables.items()}
        }


@dataclass
class InteractionEdge:
    """Edge representing an interaction between components"""
    source_id: str
    target_id: str
    event_type: EventType
    handler_fn_id: str
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "event_type": self.event_type.value,
            "handler_fn_id": self.handler_fn_id,
            "conditions": self.conditions
        }


@dataclass
class InteractionGraph:
    """Graph of all interactions in the UI"""
    edges: List[InteractionEdge] = field(default_factory=list)
    handler_functions: Dict[str, IRFunction] = field(default_factory=dict)
    state_variables: Dict[str, IRValue] = field(default_factory=dict)
    
    def add_edge(self, edge: InteractionEdge) -> None:
        """Add an interaction edge"""
        self.edges.append(edge)
    
    def add_handler(self, fn_id: str, fn: IRFunction) -> None:
        """Add an event handler function"""
        self.handler_functions[fn_id] = fn
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges": [e.to_dict() for e in self.edges],
            "handler_functions": {fid: str(fn) for fid, fn in self.handler_functions.items()},
            "state_variables": {k: str(v) for k, v in self.state_variables.items()}
        }


@dataclass
class LayoutConstraint:
    """Layout constraint for component positioning"""
    component_id: str
    constraint_type: str  # "width", "height", "position", "flex", "grid"
    value: IRValue
    priority: int = 1
    conditions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "constraint_type": self.constraint_type,
            "value": str(self.value),
            "priority": self.priority,
            "conditions": self.conditions
        }


@dataclass
class LayoutGraph:
    """Graph representing layout relationships"""
    layout_type: LayoutType
    constraints: List[LayoutConstraint] = field(default_factory=list)
    hierarchy: Dict[str, List[str]] = field(default_factory=dict)  # parent -> children
    computed_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_constraint(self, constraint: LayoutConstraint) -> None:
        """Add a layout constraint"""
        self.constraints.append(constraint)
    
    def add_hierarchy(self, parent_id: str, child_id: str) -> None:
        """Add parent-child relationship"""
        if parent_id not in self.hierarchy:
            self.hierarchy[parent_id] = []
        self.hierarchy[parent_id].append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layout_type": self.layout_type.value,
            "constraints": [c.to_dict() for c in self.constraints],
            "hierarchy": self.hierarchy,
            "computed_positions": self.computed_positions
        }


class VisualIRFunction(IRFunction):
    """Extended IR function for visual generation"""
    
    def __init__(
        self,
        fn: str,
        params: Optional[Dict[str, Any]] = None,
        blocks: Optional[List[Dict[str, Any]]] = None,
        contracts: Optional[Dict[str, List[str]]] = None
    ):
        super().__init__(fn, params or {}, blocks or [], contracts or {})
        
        # Visual-specific fields
        self.ui_components: Dict[str, ComponentNode] = {}
        self.style_graph: StyleGraph = StyleGraph()
        self.interaction_graph: InteractionGraph = InteractionGraph()
        self.layout_graph: Optional[LayoutGraph] = None
        self.root_component_id: Optional[str] = None
        
        # Metadata
        self.accessibility_report: Dict[str, Any] = {}
        self.performance_metrics: Dict[str, Any] = {}
        self.security_validation: Dict[str, Any] = {}
    
    def add_component(self, component: ComponentNode) -> None:
        """Add a UI component to the IR"""
        self.ui_components[component.id] = component
        
        if self.root_component_id is None:
            self.root_component_id = component.id
    
    def get_component(self, component_id: str) -> Optional[ComponentNode]:
        """Get a component by ID"""
        return self.ui_components.get(component_id)
    
    def get_all_components(self) -> List[ComponentNode]:
        """Get all components in order"""
        if not self.root_component_id:
            return list(self.ui_components.values())
        
        # BFS traversal from root
        visited = set()
        result = []
        queue = [self.root_component_id]
        
        while queue:
            comp_id = queue.pop(0)
            if comp_id in visited:
                continue
            
            visited.add(comp_id)
            component = self.ui_components.get(comp_id)
            
            if component:
                result.append(component)
                queue.extend(component.children_refs)
        
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base_dict = super().to_dict() if hasattr(super(), 'to_dict') else {
            "fn": self.fn,
            "params": self.params,
            "blocks": self.blocks,
            "contracts": self.contracts
        }
        
        visual_dict = {
            "ui_components": {cid: c.to_dict() for cid, c in self.ui_components.items()},
            "style_graph": self.style_graph.to_dict(),
            "interaction_graph": self.interaction_graph.to_dict(),
            "layout_graph": self.layout_graph.to_dict() if self.layout_graph else None,
            "root_component_id": self.root_component_id,
            "accessibility_report": self.accessibility_report,
            "performance_metrics": self.performance_metrics,
            "security_validation": self.security_validation
        }
        
        return {**base_dict, **visual_dict}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)


# Helper functions for creating visual IR

def create_component_node(
    component_id: str,
    component_type: ComponentType,
    properties: Optional[Dict[str, Any]] = None,
    children: Optional[List[str]] = None
) -> ComponentNode:
    """Create a component node"""
    props = {}
    if properties:
        for key, value in properties.items():
            # Convert to IRValue (simplified for now)
            props[key] = IRValue(ScalarType("string"), value)
    
    return ComponentNode(
        id=component_id,
        type=component_type,
        props=props,
        children_refs=children or []
    )


def create_style_node(
    style_id: str,
    properties: Optional[Dict[StyleProperty, Any]] = None
) -> StyleNode:
    """Create a style node"""
    props = {}
    if properties:
        for prop, value in properties.items():
            # Convert to IRValue (simplified for now)
            props[prop] = IRValue(ScalarType("string"), str(value))
    
    return StyleNode(id=style_id, properties=props)


def create_interaction_edge(
    source_id: str,
    target_id: str,
    event_type: EventType,
    handler_fn_id: str
) -> InteractionEdge:
    """Create an interaction edge"""
    return InteractionEdge(
        source_id=source_id,
        target_id=target_id,
        event_type=event_type,
        handler_fn_id=handler_fn_id
    )


if __name__ == "__main__":
    # Example: Create a simple button with click handler
    visual_ir = VisualIRFunction(fn="simple_button_ui")
    
    # Create button component
    button = create_component_node(
        component_id="btn_1",
        component_type=ComponentType.BUTTON,
        properties={"text": "Click Me", "type": "button"}
    )
    
    # Create style for button
    button_style = create_style_node(
        style_id="btn_style_1",
        properties={
            StyleProperty.BACKGROUND: "#007bff",
            StyleProperty.COLOR: "#ffffff",
            StyleProperty.PADDING: "10px 20px",
            StyleProperty.BORDER_RADIUS: "4px"
        }
    )
    
    # Link style to button
    button.style_ref = "btn_style_1"
    
    # Add to IR
    visual_ir.add_component(button)
    visual_ir.style_graph.add_style(button_style)
    
    print("Visual IR Example:")
    print(visual_ir.to_json())

