"""
Visual Task Specification Language (Visual TSL)
Declarative specifications for UI generation through IIR
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json
from pathlib import Path

from .tsl import TaskSpec, IOType, Budget


class ComponentType(Enum):
    """UI component types"""
    BUTTON = "button"
    INPUT = "input"
    TEXT = "text"
    CONTAINER = "container"
    IMAGE = "image"
    LINK = "link"
    FORM = "form"
    LIST = "list"
    CARD = "card"
    NAVIGATION = "navigation"
    HEADER = "header"
    FOOTER = "footer"
    MODAL = "modal"
    DROPDOWN = "dropdown"


class LayoutType(Enum):
    """Layout types"""
    FLEX = "flex"
    GRID = "grid"
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    FLOW = "flow"


class EventType(Enum):
    """Interaction event types"""
    CLICK = "click"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    CHANGE = "change"
    SUBMIT = "submit"
    KEYDOWN = "keydown"
    KEYUP = "keyup"
    SCROLL = "scroll"


@dataclass
class Interaction:
    """Interaction specification"""
    event_type: EventType
    handler_description: str
    target_selector: Optional[str] = None
    contracts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "handler_description": self.handler_description,
            "target_selector": self.target_selector,
            "contracts": self.contracts
        }


@dataclass
class StyleSpec:
    """Style specification"""
    colors: Dict[str, str] = field(default_factory=dict)
    fonts: Dict[str, str] = field(default_factory=dict)
    spacing: Dict[str, str] = field(default_factory=dict)
    borders: Dict[str, str] = field(default_factory=dict)
    shadows: Dict[str, str] = field(default_factory=dict)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LayoutSpec:
    """Layout specification"""
    layout_type: LayoutType
    direction: str = "row"  # row, column
    gap: str = "0"
    padding: str = "0"
    margin: str = "0"
    align_items: str = "flex-start"
    justify_content: str = "flex-start"
    wrap: bool = False
    grid_template: Optional[str] = None
    responsive_breakpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["layout_type"] = self.layout_type.value
        return result


@dataclass
class ThemeSpec:
    """Theme specification"""
    name: str = "default"
    primary_color: str = "#007bff"
    secondary_color: str = "#6c757d"
    background_color: str = "#ffffff"
    text_color: str = "#212529"
    font_family: str = "system-ui, -apple-system, sans-serif"
    font_size_base: str = "16px"
    border_radius: str = "4px"
    shadow_base: str = "0 2px 4px rgba(0,0,0,0.1)"
    custom_properties: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AccessibilitySpec:
    """Accessibility requirements"""
    wcag_level: str = "AA"  # A, AA, AAA
    min_contrast_ratio: float = 4.5
    keyboard_navigable: bool = True
    screen_reader_compatible: bool = True
    aria_labels_required: bool = True
    focus_indicators: bool = True
    alt_text_required: bool = True
    semantic_html: bool = True
    custom_requirements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceBudget:
    """Performance budget constraints"""
    max_load_time_ms: int = 100
    max_bundle_size_kb: int = 50
    max_dom_nodes: int = 1000
    max_reflows: int = 5
    max_repaints: int = 10
    cumulative_layout_shift: float = 0.1
    first_contentful_paint_ms: int = 100
    largest_contentful_paint_ms: int = 250
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VisualComponent:
    """Visual component specification"""
    component_type: ComponentType
    id: Optional[str] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    style: Optional[StyleSpec] = None
    children: List['VisualComponent'] = field(default_factory=list)
    interactions: List[Interaction] = field(default_factory=list)
    contracts: List[str] = field(default_factory=list)
    accessibility: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_type": self.component_type.value,
            "id": self.id,
            "properties": self.properties,
            "style": self.style.to_dict() if self.style else None,
            "children": [c.to_dict() for c in self.children],
            "interactions": [i.to_dict() for i in self.interactions],
            "contracts": self.contracts,
            "accessibility": self.accessibility
        }


@dataclass
class UISpec(TaskSpec):
    """Complete UI specification extending TaskSpec"""
    # Visual-specific fields
    components: List[VisualComponent] = field(default_factory=list)
    layout: Optional[LayoutSpec] = None
    theme: Optional[ThemeSpec] = None
    interactions: List[Interaction] = field(default_factory=list)
    accessibility_requirements: Optional[AccessibilitySpec] = None
    performance_budget: Optional[PerformanceBudget] = None
    security_contracts: List[str] = field(default_factory=list)
    
    def __init__(self, name: str, components: List[VisualComponent] = None, 
                 layout: Optional[LayoutSpec] = None, theme: Optional[ThemeSpec] = None,
                 interactions: List[Interaction] = None, 
                 accessibility_requirements: Optional[AccessibilitySpec] = None,
                 performance_budget: Optional[PerformanceBudget] = None,
                 security_contracts: List[str] = None, **kwargs):
        """Initialize UISpec with default inputs/outputs"""
        # Set defaults
        inputs = kwargs.get('inputs', [IOType(name="ui_description", type="string")])
        outputs = kwargs.get('outputs', [IOType(name="ui_artifacts", type="object")])
        
        # Call parent constructor
        super().__init__(
            name=name,
            inputs=inputs,
            outputs=outputs,
            pre=kwargs.get('pre', []),
            post=kwargs.get('post', []),
            invariants=kwargs.get('invariants', []),
            effects=kwargs.get('effects', []),
            budgets=kwargs.get('budgets', Budget())
        )
        
        # Set visual-specific fields
        self.components = components or []
        self.layout = layout
        self.theme = theme
        self.interactions = interactions or []
        self.accessibility_requirements = accessibility_requirements
        self.performance_budget = performance_budget
        self.security_contracts = security_contracts or []
    
    # Metadata
    version: str = "1.0"
    author: str = "ICEBURG"
    created_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        base_dict = super().to_dict() if hasattr(super(), 'to_dict') else {}
        
        visual_dict = {
            "name": self.name,
            "version": self.version,
            "author": self.author,
            "created_at": self.created_at,
            "components": [c.to_dict() for c in self.components],
            "layout": self.layout.to_dict() if self.layout else None,
            "theme": self.theme.to_dict() if self.theme else None,
            "interactions": [i.to_dict() for i in self.interactions],
            "accessibility_requirements": self.accessibility_requirements.to_dict() if self.accessibility_requirements else None,
            "performance_budget": self.performance_budget.to_dict() if self.performance_budget else None,
            "security_contracts": self.security_contracts
        }
        
        return {**base_dict, **visual_dict}
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save specification to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(self.to_json())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UISpec':
        """Create UISpec from dictionary"""
        # Parse components
        components = []
        for comp_data in data.get("components", []):
            component = cls._parse_component(comp_data)
            components.append(component)
        
        # Parse layout
        layout = None
        if data.get("layout"):
            layout_data = data["layout"]
            layout = LayoutSpec(
                layout_type=LayoutType(layout_data["layout_type"]),
                direction=layout_data.get("direction", "row"),
                gap=layout_data.get("gap", "0"),
                padding=layout_data.get("padding", "0"),
                margin=layout_data.get("margin", "0"),
                align_items=layout_data.get("align_items", "flex-start"),
                justify_content=layout_data.get("justify_content", "flex-start"),
                wrap=layout_data.get("wrap", False),
                grid_template=layout_data.get("grid_template"),
                responsive_breakpoints=layout_data.get("responsive_breakpoints", {})
            )
        
        # Parse theme
        theme = None
        if data.get("theme"):
            theme = ThemeSpec(**data["theme"])
        
        # Parse interactions
        interactions = []
        for int_data in data.get("interactions", []):
            interaction = Interaction(
                event_type=EventType(int_data["event_type"]),
                handler_description=int_data["handler_description"],
                target_selector=int_data.get("target_selector"),
                contracts=int_data.get("contracts", [])
            )
            interactions.append(interaction)
        
        # Parse accessibility requirements
        accessibility_requirements = None
        if data.get("accessibility_requirements"):
            accessibility_requirements = AccessibilitySpec(**data["accessibility_requirements"])
        
        # Parse performance budget
        performance_budget = None
        if data.get("performance_budget"):
            performance_budget = PerformanceBudget(**data["performance_budget"])
        
        return cls(
            name=data.get("name", "unnamed_ui"),
            components=components,
            layout=layout,
            theme=theme,
            interactions=interactions,
            accessibility_requirements=accessibility_requirements,
            performance_budget=performance_budget,
            security_contracts=data.get("security_contracts", []),
            version=data.get("version", "1.0"),
            author=data.get("author", "ICEBURG"),
            created_at=data.get("created_at")
        )
    
    @classmethod
    def _parse_component(cls, data: Dict[str, Any]) -> VisualComponent:
        """Parse component from dictionary"""
        # Parse style
        style = None
        if data.get("style"):
            style = StyleSpec(**data["style"])
        
        # Parse children recursively
        children = []
        for child_data in data.get("children", []):
            child = cls._parse_component(child_data)
            children.append(child)
        
        # Parse interactions
        interactions = []
        for int_data in data.get("interactions", []):
            interaction = Interaction(
                event_type=EventType(int_data["event_type"]),
                handler_description=int_data["handler_description"],
                target_selector=int_data.get("target_selector"),
                contracts=int_data.get("contracts", [])
            )
            interactions.append(interaction)
        
        return VisualComponent(
            component_type=ComponentType(data["component_type"]),
            id=data.get("id"),
            properties=data.get("properties", {}),
            style=style,
            children=children,
            interactions=interactions,
            contracts=data.get("contracts", []),
            accessibility=data.get("accessibility", {})
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> 'UISpec':
        """Create UISpec from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'UISpec':
        """Load specification from file"""
        filepath = Path(filepath)
        json_str = filepath.read_text()
        return cls.from_json(json_str)


# Convenience functions for common UI patterns
def create_button_spec(
    text: str,
    on_click: str,
    id: Optional[str] = None,
    style: Optional[Dict[str, str]] = None
) -> VisualComponent:
    """Create a button component specification"""
    return VisualComponent(
        component_type=ComponentType.BUTTON,
        id=id,
        properties={"text": text, "type": "button"},
        style=StyleSpec(custom=style or {}),
        interactions=[Interaction(
            event_type=EventType.CLICK,
            handler_description=on_click
        )],
        accessibility={"aria-label": text, "role": "button"}
    )


def create_input_spec(
    input_type: str,
    placeholder: str = "",
    id: Optional[str] = None,
    required: bool = False
) -> VisualComponent:
    """Create an input component specification"""
    return VisualComponent(
        component_type=ComponentType.INPUT,
        id=id,
        properties={
            "type": input_type,
            "placeholder": placeholder,
            "required": required
        },
        accessibility={"aria-label": placeholder, "role": "textbox"}
    )


def create_container_spec(
    children: List[VisualComponent],
    layout_type: LayoutType = LayoutType.FLEX,
    id: Optional[str] = None
) -> VisualComponent:
    """Create a container component specification"""
    return VisualComponent(
        component_type=ComponentType.CONTAINER,
        id=id,
        children=children,
        properties={"layout": layout_type.value}
    )


if __name__ == "__main__":
    # Example usage
    button = create_button_spec(
        text="Click Me",
        on_click="Alert user",
        id="submit-button"
    )
    
    ui_spec = UISpec(
        name="simple_button_example",
        components=[button],
        theme=ThemeSpec(),
        accessibility_requirements=AccessibilitySpec(),
        performance_budget=PerformanceBudget(),
        security_contracts=[
            "no_external_scripts",
            "no_inline_eval",
            "sanitized_content"
        ]
    )
    
    print("Visual TSL Example:")
    print(ui_spec.to_json())

