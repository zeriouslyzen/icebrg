"""
React Backend
Compiles Visual IR to React components with TypeScript support
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from ..visual_ir import VisualIRFunction, ComponentNode, StyleNode, StyleProperty
from ..visual_tsl import ComponentType, EventType


@dataclass
class ReactComponents:
    """Container for generated React artifacts"""
    components: str
    hooks: str
    styles: str
    typescript_defs: str
    package_json: str
    
    def save_to_directory(self, directory: Path) -> None:
        """Save all artifacts to a directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Create src directory
        src_dir = directory / "src"
        src_dir.mkdir(exist_ok=True)
        
        (src_dir / "App.tsx").write_text(self.components)
        (src_dir / "hooks.ts").write_text(self.hooks)
        (src_dir / "styles.module.css").write_text(self.styles)
        (src_dir / "types.ts").write_text(self.typescript_defs)
        (directory / "package.json").write_text(self.package_json)


class ReactBackend:
    """Compiles Visual IR to React components"""
    
    def __init__(self):
        self.component_counter = 0
    
    def compile(self, visual_ir: VisualIRFunction) -> ReactComponents:
        """Compile Visual IR to React artifacts"""
        # Generate main component
        components = self._generate_components(visual_ir)
        
        # Generate custom hooks
        hooks = self._generate_hooks(visual_ir)
        
        # Generate CSS modules
        styles = self._generate_css_modules(visual_ir)
        
        # Generate TypeScript definitions
        typescript_defs = self._generate_types(visual_ir)
        
        # Generate package.json
        package_json = self._generate_package_json(visual_ir)
        
        return ReactComponents(
            components=components,
            hooks=hooks,
            styles=styles,
            typescript_defs=typescript_defs,
            package_json=package_json
        )
    
    def _generate_components(self, visual_ir: VisualIRFunction) -> str:
        """Generate React component code"""
        imports = [
            "import React from 'react';",
            "import styles from './styles.module.css';",
            "import { useInteractions } from './hooks';"
        ]
        
        if not visual_ir.root_component_id:
            return "\n".join(imports) + "\n\nexport default function App() {\n  return <div>No components</div>;\n}\n"
        
        root_component = visual_ir.get_component(visual_ir.root_component_id)
        if not root_component:
            return "\n".join(imports) + "\n\nexport default function App() {\n  return <div>Root not found</div>;\n}\n"
        
        # Generate component tree
        jsx = self._component_to_jsx(root_component, visual_ir, depth=2)
        
        # Build full component
        component_code = f"""import React from 'react';
import styles from './styles.module.css';
import {{ useInteractions }} from './hooks';

export default function App() {{
  const {{ handleEvent }} = useInteractions();
  
  return (
{jsx}
  );
}}
"""
        
        return component_code
    
    def _component_to_jsx(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int = 0
    ) -> str:
        """Convert component node to JSX"""
        indent = "  " * depth
        
        # Map component types to React elements
        element_map = {
            ComponentType.BUTTON: "button",
            ComponentType.INPUT: "input",
            ComponentType.TEXT: "span",
            ComponentType.CONTAINER: "div",
            ComponentType.IMAGE: "img",
            ComponentType.LINK: "a",
            ComponentType.FORM: "form",
            ComponentType.LIST: "ul",
            ComponentType.CARD: "article",
            ComponentType.NAVIGATION: "nav",
            ComponentType.HEADER: "header",
            ComponentType.FOOTER: "footer",
            ComponentType.MODAL: "dialog",
            ComponentType.DROPDOWN: "select"
        }
        
        element = element_map.get(component.type, "div")
        
        # Build props
        props = []
        
        # ID
        if component.id:
            props.append(f'id="{component.id}"')
        
        # Class name from style
        if component.style_ref:
            # Convert style ref to CSS module class
            class_name = self._style_ref_to_class(component.style_ref)
            props.append(f'className={{styles.{class_name}}}')
        
        # Accessibility attributes
        for attr_name, attr_value in component.accessibility_attrs.items():
            attr_react = attr_name.replace('-', '_') if '-' in attr_name else attr_name
            props.append(f'{attr_react}="{attr_value}"')
        
        # Component properties
        for prop_name, prop_value in component.props.items():
            if prop_name == 'text':
                continue  # Handled as children
            
            if hasattr(prop_value, 'name'):
                # This is an IRValue object
                value = str(prop_value.name)
            else:
                # This is a regular value
                value = str(prop_value)
            
            # Boolean props
            if value.lower() in ['true', 'false']:
                props.append(f'{prop_name}={{{value.lower()}}}')
            else:
                props.append(f'{prop_name}="{value}"')
        
        # Event handlers
        for event_type, handler_id in component.event_handlers.items():
            event_name = f"on{event_type.value.capitalize()}"
            props.append(f'{event_name}={{() => handleEvent("{handler_id}")}}')
        
        props_str = " " + " ".join(props) if props else ""
        
        # Self-closing elements
        if element in ["input", "img", "br", "hr"] and not component.children_refs:
            return f"{indent}<{element}{props_str} />"
        
        # Build children
        children_parts = []
        
        # Text content
        if 'text' in component.props:
            text_prop = component.props['text']
            if hasattr(text_prop, 'name'):
                # This is an IRValue object
                text_value = str(text_prop.name)
            else:
                # This is a regular value
                text_value = str(text_prop)
            children_parts.append(f"{indent}  {text_value}")
        
        # Child components
        for child_id in component.children_refs:
            child_component = visual_ir.get_component(child_id)
            if child_component:
                child_jsx = self._component_to_jsx(child_component, visual_ir, depth + 1)
                children_parts.append(child_jsx)
        
        if children_parts:
            children = "\n".join(children_parts)
            return f"{indent}<{element}{props_str}>\n{children}\n{indent}</{element}>"
        else:
            return f"{indent}<{element}{props_str}></{element}>"
    
    def _style_ref_to_class(self, style_ref: str) -> str:
        """Convert style reference to CSS module class name"""
        # Generate valid CSS module class name
        return style_ref.replace('-', '_').replace(':', '_')
    
    def _generate_hooks(self, visual_ir: VisualIRFunction) -> str:
        """Generate custom React hooks"""
        hooks_code = """import { useCallback, useState } from 'react';

export function useInteractions() {
  const [state, setState] = useState({});
  
  const handleEvent = useCallback((handlerId: string) => {
    // Custom event handling logic
  }, []);
  
  return {
    handleEvent,
    state
  };
}
"""
        return hooks_code
    
    def _generate_css_modules(self, visual_ir: VisualIRFunction) -> str:
        """Generate CSS Modules"""
        css_parts = []
        
        # Global styles
        css_parts.append("""/* ICEBURG Generated Styles */
:global(*) {
  box-sizing: border-box;
}

:global(body) {
  margin: 0;
  font-family: system-ui, -apple-system, sans-serif;
  line-height: 1.5;
}
""")
        
        # Component styles
        for style_id, style_node in visual_ir.style_graph.styles.items():
            class_name = self._style_ref_to_class(style_id)
            css = self._style_node_to_css_module(class_name, style_node)
            css_parts.append(css)
        
        return "\n".join(css_parts)
    
    def _style_node_to_css_module(self, class_name: str, style_node: StyleNode) -> str:
        """Convert style node to CSS Module rule"""
        css_parts = []
        
        if style_node.properties:
            css_parts.append(f".{class_name} {{")
            for prop, value in style_node.properties.items():
                css_prop = prop.value
                css_value = str(value.value if hasattr(value, 'value') else value)
                css_parts.append(f"  {css_prop}: {css_value};")
            css_parts.append("}")
        
        # Pseudo-states
        for pseudo_state, props in style_node.pseudo_states.items():
            css_parts.append(f".{class_name}:{pseudo_state} {{")
            for prop, value in props.items():
                css_prop = prop.value
                css_value = str(value.value if hasattr(value, 'value') else value)
                css_parts.append(f"  {css_prop}: {css_value};")
            css_parts.append("}")
        
        return "\n".join(css_parts) + "\n"
    
    def _generate_types(self, visual_ir: VisualIRFunction) -> str:
        """Generate TypeScript type definitions"""
        types_code = """// ICEBURG Generated Types

export interface InteractionState {
  [key: string]: any;
}

export interface EventHandler {
  (handlerId: string): void;
}

export interface ComponentProps {
  id?: string;
  className?: string;
  children?: React.ReactNode;
}
"""
        return types_code
    
    def _generate_package_json(self, visual_ir: VisualIRFunction) -> str:
        """Generate package.json"""
        package = {
            "name": "iceburg-generated-app",
            "version": "1.0.0",
            "description": "Generated by ICEBURG Visual IR",
            "dependencies": {
                "react": "^18.2.0",
                "react-dom": "^18.2.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-dom": "^18.2.0",
                "typescript": "^5.0.0",
                "vite": "^4.3.0",
                "@vitejs/plugin-react": "^4.0.0"
            },
            "scripts": {
                "dev": "vite",
                "build": "tsc && vite build",
                "preview": "vite preview"
            }
        }
        return json.dumps(package, indent=2)


if __name__ == "__main__":
    # Example usage
    from ..visual_ir import create_component_node, create_style_node, VisualIRFunction
    from ..visual_tsl import ComponentType
    
    visual_ir = VisualIRFunction(fn="example_react_button")
    
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
            StyleProperty.BORDER_RADIUS: "4px"
        }
    )
    
    button.style_ref = "btn_style_1"
    visual_ir.add_component(button)
    visual_ir.style_graph.add_style(button_style)
    
    backend = ReactBackend()
    artifacts = backend.compile(visual_ir)
    
    print("React Component:")
    print(artifacts.components)
    print("\nHooks:")
    print(artifacts.hooks)
    print("\nCSS Modules:")
    print(artifacts.styles)

