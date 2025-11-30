"""
HTML5/CSS/JS Backend
Compiles Visual IR to vanilla HTML5, CSS3, and JavaScript
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path
import html
import json
import hashlib

from ..visual_ir import VisualIRFunction, ComponentNode, StyleNode, StyleProperty, InteractionEdge
from ..visual_tsl import ComponentType, EventType


@dataclass
class GeneratedArtifacts:
    """Container for generated HTML/CSS/JS artifacts"""
    html: str
    css: str
    js: str
    metadata: Dict[str, Any]
    
    def save_to_directory(self, directory: Path) -> None:
        """Save all artifacts to a directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        (directory / "index.html").write_text(self.html)
        (directory / "styles.css").write_text(self.css)
        (directory / "script.js").write_text(self.js)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
    
    def render_preview(self) -> str:
        """Render as a single HTML file for preview"""
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Preview</title>
    <style>
{self.css}
    </style>
</head>
<body>
{self.html}
<script>
{self.js}
</script>
</body>
</html>"""


class HTML5Backend:
    """Compiles Visual IR to HTML5/CSS/JS"""
    
    def __init__(self):
        self.component_counter = 0
        self.style_counter = 0
        self.generated_classes: Dict[str, str] = {}
    
    def compile(self, visual_ir: VisualIRFunction) -> GeneratedArtifacts:
        """Compile Visual IR to HTML/CSS/JS artifacts"""
        # Generate HTML
        html = self._generate_html(visual_ir)
        
        # Generate CSS
        css = self._generate_css(visual_ir)
        
        # Generate JavaScript
        js = self._generate_js(visual_ir)
        
        # Apply optimizations
        html = self._optimize_html(html)
        css = self._optimize_css(css)
        js = self._optimize_js(js)
        
        # Calculate metadata
        metadata = {
            "component_count": len(visual_ir.ui_components),
            "style_count": len(visual_ir.style_graph.styles),
            "interaction_count": len(visual_ir.interaction_graph.edges),
            "estimated_size_bytes": len(html) + len(css) + len(js),
            "estimated_size_kb": (len(html) + len(css) + len(js)) / 1024
        }
        
        return GeneratedArtifacts(
            html=html,
            css=css,
            js=js,
            metadata=metadata
        )
    
    def _generate_html(self, visual_ir: VisualIRFunction) -> str:
        """Generate semantic HTML5"""
        if not visual_ir.root_component_id:
            return "<div><!-- No components --></div>"
        
        root_component = visual_ir.get_component(visual_ir.root_component_id)
        if not root_component:
            return "<div><!-- Root component not found --></div>"
        
        return self._component_to_html(root_component, visual_ir)
    
    def _component_to_html(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int = 0
    ) -> str:
        """Convert a component node to HTML"""
        indent = "  " * depth
        
        # Map component types to HTML tags
        tag_map = {
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
        
        tag = tag_map.get(component.type, "div")
        
        # Build attributes
        attrs = []
        
        # ID
        if component.id:
            attrs.append(f'id="{html.escape(component.id)}"')
        
        # Classes (from style reference)
        if component.style_ref:
            class_name = self._get_or_create_class(component.style_ref)
            attrs.append(f'class="{class_name}"')
        
        # Accessibility attributes
        for attr_name, attr_value in component.accessibility_attrs.items():
            attrs.append(f'{html.escape(attr_name)}="{html.escape(attr_value)}"')
        
        # Component-specific properties
        props_html = self._generate_props_html(component, tag)
        if props_html:
            attrs.append(props_html)
        
        # Event handlers (as data attributes for JS to hook into)
        for event_type, handler_id in component.event_handlers.items():
            attrs.append(f'data-on{event_type.value}="{html.escape(handler_id)}"')
        
        attrs_str = " " + " ".join(attrs) if attrs else ""
        
        # Self-closing tags
        if tag in ["input", "img", "br", "hr"]:
            return f"{indent}<{tag}{attrs_str}>\n"
        
        # Build content
        content_parts = []
        
        # Text content from props
        if 'text' in component.props:
            text_prop = component.props['text']
            if hasattr(text_prop, 'name'):
                # This is an IRValue object
                text_value = str(text_prop.name)
            else:
                # This is a regular value
                text_value = str(text_prop)
            content_parts.append(html.escape(text_value))
        
        # Children
        for child_id in component.children_refs:
            child_component = visual_ir.get_component(child_id)
            if child_component:
                content_parts.append(self._component_to_html(child_component, visual_ir, depth + 1))
        
        if content_parts:
            content = "\n".join(content_parts)
            return f"{indent}<{tag}{attrs_str}>\n{content}\n{indent}</{tag}>\n"
        else:
            return f"{indent}<{tag}{attrs_str}></{tag}>\n"
    
    def _generate_props_html(self, component: ComponentNode, tag: str) -> str:
        """Generate HTML attributes from component properties"""
        attrs = []
        
        for prop_name, prop_value in component.props.items():
            if prop_name == 'text':
                continue  # Handled separately as content
            
            # Extract value from IRValue
            if hasattr(prop_value, 'name'):
                # This is an IRValue object
                value = str(prop_value.name)
            else:
                # This is a regular value
                value = str(prop_value)
            
            # Sanitize
            safe_value = html.escape(value)
            
            # Map prop names to HTML attributes
            if prop_name in ['type', 'placeholder', 'value', 'name', 'href', 'src', 'alt', 'title']:
                attrs.append(f'{prop_name}="{safe_value}"')
            elif prop_name == 'required' and value:
                attrs.append('required')
            elif prop_name == 'disabled' and value:
                attrs.append('disabled')
        
        return " ".join(attrs)
    
    def _get_or_create_class(self, style_ref: str) -> str:
        """Get or create a CSS class name for a style reference"""
        if style_ref in self.generated_classes:
            return self.generated_classes[style_ref]
        
        # Generate a short, semantic class name
        class_name = f"iceburg-{self._hash_string(style_ref)[:8]}"
        self.generated_classes[style_ref] = class_name
        return class_name
    
    def _hash_string(self, s: str) -> str:
        """Generate a hash for string"""
        return hashlib.md5(s.encode()).hexdigest()
    
    def _generate_css(self, visual_ir: VisualIRFunction) -> str:
        """Generate CSS stylesheet"""
        css_parts = []
        
        # Reset styles
        css_parts.append("""/* ICEBURG Generated Styles */
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  line-height: 1.5;
}
""")
        
        # Global theme variables
        if visual_ir.style_graph.theme_variables:
            css_parts.append("\n:root {")
            for var_name, var_value in visual_ir.style_graph.theme_variables.items():
                if hasattr(var_value, 'name'):
                    # This is an IRValue object
                    value = str(var_value.name)
                else:
                    # This is a regular value
                    value = str(var_value)
                css_parts.append(f"  --{var_name}: {value};")
            css_parts.append("}\n")
        
        # Component styles
        for style_id, style_node in visual_ir.style_graph.styles.items():
            class_name = self._get_or_create_class(style_id)
            css = self._style_node_to_css(class_name, style_node)
            css_parts.append(css)
        
        return "\n".join(css_parts)
    
    def _style_node_to_css(self, class_name: str, style_node: StyleNode) -> str:
        """Convert a style node to CSS rules"""
        css_parts = []
        
        # Main rule
        if style_node.properties:
            css_parts.append(f".{class_name} {{")
            for prop, value in style_node.properties.items():
                css_prop = prop.value  # Get CSS property name from enum
                if hasattr(value, 'name'):
                    # This is an IRValue object
                    css_value = str(value.name)
                else:
                    # This is a regular value
                    css_value = str(value)
                css_parts.append(f"  {css_prop}: {css_value};")
            css_parts.append("}")
        
        # Pseudo-states (:hover, :focus, etc.)
        for pseudo_state, props in style_node.pseudo_states.items():
            css_parts.append(f".{class_name}:{pseudo_state} {{")
            for prop, value in props.items():
                css_prop = prop.value
                css_value = str(value.value if hasattr(value, 'value') else value)
                css_parts.append(f"  {css_prop}: {css_value};")
            css_parts.append("}")
        
        # Media queries
        for media_query, props in style_node.media_queries.items():
            css_parts.append(f"@media {media_query} {{")
            css_parts.append(f"  .{class_name} {{")
            for prop, value in props.items():
                css_prop = prop.value
                css_value = str(value.value if hasattr(value, 'value') else value)
                css_parts.append(f"    {css_prop}: {css_value};")
            css_parts.append("  }")
            css_parts.append("}")
        
        return "\n".join(css_parts) + "\n"
    
    def _generate_js(self, visual_ir: VisualIRFunction) -> str:
        """Generate JavaScript for interactions"""
        js_parts = []
        
        # Preamble
        js_parts.append("""// ICEBURG Generated JavaScript
(function() {
  'use strict';
  
  // Initialize event handlers
  document.addEventListener('DOMContentLoaded', function() {
""")
        
        # Generate event handlers
        if visual_ir.interaction_graph.edges:
            for edge in visual_ir.interaction_graph.edges:
                handler_js = self._generate_event_handler(edge, visual_ir)
                js_parts.append(handler_js)
        
        # Closing
        js_parts.append("""  });
})();
""")
        
        return "\n".join(js_parts)
    
    def _generate_event_handler(
        self,
        edge: InteractionEdge,
        visual_ir: VisualIRFunction
    ) -> str:
        """Generate JavaScript event handler"""
        source_component = visual_ir.get_component(edge.source_id)
        if not source_component:
            return "    // Component not found\n"
        
        event_type = edge.event_type.value
        handler_fn = visual_ir.interaction_graph.handler_functions.get(edge.handler_fn_id)
        
        # Get the handler description or function
        handler_description = str(handler_fn) if handler_fn else "No-op handler"
        
        # For now, generate a simple event listener
        # In production, would compile the IRFunction to JavaScript
        js = f"""    // Handler for {edge.source_id}.{event_type}
    var element_{edge.source_id} = document.getElementById('{edge.source_id}');
    if (element_{edge.source_id}) {{
      element_{edge.source_id}.addEventListener('{event_type}', function(event) {{
        // {handler_description}
      }});
    }}
"""
        return js
    
    def _optimize_html(self, html: str) -> str:
        """Optimize HTML (minification, etc.)"""
        # Simple optimizations
        # Remove extra whitespace between tags
        import re
        html = re.sub(r'>\s+<', '><', html)
        return html.strip()
    
    def _optimize_css(self, css: str) -> str:
        """Optimize CSS"""
        # In production: minification, deduplication, critical CSS extraction
        return css.strip()
    
    def _optimize_js(self, js: str) -> str:
        """Optimize JavaScript"""
        # In production: minification, tree-shaking
        return js.strip()


if __name__ == "__main__":
    # Example usage
    from ..visual_ir import create_component_node, create_style_node, VisualIRFunction
    from ..visual_tsl import ComponentType
    
    # Create a simple button
    visual_ir = VisualIRFunction(fn="example_button")
    
    button = create_component_node(
        component_id="btn_1",
        component_type=ComponentType.BUTTON,
        properties={"text": "Click Me", "type": "button"}
    )
    button.accessibility_attrs = {"aria-label": "Click Me", "role": "button"}
    
    button_style = create_style_node(
        style_id="btn_style_1",
        properties={
            StyleProperty.BACKGROUND: "#007bff",
            StyleProperty.COLOR: "#ffffff",
            StyleProperty.PADDING: "10px 20px",
            StyleProperty.BORDER_RADIUS: "4px",
            StyleProperty.BORDER: "none",
            StyleProperty.FONT_SIZE: "16px"
        }
    )
    
    button.style_ref = "btn_style_1"
    visual_ir.add_component(button)
    visual_ir.style_graph.add_style(button_style)
    
    # Compile to HTML5
    backend = HTML5Backend()
    artifacts = backend.compile(visual_ir)
    
    print("HTML:")
    print(artifacts.html)
    print("\nCSS:")
    print(artifacts.css)
    print("\nJS:")
    print(artifacts.js)
    print("\nMetadata:")
    print(json.dumps(artifacts.metadata, indent=2))

