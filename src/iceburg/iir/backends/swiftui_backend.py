"""
SwiftUI Backend
Compiles Visual IR to native macOS/iOS SwiftUI code
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..visual_ir import VisualIRFunction, ComponentNode, StyleNode, StyleProperty
from ..visual_tsl import ComponentType, EventType


@dataclass
class SwiftUICode:
    """Container for generated SwiftUI artifacts"""
    views: str
    view_models: str
    app_delegate: str
    
    def save_to_directory(self, directory: Path) -> None:
        """Save all artifacts to a directory"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        (directory / "ContentView.swift").write_text(self.views)
        (directory / "ViewModel.swift").write_text(self.view_models)
        (directory / "App.swift").write_text(self.app_delegate)


class SwiftUIBackend:
    """Compiles Visual IR to SwiftUI"""
    
    def __init__(self):
        self.component_counter = 0
    
    def compile(self, visual_ir: VisualIRFunction) -> SwiftUICode:
        """Compile Visual IR to SwiftUI artifacts"""
        views = self._generate_views(visual_ir)
        view_models = self._generate_view_models(visual_ir)
        app_delegate = self._generate_app_delegate()
        
        return SwiftUICode(
            views=views,
            view_models=view_models,
            app_delegate=app_delegate
        )
    
    def _generate_views(self, visual_ir: VisualIRFunction) -> str:
        """Generate SwiftUI views"""
        if not visual_ir.root_component_id:
            return """import SwiftUI

struct ContentView: View {
    var body: some View {
        Text("No components")
    }
}
"""
        
        root_component = visual_ir.get_component(visual_ir.root_component_id)
        if not root_component:
            return """import SwiftUI

struct ContentView: View {
    var body: some View {
        Text("Root component not found")
    }
}
"""
        
        # Generate view code
        view_body = self._component_to_swiftui(root_component, visual_ir, depth=2)
        
        view_code = f"""import SwiftUI

struct ContentView: View {{
    @StateObject private var viewModel = ViewModel()
    
    var body: some View {{
{view_body}
    }}
}}

struct ContentView_Previews: PreviewProvider {{
    static var previews: some View {{
        ContentView()
    }}
}}
"""
        
        return view_code
    
    def _component_to_swiftui(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int = 0
    ) -> str:
        """Convert component node to SwiftUI view"""
        indent = "    " * depth
        
        # Map component types to SwiftUI views
        view_map = {
            ComponentType.BUTTON: self._generate_button,
            ComponentType.TEXT: self._generate_text,
            ComponentType.INPUT: self._generate_textfield,
            ComponentType.CONTAINER: self._generate_vstack,
            ComponentType.IMAGE: self._generate_image,
            ComponentType.LINK: self._generate_link,
            ComponentType.LIST: self._generate_list
        }
        
        generator = view_map.get(component.type, self._generate_default)
        view_code = generator(component, visual_ir, depth)
        
        # Apply modifiers from style
        if component.style_ref:
            style_node = visual_ir.style_graph.get_style(component.style_ref)
            if style_node:
                modifiers = self._generate_style_modifiers(style_node, depth)
                view_code = f"{view_code}\n{modifiers}"
        
        return view_code
    
    def _generate_button(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI Button"""
        indent = "    " * depth
        
        # Get button text
        text = "Button"
        if 'text' in component.props:
            text = str(component.props['text'].value if hasattr(component.props['text'], 'value') else component.props['text'])
        
        # Get action
        action = "viewModel.handleAction()"
        if component.event_handlers:
            handler_id = list(component.event_handlers.values())[0]
            action = f'viewModel.handleEvent("{handler_id}")'
        
        return f'{indent}Button("{text}") {{\n{indent}    {action}\n{indent}}}'
    
    def _generate_text(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI Text"""
        indent = "    " * depth
        
        text = "Text"
        if 'text' in component.props:
            text = str(component.props['text'].value if hasattr(component.props['text'], 'value') else component.props['text'])
        
        return f'{indent}Text("{text}")'
    
    def _generate_textfield(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI TextField"""
        indent = "    " * depth
        
        placeholder = "Enter text"
        if 'placeholder' in component.props:
            placeholder = str(component.props['placeholder'].value if hasattr(component.props['placeholder'], 'value') else component.props['placeholder'])
        
        return f'{indent}TextField("{placeholder}", text: $viewModel.inputText)'
    
    def _generate_vstack(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI VStack (container)"""
        indent = "    " * depth
        
        lines = [f"{indent}VStack {{"]
        
        # Generate children
        for child_id in component.children_refs:
            child_component = visual_ir.get_component(child_id)
            if child_component:
                child_code = self._component_to_swiftui(child_component, visual_ir, depth + 1)
                lines.append(child_code)
        
        lines.append(f"{indent}}}")
        
        return "\n".join(lines)
    
    def _generate_image(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI Image"""
        indent = "    " * depth
        
        image_name = "placeholder"
        if 'src' in component.props:
            image_name = str(component.props['src'].value if hasattr(component.props['src'], 'value') else component.props['src'])
        
        return f'{indent}Image("{image_name}")'
    
    def _generate_link(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI Link"""
        indent = "    " * depth
        
        text = "Link"
        if 'text' in component.props:
            text = str(component.props['text'].value if hasattr(component.props['text'], 'value') else component.props['text'])
        
        url = "https://example.com"
        if 'href' in component.props:
            url = str(component.props['href'].value if hasattr(component.props['href'], 'value') else component.props['href'])
        
        return f'{indent}Link("{text}", destination: URL(string: "{url}")!)'
    
    def _generate_list(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate SwiftUI List"""
        indent = "    " * depth
        
        lines = [f"{indent}List {{"]
        
        # Generate children
        for child_id in component.children_refs:
            child_component = visual_ir.get_component(child_id)
            if child_component:
                child_code = self._component_to_swiftui(child_component, visual_ir, depth + 1)
                lines.append(child_code)
        
        lines.append(f"{indent}}}")
        
        return "\n".join(lines)
    
    def _generate_default(
        self,
        component: ComponentNode,
        visual_ir: VisualIRFunction,
        depth: int
    ) -> str:
        """Generate default view"""
        indent = "    " * depth
        return f'{indent}Text("Component: {component.type.value}")'
    
    def _generate_style_modifiers(self, style_node: StyleNode, depth: int) -> str:
        """Generate SwiftUI view modifiers from style"""
        indent = "    " * depth
        modifiers = []
        
        for prop, value in style_node.properties.items():
            css_value = str(value.value if hasattr(value, 'value') else value)
            
            # Map CSS properties to SwiftUI modifiers
            if prop == StyleProperty.COLOR:
                color = self._css_color_to_swift(css_value)
                modifiers.append(f"{indent}.foregroundColor({color})")
            
            elif prop == StyleProperty.BACKGROUND:
                color = self._css_color_to_swift(css_value)
                modifiers.append(f"{indent}.background({color})")
            
            elif prop == StyleProperty.PADDING:
                padding = self._css_padding_to_swift(css_value)
                modifiers.append(f"{indent}.padding({padding})")
            
            elif prop == StyleProperty.FONT_SIZE:
                size = css_value.replace('px', '')
                modifiers.append(f"{indent}.font(.system(size: {size}))")
            
            elif prop == StyleProperty.BORDER_RADIUS:
                radius = css_value.replace('px', '')
                modifiers.append(f"{indent}.cornerRadius({radius})")
        
        return "\n".join(modifiers)
    
    def _css_color_to_swift(self, css_color: str) -> str:
        """Convert CSS color to SwiftUI Color"""
        # Simple color mapping
        color_map = {
            "#ffffff": "Color.white",
            "#000000": "Color.black",
            "#007bff": "Color.blue",
            "#6c757d": "Color.gray",
            "#28a745": "Color.green",
            "#dc3545": "Color.red",
            "#ffc107": "Color.yellow"
        }
        
        css_color_lower = css_color.lower()
        if css_color_lower in color_map:
            return color_map[css_color_lower]
        
        # Try to parse hex color
        if css_color.startswith('#'):
            hex_color = css_color[1:]
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16) / 255.0
                g = int(hex_color[2:4], 16) / 255.0
                b = int(hex_color[4:6], 16) / 255.0
                return f"Color(red: {r:.3f}, green: {g:.3f}, blue: {b:.3f})"
        
        return "Color.primary"
    
    def _css_padding_to_swift(self, css_padding: str) -> str:
        """Convert CSS padding to SwiftUI padding"""
        # Parse padding value
        padding = css_padding.replace('px', '').strip()
        
        if ' ' in padding:
            # Multiple values - use first one for simplicity
            padding = padding.split()[0]
        
        return padding
    
    def _generate_view_models(self, visual_ir: VisualIRFunction) -> str:
        """Generate ViewModel for state management"""
        view_model = """import SwiftUI
import Combine

class ViewModel: ObservableObject {
    @Published var inputText: String = ""
    @Published var state: [String: Any] = [:]
    
    func handleEvent(_ handlerId: String) {
        print("Event handler triggered: \\(handlerId)")
        // Custom event handling logic
    }
    
    func handleAction() {
        print("Button action triggered")
    }
}
"""
        return view_model
    
    def _generate_app_delegate(self) -> str:
        """Generate main App file"""
        app_code = """import SwiftUI

@main
struct ICEBURGApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
"""
        return app_code


if __name__ == "__main__":
    # Example usage
    from ..visual_ir import create_component_node, create_style_node, VisualIRFunction
    from ..visual_tsl import ComponentType
    
    visual_ir = VisualIRFunction(fn="example_swiftui_button")
    
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
    
    backend = SwiftUIBackend()
    artifacts = backend.compile(visual_ir)
    
    print("SwiftUI View:")
    print(artifacts.views)
    print("\nViewModel:")
    print(artifacts.view_models)

