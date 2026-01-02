# Visual Generation Integration Examples

## Overview

This document provides practical examples for integrating ICEBURG's Visual Generation System, including Visual TSL specification, multi-platform compilation (HTML5, React, SwiftUI), and security validation.

## Basic Setup

### 1. Initialize Visual Generation

```python
from iceburg.agents import VisualArchitect, VisualRedTeam, ComponentLibrary

# Initialize visual architect
visual_architect = VisualArchitect({
    "generation": {
        "enabled": True,
        "tsl_enabled": True,
        "multi_platform": True
    }
})

# Initialize visual red team
red_team = VisualRedTeam({
    "security": {
        "red_team_enabled": True,
        "vulnerability_testing": True,
        "contract_validation": True
    }
})

# Initialize component library
component_library = ComponentLibrary()
```

### 2. Basic Visual TSL Generation

```python
import asyncio
from iceburg.agents import VisualArchitect

async def basic_visual_generation():
    """Basic visual TSL generation example"""
    
    visual_architect = VisualArchitect()
    
    try:
        # Generate Visual TSL from user intent
        user_intent = "Create a modern dashboard with charts and data tables"
        tsl_spec = await visual_architect.generate_visual_tsl(user_intent)
        print(f"Generated Visual TSL:\n{tsl_spec}")
        
        # Validate contract
        validation = await visual_architect.validate_contract(tsl_spec)
        print(f"Contract validation: {validation}")
        
    except Exception as e:
        print(f"Error in visual generation: {e}")

# Run the example
asyncio.run(basic_visual_generation())
```

### 3. Multi-Platform Compilation

```python
import asyncio
from iceburg.agents import VisualArchitect

async def multi_platform_compilation():
    """Multi-platform compilation example"""
    
    visual_architect = VisualArchitect()
    
    try:
        # Generate Visual TSL
        user_intent = "Create a responsive e-commerce product page with image gallery and purchase button"
        tsl_spec = await visual_architect.generate_visual_tsl(user_intent)
        print(f"Generated TSL:\n{tsl_spec}")
        
        # Compile to HTML5
        print("\nCompiling to HTML5...")
        html5_result = await visual_architect.compile_to_html5(tsl_spec)
        print(f"HTML5 compilation result: {html5_result}")
        
        # Compile to React
        print("\nCompiling to React...")
        react_result = await visual_architect.compile_to_react(tsl_spec)
        print(f"React compilation result: {react_result}")
        
        # Compile to SwiftUI
        print("\nCompiling to SwiftUI...")
        swiftui_result = await visual_architect.compile_to_swiftui(tsl_spec)
        print(f"SwiftUI compilation result: {swiftui_result}")
        
    except Exception as e:
        print(f"Error in multi-platform compilation: {e}")

# Run the example
asyncio.run(multi_platform_compilation())
```

## Advanced Examples

### 4. Security Testing with Visual Red Team

```python
import asyncio
from iceburg.agents import VisualArchitect, VisualRedTeam

async def security_testing_example():
    """Security testing with visual red team example"""
    
    visual_architect = VisualArchitect()
    red_team = VisualRedTeam()
    
    try:
        # Generate visual output
        user_intent = "Create a secure login form with authentication"
        tsl_spec = await visual_architect.generate_visual_tsl(user_intent)
        visual_output = await visual_architect.compile_to_html5(tsl_spec)
        
        print("Generated visual output for security testing")
        
        # Test security vulnerabilities
        print("\nTesting security vulnerabilities...")
        security_results = await red_team.test_security_vulnerabilities(visual_output)
        print(f"Security test results: {security_results}")
        
        # Test contract violations
        print("\nTesting contract violations...")
        contract_results = await red_team.test_contract_violations(tsl_spec)
        print(f"Contract test results: {contract_results}")
        
        # Perform adversarial testing
        print("\nPerforming adversarial testing...")
        adversarial_results = await red_team.perform_adversarial_testing(visual_output)
        print(f"Adversarial test results: {adversarial_results}")
        
        # Validate accessibility
        print("\nValidating accessibility...")
        accessibility_results = await red_team.validate_accessibility(visual_output)
        print(f"Accessibility validation: {accessibility_results}")
        
    except Exception as e:
        print(f"Error in security testing: {e}")

# Run the example
asyncio.run(security_testing_example())
```

### 5. Component Library Management

```python
import asyncio
from iceburg.agents import ComponentLibrary

async def component_library_example():
    """Component library management example"""
    
    component_library = ComponentLibrary()
    
    try:
        # Get available components for HTML5
        html5_components = await component_library.get_available_components("html5")
        print(f"Available HTML5 components: {html5_components}")
        
        # Get available components for React
        react_components = await component_library.get_available_components("react")
        print(f"Available React components: {react_components}")
        
        # Get available components for SwiftUI
        swiftui_components = await component_library.get_available_components("swiftui")
        print(f"Available SwiftUI components: {swiftui_components}")
        
        # Get specific component
        button_component = await component_library.get_component("button", "html5")
        print(f"Button component: {button_component}")
        
        # Add custom component
        custom_component = {
            "type": "custom_card",
            "platform": "html5",
            "template": "<div class='custom-card'>{content}</div>",
            "styles": ".custom-card { border: 1px solid #ccc; padding: 20px; }",
            "props": ["content", "title", "description"]
        }
        
        success = await component_library.add_component(custom_component)
        if success:
            print("Custom component added successfully")
        
        # Update component
        updated_component = {
            "type": "custom_card",
            "platform": "html5",
            "template": "<div class='custom-card enhanced'>{content}</div>",
            "styles": ".custom-card { border: 2px solid #007bff; padding: 25px; }",
            "props": ["content", "title", "description", "image"]
        }
        
        success = await component_library.update_component("custom_card", updated_component)
        if success:
            print("Component updated successfully")
        
    except Exception as e:
        print(f"Error in component library management: {e}")

# Run the example
asyncio.run(component_library_example())
```

### 6. Layout Engine Integration

```python
import asyncio
from iceburg.agents import LayoutEngine

async def layout_engine_example():
    """Layout engine integration example"""
    
    layout_engine = LayoutEngine()
    
    try:
        # Define components for layout
        components = [
            {
                "id": "header",
                "type": "header",
                "content": "Dashboard Header",
                "position": "top"
            },
            {
                "id": "sidebar",
                "type": "sidebar",
                "content": "Navigation Menu",
                "position": "left"
            },
            {
                "id": "main_content",
                "type": "main",
                "content": "Main Content Area",
                "position": "center"
            },
            {
                "id": "footer",
                "type": "footer",
                "content": "Dashboard Footer",
                "position": "bottom"
            }
        ]
        
        # Define layout constraints
        constraints = {
            "responsive": True,
            "grid_system": "12-column",
            "breakpoints": {
                "mobile": "768px",
                "tablet": "1024px",
                "desktop": "1200px"
            },
            "spacing": "16px"
        }
        
        # Generate layout
        print("Generating layout...")
        layout = await layout_engine.generate_layout(components, constraints)
        print(f"Generated layout: {layout}")
        
        # Optimize layout
        print("\nOptimizing layout...")
        optimized_layout = await layout_engine.optimize_layout(layout)
        print(f"Optimized layout: {optimized_layout}")
        
        # Validate layout
        print("\nValidating layout...")
        validation = await layout_engine.validate_layout(optimized_layout)
        print(f"Layout validation: {validation}")
        
    except Exception as e:
        print(f"Error in layout engine: {e}")

# Run the example
asyncio.run(layout_engine_example())
```

## Application-Specific Examples

### 7. E-commerce Dashboard Generator

```python
import asyncio
from iceburg.agents import VisualArchitect, VisualRedTeam

class EcommerceDashboardGenerator:
    """E-commerce dashboard generator with visual interface"""
    
    def __init__(self):
        self.visual_architect = VisualArchitect()
        self.red_team = VisualRedTeam()
    
    async def generate_dashboard(self, requirements: dict):
        """Generate e-commerce dashboard"""
        
        try:
            # Generate Visual TSL for dashboard
            user_intent = f"""
            Create an e-commerce dashboard with:
            - Sales analytics charts
            - Product management table
            - Order tracking system
            - Customer insights
            - Revenue metrics
            - Responsive design for {requirements.get('devices', 'desktop, tablet, mobile')}
            """
            
            tsl_spec = await self.visual_architect.generate_visual_tsl(user_intent)
            print("Generated Visual TSL for e-commerce dashboard")
            
            # Compile to multiple platforms
            platforms = ["html5", "react", "swiftui"]
            results = {}
            
            for platform in platforms:
                print(f"\nCompiling to {platform}...")
                
                if platform == "html5":
                    result = await self.visual_architect.compile_to_html5(tsl_spec)
                elif platform == "react":
                    result = await self.visual_architect.compile_to_react(tsl_spec)
                elif platform == "swiftui":
                    result = await self.visual_architect.compile_to_swiftui(tsl_spec)
                
                results[platform] = result
                print(f"{platform} compilation complete")
            
            # Security testing
            print("\nPerforming security testing...")
            security_results = await self.red_team.test_security_vulnerabilities(results["html5"])
            print(f"Security test results: {security_results}")
            
            # Accessibility validation
            print("\nValidating accessibility...")
            accessibility_results = await self.red_team.validate_accessibility(results["html5"])
            print(f"Accessibility validation: {accessibility_results}")
            
            return {
                "status": "success",
                "tsl_spec": tsl_spec,
                "compilations": results,
                "security": security_results,
                "accessibility": accessibility_results
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Usage example
async def ecommerce_dashboard_example():
    generator = EcommerceDashboardGenerator()
    
    requirements = {
        "devices": "desktop, tablet, mobile",
        "features": ["analytics", "product_management", "order_tracking"],
        "security": "high",
        "accessibility": "wcag_aa"
    }
    
    result = await generator.generate_dashboard(requirements)
    print(f"Dashboard generation result: {result}")

# Run the example
asyncio.run(ecommerce_dashboard_example())
```

### 8. Mobile App Generator

```python
import asyncio
from iceburg.agents import VisualArchitect, ComponentLibrary

class MobileAppGenerator:
    """Mobile app generator with visual interface"""
    
    def __init__(self):
        self.visual_architect = VisualArchitect()
        self.component_library = ComponentLibrary()
    
    async def generate_mobile_app(self, app_spec: dict):
        """Generate mobile app from specification"""
        
        try:
            # Generate Visual TSL for mobile app
            user_intent = f"""
            Create a mobile app with:
            - {app_spec.get('screens', 'login, home, profile')} screens
            - {app_spec.get('features', 'navigation, forms, lists')} features
            - {app_spec.get('style', 'modern, clean')} design
            - {app_spec.get('platform', 'iOS and Android')} support
            """
            
            tsl_spec = await self.visual_architect.generate_visual_tsl(user_intent)
            print("Generated Visual TSL for mobile app")
            
            # Get mobile-specific components
            mobile_components = await self.component_library.get_available_components("swiftui")
            print(f"Available mobile components: {len(mobile_components)}")
            
            # Compile to SwiftUI
            swiftui_result = await self.visual_architect.compile_to_swiftui(tsl_spec)
            print("SwiftUI compilation complete")
            
            # Generate React Native version
            react_native_result = await self.visual_architect.compile_to_react(tsl_spec)
            print("React Native compilation complete")
            
            return {
                "status": "success",
                "tsl_spec": tsl_spec,
                "swiftui": swiftui_result,
                "react_native": react_native_result,
                "components": mobile_components
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Usage example
async def mobile_app_example():
    generator = MobileAppGenerator()
    
    app_spec = {
        "screens": "login, home, profile, settings",
        "features": "navigation, forms, lists, charts",
        "style": "modern, clean, minimalist",
        "platform": "iOS and Android"
    }
    
    result = await generator.generate_mobile_app(app_spec)
    print(f"Mobile app generation result: {result}")

# Run the example
asyncio.run(mobile_app_example())
```

### 9. Data Visualization Generator

```python
import asyncio
from iceburg.agents import VisualArchitect, LayoutEngine

class DataVisualizationGenerator:
    """Data visualization generator with visual interface"""
    
    def __init__(self):
        self.visual_architect = VisualArchitect()
        self.layout_engine = LayoutEngine()
    
    async def generate_visualization(self, data_spec: dict):
        """Generate data visualization from specification"""
        
        try:
            # Generate Visual TSL for data visualization
            user_intent = f"""
            Create a data visualization dashboard with:
            - {data_spec.get('chart_types', 'bar, line, pie')} charts
            - {data_spec.get('data_sources', 'CSV, API, database')} data sources
            - {data_spec.get('interactivity', 'hover, click, zoom')} features
            - {data_spec.get('responsive', 'desktop, tablet, mobile')} design
            - {data_spec.get('theme', 'modern, professional')} styling
            """
            
            tsl_spec = await self.visual_architect.generate_visual_tsl(user_intent)
            print("Generated Visual TSL for data visualization")
            
            # Compile to HTML5 with D3.js
            html5_result = await self.visual_architect.compile_to_html5(tsl_spec)
            print("HTML5 compilation complete")
            
            # Compile to React with Chart.js
            react_result = await self.visual_architect.compile_to_react(tsl_spec)
            print("React compilation complete")
            
            # Generate layout for visualization
            components = [
                {"id": "chart_1", "type": "chart", "content": "Sales Chart"},
                {"id": "chart_2", "type": "chart", "content": "Revenue Chart"},
                {"id": "table_1", "type": "table", "content": "Data Table"}
            ]
            
            constraints = {
                "responsive": True,
                "grid_system": "12-column",
                "chart_interactivity": True,
                "data_filtering": True
            }
            
            layout = await self.layout_engine.generate_layout(components, constraints)
            print("Layout generation complete")
            
            return {
                "status": "success",
                "tsl_spec": tsl_spec,
                "html5": html5_result,
                "react": react_result,
                "layout": layout
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Usage example
async def data_visualization_example():
    generator = DataVisualizationGenerator()
    
    data_spec = {
        "chart_types": "bar, line, pie, scatter",
        "data_sources": "CSV, API, database",
        "interactivity": "hover, click, zoom, filter",
        "responsive": "desktop, tablet, mobile",
        "theme": "modern, professional, dark"
    }
    
    result = await generator.generate_visualization(data_spec)
    print(f"Data visualization generation result: {result}")

# Run the example
asyncio.run(data_visualization_example())
```

## Error Handling and Best Practices

### 10. Robust Error Handling

```python
import asyncio
import logging
from iceburg.agents import VisualArchitect, VisualRedTeam

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def robust_visual_generation():
    """Robust visual generation with error handling"""
    
    visual_architect = VisualArchitect()
    red_team = VisualRedTeam()
    
    try:
        # Test visual generation with error handling
        test_cases = [
            "Create a simple button",
            "Create a complex dashboard with charts and tables",
            "Create a responsive e-commerce product page",
            "Create a mobile app with navigation and forms"
        ]
        
        for i, user_intent in enumerate(test_cases):
            try:
                print(f"\nTest case {i+1}: {user_intent}")
                
                # Generate Visual TSL
                tsl_spec = await visual_architect.generate_visual_tsl(user_intent)
                logger.info(f"Generated TSL for test case {i+1}")
                
                # Validate contract
                validation = await visual_architect.validate_contract(tsl_spec)
                if validation["valid"]:
                    logger.info(f"Contract validation passed for test case {i+1}")
                else:
                    logger.warning(f"Contract validation failed for test case {i+1}: {validation['errors']}")
                
                # Compile to HTML5
                html5_result = await visual_architect.compile_to_html5(tsl_spec)
                if html5_result["success"]:
                    logger.info(f"HTML5 compilation successful for test case {i+1}")
                else:
                    logger.error(f"HTML5 compilation failed for test case {i+1}: {html5_result['error']}")
                
                # Security testing
                security_results = await red_team.test_security_vulnerabilities(html5_result)
                if security_results["vulnerabilities"] == 0:
                    logger.info(f"Security test passed for test case {i+1}")
                else:
                    logger.warning(f"Security test found {security_results['vulnerabilities']} vulnerabilities for test case {i+1}")
                
            except Exception as e:
                logger.error(f"Error in test case {i+1}: {e}")
                continue
        
    except Exception as e:
        logger.error(f"Critical error in visual generation: {e}")

# Run the example
asyncio.run(robust_visual_generation())
```

### 11. Configuration Management

```python
import asyncio
import json
from iceburg.agents import VisualArchitect

class VisualConfigManager:
    """Configuration manager for visual generation"""
    
    def __init__(self, config_file: str = "visual_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                "generation": {
                    "enabled": True,
                    "tsl_enabled": True,
                    "multi_platform": True
                },
                "compilation": {
                    "html5": True,
                    "react": True,
                    "swiftui": True
                },
                "security": {
                    "red_team_enabled": True,
                    "vulnerability_testing": True,
                    "contract_validation": True
                }
            }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
    
    async def run_with_config(self):
        """Run visual generation with loaded configuration"""
        
        if not self.config["generation"]["enabled"]:
            print("Visual generation is disabled in configuration")
            return
        
        visual_architect = VisualArchitect(self.config)
        
        try:
            # Test visual generation based on configuration
            user_intent = "Create a simple dashboard with charts and tables"
            tsl_spec = await visual_architect.generate_visual_tsl(user_intent)
            print(f"Generated TSL: {tsl_spec}")
            
            # Compile based on configuration
            if self.config["compilation"]["html5"]:
                html5_result = await visual_architect.compile_to_html5(tsl_spec)
                print(f"HTML5 compilation: {html5_result}")
            
            if self.config["compilation"]["react"]:
                react_result = await visual_architect.compile_to_react(tsl_spec)
                print(f"React compilation: {react_result}")
            
            if self.config["compilation"]["swiftui"]:
                swiftui_result = await visual_architect.compile_to_swiftui(tsl_spec)
                print(f"SwiftUI compilation: {swiftui_result}")
            
        except Exception as e:
            print(f"Error in configured visual generation: {e}")

# Usage example
async def config_manager_example():
    config_manager = VisualConfigManager()
    await config_manager.run_with_config()

# Run the example
asyncio.run(config_manager_example())
```

## Summary

These examples demonstrate how to integrate ICEBURG's Visual Generation System with various applications, from basic TSL generation to advanced multi-platform compilation. The system provides:

- **Visual TSL specification** (TypeScript-like language for visual interfaces)
- **Multi-platform compilation** (HTML5, React, SwiftUI)
- **Security testing** (vulnerability detection, contract validation)
- **Component library management** (reusable components, templates)
- **Layout engine** (responsive design, grid systems)
- **Application-specific generators** (e-commerce, mobile apps, data visualization)

The examples show both basic usage and advanced integration patterns, with proper error handling and configuration management for production use.
