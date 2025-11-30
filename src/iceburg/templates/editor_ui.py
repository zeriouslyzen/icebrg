"""
Enhanced Editor UI Components

Provides:
- Line numbers
- Syntax highlighting
- Themes (dark/light)
- Code folding
- Search and replace
- Multi-cursor editing
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json


class EditorUIComponents:
    """Enhanced editor UI components for IDE applications."""
    
    def __init__(self):
        self.themes = {
            'dark': {
                'background': '#1e1e1e',
                'foreground': '#d4d4d4',
                'selection': '#264f78',
                'lineHighlight': '#2d2d30',
                'cursor': '#aeafad'
            },
            'light': {
                'background': '#ffffff',
                'foreground': '#000000',
                'selection': '#add6ff',
                'lineHighlight': '#f0f0f0',
                'cursor': '#000000'
            }
        }
    
    def generate_monaco_editor_html(self, 
                                  content: str = "",
                                  language: str = "swift",
                                  theme: str = "dark",
                                  features: List[str] = None) -> str:
        """
        Generate Monaco editor HTML with enhanced features.
        
        Args:
            content: Initial content
            language: Programming language
            theme: Editor theme (dark/light)
            features: List of editor features
            
        Returns:
            HTML content with Monaco editor
        """
        features = features or []
        
        # Monaco editor configuration
        config = {
            'value': content,
            'language': language,
            'theme': f'vs-{theme}',
            'automaticLayout': True,
            'minimap': {'enabled': 'minimap' in features},
            'lineNumbers': 'on',
            'wordWrap': 'on',
            'scrollBeyondLastLine': False,
            'fontSize': 14,
            'fontFamily': 'Monaco, Menlo, "Ubuntu Mono", monospace',
            'cursorBlinking': 'blink',
            'cursorSmoothCaretAnimation': True,
            'smoothScrolling': True,
            'mouseWheelZoom': True,
            'contextmenu': True,
            'selectOnLineNumbers': True,
            'roundedSelection': False,
            'readOnly': False,
            'scrollbar': {
                'vertical': 'auto',
                'horizontal': 'auto',
                'useShadows': True
            }
        }
        
        # Add feature-specific configurations
        if 'code-folding' in features:
            config['folding'] = True
            config['foldingStrategy'] = 'indentation'
        
        if 'multi-cursor' in features:
            config['multiCursorModifier'] = 'ctrlCmd'
        
        if 'search' in features:
            config['find'] = {
                'addExtraSpaceOnTop': True,
                'autoFindInSelection': 'never'
            }
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Monaco Editor</title>
    <style>
        body {{ 
            margin: 0; 
            padding: 0; 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }}
        #container {{ 
            height: 100vh; 
            width: 100vw; 
            background: {self.themes[theme]['background']};
        }}
        .editor-toolbar {{
            background: {self.themes[theme]['lineHighlight']};
            padding: 8px;
            border-bottom: 1px solid {self.themes[theme]['selection']};
            display: flex;
            gap: 8px;
        }}
        .toolbar-button {{
            background: transparent;
            border: 1px solid {self.themes[theme]['selection']};
            color: {self.themes[theme]['foreground']};
            padding: 4px 8px;
            border-radius: 4px;
            cursor: pointer;
        }}
        .toolbar-button:hover {{
            background: {self.themes[theme]['selection']};
        }}
    </style>
</head>
<body>
    <div id="container">
        <div class="editor-toolbar">
            <button class="toolbar-button" onclick="saveFile()">Save</button>
            <button class="toolbar-button" onclick="findInFile()">Find</button>
            <button class="toolbar-button" onclick="toggleTheme()">Theme</button>
            <button class="toolbar-button" onclick="formatCode()">Format</button>
        </div>
        <div id="editor" style="height: calc(100vh - 40px);"></div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs/loader.js"></script>
    <script>
        let editor;
        let currentTheme = '{theme}';
        
        require.config({{ paths: {{ 'vs': 'https://cdn.jsdelivr.net/npm/monaco-editor@0.44.0/min/vs' }} }});
        require(['vs/editor/editor.main'], function() {{
            editor = monaco.editor.create(document.getElementById('editor'), {json.dumps(config)});
            
            // Add keyboard shortcuts
            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, function() {{
                saveFile();
            }});
            
            editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyF, function() {{
                findInFile();
            }});
            
            // Send telemetry
            window.webkit.messageHandlers.telemetry.postMessage({{
                event: 'monaco_loaded',
                data: {{ 
                    language: '{language}',
                    theme: '{theme}',
                    features: {json.dumps(features)}
                }}
            }});
        }});
        
        function saveFile() {{
            const content = editor.getValue();
            window.webkit.messageHandlers.telemetry.postMessage({{
                event: 'file_saved',
                data: {{ content: content.substring(0, 100) + '...' }}
            }});
        }}
        
        function findInFile() {{
            editor.getAction('actions.find').run();
        }}
        
        function toggleTheme() {{
            currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
            monaco.editor.setTheme('vs-' + currentTheme);
            
            window.webkit.messageHandlers.telemetry.postMessage({{
                event: 'theme_changed',
                data: {{ theme: currentTheme }}
            }});
        }}
        
        function formatCode() {{
            editor.getAction('editor.action.formatDocument').run();
        }}
    </script>
</body>
</html>
"""
        
        return html
    
    def generate_syntax_highlighter_config(self, language: str) -> Dict[str, Any]:
        """
        Generate syntax highlighting configuration for a language.
        
        Args:
            language: Programming language
            
        Returns:
            Syntax highlighting configuration
        """
        configs = {
            'swift': {
                'keywords': ['import', 'struct', 'class', 'func', 'var', 'let', 'if', 'else', 'for', 'while', 'return'],
                'types': ['String', 'Int', 'Double', 'Bool', 'Array', 'Dictionary', 'Optional'],
                'strings': ['"', "'"],
                'comments': ['//', '/*', '*/']
            },
            'python': {
                'keywords': ['import', 'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'return', 'yield'],
                'types': ['str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set'],
                'strings': ['"', "'", '"""', "'''"],
                'comments': ['#']
            },
            'javascript': {
                'keywords': ['function', 'var', 'let', 'const', 'if', 'else', 'for', 'while', 'return', 'class'],
                'types': ['String', 'Number', 'Boolean', 'Array', 'Object'],
                'strings': ['"', "'", '`'],
                'comments': ['//', '/*', '*/']
            }
        }
        
        return configs.get(language, configs['swift'])
    
    def generate_editor_css(self, theme: str = "dark") -> str:
        """
        Generate CSS for editor styling.
        
        Args:
            theme: Theme name
            
        Returns:
            CSS content
        """
        colors = self.themes[theme]
        
        css = f"""
.editor-container {{
    background: {colors['background']};
    color: {colors['foreground']};
    font-family: Monaco, Menlo, "Ubuntu Mono", monospace;
    font-size: 14px;
    line-height: 1.5;
}}

.editor-line-numbers {{
    background: {colors['lineHighlight']};
    color: {colors['foreground']};
    padding: 0 8px;
    border-right: 1px solid {colors['selection']};
    user-select: none;
}}

.editor-content {{
    background: {colors['background']};
    color: {colors['foreground']};
    padding: 0 8px;
}}

.editor-selection {{
    background: {colors['selection']};
}}

.editor-cursor {{
    background: {colors['cursor']};
    width: 2px;
}}

.editor-toolbar {{
    background: {colors['lineHighlight']};
    border-bottom: 1px solid {colors['selection']};
    padding: 8px;
}}

.toolbar-button {{
    background: transparent;
    border: 1px solid {colors['selection']};
    color: {colors['foreground']};
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    margin-right: 4px;
}}

.toolbar-button:hover {{
    background: {colors['selection']};
}}

.toolbar-button.active {{
    background: {colors['selection']};
    color: {colors['background']};
}}
"""
        
        return css
