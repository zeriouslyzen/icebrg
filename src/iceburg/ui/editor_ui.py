"""
Editor UI for ICEBURG
Implements editor UI with line numbers, themes, and syntax highlighting.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EditorTheme:
    """Editor theme configuration."""
    name: str
    background: str
    foreground: str
    selection: str
    cursor: str
    line_numbers: str
    keywords: str
    strings: str
    comments: str
    functions: str
    variables: str
    operators: str


@dataclass
class EditorSettings:
    """Editor settings configuration."""
    font_family: str = "Monaco"
    font_size: int = 14
    tab_size: int = 4
    line_numbers: bool = True
    word_wrap: bool = True
    auto_indent: bool = True
    syntax_highlighting: bool = True
    bracket_matching: bool = True
    minimap: bool = True
    theme: str = "default"


class EditorUI:
    """
    Editor UI for ICEBURG with advanced features.
    
    Features:
    - Line numbers
    - Syntax highlighting
    - Themes
    - Bracket matching
    - Auto-indentation
    - Minimap
    - Find and replace
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize Editor UI.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.settings = EditorSettings()
        self.themes = self._load_themes()
        self.current_theme = self.themes.get("default")
        
        # Editor state
        self.content = ""
        self.cursor_position = (0, 0)
        self.selection_start = None
        self.selection_end = None
        self.undo_stack = []
        self.redo_stack = []
    
    def _load_themes(self) -> Dict[str, EditorTheme]:
        """Load editor themes."""
        themes = {}
        
        # Default theme
        themes["default"] = EditorTheme(
            name="Default",
            background="#FFFFFF",
            foreground="#000000",
            selection="#0078D4",
            cursor="#000000",
            line_numbers="#6A6A6A",
            keywords="#0000FF",
            strings="#A31515",
            comments="#008000",
            functions="#795E26",
            variables="#001080",
            operators="#000000"
        )
        
        # Dark theme
        themes["dark"] = EditorTheme(
            name="Dark",
            background="#1E1E1E",
            foreground="#D4D4D4",
            selection="#264F78",
            cursor="#FFFFFF",
            line_numbers="#858585",
            keywords="#569CD6",
            strings="#CE9178",
            comments="#6A9955",
            functions="#DCDCAA",
            variables="#9CDCFE",
            operators="#D4D4D4"
        )
        
        # Monokai theme
        themes["monokai"] = EditorTheme(
            name="Monokai",
            background="#272822",
            foreground="#F8F8F2",
            selection="#49483E",
            cursor="#F8F8F2",
            line_numbers="#75715E",
            keywords="#F92672",
            strings="#E6DB74",
            comments="#75715E",
            functions="#A6E22E",
            variables="#F8F8F2",
            operators="#F92672"
        )
        
        return themes
    
    def set_theme(self, theme_name: str) -> bool:
        """
        Set editor theme.
        
        Args:
            theme_name: Name of the theme
            
        Returns:
            True if theme set successfully
        """
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
            self.settings.theme = theme_name
            return True
        return False
    
    def set_content(self, content: str):
        """Set editor content."""
        self.content = content
        self._update_undo_stack()
    
    def get_content(self) -> str:
        """Get editor content."""
        return self.content
    
    def insert_text(self, text: str, position: Tuple[int, int] = None):
        """Insert text at position."""
        if position is None:
            position = self.cursor_position
        
        lines = self.content.split('\n')
        row, col = position
        
        if row < len(lines):
            lines[row] = lines[row][:col] + text + lines[row][col:]
        else:
            # Add new lines if needed
            while len(lines) <= row:
                lines.append("")
            lines[row] = text
        
        self.content = '\n'.join(lines)
        self._update_undo_stack()
    
    def delete_text(self, start: Tuple[int, int], end: Tuple[int, int]):
        """Delete text between positions."""
        lines = self.content.split('\n')
        start_row, start_col = start
        end_row, end_col = end
        
        if start_row == end_row:
            # Delete within same line
            lines[start_row] = lines[start_row][:start_col] + lines[start_row][end_col:]
        else:
            # Delete across multiple lines
            lines[start_row] = lines[start_row][:start_col] + lines[end_row][end_col:]
            # Remove lines in between
            del lines[start_row + 1:end_row + 1]
        
        self.content = '\n'.join(lines)
        self._update_undo_stack()
    
    def get_line_numbers(self) -> List[str]:
        """Get line numbers for display."""
        if not self.settings.line_numbers:
            return []
        
        lines = self.content.split('\n')
        max_width = len(str(len(lines)))
        
        return [str(i + 1).rjust(max_width) for i in range(len(lines))]
    
    def get_syntax_highlighted_lines(self) -> List[List[Dict[str, Any]]]:
        """Get syntax highlighted lines."""
        if not self.settings.syntax_highlighting:
            return []
        
        lines = self.content.split('\n')
        highlighted_lines = []
        
        for line in lines:
            highlighted_line = self._highlight_line(line)
            highlighted_lines.append(highlighted_line)
        
        return highlighted_lines
    
    def _highlight_line(self, line: str) -> List[Dict[str, Any]]:
        """Highlight a single line."""
        tokens = []
        i = 0
        
        while i < len(line):
            # Keywords
            for keyword in ["if", "else", "for", "while", "def", "class", "import", "from", "return", "True", "False", "None"]:
                if line[i:].startswith(keyword) and (i + len(keyword) >= len(line) or not line[i + len(keyword)].isalnum()):
                    tokens.append({
                        "text": keyword,
                        "type": "keyword",
                        "color": self.current_theme.keywords
                    })
                    i += len(keyword)
                    break
            else:
                # Strings
                if line[i] in ['"', "'"]:
                    quote = line[i]
                    start = i
                    i += 1
                    while i < len(line) and line[i] != quote:
                        if line[i] == '\\' and i + 1 < len(line):
                            i += 2  # Skip escaped character
                        else:
                            i += 1
                    if i < len(line):
                        i += 1  # Skip closing quote
                    tokens.append({
                        "text": line[start:i],
                        "type": "string",
                        "color": self.current_theme.strings
                    })
                # Comments
                elif line[i:i+2] == "//" or line[i:i+2] == "#":
                    tokens.append({
                        "text": line[i:],
                        "type": "comment",
                        "color": self.current_theme.comments
                    })
                    break
                # Functions
                elif line[i].isalpha() and (i == 0 or not line[i-1].isalnum()):
                    start = i
                    while i < len(line) and (line[i].isalnum() or line[i] == '_'):
                        i += 1
                    if i < len(line) and line[i] == '(':
                        tokens.append({
                            "text": line[start:i],
                            "type": "function",
                            "color": self.current_theme.functions
                        })
                    else:
                        tokens.append({
                            "text": line[start:i],
                            "type": "variable",
                            "color": self.current_theme.variables
                        })
                # Operators
                elif line[i] in "+-*/=<>!&|":
                    tokens.append({
                        "text": line[i],
                        "type": "operator",
                        "color": self.current_theme.operators
                    })
                    i += 1
                # Default
                else:
                    tokens.append({
                        "text": line[i],
                        "type": "default",
                        "color": self.current_theme.foreground
                    })
                    i += 1
        
        return tokens
    
    def find_text(self, text: str, start_position: Tuple[int, int] = None) -> Optional[Tuple[int, int]]:
        """Find text in content."""
        if start_position is None:
            start_position = (0, 0)
        
        lines = self.content.split('\n')
        start_row, start_col = start_position
        
        for row in range(start_row, len(lines)):
            line = lines[row]
            if row == start_row:
                search_text = line[start_col:]
            else:
                search_text = line
            
            pos = search_text.find(text)
            if pos != -1:
                if row == start_row:
                    return (row, start_col + pos)
                else:
                    return (row, pos)
        
        return None
    
    def replace_text(self, old_text: str, new_text: str, start_position: Tuple[int, int] = None) -> int:
        """Replace text in content."""
        if start_position is None:
            start_position = (0, 0)
        
        lines = self.content.split('\n')
        start_row, start_col = start_position
        replacements = 0
        
        for row in range(start_row, len(lines)):
            line = lines[row]
            if row == start_row:
                search_text = line[start_col:]
                pos = search_text.find(old_text)
                if pos != -1:
                    lines[row] = line[:start_col] + search_text.replace(old_text, new_text)
                    replacements += search_text.count(old_text)
            else:
                if old_text in line:
                    lines[row] = line.replace(old_text, new_text)
                    replacements += line.count(old_text)
        
        self.content = '\n'.join(lines)
        self._update_undo_stack()
        return replacements
    
    def get_bracket_matches(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get matching brackets for position."""
        if not self.settings.bracket_matching:
            return []
        
        lines = self.content.split('\n')
        row, col = position
        
        if row >= len(lines):
            return []
        
        line = lines[row]
        if col >= len(line):
            return []
        
        char = line[col]
        bracket_pairs = {'(': ')', '[': ']', '{': '}'}
        
        if char not in bracket_pairs and char not in bracket_pairs.values():
            return []
        
        # Find matching bracket
        if char in bracket_pairs:
            # Opening bracket - find closing
            target = bracket_pairs[char]
            depth = 1
            i = col + 1
            
            while i < len(line) and depth > 0:
                if line[i] == char:
                    depth += 1
                elif line[i] == target:
                    depth -= 1
                i += 1
            
            if depth == 0:
                return [(row, i - 1)]
        else:
            # Closing bracket - find opening
            target = None
            for open_bracket, close_bracket in bracket_pairs.items():
                if char == close_bracket:
                    target = open_bracket
                    break
            
            if target:
                depth = 1
                i = col - 1
                
                while i >= 0 and depth > 0:
                    if line[i] == char:
                        depth += 1
                    elif line[i] == target:
                        depth -= 1
                    i -= 1
                
                if depth == 0:
                    return [(row, i + 1)]
        
        return []
    
    def get_minimap_data(self) -> List[str]:
        """Get minimap data for display."""
        if not self.settings.minimap:
            return []
        
        lines = self.content.split('\n')
        minimap_lines = []
        
        for line in lines:
            # Create minimap representation
            minimap_line = ""
            for char in line:
                if char.isspace():
                    minimap_line += " "
                elif char.isalnum():
                    minimap_line += "█"
                else:
                    minimap_line += "░"
            
            minimap_lines.append(minimap_line)
        
        return minimap_lines
    
    def undo(self) -> bool:
        """Undo last action."""
        if not self.undo_stack:
            return False
        
        # Move current state to redo stack
        self.redo_stack.append(self.content)
        
        # Restore previous state
        self.content = self.undo_stack.pop()
        return True
    
    def redo(self) -> bool:
        """Redo last undone action."""
        if not self.redo_stack:
            return False
        
        # Move current state to undo stack
        self.undo_stack.append(self.content)
        
        # Restore next state
        self.content = self.redo_stack.pop()
        return True
    
    def _update_undo_stack(self):
        """Update undo stack with current state."""
        self.undo_stack.append(self.content)
        
        # Limit undo stack size
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        
        # Clear redo stack when new action is performed
        self.redo_stack.clear()
    
    def get_editor_html(self) -> str:
        """Generate HTML for editor display."""
        lines = self.content.split('\n')
        line_numbers = self.get_line_numbers()
        highlighted_lines = self.get_syntax_highlighted_lines()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ICEBURG Editor</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            font-family: {self.settings.font_family}, monospace;
            font-size: {self.settings.font_size}px;
            background-color: {self.current_theme.background};
            color: {self.current_theme.foreground};
        }}
        
        .editor-container {{
            display: flex;
            height: 100vh;
        }}
        
        .line-numbers {{
            background-color: {self.current_theme.background};
            color: {self.current_theme.line_numbers};
            padding: 10px 5px;
            border-right: 1px solid {self.current_theme.line_numbers};
            text-align: right;
            user-select: none;
            min-width: 50px;
        }}
        
        .editor-content {{
            flex: 1;
            padding: 10px;
            overflow: auto;
        }}
        
        .line {{
            white-space: pre;
            min-height: {self.settings.font_size + 4}px;
        }}
        
        .keyword {{ color: {self.current_theme.keywords}; }}
        .string {{ color: {self.current_theme.strings}; }}
        .comment {{ color: {self.current_theme.comments}; }}
        .function {{ color: {self.current_theme.functions}; }}
        .variable {{ color: {self.current_theme.variables}; }}
        .operator {{ color: {self.current_theme.operators}; }}
    </style>
</head>
<body>
    <div class="editor-container">
        <div class="line-numbers">
            {chr(10).join(line_numbers)}
        </div>
        <div class="editor-content">
            {self._generate_highlighted_html(highlighted_lines)}
        </div>
    </div>
</body>
</html>
"""
        return html
    
    def _generate_highlighted_html(self, highlighted_lines: List[List[Dict[str, Any]]]) -> str:
        """Generate HTML for highlighted lines."""
        html_lines = []
        
        for line_tokens in highlighted_lines:
            line_html = '<div class="line">'
            for token in line_tokens:
                css_class = token["type"]
                text = token["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                line_html += f'<span class="{css_class}">{text}</span>'
            line_html += '</div>'
            html_lines.append(line_html)
        
        return '\n'.join(html_lines)
    
    def save_to_file(self, file_path: str):
        """Save editor content to file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.content)
    
    def load_from_file(self, file_path: str):
        """Load editor content from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            self.content = f.read()
        self._update_undo_stack()
    
    def get_editor_settings(self) -> Dict[str, Any]:
        """Get current editor settings."""
        return {
            "font_family": self.settings.font_family,
            "font_size": self.settings.font_size,
            "tab_size": self.settings.tab_size,
            "line_numbers": self.settings.line_numbers,
            "word_wrap": self.settings.word_wrap,
            "auto_indent": self.settings.auto_indent,
            "syntax_highlighting": self.settings.syntax_highlighting,
            "bracket_matching": self.settings.bracket_matching,
            "minimap": self.settings.minimap,
            "theme": self.settings.theme
        }
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update editor settings."""
        for key, value in settings.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        
        # Update theme if changed
        if "theme" in settings:
            self.set_theme(settings["theme"])


# Convenience functions
def create_editor_ui(config: Dict[str, Any] = None) -> EditorUI:
    """Create Editor UI."""
    return EditorUI(config)


def create_editor_with_content(content: str, theme: str = "default") -> EditorUI:
    """Create Editor UI with content."""
    editor = EditorUI()
    editor.set_content(content)
    editor.set_theme(theme)
    return editor
