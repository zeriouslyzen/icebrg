#!/usr/bin/env python3
"""
Fix all syntax errors in protocol.py
"""

import re

def fix_syntax_errors():
    file_path = "src/iceburg/protocol.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all if verbose: statements that are missing indented blocks
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        fixed_lines.append(line)
        
        # Check if this is an if verbose: statement
        if re.match(r'^\s*if verbose:\s*$', line):
            # Check if the next line is not indented (missing block)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip() and not next_line.startswith(' '):
                    # Add a pass statement
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(' ' * (indent + 4) + 'pass')
        
        i += 1
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Fixed syntax errors in protocol.py")

if __name__ == "__main__":
    fix_syntax_errors()
