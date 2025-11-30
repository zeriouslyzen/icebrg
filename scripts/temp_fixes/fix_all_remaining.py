#!/usr/bin/env python3
"""
Fix ALL remaining syntax errors in protocol.py
"""

import re

def fix_all_remaining():
    file_path = "src/iceburg/protocol.py"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find all if statements that end with : and are missing indented blocks
    lines = content.split('\n')
    fixed_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check if this is an if statement ending with :
        if re.match(r'^\s*if\s+.*:\s*$', line):
            # Look ahead to find the next non-empty line
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            
            if j < len(lines):
                next_line = lines[j]
                # If the next line is not indented, we need to add a pass
                if next_line.strip() and not next_line.startswith(' '):
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line)
                    fixed_lines.append(' ' * (indent + 4) + 'pass')
                    i += 1
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.write('\n'.join(fixed_lines))
    
    print("Fixed ALL remaining syntax errors in protocol.py")

if __name__ == "__main__":
    fix_all_remaining()
