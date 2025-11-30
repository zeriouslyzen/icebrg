#!/usr/bin/env python3
"""
Fix syntax errors in protocol.py by adding missing indented blocks after 'if verbose:' statements.
"""

import re

def fix_protocol_syntax():
    """Fix syntax errors in protocol.py."""
    
    # Read the file
    with open('src/iceburg/protocol.py', 'r') as f:
        content = f.read()
    
    # Pattern to find 'if verbose:' followed by empty line or non-indented line
    pattern = r'(if verbose:\s*\n)(\s*\n|\s*[^ \t])'
    
    def replace_func(match):
        if_stmt = match.group(1)
        next_line = match.group(2)
        
        # If the next line is empty or not indented, add a pass statement
        if next_line.strip() == '' or not next_line.startswith('    '):
            return if_stmt + '        pass  # Placeholder for verbose output\n' + next_line
        else:
            return match.group(0)
    
    # Apply the fix
    fixed_content = re.sub(pattern, replace_func, content)
    
    # Write the fixed content back
    with open('src/iceburg/protocol.py', 'w') as f:
        f.write(fixed_content)
    
    print("Fixed syntax errors in protocol.py")

if __name__ == "__main__":
    fix_protocol_syntax()
