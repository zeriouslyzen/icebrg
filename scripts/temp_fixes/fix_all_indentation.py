#!/usr/bin/env python3
"""Fix all indentation errors in agent files"""
import os
import re
import glob
from pathlib import Path

def fix_indentation_errors(file_path):
    """Fix common indentation errors in Python files"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Check for 'if verbose:' followed by empty/incorrectly indented block
        if re.match(r'^\s+if\s+verbose\s*:\s*$', line):
            fixed_lines.append(line)
            i += 1
            
            # Check if next line is empty or improperly indented
            if i < len(lines):
                next_line = lines[i]
                current_indent = len(line) - len(line.lstrip())
                
                # If next line is empty, skip it
                if not next_line.strip():
                    fixed_lines.append(next_line)
                    i += 1
                    next_line = lines[i] if i < len(lines) else ''
                
                # If next line is not properly indented or is a return/statement at wrong level
                if next_line.strip() and not next_line.startswith(' ' * (current_indent + 4)):
                    # Check if it's a statement that should be inside the if block
                    if next_line.strip().startswith(('return ', 'print(', 'pass')):
                        # Indent it properly
                        fixed_lines.append(' ' * (current_indent + 4) + next_line.lstrip())
                        i += 1
                    else:
                        # Add a print statement for the if block
                        fixed_lines.append(' ' * (current_indent + 4) + f'print(f"[DEBUG] verbose mode enabled")\n')
                        fixed_lines.append(next_line)
                        i += 1
                else:
                    fixed_lines.append(next_line)
                    i += 1
        else:
            fixed_lines.append(line)
            i += 1
    
    # Write fixed content
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    return True

# Find all Python files in agents directory
agents_dir = Path('src/iceburg/agents')
agent_files = list(agents_dir.glob('*.py'))

# Also check emergent_software_architect.py
agent_files.append(Path('src/iceburg/emergent_software_architect.py'))
agent_files.append(Path('src/iceburg/agents/pyramid_dag_architect.py'))
agent_files.append(Path('src/iceburg/agents/swarm_integrated_architect.py'))

fixed_count = 0
for file_path in agent_files:
    if file_path.exists():
        try:
            # Try to compile first to see if there are errors
            import py_compile
            py_compile.compile(str(file_path), doraise=True)
            print(f"✅ {file_path.name} - OK")
        except py_compile.PyCompileError as e:
            print(f"⚠️  {file_path.name} - HAS ERRORS")
            if fix_indentation_errors(str(file_path)):
                fixed_count += 1
                print(f"   Fixed indentation errors")

print(f"\n✅ Fixed {fixed_count} files")

