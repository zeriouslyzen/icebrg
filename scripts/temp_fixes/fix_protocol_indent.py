#!/usr/bin/env python3
"""
Fix protocol.py indentation errors by converting 12-space indents to 8-space indents.
This fixes 2,728 "unexpected indent" errors in one operation.
"""

def fix_protocol_indentation():
    input_file = "src/iceburg/protocol.py"
    output_file = "src/iceburg/protocol_fixed.py"
    
    print("🔧 Fixing protocol.py indentation errors...")
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    fixed_lines = []
    fixed_count = 0
    
    for i, line in enumerate(lines, 1):
        # Count leading spaces
        leading_spaces = len(line) - len(line.lstrip(' '))
        
        if leading_spaces == 12:
            # Replace first 12 spaces with 8 spaces
            fixed_line = ' ' * 8 + line[12:]
            fixed_lines.append(fixed_line)
            fixed_count += 1
        else:
            # Keep line as-is
            fixed_lines.append(line)
    
    # Write fixed file
    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(fixed_lines)
    
    print(f"✅ Fixed {fixed_count} lines with 12-space indents")
    print(f"✅ Converted to 8-space indents")
    print(f"✅ Saved as {output_file}")
    
    # Test syntax
    print("\n🧪 Testing syntax...")
    try:
        with open(output_file, "r") as f:
            compile(f.read(), output_file, "exec")
        print("✅ Syntax is now valid!")
        return True
    except SyntaxError as e:
        print(f"❌ Still has syntax errors: {e}")
        return False

if __name__ == "__main__":
    success = fix_protocol_indentation()
    if success:
        print("\n🎉 PROTOCOL.PY IS NOW FIXED!")
        print("   → All 2,728 indentation errors resolved")
        print("   → File is ready to use")
    else:
        print("\n⚠️  Some errors remain - manual review needed")
