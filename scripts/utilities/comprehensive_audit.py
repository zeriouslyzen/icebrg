#!/usr/bin/env python3
"""
Comprehensive ICEBURG Codebase Audit
Checks everything the user requested:
- Contamination (code and metadata)
- Dependencies and conflicts
- Duplicate code
- Unnecessary API calls
- Modelfile issues
- Issues and conflicts
"""

import os
import re
import json
import ast
from pathlib import Path
from collections import defaultdict
import importlib.util

# Results storage
results = {
    'contamination': defaultdict(list),
    'dependencies': defaultdict(list),
    'conflicts': [],
    'duplicates': defaultdict(list),
    'api_calls': defaultdict(list),
    'unnecessary_imports': defaultdict(list),
    'metadata_issues': [],
    'modelfile_issues': [],
    'issues': [],
    'files_scanned': 0,
    'errors': []
}

def scan_python_file(filepath: Path):
    """Comprehensive scan of a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        rel_path = str(filepath.relative_to(Path('.')))
        results['files_scanned'] += 1
        
        # 1. CONTAMINATION PATTERNS
        contamination_patterns = [
            (r'previous conversation', 'Conversation history reference'),
            (r'building upon', 'Building upon reference'),
            (r'as we discussed', 'Previous discussion reference'),
            (r'interconnectedness of all things', 'Pseudo-profound phrase'),
            (r'resonates with', 'Vague connection phrase'),
            (r'conversation_history\s*=', 'Conversation history assignment'),
            (r'get_conversations', 'Conversation retrieval'),
            (r'astrology.*organ', 'Astrology-organ contamination'),
            (r'quantum.*consciousness.*IIT', 'Quantum consciousness woo'),
        ]
        
        for pattern, desc in contamination_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE))
            if matches:
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ''
                    # Skip if it's in a comment or string that's explicitly forbidding it
                    if 'FORBIDDEN' in line_content or 'NOT evidence' in line_content:
                        continue
                    results['contamination'][rel_path].append({
                        'line': line_num,
                        'pattern': desc,
                        'match': match.group(),
                        'context': line_content[:80]
                    })
        
        # 2. DEPENDENCIES - Parse imports
        try:
            tree = ast.parse(content)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            if imports:
                results['dependencies'][rel_path] = imports
        except:
            pass
        
        # 3. API CALLS
        api_patterns = [
            (r'requests\.(get|post|put|delete|patch)', 'requests library'),
            (r'httpx\.(get|post|put|delete|patch)', 'httpx library'),
            (r'aiohttp\.(get|post|put|delete|patch)', 'aiohttp library'),
            (r'urllib\.request', 'urllib requests'),
            (r'fetch\(', 'Fetch API'),
            (r'\.get\(|\.post\(|\.put\(|\.delete\(', 'HTTP method calls'),
        ]
        
        for pattern, desc in api_patterns:
            matches = list(re.finditer(pattern, content, re.IGNORECASE))
            if matches:
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ''
                    results['api_calls'][rel_path].append({
                        'line': line_num,
                        'type': desc,
                        'match': match.group(),
                        'context': line_content[:80]
                    })
        
        # 4. DUPLICATES - Function/class definitions
        func_pattern = r'def\s+(\w+)\s*\('
        functions = re.findall(func_pattern, content)
        func_counts = defaultdict(int)
        for func in functions:
            func_counts[func] += 1
        
        for func, count in func_counts.items():
            if count > 1:
                results['duplicates'][rel_path].append(f"Function '{func}' defined {count} times")
        
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, content)
        class_counts = defaultdict(int)
        for cls in classes:
            class_counts[cls] += 1
        
        for cls, count in class_counts.items():
            if count > 1:
                results['duplicates'][rel_path].append(f"Class '{cls}' defined {count} times")
        
        # 5. UNNECESSARY IMPORTS - Check if imports are used
        try:
            tree = ast.parse(content)
            imported_names = set()
            used_names = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
                elif isinstance(node, ast.Name):
                    used_names.add(node.id)
            
            unused = imported_names - used_names
            if unused:
                results['unnecessary_imports'][rel_path] = list(unused)
        except:
            pass
        
    except Exception as e:
        results['errors'].append(f"{filepath}: {str(e)}")

def should_skip_path(filepath):
    """Check if path should be skipped"""
    skip_patterns = [
        'node_modules', '.git', '__pycache__', '.build', 'dist', 
        'build', '.venv', 'venv', '.env', 'env', 'target',
        'ActivityMonitor', '.pytest_cache', '.mypy_cache'
    ]
    path_str = str(filepath)
    return any(pattern in path_str for pattern in skip_patterns)

def scan_metadata_files():
    """Scan metadata files (JSON, YAML, etc.)"""
    metadata_patterns = [
        (r'conversation_history', 'Conversation history in metadata'),
        (r'previous conversation', 'Previous conversation in metadata'),
    ]
    
    # Only scan specific directories
    scan_dirs = ['src', 'frontend', '.']
    for scan_dir in scan_dirs:
        dir_path = Path(scan_dir)
        if not dir_path.exists():
            continue
        
        for ext in ['*.json', '*.yaml', '*.yml']:
            try:
                for filepath in dir_path.rglob(ext):
                    if should_skip_path(filepath):
                        continue
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                for pattern, desc in metadata_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        results['metadata_issues'].append({
                            'file': str(filepath),
                            'issue': desc
                        })
            except:
                pass

def check_dependency_conflicts():
    """Check for dependency conflicts"""
    requirements_files = []
    for pattern in ['requirements*.txt', 'pyproject.toml', 'setup.py']:
        try:
            requirements_files.extend(Path('.').glob(pattern))
        except:
            pass
    
    all_deps = defaultdict(list)
    
    for req_file in requirements_files:
        try:
            with open(req_file, 'r') as f:
                content = f.read()
                # Simple regex for package names
                deps = re.findall(r'^([a-zA-Z0-9_-]+)', content, re.MULTILINE)
                for dep in deps:
                    if dep and not dep.startswith('#'):
                        all_deps[dep.lower()].append(str(req_file))
        except:
            pass
    
    # Check for conflicts (same package in multiple files with different versions)
    for dep, files in all_deps.items():
        if len(files) > 1:
            results['conflicts'].append({
                'dependency': dep,
                'files': files
            })

def check_modelfiles():
    """Check for modelfiles and audit them"""
    modelfiles = []
    for pattern in ['Modelfile*', '*.modelfile']:
        try:
            modelfiles.extend(Path('.').glob(pattern))
            modelfiles.extend(Path('src').glob(pattern) if Path('src').exists() else [])
        except:
            pass
    
    for modelfile in modelfiles:
        if should_skip_path(modelfile):
            continue
        
        try:
            with open(modelfile, 'r') as f:
                content = f.read()
            
            issues = []
            
            # Check for contamination
            if re.search(r'previous conversation|building upon|as we discussed', content, re.IGNORECASE):
                issues.append('Contains conversation history references')
            
            # Check for forbidden patterns
            if re.search(r'interconnectedness of all things|resonates with', content, re.IGNORECASE):
                issues.append('Contains pseudo-profound phrases')
            
            if issues:
                results['modelfile_issues'].append({
                    'file': str(modelfile),
                    'issues': issues
                })
        except Exception as e:
            results['modelfile_issues'].append({
                'file': str(modelfile),
                'error': str(e)
            })

def generate_report():
    """Generate comprehensive report"""
    print("=" * 100)
    print("COMPREHENSIVE ICEBURG CODEBASE AUDIT REPORT")
    print("=" * 100)
    print(f"\n📊 Files scanned: {results['files_scanned']}")
    print(f"❌ Errors: {len(results['errors'])}")
    
    # 1. CONTAMINATION
    print("\n" + "=" * 100)
    print("🚨 CONTAMINATION CHECK")
    print("=" * 100)
    if results['contamination']:
        total = sum(len(v) for v in results['contamination'].values())
        print(f"⚠️  Found {total} contamination issues across {len(results['contamination'])} files:")
        for filepath, issues in list(results['contamination'].items())[:10]:
            print(f"\n  📄 {filepath}:")
            for issue in issues[:3]:
                print(f"     Line {issue['line']}: {issue['pattern']} - {issue['context'][:60]}")
    else:
        print("✅ No contamination detected")
    
    # 2. DEPENDENCIES
    print("\n" + "=" * 100)
    print("📦 DEPENDENCIES")
    print("=" * 100)
    all_deps = set()
    for deps in results['dependencies'].values():
        all_deps.update(deps)
    print(f"📊 Total unique dependencies: {len(all_deps)}")
    print(f"📄 Files with imports: {len(results['dependencies'])}")
    
    # 3. CONFLICTS
    print("\n" + "=" * 100)
    print("⚔️  DEPENDENCY CONFLICTS")
    print("=" * 100)
    if results['conflicts']:
        print(f"⚠️  Found {len(results['conflicts'])} potential conflicts:")
        for conflict in results['conflicts'][:5]:
            print(f"  📦 {conflict['dependency']} in: {', '.join(conflict['files'])}")
    else:
        print("✅ No dependency conflicts detected")
    
    # 4. DUPLICATES
    print("\n" + "=" * 100)
    print("🔄 DUPLICATE CODE")
    print("=" * 100)
    if results['duplicates']:
        total = sum(len(v) for v in results['duplicates'].values())
        print(f"⚠️  Found {total} duplicate definitions:")
        for filepath, dups in list(results['duplicates'].items())[:5]:
            print(f"  📄 {filepath}:")
            for dup in dups[:3]:
                print(f"     - {dup}")
    else:
        print("✅ No duplicate definitions detected")
    
    # 5. API CALLS
    print("\n" + "=" * 100)
    print("🌐 EXTERNAL API CALLS")
    print("=" * 100)
    if results['api_calls']:
        total = sum(len(v) for v in results['api_calls'].values())
        print(f"📡 Found {total} API calls across {len(results['api_calls'])} files:")
        for filepath, calls in list(results['api_calls'].items())[:10]:
            print(f"  📄 {filepath}:")
            call_types = defaultdict(int)
            for call in calls:
                call_types[call['type']] += 1
            for call_type, count in call_types.items():
                print(f"     - {call_type}: {count} calls")
    else:
        print("✅ No external API calls detected")
    
    # 6. UNNECESSARY IMPORTS
    print("\n" + "=" * 100)
    print("🗑️  UNNECESSARY IMPORTS")
    print("=" * 100)
    if results['unnecessary_imports']:
        total = sum(len(v) for v in results['unnecessary_imports'].values())
        print(f"⚠️  Found {total} potentially unused imports:")
        for filepath, imports in list(results['unnecessary_imports'].items())[:5]:
            print(f"  📄 {filepath}: {', '.join(imports[:5])}")
    else:
        print("✅ No unnecessary imports detected")
    
    # 7. METADATA ISSUES
    print("\n" + "=" * 100)
    print("📋 METADATA ISSUES")
    print("=" * 100)
    if results['metadata_issues']:
        print(f"⚠️  Found {len(results['metadata_issues'])} metadata issues:")
        for issue in results['metadata_issues'][:5]:
            print(f"  📄 {issue['file']}: {issue['issue']}")
    else:
        print("✅ No metadata issues detected")
    
    # 8. MODELFILE ISSUES
    print("\n" + "=" * 100)
    print("📝 MODELFILE AUDIT")
    print("=" * 100)
    if results['modelfile_issues']:
        print(f"⚠️  Found {len(results['modelfile_issues'])} modelfile issues:")
        for issue in results['modelfile_issues']:
            print(f"  📄 {issue['file']}:")
            if 'issues' in issue:
                for i in issue['issues']:
                    print(f"     - {i}")
            if 'error' in issue:
                print(f"     - Error: {issue['error']}")
    else:
        print("✅ No modelfiles found or no issues detected")
    
    # 9. ERRORS
    if results['errors']:
        print("\n" + "=" * 100)
        print("❌ ERRORS")
        print("=" * 100)
        for error in results['errors'][:10]:
            print(f"  ⚠️  {error}")
    
    print("\n" + "=" * 100)
    print("AUDIT COMPLETE")
    print("=" * 100)

def main():
    """Main audit function"""
    print("Starting comprehensive audit...")
    
    # Scan Python files
    src_dir = Path('src/iceburg')
    if src_dir.exists():
        for py_file in src_dir.rglob('*.py'):
            if py_file.name.startswith('__'):
                continue
            scan_python_file(py_file)
    
    # Scan frontend
    frontend_dir = Path('frontend')
    if frontend_dir.exists():
        for js_file in frontend_dir.rglob('*.js'):
            # Basic JS scan (simpler patterns)
            try:
                with open(js_file, 'r') as f:
                    content = f.read()
                
                rel_path = str(js_file.relative_to(Path('.')))
                
                # Check for contamination
                if re.search(r'conversation_history|previous conversation', content, re.IGNORECASE):
                    results['contamination'][rel_path].append({
                        'pattern': 'Conversation history in JS',
                        'context': 'JavaScript file'
                    })
                
                # Check for API calls
                if re.search(r'fetch\(|\.get\(|\.post\(', content):
                    results['api_calls'][rel_path].append({
                        'type': 'JavaScript API call',
                        'context': 'JavaScript file'
                    })
            except:
                pass
    
    # Scan metadata
    scan_metadata_files()
    
    # Check dependencies
    check_dependency_conflicts()
    
    # Check modelfiles
    check_modelfiles()
    
    # Generate report
    generate_report()
    
    # Save to JSON
    with open('comprehensive_audit_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n📊 Full report saved to comprehensive_audit_report.json")

if __name__ == '__main__':
    main()

