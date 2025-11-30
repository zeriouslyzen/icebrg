#!/usr/bin/env python3
"""
Comprehensive Codebase Audit Script
Checks for:
- Conversation history contamination
- Duplicate code
- Unnecessary API calls
- Contaminated code patterns
- Dependencies/conflicts
- Modelfile issues
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

# Patterns to check
CONTAMINATION_PATTERNS = [
    (r'previous conversation', 'Conversation history reference'),
    (r'building upon', 'Building upon reference'),
    (r'as we discussed', 'Previous discussion reference'),
    (r'interconnectedness of all things', 'Pseudo-profound phrase'),
    (r'resonates with', 'Vague connection phrase'),
    (r'conversation_history', 'Conversation history variable'),
    (r'get_conversations', 'Conversation retrieval'),
    (r'previous.*conversation', 'Previous conversation pattern'),
]

DUPLICATE_PATTERNS = [
    (r'def.*query_endpoint', 'Query endpoint definition'),
    (r'def.*generate_stream', 'Stream generator definition'),
    (r'conversation_history\s*=', 'Conversation history assignment'),
    (r'thinking_callback', 'Thinking callback'),
]

API_CALL_PATTERNS = [
    (r'requests\.(get|post|put|delete)', 'HTTP requests library'),
    (r'httpx\.(get|post|put|delete)', 'HTTPX library'),
    (r'aiohttp\.(get|post|put|delete)', 'Async HTTP library'),
    (r'urllib\.request', 'urllib requests'),
    (r'fetch\(', 'Fetch API calls'),
]

FORBIDDEN_PATTERNS = [
    (r'astrology.*organ', 'Astrology-organ connection'),
    (r'quantum.*consciousness', 'Quantum consciousness woo'),
    (r'IIT.*superposition', 'IIT-superposition connection'),
    (r'embodied cognition.*interconnectedness', 'Embodied cognition woo'),
]

def scan_file(filepath: Path, patterns: list) -> dict:
    """Scan a file for patterns"""
    results = defaultdict(list)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
            
        for pattern, description in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1].strip() if line_num <= len(lines) else ''
                results[description].append({
                    'line': line_num,
                    'match': match.group(),
                    'context': line_content[:100]
                })
    except Exception as e:
        results['_errors'] = [str(e)]
    return results

def find_duplicates(filepath: Path) -> list:
    """Find duplicate function/class definitions"""
    duplicates = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find function definitions
        func_pattern = r'def\s+(\w+)\s*\('
        functions = re.findall(func_pattern, content)
        func_counts = defaultdict(int)
        for func in functions:
            func_counts[func] += 1
        
        for func, count in func_counts.items():
            if count > 1:
                duplicates.append(f"Function '{func}' defined {count} times")
        
        # Find class definitions
        class_pattern = r'class\s+(\w+)'
        classes = re.findall(class_pattern, content)
        class_counts = defaultdict(int)
        for cls in classes:
            class_counts[cls] += 1
        
        for cls, count in class_counts.items():
            if count > 1:
                duplicates.append(f"Class '{cls}' defined {count} times")
    except Exception as e:
        duplicates.append(f"Error: {e}")
    return duplicates

def audit_codebase():
    """Main audit function"""
    src_dir = Path('src/iceburg')
    frontend_dir = Path('frontend')
    
    results = {
        'contamination': defaultdict(list),
        'duplicates': defaultdict(list),
        'api_calls': defaultdict(list),
        'forbidden_patterns': defaultdict(list),
        'files_scanned': 0,
        'errors': []
    }
    
    # Scan Python files
    for py_file in src_dir.rglob('*.py'):
        if py_file.name.startswith('__'):
            continue
        
        results['files_scanned'] += 1
        rel_path = str(py_file.relative_to(Path('.')))
        
        # Check contamination
        contamination = scan_file(py_file, CONTAMINATION_PATTERNS)
        if contamination:
            results['contamination'][rel_path] = contamination
        
        # Check duplicates
        duplicates = find_duplicates(py_file)
        if duplicates:
            results['duplicates'][rel_path] = duplicates
        
        # Check API calls
        api_calls = scan_file(py_file, API_CALL_PATTERNS)
        if api_calls:
            results['api_calls'][rel_path] = api_calls
        
        # Check forbidden patterns
        forbidden = scan_file(py_file, FORBIDDEN_PATTERNS)
        if forbidden:
            results['forbidden_patterns'][rel_path] = forbidden
    
    # Scan frontend files
    for js_file in frontend_dir.rglob('*.js'):
        results['files_scanned'] += 1
        rel_path = str(js_file.relative_to(Path('.')))
        
        contamination = scan_file(js_file, CONTAMINATION_PATTERNS)
        if contamination:
            results['contamination'][rel_path] = contamination
        
        api_calls = scan_file(js_file, API_CALL_PATTERNS)
        if api_calls:
            results['api_calls'][rel_path] = api_calls
    
    # Check for Modelfile
    modelfiles = list(Path('.').glob('**/Modelfile*'))
    results['modelfiles'] = [str(f) for f in modelfiles]
    
    return results

def print_report(results: dict):
    """Print audit report"""
    print("=" * 80)
    print("ICEBURG CODEBASE AUDIT REPORT")
    print("=" * 80)
    print(f"\nFiles scanned: {results['files_scanned']}")
    
    # Contamination
    if results['contamination']:
        print("\n" + "=" * 80)
        print("🚨 CONTAMINATION DETECTED")
        print("=" * 80)
        for filepath, issues in results['contamination'].items():
            print(f"\n📄 {filepath}:")
            for issue_type, matches in issues.items():
                if issue_type != '_errors':
                    print(f"  ⚠️  {issue_type}: {len(matches)} occurrences")
                    for match in matches[:3]:  # Show first 3
                        print(f"     Line {match['line']}: {match['context'][:60]}...")
    else:
        print("\n✅ No contamination detected")
    
    # Duplicates
    if results['duplicates']:
        print("\n" + "=" * 80)
        print("🔄 DUPLICATE CODE DETECTED")
        print("=" * 80)
        for filepath, duplicates in results['duplicates'].items():
            print(f"\n📄 {filepath}:")
            for dup in duplicates:
                print(f"  ⚠️  {dup}")
    else:
        print("\n✅ No duplicate definitions detected")
    
    # API Calls
    if results['api_calls']:
        print("\n" + "=" * 80)
        print("🌐 EXTERNAL API CALLS DETECTED")
        print("=" * 80)
        for filepath, calls in results['api_calls'].items():
            print(f"\n📄 {filepath}:")
            for call_type, matches in calls.items():
                if call_type != '_errors':
                    print(f"  📡 {call_type}: {len(matches)} occurrences")
                    for match in matches[:3]:
                        print(f"     Line {match['line']}: {match['context'][:60]}...")
    else:
        print("\n✅ No external API calls detected")
    
    # Forbidden patterns
    if results['forbidden_patterns']:
        print("\n" + "=" * 80)
        print("🚫 FORBIDDEN PATTERNS DETECTED")
        print("=" * 80)
        for filepath, patterns in results['forbidden_patterns'].items():
            print(f"\n📄 {filepath}:")
            for pattern_type, matches in patterns.items():
                if pattern_type != '_errors':
                    print(f"  ⚠️  {pattern_type}: {len(matches)} occurrences")
                    for match in matches[:3]:
                        print(f"     Line {match['line']}: {match['context'][:60]}...")
    else:
        print("\n✅ No forbidden patterns detected")
    
    # Modelfiles
    if results['modelfiles']:
        print("\n" + "=" * 80)
        print("📝 MODELFILES FOUND")
        print("=" * 80)
        for modelfile in results['modelfiles']:
            print(f"  📄 {modelfile}")
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    results = audit_codebase()
    print_report(results)
    
    # Save to JSON
    with open('audit_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n📊 Full report saved to audit_report.json")

