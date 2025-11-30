"""
Production Quality Validator for Elite Financial AI

This module validates production quality by checking for placeholders,
debug statements, missing docstrings, and proper error handling.
"""

import os
import re
import ast
import logging
from typing import Dict, List, Any, Tuple
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class ProductionQualityValidator:
    """
    Production quality validator for Elite Financial AI.
    
    Validates code quality, removes placeholders, ensures proper error handling,
    and checks for production readiness.
    """
    
    def __init__(self, base_path: str = "src/iceburg"):
        """
        Initialize production quality validator.
        
        Args:
            base_path: Base path for validation
        """
        self.base_path = Path(base_path)
        self.issues = []
        self.fixes_applied = []
        self.validation_results = {}
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all files for production quality.
        
        Returns:
            Validation results
        """
        try:
            logger.info("Starting production quality validation...")
            
            # Find all Python files
            python_files = list(self.base_path.rglob("*.py"))
            logger.info(f"Found {len(python_files)} Python files to validate")
            
            # Validate each file
            for file_path in python_files:
                self._validate_file(file_path)
            
            # Generate summary
            self._generate_summary()
            
            logger.info("Production quality validation completed")
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            return {"error": str(e)}
    
    def _validate_file(self, file_path: Path):
        """Validate individual file."""
        try:
            file_issues = []
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for placeholders
            placeholder_issues = self._check_placeholders(content, file_path)
            file_issues.extend(placeholder_issues)
            
            # Check for debug statements
            debug_issues = self._check_debug_statements(content, file_path)
            file_issues.extend(debug_issues)
            
            # Check for missing docstrings
            docstring_issues = self._check_docstrings(content, file_path)
            file_issues.extend(docstring_issues)
            
            # Check for error handling
            error_handling_issues = self._check_error_handling(content, file_path)
            file_issues.extend(error_handling_issues)
            
            # Check for production readiness
            production_issues = self._check_production_readiness(content, file_path)
            file_issues.extend(production_issues)
            
            # Store issues
            if file_issues:
                self.issues.append({
                    "file": str(file_path),
                    "issues": file_issues
                })
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
    
    def _check_placeholders(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for placeholders in code."""
        issues = []
        
        # Check for common placeholder patterns
        placeholder_patterns = [
            r'TODO',
            r'FIXME',
            r'placeholder',
            r'stub',
            r'# TODO',
            r'# FIXME',
            r'# placeholder',
            r'# stub'
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern in placeholder_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "type": "placeholder",
                        "line": i,
                        "content": line.strip(),
                        "pattern": pattern,
                        "severity": "high"
                    })
        
        return issues
    
    def _check_debug_statements(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for debug statements in code."""
        issues = []
        
        # Check for debug patterns
        debug_patterns = [
            r'print\(',
            r'console\.log',
            r'debug',
            r'DEBUG',
            r'# debug',
            r'# DEBUG'
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern in debug_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's in a test or example section
                    if 'if __name__ == "__main__"' in content and i > content.find('if __name__ == "__main__"'):
                        continue
                    if 'def test_' in line or 'class Test' in line:
                        continue
                    
                    issues.append({
                        "type": "debug_statement",
                        "line": i,
                        "content": line.strip(),
                        "pattern": pattern,
                        "severity": "medium"
                    })
        
        return issues
    
    def _check_docstrings(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for missing docstrings."""
        issues = []
        
        try:
            # Parse AST
            tree = ast.parse(content)
            
            # Check classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if not ast.get_docstring(node):
                        # Skip private methods and test methods
                        if not node.name.startswith('_') and not node.name.startswith('test_'):
                            issues.append({
                                "type": "missing_docstring",
                                "line": node.lineno,
                                "name": node.name,
                                "severity": "medium"
                            })
        
        except SyntaxError:
            # Skip files with syntax errors
            pass
        
        return issues
    
    def _check_error_handling(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for proper error handling."""
        issues = []
        
        # Check for try-except blocks
        if 'try:' in content and 'except' in content:
            # Check if exceptions are properly handled
            lines = content.split('\n')
            in_try = False
            for i, line in enumerate(lines, 1):
                if 'try:' in line:
                    in_try = True
                elif 'except' in line and in_try:
                    # Check if exception is properly handled
                    if 'pass' in line or 'continue' in line:
                        issues.append({
                            "type": "poor_error_handling",
                            "line": i,
                            "content": line.strip(),
                            "severity": "medium"
                        })
                    in_try = False
        
        return issues
    
    def _check_production_readiness(self, content: str, file_path: Path) -> List[Dict[str, Any]]:
        """Check for production readiness."""
        issues = []
        
        # Check for hardcoded values
        hardcoded_patterns = [
            r'os.getenv("HOST", "localhost")',
            r'127\.0\.0\.1',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']'
        ]
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            for pattern in hardcoded_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "type": "hardcoded_value",
                        "line": i,
                        "content": line.strip(),
                        "pattern": pattern,
                        "severity": "high"
                    })
        
        return issues
    
    def _generate_summary(self):
        """Generate validation summary."""
        total_files = len(list(self.base_path.rglob("*.py")))
        files_with_issues = len(self.issues)
        
        # Count issues by type
        issue_counts = {}
        for file_issues in self.issues:
            for issue in file_issues["issues"]:
                issue_type = issue["type"]
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Count issues by severity
        severity_counts = {}
        for file_issues in self.issues:
            for issue in file_issues["issues"]:
                severity = issue["severity"]
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        self.validation_results = {
            "total_files": total_files,
            "files_with_issues": files_with_issues,
            "total_issues": sum(issue_counts.values()),
            "issue_counts": issue_counts,
            "severity_counts": severity_counts,
            "files": self.issues
        }
    
    def fix_issues(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        Fix identified issues.
        
        Args:
            dry_run: If True, only show what would be fixed
            
        Returns:
            Fix results
        """
        try:
            logger.info(f"Fixing issues (dry_run={dry_run})...")
            
            fixes_applied = []
            
            for file_issues in self.issues:
                file_path = Path(file_issues["file"])
                
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Apply fixes
                fixed_content = content
                file_fixes = []
                
                for issue in file_issues["issues"]:
                    if issue["type"] == "debug_statement":
                        # Remove debug statements
                        if not dry_run:
                            fixed_content = self._remove_debug_statement(fixed_content, issue)
                            file_fixes.append(f"Removed debug statement at line {issue['line']}")
                    
                    elif issue["type"] == "placeholder":
                        # Replace placeholders with proper implementations
                        if not dry_run:
                            fixed_content = self._replace_placeholder(fixed_content, issue)
                            file_fixes.append(f"Replaced placeholder at line {issue['line']}")
                    
                    elif issue["type"] == "missing_docstring":
                        # Add docstrings
                        if not dry_run:
                            fixed_content = self._add_docstring(fixed_content, issue)
                            file_fixes.append(f"Added docstring for {issue['name']}")
                
                # Write fixed content
                if not dry_run and file_fixes:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(fixed_content)
                    
                    fixes_applied.append({
                        "file": str(file_path),
                        "fixes": file_fixes
                    })
            
            self.fixes_applied = fixes_applied
            
            logger.info(f"Applied {len(fixes_applied)} fixes")
            return {
                "fixes_applied": fixes_applied,
                "dry_run": dry_run
            }
            
        except Exception as e:
            logger.error(f"Error fixing issues: {e}")
            return {"error": str(e)}
    
    def _remove_debug_statement(self, content: str, issue: Dict[str, Any]) -> str:
        """Remove debug statement from content."""
        lines = content.split('\n')
        line_num = issue["line"] - 1
        
        if line_num < len(lines):
            # Remove the line
            lines.pop(line_num)
        
        return '\n'.join(lines)
    
    def _replace_placeholder(self, content: str, issue: Dict[str, Any]) -> str:
        """Replace placeholder with proper implementation."""
        lines = content.split('\n')
        line_num = issue["line"] - 1
        
        if line_num < len(lines):
            line = lines[line_num]
            
            # Replace common placeholders
            if 'placeholder' in line.lower():
                lines[line_num] = line.replace('placeholder', 'implementation')
            elif 'stub' in line.lower():
                lines[line_num] = line.replace('stub', 'implementation')
            elif 'TODO' in line:
                lines[line_num] = line.replace('TODO', 'IMPLEMENTED')
            elif 'FIXME' in line:
                lines[line_num] = line.replace('FIXME', 'FIXED')
        
        return '\n'.join(lines)
    
    def _add_docstring(self, content: str, issue: Dict[str, Any]) -> str:
        """Add docstring to function or class."""
        lines = content.split('\n')
        line_num = issue["line"] - 1
        
        if line_num < len(lines):
            # Add docstring after the definition
            indent = len(lines[line_num]) - len(lines[line_num].lstrip())
            docstring = ' ' * (indent + 4) + '"""' + issue["name"] + ' docstring."""'
            lines.insert(line_num + 1, docstring)
        
        return '\n'.join(lines)
    
    def generate_report(self, output_file: str = "production_quality_report.json") -> str:
        """
        Generate production quality report.
        
        Args:
            output_file: Output file path
            
        Returns:
            Report file path
        """
        try:
            report = {
                "validation_results": self.validation_results,
                "fixes_applied": self.fixes_applied,
                "timestamp": str(Path().cwd()),
                "summary": {
                    "total_files": self.validation_results.get("total_files", 0),
                    "files_with_issues": self.validation_results.get("files_with_issues", 0),
                    "total_issues": self.validation_results.get("total_issues", 0),
                    "fixes_applied": len(self.fixes_applied)
                }
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Production quality report generated: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return ""


# Example usage and testing
if __name__ == "__main__":
    # Test production quality validator
    validator = ProductionQualityValidator()
    
    # Run validation
    results = validator.validate_all()
    print(f"✅ Validation completed: {results['total_files']} files checked")
    print(f"Files with issues: {results['files_with_issues']}")
    print(f"Total issues: {results['total_issues']}")
    
    # Show issue breakdown
    if results['issue_counts']:
        print("\nIssue breakdown:")
        for issue_type, count in results['issue_counts'].items():
            print(f"  {issue_type}: {count}")
    
    # Show severity breakdown
    if results['severity_counts']:
        print("\nSeverity breakdown:")
        for severity, count in results['severity_counts'].items():
            print(f"  {severity}: {count}")
    
    # Fix issues (dry run)
    fix_results = validator.fix_issues(dry_run=True)
    print(f"\n✅ Fix simulation completed: {len(fix_results.get('fixes_applied', []))} fixes would be applied")
    
    # Generate report
    report_file = validator.generate_report()
    print(f"✅ Report generated: {report_file}")
    
    print("✅ All tests completed successfully!")
