"""
Code Validator for Generated Agent Code
Validates generated code before deployment
"""

import ast
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class CodeValidator:
    """Validates generated code before deployment."""
    
    def __init__(self):
        """Initialize code validator."""
        self.required_imports = [
            "from __future__ import annotations",
            "from typing import",
            "from ..config import IceburgConfig",
            "import logging"
        ]
        
        self.required_methods = [
            "__init__",
            "run"
        ]
    
    def validate_code(self, code: str) -> bool:
        """
        Validate generated code.
        
        Args:
            code: Generated code string
            
        Returns:
            True if code is valid, False otherwise
        """
        try:
            # 1. Check syntax
            if not self._validate_syntax(code):
                logger.warning("Code validation failed: syntax error")
                return False
            
            # 2. Check required imports
            if not self._validate_imports(code):
                logger.warning("Code validation failed: missing required imports")
                return False
            
            # 3. Check required methods
            if not self._validate_methods(code):
                logger.warning("Code validation failed: missing required methods")
                return False
            
            # 4. Check for dangerous operations
            if not self._validate_safety(code):
                logger.warning("Code validation failed: unsafe operations detected")
                return False
            
            logger.info("Code validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Code validation error: {e}")
            return False
    
    def _validate_syntax(self, code: str) -> bool:
        """Validate Python syntax."""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error: {e}")
            return False
    
    def _validate_imports(self, code: str) -> bool:
        """Validate required imports."""
        # Check for at least some required imports
        found_imports = 0
        for required_import in self.required_imports:
            if required_import in code:
                found_imports += 1
        
        # At least 2 required imports should be present
        return found_imports >= 2
    
    def _validate_methods(self, code: str) -> bool:
        """Validate required methods."""
        # Check for required methods
        found_methods = 0
        for required_method in self.required_methods:
            if f"def {required_method}" in code:
                found_methods += 1
        
        # All required methods should be present
        return found_methods == len(self.required_methods)
    
    def _validate_safety(self, code: str) -> bool:
        """Validate code safety (no dangerous operations)."""
        # Check for dangerous operations
        dangerous_patterns = [
            "eval(",
            "exec(",
            "__import__",
            "open(",
            "subprocess",
            "os.system",
            "shutil.rmtree"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in code:
                logger.warning(f"Unsafe operation detected: {pattern}")
                return False
        
        return True
    
    def get_validation_details(self, code: str) -> Dict[str, Any]:
        """
        Get detailed validation results.
        
        Args:
            code: Generated code string
            
        Returns:
            Detailed validation results
        """
        results = {
            "valid": False,
            "syntax_valid": False,
            "imports_valid": False,
            "methods_valid": False,
            "safety_valid": False,
            "errors": []
        }
        
        try:
            # Check syntax
            try:
                ast.parse(code)
                results["syntax_valid"] = True
            except SyntaxError as e:
                results["errors"].append(f"Syntax error: {e}")
            
            # Check imports
            found_imports = 0
            for required_import in self.required_imports:
                if required_import in code:
                    found_imports += 1
            results["imports_valid"] = found_imports >= 2
            if not results["imports_valid"]:
                results["errors"].append(f"Missing required imports (found {found_imports}/{len(self.required_imports)})")
            
            # Check methods
            found_methods = 0
            for required_method in self.required_methods:
                if f"def {required_method}" in code:
                    found_methods += 1
            results["methods_valid"] = found_methods == len(self.required_methods)
            if not results["methods_valid"]:
                results["errors"].append(f"Missing required methods (found {found_methods}/{len(self.required_methods)})")
            
            # Check safety
            dangerous_patterns = [
                "eval(", "exec(", "__import__", "open(", "subprocess", "os.system", "shutil.rmtree"
            ]
            unsafe_operations = [p for p in dangerous_patterns if p in code]
            results["safety_valid"] = len(unsafe_operations) == 0
            if not results["safety_valid"]:
                results["errors"].append(f"Unsafe operations detected: {unsafe_operations}")
            
            # Overall validity
            results["valid"] = (
                results["syntax_valid"] and
                results["imports_valid"] and
                results["methods_valid"] and
                results["safety_valid"]
            )
            
        except Exception as e:
            results["errors"].append(f"Validation error: {e}")
        
        return results

