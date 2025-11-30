"""
ICEBURG Robust JSON Parser
Enhanced JSON parsing with comprehensive error handling and recovery strategies
"""

import json
import re
import logging
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class ParseResult:
    """Result of parsing operation"""
    success: bool
    data: Any
    method: str
    error: Optional[str] = None
    attempts: int = 1


class RobustJSONParser:
    """Robust JSON parser with multiple fallback strategies"""
    
    def __init__(self, max_retries: int = 3, verbose: bool = False):
        self.max_retries = max_retries
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
    def parse_with_retry(self, json_str: str, expected_structure: Optional[Dict] = None) -> ParseResult:
        """
        Parse JSON with multiple retry strategies and fallbacks
        """
        if not json_str or not json_str.strip():
            return ParseResult(
                success=False,
                data=None,
                method="empty_input",
                error="Empty or None input provided"
            )
        
        # Strategy 1: Direct JSON parsing
        result = self._try_direct_json(json_str)
        if result.success:
            return result
        
        # Strategy 2: Clean and retry
        result = self._try_cleaned_json(json_str)
        if result.success:
            return result
        
        # Strategy 3: Extract JSON blocks
        result = self._try_extract_json_blocks(json_str)
        if result.success:
            return result
        
        # Strategy 4: Fix common issues
        result = self._try_fix_common_issues(json_str)
        if result.success:
            return result
        
        # Strategy 5: Partial parsing
        result = self._try_partial_parsing(json_str)
        if result.success:
            return result
        
        # Strategy 6: Fallback structure
        result = self._create_fallback_structure(json_str, expected_structure)
        
        return result
    
    def _try_direct_json(self, json_str: str) -> ParseResult:
        """Try direct JSON parsing"""
        try:
            data = json.loads(json_str.strip())
            if self.verbose:
                print("Direct JSON parsing succeeded")
            return ParseResult(success=True, data=data, method="direct_json")
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Direct JSON parsing failed: {e}")
            return ParseResult(success=False, data=None, method="direct_json", error=str(e))
    
    def _try_cleaned_json(self, json_str: str) -> ParseResult:
        """Try parsing after cleaning the input"""
        try:
            # Remove common prefixes/suffixes
            cleaned = json_str.strip()
            
            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*$', '', cleaned)
            
            # Remove leading/trailing whitespace
            cleaned = cleaned.strip()
            
            # Try parsing cleaned version
            data = json.loads(cleaned)
            if self.verbose:
                print("Cleaned JSON parsing succeeded")
            return ParseResult(success=True, data=data, method="cleaned_json")
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Cleaned JSON parsing failed: {e}")
            return ParseResult(success=False, data=None, method="cleaned_json", error=str(e))
    
    def _try_extract_json_blocks(self, json_str: str) -> ParseResult:
        """Try to extract JSON from text blocks"""
        try:
            # Look for JSON-like structures in the text
            json_patterns = [
                r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
                r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Nested arrays
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, json_str, re.DOTALL)
                for match in matches:
                    try:
                        data = json.loads(match)
                        if self.verbose:
                            print("Extracted JSON block parsing succeeded")
                        return ParseResult(success=True, data=data, method="extract_blocks")
                    except json.JSONDecodeError:
                        continue
            
            return ParseResult(success=False, data=None, method="extract_blocks", error="No valid JSON blocks found")
        except Exception as e:
            return ParseResult(success=False, data=None, method="extract_blocks", error=str(e))
    
    def _try_fix_common_issues(self, json_str: str) -> ParseResult:
        """Try to fix common JSON formatting issues"""
        try:
            fixed = json_str.strip()
            
            # Fix trailing commas
            fixed = re.sub(r',\s*}', '}', fixed)
            fixed = re.sub(r',\s*]', ']', fixed)
            
            # Fix single quotes to double quotes (but be careful with strings)
            # This is a simplified approach - in production, you'd want more sophisticated handling
            fixed = re.sub(r"'([^']*)':", r'"\1":', fixed)
            
            # Fix unquoted keys
            fixed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
            
            # Remove comments (simple approach)
            lines = fixed.split('\n')
            cleaned_lines = []
            for line in lines:
                if '//' in line:
                    line = line[:line.index('//')]
                cleaned_lines.append(line)
            fixed = '\n'.join(cleaned_lines)
            
            # Try parsing fixed version
            data = json.loads(fixed)
            if self.verbose:
                print("Fixed common issues JSON parsing succeeded")
            return ParseResult(success=True, data=data, method="fix_common_issues")
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Fixed common issues JSON parsing failed: {e}")
            return ParseResult(success=False, data=None, method="fix_common_issues", error=str(e))
    
    def _try_partial_parsing(self, json_str: str) -> ParseResult:
        """Try to parse partial or truncated JSON"""
        try:
            # Try to find the largest valid JSON structure
            for i in range(len(json_str), 0, -1):
                try:
                    partial = json_str[:i]
                    data = json.loads(partial)
                    if self.verbose:
                        print("Partial JSON parsing succeeded")
                    return ParseResult(success=True, data=data, method="partial_parsing")
                except json.JSONDecodeError:
                    continue
            
            return ParseResult(success=False, data=None, method="partial_parsing", error="No valid partial JSON found")
        except Exception as e:
            return ParseResult(success=False, data=None, method="partial_parsing", error=str(e))
    
    def _create_fallback_structure(self, json_str: str, expected_structure: Optional[Dict] = None) -> ParseResult:
        """Create a fallback structure when all parsing fails"""
        try:
            # Extract key information from the text
            fallback_data = {
                "status": "parsing_failed",
                "raw_input": json_str[:1000],  # Truncate for safety
                "error": "All JSON parsing strategies failed",
                "timestamp": self._get_timestamp(),
                "fallback": True
            }
            
            # If we have an expected structure, try to populate it with defaults
            if expected_structure:
                fallback_data.update(self._create_default_structure(expected_structure))
            
            if self.verbose:
                print("Using fallback structure")
            
            return ParseResult(success=True, data=fallback_data, method="fallback_structure")
        except Exception as e:
            return ParseResult(success=False, data=None, method="fallback_structure", error=str(e))
    
    def _create_default_structure(self, expected_structure: Dict) -> Dict:
        """Create default values based on expected structure"""
        defaults = {}
        for key, value_type in expected_structure.items():
            if isinstance(value_type, type):
                if value_type == str:
                    defaults[key] = ""
                elif value_type == int:
                    defaults[key] = 0
                elif value_type == float:
                    defaults[key] = 0.0
                elif value_type == bool:
                    defaults[key] = False
                elif value_type == list:
                    defaults[key] = []
                elif value_type == dict:
                    defaults[key] = {}
                else:
                    defaults[key] = None
            else:
                defaults[key] = value_type
        return defaults
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_parsed_data(self, data: Any, expected_structure: Optional[Dict] = None) -> bool:
        """Validate parsed data against expected structure"""
        if expected_structure is None:
            return True
        
        try:
            if not isinstance(data, dict):
                return False
            
            for key, expected_type in expected_structure.items():
                if key not in data:
                    return False
                
                if isinstance(expected_type, type):
                    if not isinstance(data[key], expected_type):
                        return False
            
            return True
        except Exception:
            return False


# Global parser instance
_robust_parser = RobustJSONParser()


def parse_json_robust(json_str: str, expected_structure: Optional[Dict] = None, verbose: bool = False) -> ParseResult:
    """
    Parse JSON with robust error handling and multiple fallback strategies
    
    Args:
        json_str: JSON string to parse
        expected_structure: Optional expected structure for validation
        verbose: Whether to print debug information
    
    Returns:
        ParseResult with success status, data, and method used
    """
    global _robust_parser
    _robust_parser.verbose = verbose
    
    result = _robust_parser.parse_with_retry(json_str, expected_structure)
    
    if verbose and not result.success:
        print(f"JSON parsing failed: {result.error}")
    
    return result


def safe_json_parse(json_str: str, fallback: Any = None, verbose: bool = False) -> Any:
    """
    Safely parse JSON with a fallback value
    
    Args:
        json_str: JSON string to parse
        fallback: Fallback value if parsing fails
        verbose: Whether to print debug information
    
    Returns:
        Parsed data or fallback value
    """
    result = parse_json_robust(json_str, verbose=verbose)
    return result.data if result.success else fallback
