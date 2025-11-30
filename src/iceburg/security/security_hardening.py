"""
ICEBURG Security Hardening

Provides:
- Input validation
- Output sanitization
- Rate limiting
- Security checks
"""

import re
import time
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting"""
    max_requests: int = 100
    time_window: float = 60.0  # seconds
    max_requests_per_agent: int = 20
    max_requests_per_ip: int = 50


@dataclass
class SecurityViolation:
    """Security violation record"""
    violation_type: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)


class InputValidator:
    """
    Validates and sanitizes input to prevent security issues.
    
    Checks for:
    - Injection attacks (SQL, command, code)
    - XSS (Cross-Site Scripting)
    - Path traversal
    - Excessive length
    - Malformed data
    """
    
    def __init__(self):
        self.max_input_length = 100000  # 100KB
        self.max_query_length = 10000  # 10KB
        
        # Dangerous patterns
        self.injection_patterns = [
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)',
            r'(\b(rm\s+-rf|del\s+/f|format\s+)\b)',
            r'(\b(eval|exec|compile|__import__|globals|locals)\b)',
            r'(\b(system|popen|shell_exec|passthru)\b)',
            r'(\.\.\/|\.\.\\\\)',  # Path traversal
            r'(\b(chmod|chown|sudo|su)\b)',
        ]
        
        self.xss_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'<iframe[^>]*>',
            r'<object[^>]*>',
            r'<embed[^>]*>',
        ]
        
        self.compiled_injection = re.compile('|'.join(self.injection_patterns), re.IGNORECASE)
        self.compiled_xss = re.compile('|'.join(self.xss_patterns), re.IGNORECASE)
    
    def validate_input(self, input_data: Any, input_type: str = "query") -> Tuple[bool, Optional[str], Optional[SecurityViolation]]:
        """
        Validate input for security issues.
        
        Args:
            input_data: Input to validate
            input_type: Type of input (query, command, file_path, etc.)
            
        Returns:
            Tuple of (is_valid, sanitized_input, violation)
        """
        if input_data is None:
            return True, None, None
        
        # Convert to string for validation
        if not isinstance(input_data, str):
            input_data = str(input_data)
        
        # Check length
        max_length = self.max_query_length if input_type == "query" else self.max_input_length
        if len(input_data) > max_length:
            violation = SecurityViolation(
                violation_type="excessive_length",
                severity="medium",
                message=f"Input exceeds maximum length of {max_length} characters",
                context={"input_type": input_type, "length": len(input_data)}
            )
            return False, None, violation
        
        # Check for injection patterns
        if self.compiled_injection.search(input_data):
            violation = SecurityViolation(
                violation_type="injection_attempt",
                severity="high",
                message="Potential injection attack detected",
                context={"input_type": input_type, "pattern": "injection"}
            )
            logger.warning(f"Potential injection attack detected in {input_type}: {input_data[:100]}")
            return False, None, violation
        
        # Check for XSS patterns
        if self.compiled_xss.search(input_data):
            violation = SecurityViolation(
                violation_type="xss_attempt",
                severity="high",
                message="Potential XSS attack detected",
                context={"input_type": input_type, "pattern": "xss"}
            )
            logger.warning(f"Potential XSS attack detected in {input_type}: {input_data[:100]}")
            return False, None, violation
        
        # Sanitize input
        sanitized = self.sanitize_input(input_data)
        
        return True, sanitized, None
    
    def sanitize_input(self, input_data: str) -> str:
        """
        Sanitize input by removing dangerous characters and patterns.
        
        Args:
            input_data: Input to sanitize
            
        Returns:
            Sanitized input
        """
        sanitized = input_data
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters (except newline, tab, carriage return)
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        # Remove potential script tags (already checked, but double-check)
        sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove potential event handlers
        sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        # Limit consecutive special characters
        sanitized = re.sub(r'[!@#$%^&*()_+=\[\]{}|;:,.<>?]{3,}', '', sanitized)
        
        return sanitized


class OutputSanitizer:
    """
    Sanitizes output to prevent security issues.
    
    Removes:
    - Sensitive information
    - Dangerous code
    - Malformed data
    """
    
    def __init__(self):
        self.sensitive_patterns = [
            r'password\s*[:=]\s*\S+',
            r'api[_-]?key\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+',
            r'token\s*[:=]\s*\S+',
            r'credential\s*[:=]\s*\S+',
        ]
        
        self.compiled_sensitive = re.compile('|'.join(self.sensitive_patterns), re.IGNORECASE)
    
    def sanitize_output(self, output: str, remove_sensitive: bool = True) -> str:
        """
        Sanitize output to prevent information leakage.
        
        Args:
            output: Output to sanitize
            remove_sensitive: Whether to remove sensitive information
            
        Returns:
            Sanitized output
        """
        sanitized = output
        
        # Remove sensitive information
        if remove_sensitive:
            sanitized = self.compiled_sensitive.sub('[REDACTED]', sanitized)
        
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Remove control characters (except newline, tab, carriage return)
        sanitized = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', sanitized)
        
        return sanitized


class RateLimiter:
    """
    Rate limiting to prevent abuse and DoS attacks.
    
    Tracks:
    - Requests per time window
    - Requests per agent
    - Requests per IP address
    """
    
    def __init__(self, config: RateLimitConfig = None):
        self.config = config or RateLimitConfig()
        self.request_history: Dict[str, List[float]] = defaultdict(list)
        self.agent_history: Dict[str, List[float]] = defaultdict(list)
        self.ip_history: Dict[str, List[float]] = defaultdict(list)
    
    def check_rate_limit(
        self,
        identifier: str,
        agent_name: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Request identifier (user ID, session ID, etc.)
            agent_name: Optional agent name for per-agent limiting
            ip_address: Optional IP address for per-IP limiting
            
        Returns:
            Tuple of (is_allowed, error_message)
        """
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(current_time)
        
        # Check global rate limit
        if not self._check_limit(
            self.request_history[identifier],
            self.config.max_requests,
            self.config.time_window,
            current_time
        ):
            return False, f"Rate limit exceeded: {self.config.max_requests} requests per {self.config.time_window}s"
        
        # Check per-agent rate limit
        if agent_name:
            if not self._check_limit(
                self.agent_history[agent_name],
                self.config.max_requests_per_agent,
                self.config.time_window,
                current_time
            ):
                return False, f"Agent rate limit exceeded: {self.config.max_requests_per_agent} requests per {self.config.time_window}s"
        
        # Check per-IP rate limit
        if ip_address:
            if not self._check_limit(
                self.ip_history[ip_address],
                self.config.max_requests_per_ip,
                self.config.time_window,
                current_time
            ):
                return False, f"IP rate limit exceeded: {self.config.max_requests_per_ip} requests per {self.config.time_window}s"
        
        # Record request
        self.request_history[identifier].append(current_time)
        if agent_name:
            self.agent_history[agent_name].append(current_time)
        if ip_address:
            self.ip_history[ip_address].append(current_time)
        
        return True, None
    
    def _check_limit(self, history: List[float], max_requests: int, time_window: float, current_time: float) -> bool:
        """Check if limit is exceeded"""
        # Remove old entries
        cutoff_time = current_time - time_window
        recent_requests = [t for t in history if t > cutoff_time]
        
        return len(recent_requests) < max_requests
    
    def _clean_old_entries(self, current_time: float):
        """Clean old entries from history"""
        cutoff_time = current_time - self.config.time_window
        
        # Clean request history
        for key in list(self.request_history.keys()):
            self.request_history[key] = [t for t in self.request_history[key] if t > cutoff_time]
            if not self.request_history[key]:
                del self.request_history[key]
        
        # Clean agent history
        for key in list(self.agent_history.keys()):
            self.agent_history[key] = [t for t in self.agent_history[key] if t > cutoff_time]
            if not self.agent_history[key]:
                del self.agent_history[key]
        
        # Clean IP history
        for key in list(self.ip_history.keys()):
            self.ip_history[key] = [t for t in self.ip_history[key] if t > cutoff_time]
            if not self.ip_history[key]:
                del self.ip_history[key]


class SecurityManager:
    """
    Centralized security management.
    
    Provides:
    - Input validation
    - Output sanitization
    - Rate limiting
    - Security violation tracking
    """
    
    def __init__(self, rate_limit_config: RateLimitConfig = None):
        self.input_validator = InputValidator()
        self.output_sanitizer = OutputSanitizer()
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.violations: List[SecurityViolation] = []
        self.max_violations = 1000
    
    def validate_and_sanitize_input(
        self,
        input_data: Any,
        input_type: str = "query"
    ) -> Tuple[bool, Optional[str], Optional[SecurityViolation]]:
        """Validate and sanitize input"""
        is_valid, sanitized, violation = self.input_validator.validate_input(input_data, input_type)
        
        if violation:
            self.record_violation(violation)
        
        return is_valid, sanitized, violation
    
    def sanitize_output(self, output: str, remove_sensitive: bool = True) -> str:
        """Sanitize output"""
        return self.output_sanitizer.sanitize_output(output, remove_sensitive)
    
    def check_rate_limit(
        self,
        identifier: str,
        agent_name: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Optional[str]]:
        """Check rate limit"""
        is_allowed, error = self.rate_limiter.check_rate_limit(identifier, agent_name, ip_address)
        
        if not is_allowed:
            violation = SecurityViolation(
                violation_type="rate_limit_exceeded",
                severity="medium",
                message=error or "Rate limit exceeded",
                context={
                    "identifier": identifier,
                    "agent_name": agent_name,
                    "ip_address": ip_address
                }
            )
            self.record_violation(violation)
        
        return is_allowed, error
    
    def record_violation(self, violation: SecurityViolation):
        """Record security violation"""
        self.violations.append(violation)
        
        # Trim violations list
        if len(self.violations) > self.max_violations:
            self.violations = self.violations[-self.max_violations:]
        
        # Log violation
        logger.warning(f"Security violation: {violation.violation_type} - {violation.message}")
    
    def get_violations(self, severity: Optional[str] = None) -> List[SecurityViolation]:
        """Get security violations, optionally filtered by severity"""
        if severity:
            return [v for v in self.violations if v.severity == severity]
        return self.violations.copy()


# Global security manager instance
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Get or create global security manager"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager

