"""
ICEBURG API Security Module
Security headers, input validation, and security utilities
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable, List, Optional
import re
import html
import logging
import os
from datetime import datetime, timedelta
import secrets
import time

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Content Security Policy
        # In development, allow all connections for network access (iPhone, etc.)
        # In production, restrict to specific origins
        if os.getenv("ENVIRONMENT") == "production":
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com; "
                "img-src 'self' data: https:; "
                "font-src 'self' https://cdn.jsdelivr.net; "
                "connect-src 'self' ws://localhost:8000 wss://localhost:8000 http://localhost:8000 https://localhost:8000; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self'"
            )
        else:
            # Development: Allow all connections for network access
            csp = (
                "default-src * 'unsafe-inline' 'unsafe-eval' data: blob:; "
                "script-src * 'unsafe-inline' 'unsafe-eval'; "
                "style-src * 'unsafe-inline'; "
                "img-src * data: blob:; "
                "font-src * data:; "
                "connect-src * ws: wss: http: https:; "
                "frame-ancestors *; "
                "base-uri *; "
                "form-action *"
            )
        response.headers["Content-Security-Policy"] = csp
        
        # HSTS (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with per-IP tracking and burst protection"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        # PHASE 1.5: Burst protection - allow short bursts but limit sustained rate
        self.burst_size = int(os.getenv("RATE_LIMIT_BURST_SIZE", "20"))  # Allow 20 requests in short burst
        self.burst_window = timedelta(seconds=10)  # 10 second window for bursts
        self.request_counts = {}  # Per-IP request timestamps
        self.last_cleanup = datetime.now()
        self.cleanup_interval = timedelta(minutes=5)  # Clean up every 5 minutes
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now()
        
        # Exempt certain paths from rate limiting
        exempt_paths = ["/api/matrix/", "/ws", "/api/health"]
        path = request.url.path
        if any(path.startswith(p) for p in exempt_paths):
            return await call_next(request)
        
        # Periodic cleanup of old entries
        if now - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(now)
            self.last_cleanup = now
        
        # Get or initialize request history for this IP
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []
        
        recent_requests = self.request_counts[client_ip]
        
        # PHASE 1.5: Burst protection - check short-term burst first
        burst_cutoff = now - self.burst_window
        burst_requests = [t for t in recent_requests if t > burst_cutoff]
        if len(burst_requests) >= self.burst_size:
            logger.warning(f"Rate limit burst exceeded for IP {client_ip}: {len(burst_requests)} requests in {self.burst_window.total_seconds()}s")
            return Response(
                content='{"error": "Rate limit exceeded (burst protection)"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "10"}
            )
        
        # Check sustained rate limit (per minute)
        minute_cutoff = now - timedelta(minutes=1)
        minute_requests = [t for t in recent_requests if t > minute_cutoff]
        if len(minute_requests) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP {client_ip}: {len(minute_requests)} requests per minute")
            # PHASE 2.1: Log security event
            try:
                security_logger = logging.getLogger("security")
                security_logger.warning(f"RATE_LIMIT_EXCEEDED: IP={client_ip}, Requests={len(minute_requests)}, Limit={self.requests_per_minute}")
            except:
                pass
            return Response(
                content='{"error": "Rate limit exceeded"}',
                status_code=429,
                media_type="application/json",
                headers={"Retry-After": "60"}
            )
        
        # Add current request
        recent_requests.append(now)
        # Keep only requests from last 2 minutes (for cleanup efficiency)
        self.request_counts[client_ip] = [t for t in recent_requests if t > (now - timedelta(minutes=2))]
        
        return await call_next(request)
    
    def _cleanup_old_entries(self, now: datetime):
        """Remove entries older than 2 minutes"""
        cutoff = now - timedelta(minutes=2)
        for ip in list(self.request_counts.keys()):
            self.request_counts[ip] = [t for t in self.request_counts[ip] if t > cutoff]
            if not self.request_counts[ip]:
                del self.request_counts[ip]


def sanitize_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user input to prevent XSS and injection attacks"""
    if not isinstance(text, str):
        return ""
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Escape HTML entities
    text = html.escape(text)
    
    # Remove potentially dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'on\w+\s*=',
        r'<iframe[^>]*>',
        r'<object[^>]*>',
        r'<embed[^>]*>',
    ]
    
    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text


def validate_query(query: str) -> tuple[bool, Optional[str]]:
    """Validate query input"""
    if not query or not isinstance(query, str):
        return False, "Query must be a non-empty string"
    
    if len(query) > 10000:
        return False, "Query is too long (max 10000 characters)"
    
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\.\.\/',  # Path traversal
        r'<script',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'eval\s*\(',  # Eval calls
        r'exec\s*\(',  # Exec calls
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"Suspicious pattern detected: {pattern}"
    
    return True, None


def validate_file_upload(filename: str, content_type: str, file_size: int) -> tuple[bool, Optional[str]]:
    """Validate file upload"""
    # Allowed file types
    allowed_types = [
        'image/jpeg', 'image/png', 'image/gif', 'image/webp',
        'application/pdf',
        'text/plain', 'text/markdown',
        'application/json',
        'text/csv',
        'text/x-python', 'application/javascript', 'text/typescript'
    ]
    
    # Allowed extensions
    allowed_extensions = [
        '.jpg', '.jpeg', '.png', '.gif', '.webp',
        '.pdf',
        '.txt', '.md',
        '.json',
        '.csv',
        '.py', '.js', '.ts'
    ]
    
    # PHASE 1.4: Check file size - configurable via environment variable
    max_size = int(os.getenv("MAX_FILE_SIZE", str(10 * 1024 * 1024)))  # Default 10MB
    if file_size > max_size:
        return False, f"File size exceeds maximum ({max_size / 1024 / 1024}MB)"
    
    # Check content type
    if content_type not in allowed_types:
        return False, f"File type not allowed: {content_type}"
    
    # Check extension
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return False, f"File extension not allowed: {filename}"
    
    # Check filename for path traversal
    if '..' in filename or '/' in filename or '\\' in filename:
        return False, "Invalid filename"
    
    return True, None


def generate_csrf_token() -> str:
    """Generate CSRF token"""
    return secrets.token_urlsafe(32)


def verify_csrf_token(token: str, session_token: str) -> bool:
    """Verify CSRF token"""
    return secrets.compare_digest(token, session_token)

