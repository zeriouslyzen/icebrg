"""
Rate Limiter
Rate limiting per API key and endpoint
"""

from typing import Any, Dict, Optional, List
import time
from collections import defaultdict
from datetime import datetime, timedelta


class RateLimiter:
    """Rate limiting manager"""
    
    def __init__(self):
        self.limits: Dict[str, Dict[str, int]] = {
            "default": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000,
                "requests_per_day": 10000
            }
        }
        self.requests: Dict[str, List[float]] = defaultdict(list)
        self.blocked: Dict[str, float] = {}
    
    def set_limit(
        self,
        api_key: str,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000,
        requests_per_day: int = 10000
    ) -> bool:
        """Set rate limit for API key"""
        self.limits[api_key] = {
            "requests_per_minute": requests_per_minute,
            "requests_per_hour": requests_per_hour,
            "requests_per_day": requests_per_day
        }
        return True
    
    def check_rate_limit(self, api_key: str) -> Dict[str, Any]:
        """Check if request is within rate limit"""
        result = {
            "allowed": True,
            "remaining": 0,
            "reset_at": None,
            "limit": 0
        }
        
        # Check if blocked
        if api_key in self.blocked:
            block_until = self.blocked[api_key]
            if time.time() < block_until:
                result["allowed"] = False
                result["reset_at"] = datetime.fromtimestamp(block_until).isoformat()
                return result
            else:
                del self.blocked[api_key]
        
        # Get limits
        limits = self.limits.get(api_key, self.limits["default"])
        
        # Get request history
        now = time.time()
        requests = self.requests[api_key]
        
        # Clean old requests
        requests[:] = [r for r in requests if now - r < 86400]  # Keep last 24 hours
        
        # Check per-minute limit
        recent_minute = [r for r in requests if now - r < 60]
        if len(recent_minute) >= limits["requests_per_minute"]:
            result["allowed"] = False
            result["limit"] = limits["requests_per_minute"]
            result["remaining"] = 0
            # Block for 1 minute
            self.blocked[api_key] = now + 60
            result["reset_at"] = datetime.fromtimestamp(now + 60).isoformat()
            return result
        
        # Check per-hour limit
        recent_hour = [r for r in requests if now - r < 3600]
        if len(recent_hour) >= limits["requests_per_hour"]:
            result["allowed"] = False
            result["limit"] = limits["requests_per_hour"]
            result["remaining"] = 0
            # Block for 1 hour
            self.blocked[api_key] = now + 3600
            result["reset_at"] = datetime.fromtimestamp(now + 3600).isoformat()
            return result
        
        # Check per-day limit
        recent_day = [r for r in requests if now - r < 86400]
        if len(recent_day) >= limits["requests_per_day"]:
            result["allowed"] = False
            result["limit"] = limits["requests_per_day"]
            result["remaining"] = 0
            # Block for 24 hours
            self.blocked[api_key] = now + 86400
            result["reset_at"] = datetime.fromtimestamp(now + 86400).isoformat()
            return result
        
        # Record request
        requests.append(now)
        
        # Calculate remaining
        result["remaining"] = limits["requests_per_minute"] - len(recent_minute)
        result["limit"] = limits["requests_per_minute"]
        
        return result
    
    def get_usage(self, api_key: str) -> Dict[str, Any]:
        """Get usage statistics for API key"""
        now = time.time()
        requests = self.requests.get(api_key, [])
        limits = self.limits.get(api_key, self.limits["default"])
        
        recent_minute = [r for r in requests if now - r < 60]
        recent_hour = [r for r in requests if now - r < 3600]
        recent_day = [r for r in requests if now - r < 86400]
        
        return {
            "api_key": api_key,
            "requests_per_minute": len(recent_minute),
            "requests_per_hour": len(recent_hour),
            "requests_per_day": len(recent_day),
            "limits": limits,
            "blocked": api_key in self.blocked,
            "blocked_until": (
                datetime.fromtimestamp(self.blocked[api_key]).isoformat()
                if api_key in self.blocked
                else None
            )
        }
    
    def reset_usage(self, api_key: str) -> bool:
        """Reset usage for API key"""
        if api_key in self.requests:
            del self.requests[api_key]
        if api_key in self.blocked:
            del self.blocked[api_key]
        return True

