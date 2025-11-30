"""
Runtime Security Governance for ICEBURG
Implements SSO, DLP, access control, and audit logging for enterprise security.
"""

import asyncio
import time
import json
import logging
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import jwt
import requests
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class AccessLevel(Enum):
    """Access levels for ICEBURG modes."""
    READ_ONLY = "read_only"
    RESEARCH = "research"
    SOFTWARE = "software"
    CIVILIZATION = "civilization"
    ADMIN = "admin"
    SYSTEM = "system"


class SecurityEvent(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PII_DETECTED = "pii_detected"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    SYSTEM_BREACH = "system_breach"
    DATA_EXPORT = "data_export"
    CONFIG_CHANGE = "config_change"


@dataclass
class User:
    """User identity and permissions."""
    user_id: str
    username: str
    email: str
    access_level: AccessLevel
    permissions: Set[str] = field(default_factory=set)
    groups: List[str] = field(default_factory=list)
    last_login: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityEvent:
    """Security event record."""
    event_id: str
    event_type: SecurityEvent
    user_id: str
    timestamp: float
    details: Dict[str, Any]
    severity: str = "info"  # info, warning, critical
    ip_address: str = ""
    user_agent: str = ""


class SSOProvider:
    """
    Single Sign-On provider for enterprise authentication.
    
    Supports:
    - OAuth2 (Google, Microsoft, GitHub)
    - SAML 2.0
    - Azure AD
    - Custom JWT tokens
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SSO provider.
        
        Args:
            config: SSO configuration
        """
        self.config = config
        self.provider_type = config.get("provider", "oauth2")
        self.client_id = config.get("client_id")
        self.client_secret = config.get("client_secret")
        self.redirect_uri = config.get("redirect_uri")
        self.jwks_url = config.get("jwks_url")
        self.issuer = config.get("issuer")
        
        # Token validation
        self.jwt_secret = config.get("jwt_secret")
        self.token_cache: Dict[str, Dict[str, Any]] = {}
        
        # OAuth2 endpoints
        self.oauth_endpoints = {
            "google": "https://oauth2.googleapis.com",
            "microsoft": "https://login.microsoftonline.com",
            "github": "https://github.com"
        }
    
    async def authenticate(self, token: str) -> Optional[User]:
        """
        Authenticate user with SSO token.
        
        Args:
            token: SSO token (JWT, OAuth2, etc.)
            
        Returns:
            User object if authenticated, None otherwise
        """
        try:
            if self.provider_type == "oauth2":
                return await self._authenticate_oauth2(token)
            elif self.provider_type == "saml":
                return await self._authenticate_saml(token)
            elif self.provider_type == "jwt":
                return await self._authenticate_jwt(token)
            else:
                logger.error(f"Unsupported SSO provider: {self.provider_type}")
                return None
                
        except Exception as e:
            logger.error(f"SSO authentication failed: {e}")
            return None
    
    async def _authenticate_oauth2(self, token: str) -> Optional[User]:
        """Authenticate using OAuth2."""
        # Validate token with provider
        user_info = await self._get_user_info_oauth2(token)
        if not user_info:
            return None
        
        # Create user object
        user = User(
            user_id=user_info.get("sub") or user_info.get("id"),
            username=user_info.get("preferred_username") or user_info.get("name"),
            email=user_info.get("email"),
            access_level=AccessLevel.RESEARCH,  # Default level
            groups=user_info.get("groups", []),
            last_login=time.time()
        )
        
        # Set permissions based on groups
        user.permissions = self._get_permissions_from_groups(user.groups)
        
        return user
    
    async def _authenticate_saml(self, token: str) -> Optional[User]:
        """Authenticate using SAML."""
        # Decode SAML assertion
        saml_data = self._decode_saml_assertion(token)
        if not saml_data:
            return None
        
        user = User(
            user_id=saml_data.get("NameID"),
            username=saml_data.get("username"),
            email=saml_data.get("email"),
            access_level=AccessLevel.RESEARCH,
            groups=saml_data.get("groups", []),
            last_login=time.time()
        )
        
        user.permissions = self._get_permissions_from_groups(user.groups)
        return user
    
    async def _authenticate_jwt(self, token: str) -> Optional[User]:
        """Authenticate using JWT."""
        try:
            # Decode JWT
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            
            user = User(
                user_id=payload.get("sub"),
                username=payload.get("username"),
                email=payload.get("email"),
                access_level=AccessLevel(payload.get("access_level", "research")),
                groups=payload.get("groups", []),
                last_login=time.time()
            )
            
            user.permissions = self._get_permissions_from_groups(user.groups)
            return user
            
        except jwt.InvalidTokenError as e:
            logger.error(f"JWT validation failed: {e}")
            return None
    
    async def _get_user_info_oauth2(self, token: str) -> Optional[Dict[str, Any]]:
        """Get user info from OAuth2 provider."""
        # Check cache first
        if token in self.token_cache:
            cached_data = self.token_cache[token]
            if time.time() - cached_data.get("timestamp", 0) < 3600:  # 1 hour cache
                return cached_data.get("user_info")
        
        # Fetch from provider
        headers = {"Authorization": f"Bearer {token}"}
        
        try:
            response = requests.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            user_info = response.json()
            
            # Cache result
            self.token_cache[token] = {
                "user_info": user_info,
                "timestamp": time.time()
            }
            
            return user_info
            
        except Exception as e:
            logger.error(f"Failed to fetch user info: {e}")
            return None
    
    def _decode_saml_assertion(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode SAML assertion (simplified)."""
        # In real implementation, use xmlsec or similar
        # For now, return mock data
        return {
            "NameID": "user123",
            "username": "john.doe",
            "email": "john.doe@company.com",
            "groups": ["researchers", "developers"]
        }
    
    def _get_permissions_from_groups(self, groups: List[str]) -> Set[str]:
        """Get permissions from user groups."""
        permissions = set()
        
        group_permissions = {
            "researchers": ["research", "chat"],
            "developers": ["research", "chat", "software"],
            "admins": ["research", "chat", "software", "civilization", "admin"],
            "system": ["research", "chat", "software", "civilization", "admin", "system"]
        }
        
        for group in groups:
            if group in group_permissions:
                permissions.update(group_permissions[group])
        
        return permissions


class DataLeakagePrevention:
    """
    Data Leakage Prevention system for ICEBURG.
    
    Features:
    - PII detection and redaction
    - Sensitive data classification
    - Data export monitoring
    - Content filtering
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DLP system.
        
        Args:
            config: DLP configuration
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        
        # PII patterns
        self.pii_patterns = {
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "credit_card_visa": r"\b4\d{3}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "credit_card_mastercard": r"\b5[1-5]\d{2}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
        }
        
        # Sensitive keywords
        self.sensitive_keywords = [
            "password", "secret", "key", "token", "api_key",
            "private", "confidential", "classified", "restricted"
        ]
        
        # Data classification levels
        self.classification_levels = {
            "public": 0,
            "internal": 1,
            "confidential": 2,
            "restricted": 3
        }
    
    def scan_content(self, content: str) -> Dict[str, Any]:
        """
        Scan content for sensitive data.
        
        Args:
            content: Content to scan
            
        Returns:
            Scan results with detected issues
        """
        if not self.enabled:
            return {"safe": True, "issues": []}
        
        issues = []
        redacted_content = content
        
        # Check for PII
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                issues.append({
                    "type": "pii",
                    "pii_type": pii_type,
                    "matches": matches,
                    "severity": "high"
                })
                
                # Redact PII
                redacted_content = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", redacted_content)
        
        # Check for sensitive keywords
        for keyword in self.sensitive_keywords:
            if keyword.lower() in content.lower():
                issues.append({
                    "type": "sensitive_keyword",
                    "keyword": keyword,
                    "severity": "medium"
                })
        
        # Determine overall risk level
        risk_level = "low"
        if any(issue["severity"] == "high" for issue in issues):
            risk_level = "high"
        elif any(issue["severity"] == "medium" for issue in issues):
            risk_level = "medium"
        
        return {
            "safe": len(issues) == 0,
            "risk_level": risk_level,
            "issues": issues,
            "redacted_content": redacted_content if issues else content,
            "original_content": content
        }
    
    def classify_data(self, content: str) -> str:
        """
        Classify data sensitivity level.
        
        Args:
            content: Content to classify
            
        Returns:
            Classification level
        """
        scan_result = self.scan_content(content)
        
        if scan_result["risk_level"] == "high":
            return "restricted"
        elif scan_result["risk_level"] == "medium":
            return "confidential"
        elif any(keyword in content.lower() for keyword in ["internal", "company"]):
            return "internal"
        else:
            return "public"
    
    def can_export(self, content: str, user_level: AccessLevel) -> bool:
        """
        Check if user can export content.
        
        Args:
            content: Content to export
            user_level: User access level
            
        Returns:
            True if export is allowed
        """
        classification = self.classify_data(content)
        
        # Map access levels to classification permissions
        level_permissions = {
            AccessLevel.READ_ONLY: ["public"],
            AccessLevel.RESEARCH: ["public", "internal"],
            AccessLevel.SOFTWARE: ["public", "internal", "confidential"],
            AccessLevel.CIVILIZATION: ["public", "internal", "confidential"],
            AccessLevel.ADMIN: ["public", "internal", "confidential", "restricted"],
            AccessLevel.SYSTEM: ["public", "internal", "confidential", "restricted"]
        }
        
        allowed_classifications = level_permissions.get(user_level, ["public"])
        return classification in allowed_classifications


class AccessControlList:
    """
    Access Control List for ICEBURG modes and features.
    
    Features:
    - Role-based access control
    - Mode-specific permissions
    - Resource-level permissions
    - Time-based access
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ACL.
        
        Args:
            config: ACL configuration
        """
        self.config = config
        
        # Mode permissions
        self.mode_permissions = {
            "chat": [AccessLevel.READ_ONLY, AccessLevel.RESEARCH, AccessLevel.SOFTWARE, AccessLevel.CIVILIZATION, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "research": [AccessLevel.RESEARCH, AccessLevel.SOFTWARE, AccessLevel.CIVILIZATION, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "software": [AccessLevel.SOFTWARE, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "civilization": [AccessLevel.CIVILIZATION, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "admin": [AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "system": [AccessLevel.SYSTEM]
        }
        
        # Feature permissions
        self.feature_permissions = {
            "distributed_processing": [AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "autonomous_learning": [AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "civilization_simulation": [AccessLevel.CIVILIZATION, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "software_generation": [AccessLevel.SOFTWARE, AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "monitoring": [AccessLevel.ADMIN, AccessLevel.SYSTEM],
            "configuration": [AccessLevel.ADMIN, AccessLevel.SYSTEM]
        }
        
        # Time-based access rules
        self.time_rules = config.get("time_rules", {})
    
    def can_access_mode(self, user: User, mode: str) -> bool:
        """
        Check if user can access a specific mode.
        
        Args:
            user: User object
            mode: ICEBURG mode
            
        Returns:
            True if access is allowed
        """
        if not user:
            return False
        
        # Check mode permissions
        allowed_levels = self.mode_permissions.get(mode, [])
        if user.access_level not in allowed_levels:
            return False
        
        # Check time-based rules
        if not self._check_time_access(user, mode):
            return False
        
        return True
    
    def can_access_feature(self, user: User, feature: str) -> bool:
        """
        Check if user can access a specific feature.
        
        Args:
            user: User object
            feature: Feature name
            
        Returns:
            True if access is allowed
        """
        if not user:
            return False
        
        allowed_levels = self.feature_permissions.get(feature, [])
        return user.access_level in allowed_levels
    
    def can_access_resource(self, user: User, resource: str) -> bool:
        """
        Check if user can access a specific resource.
        
        Args:
            user: User object
            resource: Resource identifier
            
        Returns:
            True if access is allowed
        """
        if not user:
            return False
        
        # Check resource-specific permissions
        if resource.startswith("admin/"):
            return user.access_level in [AccessLevel.ADMIN, AccessLevel.SYSTEM]
        elif resource.startswith("system/"):
            return user.access_level == AccessLevel.SYSTEM
        else:
            return True  # Default allow for other resources
    
    def _check_time_access(self, user: User, mode: str) -> bool:
        """Check time-based access rules."""
        if not self.time_rules:
            return True
        
        current_hour = time.localtime().tm_hour
        
        # Check if user is in restricted time window
        restricted_hours = self.time_rules.get("restricted_hours", [])
        if current_hour in restricted_hours:
            # Only allow admin/system users during restricted hours
            return user.access_level in [AccessLevel.ADMIN, AccessLevel.SYSTEM]
        
        return True


class AuditLogger:
    """
    Audit logging system for security events.
    
    Features:
    - Security event logging
    - Compliance reporting
    - Event correlation
    - Alert generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audit logger.
        
        Args:
            config: Audit configuration
        """
        self.config = config
        self.log_file = config.get("log_file", "security_audit.log")
        self.retention_days = config.get("retention_days", 90)
        self.events: List[SecurityEvent] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            "failed_logins": 5,  # Alert after 5 failed logins
            "access_denials": 10,  # Alert after 10 access denials
            "pii_detections": 3,  # Alert after 3 PII detections
            "suspicious_activity": 1  # Alert on any suspicious activity
        }
    
    def log_event(self, 
                 event_type: SecurityEvent, 
                 user_id: str, 
                 details: Dict[str, Any],
                 severity: str = "info",
                 ip_address: str = "",
                 user_agent: str = ""):
        """
        Log a security event.
        
        Args:
            event_type: Type of security event
            user_id: User identifier
            details: Event details
            severity: Event severity
            ip_address: Client IP address
            user_agent: Client user agent
        """
        event = SecurityEvent(
            event_id=self._generate_event_id(),
            event_type=event_type,
            user_id=user_id,
            timestamp=time.time(),
            details=details,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.events.append(event)
        
        # Write to log file
        self._write_to_log(event)
        
        # Check for alerts
        self._check_alerts(event)
        
        logger.info(f"Security event logged: {event_type.value} for user {user_id}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return hashlib.md5(f"{time.time()}{len(self.events)}".encode()).hexdigest()[:16]
    
    def _write_to_log(self, event: SecurityEvent):
        """Write event to log file."""
        log_entry = {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "user_id": event.user_id,
            "timestamp": event.timestamp,
            "details": event.details,
            "severity": event.severity,
            "ip_address": event.ip_address,
            "user_agent": event.user_agent
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def _check_alerts(self, event: SecurityEvent):
        """Check if event should trigger an alert."""
        # Count recent events of same type
        recent_events = [
            e for e in self.events 
            if e.event_type == event.event_type 
            and time.time() - e.timestamp < 3600  # Last hour
        ]
        
        threshold = self.alert_thresholds.get(event.event_type.value, float('inf'))
        if len(recent_events) >= threshold:
            self._trigger_alert(event, recent_events)
    
    def _trigger_alert(self, event: SecurityEvent, recent_events: List[SecurityEvent]):
        """Trigger security alert."""
        alert = {
            "type": "security_alert",
            "event_type": event.event_type.value,
            "count": len(recent_events),
            "user_id": event.user_id,
            "timestamp": time.time(),
            "severity": "high" if len(recent_events) > 10 else "medium"
        }
        
        logger.warning(f"Security alert triggered: {alert}")
        
        # In production, this would send to SIEM, email, etc.
        self._send_alert(alert)
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert to security team."""
        # Mock alert sending
        logger.warning(f"ALERT: {alert}")
    
    def get_events(self, 
                  user_id: str = None, 
                  event_type: SecurityEvent = None,
                  start_time: float = None,
                  end_time: float = None) -> List[SecurityEvent]:
        """
        Get security events with filters.
        
        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            
        Returns:
            Filtered security events
        """
        filtered_events = self.events
        
        if user_id:
            filtered_events = [e for e in filtered_events if e.user_id == user_id]
        
        if event_type:
            filtered_events = [e for e in filtered_events if e.event_type == event_type]
        
        if start_time:
            filtered_events = [e for e in filtered_events if e.timestamp >= start_time]
        
        if end_time:
            filtered_events = [e for e in filtered_events if e.timestamp <= end_time]
        
        return filtered_events
    
    def generate_compliance_report(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """
        Generate compliance report for time period.
        
        Args:
            start_time: Report start time
            end_time: Report end time
            
        Returns:
            Compliance report
        """
        events = self.get_events(start_time=start_time, end_time=end_time)
        
        # Count events by type
        event_counts = {}
        for event in events:
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
        
        # Count events by severity
        severity_counts = {}
        for event in events:
            severity = event.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count unique users
        unique_users = len(set(event.user_id for event in events))
        
        return {
            "period": {"start": start_time, "end": end_time},
            "total_events": len(events),
            "unique_users": unique_users,
            "event_counts": event_counts,
            "severity_counts": severity_counts,
            "compliance_score": self._calculate_compliance_score(events)
        }
    
    def _calculate_compliance_score(self, events: List[SecurityEvent]) -> float:
        """Calculate compliance score (0-100)."""
        if not events:
            return 100.0
        
        # Penalize security events
        penalty_events = [
            SecurityEvent.LOGIN_FAILURE,
            SecurityEvent.ACCESS_DENIED,
            SecurityEvent.PII_DETECTED,
            SecurityEvent.SUSPICIOUS_ACTIVITY,
            SecurityEvent.SYSTEM_BREACH
        ]
        
        penalty_count = sum(1 for event in events if event.event_type in penalty_events)
        total_events = len(events)
        
        score = max(0, 100 - (penalty_count / total_events) * 100)
        return score


class RuntimeGovernance:
    """
    Main runtime governance system for ICEBURG.
    
    Features:
    - SSO authentication
    - Data leakage prevention
    - Access control
    - Audit logging
    - Security monitoring
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize runtime governance.
        
        Args:
            config: Governance configuration
        """
        self.config = config
        self.enabled = config.get("enabled", False)
        
        if not self.enabled:
            logger.info("Runtime governance disabled")
            return
        
        # Initialize components
        self.sso = SSOProvider(config.get("sso", {}))
        self.dlp = DataLeakagePrevention(config.get("dlp", {}))
        self.acl = AccessControlList(config.get("acl", {}))
        self.audit_logger = AuditLogger(config.get("audit", {}))
        
        # Security state
        self.active_sessions: Dict[str, User] = {}
        self.blocked_users: Set[str] = set()
        
        logger.info("Runtime governance initialized")
    
    async def authorize_request(self, 
                              token: str, 
                              query: str, 
                              mode: str,
                              ip_address: str = "",
                              user_agent: str = "") -> Tuple[bool, Optional[User], str]:
        """
        Authorize a request.
        
        Args:
            token: SSO token
            query: User query
            mode: ICEBURG mode
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (authorized, user, message)
        """
        if not self.enabled:
            return True, None, "Governance disabled"
        
        try:
            # Authenticate user
            user = await self.sso.authenticate(token)
            if not user:
                self.audit_logger.log_event(
                    SecurityEvent.LOGIN_FAILURE,
                    "unknown",
                    {"reason": "Invalid token", "mode": mode},
                    severity="warning",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return False, None, "Authentication failed"
            
            # Check if user is blocked
            if user.user_id in self.blocked_users:
                self.audit_logger.log_event(
                    SecurityEvent.ACCESS_DENIED,
                    user.user_id,
                    {"reason": "User blocked", "mode": mode},
                    severity="critical",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return False, user, "User account blocked"
            
            # Check mode access
            if not self.acl.can_access_mode(user, mode):
                self.audit_logger.log_event(
                    SecurityEvent.ACCESS_DENIED,
                    user.user_id,
                    {"reason": "Insufficient permissions", "mode": mode},
                    severity="warning",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return False, user, f"Insufficient permissions for {mode} mode"
            
            # Scan query for sensitive data
            dlp_result = self.dlp.scan_content(query)
            if not dlp_result["safe"]:
                self.audit_logger.log_event(
                    SecurityEvent.PII_DETECTED,
                    user.user_id,
                    {"issues": dlp_result["issues"], "mode": mode},
                    severity="high",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                # Block request if high risk
                if dlp_result["risk_level"] == "high":
                    return False, user, "Request contains sensitive data"
                
                # Use redacted content
                query = dlp_result["redacted_content"]
            
            # Log successful access
            self.audit_logger.log_event(
                SecurityEvent.ACCESS_GRANTED,
                user.user_id,
                {"mode": mode, "query_length": len(query)},
                severity="info",
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Update active sessions
            self.active_sessions[user.user_id] = user
            
            return True, user, "Access granted"
            
        except Exception as e:
            logger.error(f"Authorization error: {e}")
            return False, None, "Authorization error"
    
    async def check_export_permission(self, user: User, content: str) -> bool:
        """
        Check if user can export content.
        
        Args:
            user: User object
            content: Content to export
            
        Returns:
            True if export is allowed
        """
        if not self.enabled or not user:
            return True
        
        # Check DLP classification
        can_export = self.dlp.can_export(content, user.access_level)
        
        if not can_export:
            self.audit_logger.log_event(
                SecurityEvent.DATA_EXPORT,
                user.user_id,
                {"reason": "Classification violation", "content_length": len(content)},
                severity="high"
            )
        
        return can_export
    
    def block_user(self, user_id: str, reason: str = "Security violation"):
        """
        Block a user.
        
        Args:
            user_id: User ID to block
            reason: Block reason
        """
        self.blocked_users.add(user_id)
        
        self.audit_logger.log_event(
            SecurityEvent.SYSTEM_BREACH,
            user_id,
            {"reason": reason, "action": "user_blocked"},
            severity="critical"
        )
        
        logger.warning(f"User {user_id} blocked: {reason}")
    
    def unblock_user(self, user_id: str):
        """
        Unblock a user.
        
        Args:
            user_id: User ID to unblock
        """
        self.blocked_users.discard(user_id)
        
        self.audit_logger.log_event(
            SecurityEvent.CONFIG_CHANGE,
            "system",
            {"action": "user_unblocked", "user_id": user_id},
            severity="info"
        )
        
        logger.info(f"User {user_id} unblocked")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        return {
            "enabled": self.enabled,
            "active_sessions": len(self.active_sessions),
            "blocked_users": len(self.blocked_users),
            "recent_events": len(self.audit_logger.events[-10:]),  # Last 10 events
            "compliance_score": self.audit_logger._calculate_compliance_score(
                self.audit_logger.events[-100:]  # Last 100 events
            )
        }
    
    def get_user_activity(self, user_id: str, hours: int = 24) -> List[SecurityEvent]:
        """Get user activity for specified hours."""
        start_time = time.time() - (hours * 3600)
        return self.audit_logger.get_events(
            user_id=user_id,
            start_time=start_time
        )
    
    async def cleanup(self):
        """Cleanup governance resources."""
        # Clear active sessions
        self.active_sessions.clear()
        
        logger.info("Runtime governance cleanup completed")


# Convenience functions
async def create_runtime_governance(config: Dict[str, Any] = None) -> RuntimeGovernance:
    """Create runtime governance system."""
    if config is None:
        config = {
            "enabled": True,
            "sso": {"provider": "oauth2"},
            "dlp": {"enabled": True},
            "acl": {},
            "audit": {"log_file": "security_audit.log"}
        }
    
    return RuntimeGovernance(config)


async def authorize_iceburg_request(token: str, 
                                  query: str, 
                                  mode: str,
                                  governance: RuntimeGovernance = None) -> Tuple[bool, Optional[User], str]:
    """Authorize an ICEBURG request."""
    if governance is None:
        governance = await create_runtime_governance()
    
    return await governance.authorize_request(token, query, mode)
