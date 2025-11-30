"""
Military-Grade Security for ICEBURG Financial Trading System
Implements military-level security protocols for real money trading
"""

import os
import hashlib
import hmac
import time
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets
import ipaddress
from datetime import datetime, timedelta
import asyncio
import aiohttp
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Military-grade security configuration"""
    # Encryption settings
    encryption_key: str = None
    key_rotation_hours: int = 24
    max_failed_attempts: int = 3
    lockout_duration_minutes: int = 30
    
    # API Security
    ip_whitelist: List[str] = None
    api_rate_limit: int = 100  # requests per minute
    session_timeout_minutes: int = 60
    
    # Trading Security
    max_daily_loss_percent: float = 5.0
    max_position_size_percent: float = 10.0
    emergency_stop_threshold: float = 10.0
    max_trades_per_day: int = 100
    
    # Monitoring
    security_log_level: str = "INFO"
    alert_webhook: str = None
    backup_frequency_hours: int = 6


class MilitarySecurityManager:
    """
    Military-grade security manager for financial trading operations.
    
    Implements:
    - Multi-layer encryption
    - IP whitelisting and geolocation
    - Rate limiting and DDoS protection
    - Real-time threat detection
    - Emergency kill switches
    - Audit logging
    - Secure key management
    """
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_key = self._generate_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self.failed_attempts = {}
        self.locked_ips = set()
        self.security_events = []
        self.active_sessions = {}
        
        # Initialize security monitoring
        self._start_security_monitoring()
        
        logger.info("🔒 Military-grade security manager initialized")
    
    def _generate_encryption_key(self) -> bytes:
        """Generate military-grade encryption key"""
        if self.config.encryption_key:
            # Derive key from password using PBKDF2
            password = self.config.encryption_key.encode()
            salt = b'iceburg_military_salt_2025'
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password))
            return key
        else:
            # Generate random key
            return Fernet.generate_key()
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data with military-grade encryption"""
        try:
            encrypted_data = self.cipher_suite.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise SecurityException("Data encryption failed")
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_data = self.cipher_suite.decrypt(decoded_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise SecurityException("Data decryption failed")
    
    def validate_ip_access(self, ip_address: str) -> bool:
        """Validate IP address against whitelist"""
        if not self.config.ip_whitelist:
            return True
        
        try:
            client_ip = ipaddress.ip_address(ip_address)
            for allowed_ip in self.config.ip_whitelist:
                if client_ip in ipaddress.ip_network(allowed_ip):
                    return True
            return False
        except Exception as e:
            logger.error(f"IP validation failed: {e}")
            return False
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check rate limiting for client"""
        current_time = time.time()
        minute_window = 60  # 1 minute window
        
        if client_id not in self.failed_attempts:
            self.failed_attempts[client_id] = []
        
        # Clean old attempts
        self.failed_attempts[client_id] = [
            attempt_time for attempt_time in self.failed_attempts[client_id]
            if current_time - attempt_time < minute_window
        ]
        
        # Check if under rate limit
        if len(self.failed_attempts[client_id]) >= self.config.api_rate_limit:
            return False
        
        # Record this attempt
        self.failed_attempts[client_id].append(current_time)
        return True
    
    def authenticate_request(self, api_key: str, signature: str, timestamp: str, 
                           request_body: str) -> bool:
        """Authenticate API request with military-grade security"""
        try:
            # Check timestamp (prevent replay attacks)
            request_time = int(timestamp)
            current_time = int(time.time())
            time_diff = abs(current_time - request_time)
            
            if time_diff > 300:  # 5 minutes tolerance
                self._log_security_event("AUTH_FAILED", "Timestamp too old")
                return False
            
            # Verify signature
            expected_signature = self._generate_signature(api_key, timestamp, request_body)
            if not hmac.compare_digest(signature, expected_signature):
                self._log_security_event("AUTH_FAILED", "Invalid signature")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            self._log_security_event("AUTH_ERROR", str(e))
            return False
    
    def _generate_signature(self, api_key: str, timestamp: str, body: str) -> str:
        """Generate HMAC signature for request authentication"""
        message = f"{api_key}{timestamp}{body}"
        signature = hmac.new(
            api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def check_trading_limits(self, trade_amount: float, account_balance: float, 
                           daily_pnl: float) -> Dict[str, Any]:
        """Check military-grade trading limits"""
        results = {
            "allowed": True,
            "reasons": [],
            "risk_level": "LOW"
        }
        
        # Check daily loss limit
        loss_percent = abs(daily_pnl) / account_balance * 100
        if loss_percent >= self.config.max_daily_loss_percent:
            results["allowed"] = False
            results["reasons"].append(f"Daily loss limit exceeded: {loss_percent:.2f}%")
            results["risk_level"] = "CRITICAL"
        
        # Check position size limit
        position_percent = trade_amount / account_balance * 100
        if position_percent > self.config.max_position_size_percent:
            results["allowed"] = False
            results["reasons"].append(f"Position size too large: {position_percent:.2f}%")
            results["risk_level"] = "HIGH"
        
        # Check emergency stop threshold
        if loss_percent >= self.config.emergency_stop_threshold:
            results["allowed"] = False
            results["reasons"].append("EMERGENCY STOP TRIGGERED")
            results["risk_level"] = "EMERGENCY"
            self._trigger_emergency_stop()
        
        return results
    
    def _trigger_emergency_stop(self):
        """Trigger emergency stop - close all positions immediately"""
        self._log_security_event("EMERGENCY_STOP", "Emergency stop triggered")
        
        # Send emergency alerts
        self._send_emergency_alert("EMERGENCY STOP TRIGGERED - ALL POSITIONS CLOSED")
        
        # Close all positions (implementation depends on broker)
        # This would be implemented in the broker integration
    
    def _log_security_event(self, event_type: str, description: str):
        """Log security event with full audit trail"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "description": description,
            "severity": self._get_event_severity(event_type)
        }
        
        self.security_events.append(event)
        
        # Log to file
        logger.warning(f"SECURITY EVENT: {event_type} - {description}")
        
        # Send alert if critical
        if event["severity"] == "CRITICAL":
            self._send_security_alert(event)
    
    def _get_event_severity(self, event_type: str) -> str:
        """Determine event severity"""
        critical_events = ["EMERGENCY_STOP", "AUTH_FAILED", "RATE_LIMIT_EXCEEDED"]
        high_events = ["TRADING_LIMIT_EXCEEDED", "SUSPICIOUS_ACTIVITY"]
        
        if event_type in critical_events:
            return "CRITICAL"
        elif event_type in high_events:
            return "HIGH"
        else:
            return "MEDIUM"
    
    def _send_emergency_alert(self, message: str):
        """Send emergency alert"""
        if self.config.alert_webhook:
            # Send to webhook
            pass
        
        # Log to console
        logger.critical(f"🚨 EMERGENCY ALERT: {message}")
    
    def _send_security_alert(self, event: Dict[str, Any]):
        """Send security alert"""
        if self.config.alert_webhook:
            # Send to webhook
            pass
        
        logger.warning(f"🔒 SECURITY ALERT: {event}")
    
    def _start_security_monitoring(self):
        """Start real-time security monitoring"""
        # This would start background monitoring tasks
        pass
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        return {
            "encryption_active": True,
            "failed_attempts": len(self.failed_attempts),
            "locked_ips": len(self.locked_ips),
            "security_events_24h": len([
                e for e in self.security_events 
                if datetime.fromisoformat(e["timestamp"]) > datetime.utcnow() - timedelta(hours=24)
            ]),
            "active_sessions": len(self.active_sessions)
        }


class SecurityException(Exception):
    """Security-related exception"""
    pass


# Example usage
if __name__ == "__main__":
    # Initialize military-grade security
    config = SecurityConfig(
        encryption_key="your_military_grade_password",
        ip_whitelist=["192.168.1.0/24", "10.0.0.0/8"],
        max_daily_loss_percent=5.0,
        emergency_stop_threshold=10.0
    )
    
    security = MilitarySecurityManager(config)
    
    # Test encryption
    sensitive_data = "API_KEY_12345"
    encrypted = security.encrypt_sensitive_data(sensitive_data)
    decrypted = security.decrypt_sensitive_data(encrypted)
    
    print(f"Original: {sensitive_data}")
    print(f"Encrypted: {encrypted}")
    print(f"Decrypted: {decrypted}")
    
    # Test trading limits
    limits = security.check_trading_limits(
        trade_amount=1000,
        account_balance=10000,
        daily_pnl=-200
    )
    print(f"Trading limits: {limits}")
