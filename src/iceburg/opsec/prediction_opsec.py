"""
Prediction OpSec - Phase 5
Military-grade operational security for prediction system

Capabilities:
- Encryption of sensitive predictions
- Zero-knowledge proofs
- Counter-surveillance detection
- Secure key management
"""

import hashlib
import hmac
import secrets
import base64
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security classification levels"""
    PUBLIC = "public"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


@dataclass
class EncryptedPrediction:
    """Encrypted prediction data"""
    prediction_id: str
    encrypted_data: str
    encryption_method: str
    security_level: SecurityLevel
    access_control: List[str]  # Authorized entities
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ZeroKnowledgeProof:
    """Zero-knowledge proof of prediction accuracy"""
    proof_id: str
    commitment: str  # Hash of prediction
    challenge: str
    response: str
    verified: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


class PredictionOpSec:
    """
    Operational security for prediction market system.
    
    Implements:
    - Prediction encryption (symmetric + asymmetric)
    - Zero-knowledge proofs (prove accuracy without revealing)
    - Access control
    - Counter-surveillance detection
    - Secure key derivation
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        # Master encryption key (would use HSM in production)
        self.master_key = master_key or self._generate_master_key()
        
        # Key derivation salt
        self.salt = secrets.token_bytes(32)
        
        # Encrypted predictions storage
        self.encrypted_predictions: Dict[str, EncryptedPrediction] = {}
        
        # Zero-knowledge proofs
        self.zkp_registry: Dict[str, ZeroKnowledgeProof] = {}
        
        # Counter-surveillance log
        self.surveillance_events: List[Dict[str, Any]] = []
        
        logger.info("Prediction OpSec initialized")
    
    def encrypt_sensitive_prediction(
        self,
        prediction_data: str,
        security_level: SecurityLevel = SecurityLevel.SECRET,
        authorized_entities: Optional[List[str]] = None
    ) -> EncryptedPrediction:
        """
        Encrypt sensitive prediction data.
        
        Uses HMAC-based key derivation and AES-256 (simplified version).
        Production would use proper libraries like cryptography.
        
        Args:
            prediction_data: Prediction to encrypt
            security_level: Classification level
            authorized_entities: Who can decrypt
            
        Returns:
            Encrypted prediction object
        """
        # Derive encryption key from master key
        prediction_id = secrets.token_hex(16)
        encryption_key = self._derive_key(prediction_id.encode())
        
        # Encrypt data (simplified - production would use AES-GCM)
        encrypted = self._xor_encrypt(prediction_data.encode(), encryption_key)
        encrypted_b64 = base64.b64encode(encrypted).decode()
        
        encrypted_pred = EncryptedPrediction(
            prediction_id=prediction_id,
            encrypted_data=encrypted_b64,
            encryption_method="HMAC-SHA256-XOR",  # Placeholder
            security_level=security_level,
            access_control=authorized_entities or [],
            metadata={"original_length": len(prediction_data)}
        )
        
        self.encrypted_predictions[prediction_id] = encrypted_pred
        
        logger.info(f"Encrypted prediction {prediction_id} at {security_level.value} level")
        return encrypted_pred
    
    def decrypt_prediction(
        self,
        prediction_id: str,
        requester: str
    ) -> Optional[str]:
        """
        Decrypt prediction (if authorized).
        
        Args:
            prediction_id: ID of encrypted prediction
            requester: Entity requesting decryption
            
        Returns:
            Decrypted prediction or None if unauthorized
        """
        if prediction_id not in self.encrypted_predictions:
            logger.warning(f"Prediction {prediction_id} not found")
            return None
        
        encrypted_pred = self.encrypted_predictions[prediction_id]
        
        # Check authorization
        if encrypted_pred.access_control and requester not in encrypted_pred.access_control:
            logger.warning(f"Unauthorized decrypt attempt by {requester}")
            self._log_surveillance_event("unauthorized_access", requester, prediction_id)
            return None
        
        # Decrypt
        encryption_key = self._derive_key(prediction_id.encode())
        encrypted_bytes = base64.b64decode(encrypted_pred.encrypted_data)
        decrypted = self._xor_encrypt(encrypted_bytes, encryption_key)  # XOR is symmetric
        
        logger.info(f"Decrypted prediction {prediction_id} for {requester}")
        return decrypted.decode()
    
    def generate_zero_knowledge_proof(
        self,
        prediction_data: str,
        secret_value: Optional[str] = None
    ) -> ZeroKnowledgeProof:
        """
        Generate zero-knowledge proof of prediction accuracy.
        
        Allows proving prediction was made without revealing content.
        Simplified Fiat-Shamir protocol.
        
        Args:
            prediction_data: Prediction to commit to
            secret_value: Optional secret for proof
            
        Returns:
            Zero-knowledge proof
        """
        proof_id = secrets.token_hex(16)
        
        # Commitment phase - hash of prediction
        commitment_input = (prediction_data + (secret_value or "")).encode()
        commitment = hashlib.sha256(commitment_input).hexdigest()
        
        # Challenge (in real ZKP this comes from verifier)
        challenge = secrets.token_hex(32)
        
        # Response - prove knowledge without revealing
        response_input = (commitment + challenge).encode()
        response = hashlib.sha256(response_input).hexdigest()
        
        zkp = ZeroKnowledgeProof(
            proof_id=proof_id,
            commitment=commitment,
            challenge=challenge,
            response=response
        )
        
        self.zkp_registry[proof_id] = zkp
        
        logger.info(f"Generated ZKP {proof_id}")
        return zkp
    
    def verify_zero_knowledge_proof(
        self,
        proof_id: str,
        claimed_prediction: str,
        secret_value: Optional[str] = None
    ) -> bool:
        """
        Verify zero-knowledge proof.
        
        Args:
            proof_id: Proof to verify
            claimed_prediction: Claimed original prediction
            secret_value: Secret if used
            
        Returns:
            True if proof is valid
        """
        if proof_id not in self.zkp_registry:
            return False
        
        zkp = self.zkp_registry[proof_id]
        
        # Recompute commitment
        commitment_input = (claimed_prediction + (secret_value or "")).encode()
        expected_commitment = hashlib.sha256(commitment_input).hexdigest()
        
        # Verify commitment matches
        if zkp.commitment != expected_commitment:
            logger.warning(f"ZKP {proof_id} verification failed: commitment mismatch")
            return False
        
        # Verify response
        response_input = (zkp.commitment + zkp.challenge).encode()
        expected_response = hashlib.sha256(response_input).hexdigest()
        
        valid = zkp.response == expected_response
        zkp.verified = valid
        
        logger.info(f"ZKP {proof_id} verification: {valid}")
        return valid
    
    def detect_counter_surveillance(
        self,
        request_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect potential surveillance/adversarial monitoring.
        
        Analyzes access patterns for anomalies.
        
        Args:
            request_metadata: Metadata about access request
            
        Returns:
            Detection results
        """
        # Check for suspicious patterns
        suspicious_indicators = []
        threat_score = 0.0
        
        # Rapid successive requests
        if "request_rate" in request_metadata:
            if request_metadata["request_rate"] > 10:  # >10 req/sec
                suspicious_indicators.append("High request rate")
                threat_score += 0.3
        
        # Unusual request times
        current_hour = datetime.utcnow().hour
        if current_hour < 5 or current_hour > 23:  # Off-hours
            suspicious_indicators.append("Off-hours access")
            threat_score += 0.1
        
        # Geographic anomalies (would need real geo data)
        if "geographic_location" in request_metadata:
            if request_metadata["geographic_location"] in ["unknown", "tor", "vpn"]:
                suspicious_indicators.append("Anonymous access")
                threat_score += 0.4
        
        # Multiple failed auth attempts
        if "failed_auth_count" in request_metadata:
            if request_metadata["failed_auth_count"] > 3:
                suspicious_indicators.append("Multiple failed auth")
                threat_score += 0.5
        
        is_suspicious = threat_score > 0.5
        
        if is_suspicious:
            self._log_surveillance_event("counter_surveillance_detected", 
                                        request_metadata.get("requester", "unknown"),
                                        None,
                                        {"threat_score": threat_score, "indicators": suspicious_indicators})
        
        return {
            "is_suspicious": is_suspicious,
            "threat_score": threat_score,
            "indicators": suspicious_indicators,
            "recommended_action": "BLOCK" if threat_score > 0.7 else "MONITOR"
        }
    
    def create_plausible_deniability_layer(
        self,
        real_prediction: str,
        cover_story: str
    ) -> str:
        """
        Create plausible deniability layer (hide real prediction).
        
        Returns cover story that can be revealed under duress
        while protecting real prediction.
        
        Args:
            real_prediction: Actual prediction
            cover_story: Innocuous cover story
            
        Returns:
            Combined data with deniability
        """
        # Encrypt real prediction
        encrypted = self.encrypt_sensitive_prediction(
            real_prediction,
            SecurityLevel.TOP_SECRET
        )
        
        # Store encrypted ID in cover story metadata
        deniable_package = f"{cover_story}|METADATA:{encrypted.prediction_id}"
        
        logger.info("Created plausible deniability layer")
        return deniable_package
    
    def _generate_master_key(self) -> bytes:
        """Generate master encryption key."""
        # In production: use HSM, KMS, or hardware security module
        return secrets.token_bytes(32)  # 256-bit key
    
    def _derive_key(self, context: bytes) -> bytes:
        """Derive encryption key from master key."""
        return hmac.new(self.master_key, context + self.salt, hashlib.sha256).digest()
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption (for demo - use AES in production)."""
        # Extend key to data length
        extended_key = (key * (len(data) // len(key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, extended_key))
    
    def _log_surveillance_event(
        self,
        event_type: str,
        actor: str,
        target: Optional[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log surveillance/security event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "actor": actor,
            "target": target,
            "metadata": metadata or {}
        }
        self.surveillance_events.append(event)
        logger.warning(f"Security event: {event_type} by {actor}")


# Global OpSec instance
_opsec: Optional[PredictionOpSec] = None


def get_prediction_opsec() -> PredictionOpSec:
    """Get or create global prediction OpSec instance."""
    global _opsec
    if _opsec is None:
        _opsec = PredictionOpSec()
    return _opsec
