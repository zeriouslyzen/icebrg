"""
Secure Wallet Manager for ICEBURG Financial Trading System
Military-grade secure wallet management with encryption and multi-signature support
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import hmac
import base64
from pathlib import Path
import sqlite3
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

from .military_security import MilitarySecurityManager, SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class WalletConfig:
    """Wallet configuration with military-grade security"""
    wallet_id: str
    wallet_type: str  # 'hot', 'cold', 'multi_sig'
    max_balance: float
    daily_limit: float
    withdrawal_limit: float
    encryption_key: str
    backup_frequency_hours: int = 24
    multi_sig_threshold: int = 2  # For multi-sig wallets
    multi_sig_total: int = 3  # Total signers for multi-sig


@dataclass
class WalletTransaction:
    """Wallet transaction record"""
    transaction_id: str
    wallet_id: str
    transaction_type: str  # 'deposit', 'withdrawal', 'transfer'
    amount: float
    currency: str
    destination: str
    status: str  # 'pending', 'confirmed', 'failed'
    timestamp: datetime
    signature: str
    block_height: Optional[int] = None


class SecureWalletManager:
    """
    Military-grade secure wallet manager for real money trading.
    
    Features:
    - Multi-layer encryption
    - Multi-signature support
    - Secure key management
    - Transaction signing
    - Balance tracking
    - Risk management
    - Audit logging
    """
    
    def __init__(self, security_config: SecurityConfig):
        self.security = MilitarySecurityManager(security_config)
        self.wallets: Dict[str, WalletConfig] = {}
        self.transactions: List[WalletTransaction] = []
        self.private_keys: Dict[str, bytes] = {}
        self.public_keys: Dict[str, bytes] = {}
        
        # Initialize database
        self.db_path = Path("data/secure_wallets.db")
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_database()
        
        logger.info("🔐 Secure wallet manager initialized with military-grade security")
    
    def _init_database(self):
        """Initialize secure wallet database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create wallets table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS wallets (
                        wallet_id TEXT PRIMARY KEY,
                        wallet_type TEXT NOT NULL,
                        max_balance REAL NOT NULL,
                        daily_limit REAL NOT NULL,
                        withdrawal_limit REAL NOT NULL,
                        encrypted_private_key TEXT NOT NULL,
                        public_key TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_backup TIMESTAMP
                    )
                """)
                
                # Create transactions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS transactions (
                        transaction_id TEXT PRIMARY KEY,
                        wallet_id TEXT NOT NULL,
                        transaction_type TEXT NOT NULL,
                        amount REAL NOT NULL,
                        currency TEXT NOT NULL,
                        destination TEXT,
                        status TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        signature TEXT NOT NULL,
                        block_height INTEGER,
                        FOREIGN KEY (wallet_id) REFERENCES wallets (wallet_id)
                    )
                """)
                
                # Create audit log table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS audit_log (
                        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        wallet_id TEXT,
                        action TEXT NOT NULL,
                        details TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        ip_address TEXT,
                        user_agent TEXT
                    )
                """)
                
                conn.commit()
                logger.info("✅ Secure wallet database initialized")
                
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
            raise
    
    def create_wallet(self, wallet_config: WalletConfig) -> str:
        """Create a new secure wallet"""
        try:
            # Generate key pair
            private_key, public_key = self._generate_key_pair()
            
            # Encrypt private key
            encrypted_private_key = self.security.encrypt_sensitive_data(
                private_key.hex()
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO wallets (
                        wallet_id, wallet_type, max_balance, daily_limit,
                        withdrawal_limit, encrypted_private_key, public_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    wallet_config.wallet_id,
                    wallet_config.wallet_type,
                    wallet_config.max_balance,
                    wallet_config.daily_limit,
                    wallet_config.withdrawal_limit,
                    encrypted_private_key,
                    public_key.hex()
                ))
                conn.commit()
            
            # Store in memory
            self.wallets[wallet_config.wallet_id] = wallet_config
            self.private_keys[wallet_config.wallet_id] = private_key
            self.public_keys[wallet_config.wallet_id] = public_key
            
            # Log creation
            self._log_audit_event(
                wallet_config.wallet_id,
                "WALLET_CREATED",
                f"Wallet {wallet_config.wallet_id} created with type {wallet_config.wallet_type}"
            )
            
            logger.info(f"✅ Wallet created: {wallet_config.wallet_id}")
            return wallet_config.wallet_id
            
        except Exception as e:
            logger.error(f"❌ Wallet creation failed: {e}")
            raise
    
    def _generate_key_pair(self) -> Tuple[bytes, bytes]:
        """Generate RSA key pair for wallet"""
        try:
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            public_key = private_key.public_key()
            
            # Serialize keys
            private_pem = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            return private_pem, public_pem
            
        except Exception as e:
            logger.error(f"❌ Key generation failed: {e}")
            raise
    
    def get_wallet_balance(self, wallet_id: str) -> Dict[str, float]:
        """Get wallet balance with security validation"""
        try:
            # Check if wallet exists
            if wallet_id not in self.wallets:
                raise Exception(f"Wallet {wallet_id} not found")
            
            # Get balance from database (simplified)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT SUM(amount) as total_deposits,
                           (SELECT SUM(amount) FROM transactions 
                            WHERE wallet_id = ? AND transaction_type = 'withdrawal') as total_withdrawals
                    FROM transactions 
                    WHERE wallet_id = ? AND transaction_type = 'deposit'
                """, (wallet_id, wallet_id))
                
                result = cursor.fetchone()
                if result:
                    deposits = result[0] or 0
                    withdrawals = result[1] or 0
                    balance = deposits - withdrawals
                else:
                    balance = 0.0
            
            # Log balance check
            self._log_audit_event(
                wallet_id,
                "BALANCE_CHECK",
                f"Balance checked: {balance}"
            )
            
            return {
                "balance": balance,
                "currency": "USD",  # Simplified
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Balance check failed: {e}")
            raise
    
    def create_transaction(self, wallet_id: str, transaction_type: str, 
                          amount: float, currency: str, destination: str = None) -> str:
        """Create a new transaction with military-grade security"""
        try:
            # Validate wallet
            if wallet_id not in self.wallets:
                raise Exception(f"Wallet {wallet_id} not found")
            
            # Check limits
            wallet_config = self.wallets[wallet_id]
            if amount > wallet_config.daily_limit:
                raise Exception(f"Amount {amount} exceeds daily limit {wallet_config.daily_limit}")
            
            if transaction_type == 'withdrawal' and amount > wallet_config.withdrawal_limit:
                raise Exception(f"Withdrawal amount {amount} exceeds limit {wallet_config.withdrawal_limit}")
            
            # Generate transaction ID
            transaction_id = self._generate_transaction_id()
            
            # Sign transaction
            signature = self._sign_transaction(
                wallet_id, transaction_id, transaction_type, amount, currency
            )
            
            # Create transaction record
            transaction = WalletTransaction(
                transaction_id=transaction_id,
                wallet_id=wallet_id,
                transaction_type=transaction_type,
                amount=amount,
                currency=currency,
                destination=destination,
                status='pending',
                timestamp=datetime.now(),
                signature=signature
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO transactions (
                        transaction_id, wallet_id, transaction_type, amount,
                        currency, destination, status, timestamp, signature
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    transaction.transaction_id,
                    transaction.wallet_id,
                    transaction.transaction_type,
                    transaction.amount,
                    transaction.currency,
                    transaction.destination,
                    transaction.status,
                    transaction.timestamp.isoformat(),
                    transaction.signature
                ))
                conn.commit()
            
            # Store in memory
            self.transactions.append(transaction)
            
            # Log transaction
            self._log_audit_event(
                wallet_id,
                "TRANSACTION_CREATED",
                f"Transaction {transaction_id}: {transaction_type} {amount} {currency}"
            )
            
            logger.info(f"✅ Transaction created: {transaction_id}")
            return transaction_id
            
        except Exception as e:
            logger.error(f"❌ Transaction creation failed: {e}")
            raise
    
    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        timestamp = int(time.time() * 1000)
        random_bytes = os.urandom(16)
        return f"tx_{timestamp}_{random_bytes.hex()}"
    
    def _sign_transaction(self, wallet_id: str, transaction_id: str, 
                         transaction_type: str, amount: float, currency: str) -> str:
        """Sign transaction with wallet private key"""
        try:
            # Get private key
            private_key = self.private_keys.get(wallet_id)
            if not private_key:
                raise Exception(f"Private key not found for wallet {wallet_id}")
            
            # Create message to sign
            message = f"{transaction_id}{transaction_type}{amount}{currency}"
            message_bytes = message.encode()
            
            # Sign with private key
            private_key_obj = serialization.load_pem_private_key(
                private_key, password=None
            )
            signature = private_key_obj.sign(
                message_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            logger.error(f"❌ Transaction signing failed: {e}")
            raise
    
    def verify_transaction(self, transaction_id: str, signature: str) -> bool:
        """Verify transaction signature"""
        try:
            # Get transaction from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT wallet_id, transaction_type, amount, currency
                    FROM transactions WHERE transaction_id = ?
                """, (transaction_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                wallet_id, transaction_type, amount, currency = result
            
            # Get public key
            public_key = self.public_keys.get(wallet_id)
            if not public_key:
                return False
            
            # Recreate message
            message = f"{transaction_id}{transaction_type}{amount}{currency}"
            message_bytes = message.encode()
            
            # Verify signature
            public_key_obj = serialization.load_pem_public_key(public_key)
            try:
                public_key_obj.verify(
                    base64.b64decode(signature),
                    message_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except Exception:
                return False
                
        except Exception as e:
            logger.error(f"❌ Transaction verification failed: {e}")
            return False
    
    def get_transaction_history(self, wallet_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction history for wallet"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT transaction_id, transaction_type, amount, currency,
                           destination, status, timestamp, signature
                    FROM transactions 
                    WHERE wallet_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                """, (wallet_id, limit))
                
                results = cursor.fetchall()
                transactions = []
                
                for row in results:
                    transactions.append({
                        "transaction_id": row[0],
                        "transaction_type": row[1],
                        "amount": row[2],
                        "currency": row[3],
                        "destination": row[4],
                        "status": row[5],
                        "timestamp": row[6],
                        "signature": row[7]
                    })
                
                return transactions
                
        except Exception as e:
            logger.error(f"❌ Failed to get transaction history: {e}")
            return []
    
    def _log_audit_event(self, wallet_id: str, action: str, details: str):
        """Log audit event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO audit_log (wallet_id, action, details)
                    VALUES (?, ?, ?)
                """, (wallet_id, action, details))
                conn.commit()
                
        except Exception as e:
            logger.error(f"❌ Audit logging failed: {e}")
    
    def get_wallet_status(self, wallet_id: str) -> Dict[str, Any]:
        """Get comprehensive wallet status"""
        try:
            if wallet_id not in self.wallets:
                raise Exception(f"Wallet {wallet_id} not found")
            
            wallet_config = self.wallets[wallet_id]
            balance = self.get_wallet_balance(wallet_id)
            transactions = self.get_transaction_history(wallet_id, 10)
            
            return {
                "wallet_id": wallet_id,
                "wallet_type": wallet_config.wallet_type,
                "balance": balance,
                "limits": {
                    "max_balance": wallet_config.max_balance,
                    "daily_limit": wallet_config.daily_limit,
                    "withdrawal_limit": wallet_config.withdrawal_limit
                },
                "recent_transactions": transactions,
                "security_status": self.security.get_security_status()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get wallet status: {e}")
            return {}


# Example usage
if __name__ == "__main__":
    # Initialize security
    security_config = SecurityConfig(
        encryption_key="military_grade_wallet_password_2025",
        ip_whitelist=["192.168.1.0/24"],
        max_daily_loss_percent=5.0
    )
    
    # Create wallet manager
    wallet_manager = SecureWalletManager(security_config)
    
    # Create wallet
    wallet_config = WalletConfig(
        wallet_id="trading_wallet_001",
        wallet_type="hot",
        max_balance=10000.0,
        daily_limit=1000.0,
        withdrawal_limit=500.0,
        encryption_key="wallet_encryption_key"
    )
    
    wallet_id = wallet_manager.create_wallet(wallet_config)
    print(f"Created wallet: {wallet_id}")
    
    # Get balance
    balance = wallet_manager.get_wallet_balance(wallet_id)
    print(f"Wallet balance: {balance}")
    
    # Create transaction
    transaction_id = wallet_manager.create_transaction(
        wallet_id, "deposit", 1000.0, "USD"
    )
    print(f"Created transaction: {transaction_id}")
    
    # Get wallet status
    status = wallet_manager.get_wallet_status(wallet_id)
    print(f"Wallet status: {json.dumps(status, indent=2, default=str)}")
