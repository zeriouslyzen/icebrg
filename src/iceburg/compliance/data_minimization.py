"""
Data Minimization
Implements data minimization principle
"""

from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta


class DataMinimization:
    """Data minimization manager"""
    
    def __init__(self, retention_period_days: int = 365):
        self.retention_period_days = retention_period_days
        self.data_records: Dict[str, Dict[str, Any]] = {}
    
    def store_data(
        self,
        key: str,
        data: Any,
        purpose: str,
        retention_days: Optional[int] = None
    ) -> bool:
        """Store data with purpose and retention period"""
        retention = retention_days or self.retention_period_days
        expires_at = datetime.now() + timedelta(days=retention)
        
        self.data_records[key] = {
            "data": data,
            "purpose": purpose,
            "stored_at": datetime.now().isoformat(),
            "expires_at": expires_at.isoformat(),
            "retention_days": retention
        }
        return True
    
    def get_data(self, key: str) -> Optional[Any]:
        """Get data if still valid"""
        if key not in self.data_records:
            return None
        
        record = self.data_records[key]
        expires_at = datetime.fromisoformat(record["expires_at"])
        
        if datetime.now() > expires_at:
            # Data expired, delete it
            del self.data_records[key]
            return None
        
        return record["data"]
    
    def cleanup_expired(self) -> int:
        """Clean up expired data"""
        now = datetime.now()
        expired_keys = []
        
        for key, record in self.data_records.items():
            expires_at = datetime.fromisoformat(record["expires_at"])
            if now > expires_at:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.data_records[key]
        
        return len(expired_keys)
    
    def get_data_by_purpose(self, purpose: str) -> List[Dict[str, Any]]:
        """Get all data for a specific purpose"""
        return [
            {"key": key, "data": record["data"], "stored_at": record["stored_at"]}
            for key, record in self.data_records.items()
            if record["purpose"] == purpose
        ]
    
    def delete_data_by_purpose(self, purpose: str) -> int:
        """Delete all data for a specific purpose"""
        keys_to_delete = [
            key for key, record in self.data_records.items()
            if record["purpose"] == purpose
        ]
        
        for key in keys_to_delete:
            del self.data_records[key]
        
        return len(keys_to_delete)

