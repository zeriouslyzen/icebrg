"""
GDPR Compliance
General Data Protection Regulation compliance
"""

from typing import Any, Dict, Optional, List
from datetime import datetime
import json


class GDPRCompliance:
    """GDPR compliance manager"""
    
    def __init__(self):
        self.consents: Dict[str, Dict[str, Any]] = {}
        self.data_subjects: Dict[str, Dict[str, Any]] = {}
        self.data_processing_records: List[Dict[str, Any]] = []
    
    def record_consent(
        self,
        data_subject_id: str,
        purpose: str,
        consent_given: bool,
        timestamp: Optional[str] = None
    ) -> bool:
        """Record consent from data subject"""
        if data_subject_id not in self.consents:
            self.consents[data_subject_id] = {}
        
        self.consents[data_subject_id][purpose] = {
            "consent_given": consent_given,
            "timestamp": timestamp or datetime.now().isoformat()
        }
        return True
    
    def has_consent(self, data_subject_id: str, purpose: str) -> bool:
        """Check if data subject has given consent"""
        if data_subject_id not in self.consents:
            return False
        if purpose not in self.consents[data_subject_id]:
            return False
        return self.consents[data_subject_id][purpose]["consent_given"]
    
    def register_data_subject(
        self,
        data_subject_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a data subject"""
        self.data_subjects[data_subject_id] = {
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        return True
    
    def record_data_processing(
        self,
        data_subject_id: str,
        purpose: str,
        data_categories: List[str],
        legal_basis: str
    ) -> bool:
        """Record data processing activity"""
        record = {
            "data_subject_id": data_subject_id,
            "purpose": purpose,
            "data_categories": data_categories,
            "legal_basis": legal_basis,
            "timestamp": datetime.now().isoformat()
        }
        self.data_processing_records.append(record)
        return True
    
    def get_data_subject_records(self, data_subject_id: str) -> List[Dict[str, Any]]:
        """Get all records for a data subject"""
        return [r for r in self.data_processing_records if r["data_subject_id"] == data_subject_id]
    
    def delete_data_subject(self, data_subject_id: str) -> bool:
        """Delete all data for a data subject (right to be forgotten)"""
        if data_subject_id in self.consents:
            del self.consents[data_subject_id]
        if data_subject_id in self.data_subjects:
            del self.data_subjects[data_subject_id]
        self.data_processing_records = [
            r for r in self.data_processing_records
            if r["data_subject_id"] != data_subject_id
        ]
        return True
    
    def export_data_subject_data(self, data_subject_id: str) -> Dict[str, Any]:
        """Export all data for a data subject (data portability)"""
        return {
            "data_subject_id": data_subject_id,
            "consents": self.consents.get(data_subject_id, {}),
            "subject_info": self.data_subjects.get(data_subject_id, {}),
            "processing_records": self.get_data_subject_records(data_subject_id)
        }

