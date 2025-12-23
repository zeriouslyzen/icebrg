"""
Quarantine Manager
==================

Handles the storage and management of flagged agent outputs (contradictions, hallucinations, high-novelty).
Instead of discarding these outputs, we "quarantine" them for future review, as they may contain
breakthrough ideas that simple filters mistook for errors.
"""

import os
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class QuarantineManager:
    """Manages the quarantine storage for flagged agent outputs."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize the quarantine manager.
        
        Args:
            data_dir: Base data directory. If None, uses defaults.
        """
        if data_dir:
            self.base_dir = Path(data_dir)
        else:
            # Try to find project root
            current_path = Path(__file__).resolve()
            # src/iceburg/core/quarantine_manager.py -> src/iceburg/core -> src/iceburg -> src -> root
            project_root = current_path.parent.parent.parent.parent
            self.base_dir = project_root / "data"
            
        self.quarantine_dir = self.base_dir / "hallucinations" / "quarantined"
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure quarantine directory exists."""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
    def quarantine_item(self, 
                       content: Any, 
                       source_agent: str, 
                       reason: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Quarantine an item (agent output or thought) for review.
        
        Args:
            content: The content to quarantine (usually string or dict)
            source_agent: Name of the agent that produced it
            reason: Why it was flagged (e.g., "contradiction", "novelty_outlier")
            metadata: Additional context (scores, query, etc.)
            
        Returns:
            str: ID of the quarantined item
        """
        item_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        quarantine_record = {
            "id": item_id,
            "timestamp": timestamp,
            "source_agent": source_agent,
            "flag_reason": reason,
            "content": content,
            "metadata": metadata or {},
            "status": "pending_review",
            "review_notes": None
        }
        
        filename = f"{timestamp.replace(':', '-')}_{source_agent}_{item_id[:8]}.json"
        file_path = self.quarantine_dir / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(quarantine_record, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Quarantined item {item_id} from {source_agent} due to {reason}")
            return item_id
            
        except Exception as e:
            logger.error(f"Failed to save quarantine item: {e}")
            return None
            
    def get_stats(self) -> Dict[str, int]:
        """Get statistics on quarantined items."""
        try:
            files = list(self.quarantine_dir.glob("*.json"))
            return {
                "total_items": len(files),
                "storage_path": str(self.quarantine_dir)
            }
        except Exception:
            return {"total_items": 0, "error": "Could not access directory"}

# Global instance
_quarantine_manager = None

def get_quarantine_manager() -> QuarantineManager:
    """Get singleton instance."""
    global _quarantine_manager
    if _quarantine_manager is None:
        _quarantine_manager = QuarantineManager()
    return _quarantine_manager
