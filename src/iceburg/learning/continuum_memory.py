"""
Continuum Memory for Sustained Evolutions
Maintains state across evolution cycles without resets
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class ContinuumMemory:
    """Continuum memory for sustained evolutions."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize continuum memory.
        
        Args:
            config: Configuration for continuum memory
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        
        # Memory storage
        if cfg and hasattr(cfg, 'data_dir'):
            self.memory_dir = Path(cfg.data_dir) / "continuum_memory"
        else:
            self.memory_dir = Path("data/continuum_memory")
        
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "continuum_state.json"
        
        # Load existing state
        self.state = self._load_state()
    
    def _load_state(self) -> Dict[str, Any]:
        """Load continuum state from disk."""
        if not self.memory_file.exists():
            return {
                "evolutions": [],
                "current_architecture": None,
                "last_updated": time.time()
            }
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
                logger.info(f"Loaded continuum state with {len(state.get('evolutions', []))} evolutions")
                return state
        except Exception as e:
            logger.error(f"Failed to load continuum state: {e}")
            return {
                "evolutions": [],
                "current_architecture": None,
                "last_updated": time.time()
            }
    
    async def load_state(self) -> Dict[str, Any]:
        """Load continuum state (async wrapper)."""
        return self.state
    
    async def store_redesign(
        self,
        current_architecture: Dict[str, Any],
        redesign: Dict[str, Any],
        metrics: Dict[str, Any]
    ):
        """
        Store redesign in continuum memory.
        
        Args:
            current_architecture: Current architecture
            redesign: Redesign proposal
            metrics: Performance metrics
        """
        # Add evolution to history
        evolution = {
            "timestamp": time.time(),
            "current_architecture": current_architecture,
            "redesign": redesign,
            "metrics": metrics
        }
        
        self.state["evolutions"].append(evolution)
        self.state["current_architecture"] = redesign
        self.state["last_updated"] = time.time()
        
        # Keep only last 100 evolutions
        if len(self.state["evolutions"]) > 100:
            self.state["evolutions"] = self.state["evolutions"][-100:]
        
        # Save to disk
        self._save_state()
        
        logger.info(f"Stored redesign in continuum memory (total evolutions: {len(self.state['evolutions'])})")
    
    def _save_state(self):
        """Save continuum state to disk."""
        try:
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save continuum state: {e}")
    
    def get_evolution_history(self) -> List[Dict[str, Any]]:
        """Get evolution history."""
        return self.state.get("evolutions", [])

