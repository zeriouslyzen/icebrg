"""
Emergence Aggregator
Aggregates emergence events across all agents and builds global patterns.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

from ..config import IceburgConfig
from ..memory.unified_memory import UnifiedMemory

logger = logging.getLogger(__name__)


class EmergenceAggregator:
    """
    Aggregates emergence across all agents.
    
    Features:
    - Collect emergence events from all agents
    - Build global emergence patterns
    - Track emergence evolution over time
    - Cross-agent emergence correlation
    """
    
    def __init__(self, cfg: IceburgConfig):
        """
        Initialize emergence aggregator.
        
        Args:
            cfg: ICEBURG configuration
        """
        self.cfg = cfg
        
        # Initialize memory for emergence storage
        self.memory = None
        try:
            self.memory = UnifiedMemory(cfg)
            logger.info("✅ Emergence aggregator memory initialized")
        except Exception as e:
            logger.warning(f"Could not initialize memory: {e}")
        
        # Emergence storage directory
        self.emergence_dir = Path(cfg.data_dir) / "emergence" / "global" if hasattr(cfg, 'data_dir') else Path("./data/emergence/global")
        self.emergence_dir.mkdir(parents=True, exist_ok=True)
        
        # Emergence statistics
        self.emergence_stats = {
            "total_events": 0,
            "events_by_agent": {},
            "events_by_type": {},
            "breakthroughs": [],
            "last_updated": datetime.now().isoformat()
        }
        
        # Load existing statistics
        self._load_stats()
    
    def _load_stats(self):
        """Load existing emergence statistics."""
        stats_file = self.emergence_dir / "emergence_stats.json"
        if stats_file.exists():
            try:
                with open(stats_file, 'r', encoding='utf-8') as f:
                    self.emergence_stats = json.load(f)
                logger.info(f"Loaded {self.emergence_stats.get('total_events', 0)} existing emergence events")
            except Exception as e:
                logger.warning(f"Could not load emergence stats: {e}")
    
    def _save_stats(self):
        """Save emergence statistics."""
        self.emergence_stats["last_updated"] = datetime.now().isoformat()
        stats_file = self.emergence_dir / "emergence_stats.json"
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.emergence_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save emergence stats: {e}")
    
    def record_emergence(
        self,
        agent: str,
        emergence: Dict[str, Any],
        query: str,
        context: Dict[str, Any]
    ):
        """
        Record an emergence event.
        
        Args:
            agent: Agent name
            emergence: Emergence detection result
            query: Original query
            context: Execution context
        """
        try:
            # Extract emergence features
            emergence_features = {
                "agent": agent,
                "emergence_score": emergence.get("emergence_score", 0.0),
                "emergence_type": emergence.get("emergence_type", "unknown"),
                "query": query[:200],
                "content_snippet": emergence.get("content", "")[:200],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in vector DB for semantic search
            if self.memory:
                try:
                    emergence_text = f"Agent: {agent}. Type: {emergence.get('emergence_type', 'unknown')}. Query: {query[:100]}"
                    self.memory.index_texts(
                        namespace="emergence_events",
                        texts=[emergence_text],
                        metadatas=[emergence_features]
                    )
                except Exception as e:
                    logger.debug(f"Could not store in vector DB: {e}")
            
            # Store in JSONL file for detailed tracking
            event_file = self.emergence_dir / f"events_{datetime.now().strftime('%Y%m')}.jsonl"
            event_data = {
                "timestamp": datetime.now().isoformat(),
                "agent": agent,
                "emergence": emergence,
                "query": query,
                "context": {k: str(v)[:200] for k, v in context.items()}
            }
            
            try:
                with open(event_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(event_data, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.debug(f"Could not write to event file: {e}")
            
            # Update statistics
            self.emergence_stats["total_events"] = self.emergence_stats.get("total_events", 0) + 1
            
            # Update agent-specific stats
            if agent not in self.emergence_stats["events_by_agent"]:
                self.emergence_stats["events_by_agent"][agent] = 0
            self.emergence_stats["events_by_agent"][agent] += 1
            
            # Update type-specific stats
            emergence_type = emergence.get("emergence_type", "unknown")
            if emergence_type not in self.emergence_stats["events_by_type"]:
                self.emergence_stats["events_by_type"][emergence_type] = 0
            self.emergence_stats["events_by_type"][emergence_type] += 1
            
            # Track breakthroughs (high emergence score)
            if emergence.get("emergence_score", 0.0) > 0.8:
                breakthrough = {
                    "timestamp": datetime.now().isoformat(),
                    "agent": agent,
                    "emergence_type": emergence_type,
                    "score": emergence.get("emergence_score", 0.0),
                    "query": query[:200]
                }
                self.emergence_stats["breakthroughs"].append(breakthrough)
                # Keep only last 100 breakthroughs
                if len(self.emergence_stats["breakthroughs"]) > 100:
                    self.emergence_stats["breakthroughs"] = self.emergence_stats["breakthroughs"][-100:]
            
            # Save statistics
            self._save_stats()
            
            logger.debug(f"Recorded emergence from agent {agent}: {emergence_type} (score: {emergence.get('emergence_score', 0.0):.2f})")
            
        except Exception as e:
            logger.warning(f"Error recording emergence: {e}")
    
    def get_emergence_stats(self) -> Dict[str, Any]:
        """Get emergence aggregation statistics."""
        return self.emergence_stats.copy()
    
    def get_agent_emergence(self, agent: str) -> Dict[str, Any]:
        """
        Get emergence statistics for a specific agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Agent emergence statistics
        """
        return {
            "agent": agent,
            "total_events": self.emergence_stats["events_by_agent"].get(agent, 0),
            "event_types": {
                k: v for k, v in self.emergence_stats["events_by_type"].items()
            }
        }
    
    def get_recent_breakthroughs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent breakthroughs.
        
        Args:
            limit: Number of breakthroughs to return
            
        Returns:
            List of recent breakthroughs
        """
        breakthroughs = self.emergence_stats.get("breakthroughs", [])
        return breakthroughs[-limit:] if breakthroughs else []

