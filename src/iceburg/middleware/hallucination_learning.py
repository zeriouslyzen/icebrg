"""
Hallucination Learning System
Learns from hallucinations and shares patterns globally across agents.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
import logging

from ..config import IceburgConfig
from ..memory.unified_memory import UnifiedMemory

logger = logging.getLogger(__name__)


class HallucinationLearning:
    """
    Learning system for hallucination patterns.
    
    Features:
    - Store hallucination patterns in vector DB
    - Cross-agent pattern matching
    - Pattern frequency tracking
    - Agent-specific vs. global patterns
    """
    
    def __init__(self, cfg: IceburgConfig):
        """
        Initialize hallucination learning system.
        
        Args:
            cfg: ICEBURG configuration
        """
        self.cfg = cfg
        
        # Initialize memory for pattern storage
        self.memory = None
        try:
            self.memory = UnifiedMemory(cfg)
            logger.info("✅ Hallucination learning memory initialized")
        except Exception as e:
            logger.warning(f"Could not initialize memory: {e}")
        
        # Pattern storage directory
        self.patterns_dir = Path(cfg.data_dir) / "hallucinations" / "patterns" if hasattr(cfg, 'data_dir') else Path("./data/hallucinations/patterns")
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        
        # Pattern statistics
        self.pattern_stats = {
            "total_patterns": 0,
            "patterns_by_agent": {},
            "patterns_by_type": {},
            "last_updated": datetime.now().isoformat()
        }
        
        # Load existing patterns
        self._load_patterns()
    
    def _load_patterns(self):
        """Load existing patterns from storage."""
        pattern_file = self.patterns_dir / "pattern_stats.json"
        if pattern_file.exists():
            try:
                with open(pattern_file, 'r', encoding='utf-8') as f:
                    self.pattern_stats = json.load(f)
                logger.info(f"Loaded {self.pattern_stats.get('total_patterns', 0)} existing patterns")
            except Exception as e:
                logger.warning(f"Could not load pattern stats: {e}")
    
    def _save_patterns(self):
        """Save pattern statistics."""
        self.pattern_stats["last_updated"] = datetime.now().isoformat()
        pattern_file = self.patterns_dir / "pattern_stats.json"
        try:
            with open(pattern_file, 'w', encoding='utf-8') as f:
                json.dump(self.pattern_stats, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Could not save pattern stats: {e}")
    
    def learn_pattern(
        self,
        agent: str,
        pattern: Dict[str, Any],
        query: str,
        result: str,
        context: Dict[str, Any]
    ):
        """
        Learn from a hallucination pattern.
        
        Args:
            agent: Agent name
            pattern: Hallucination detection result
            query: Original query
            result: Agent result
            context: Execution context
        """
        try:
            # Extract pattern features
            pattern_features = {
                "agent": agent,
                "hallucination_score": pattern.get("hallucination_score", 0.0),
                "detection_layers": pattern.get("detection_layers", {}),
                "flags": pattern.get("flags", []),
                "patterns_detected": pattern.get("patterns_detected", []),
                "query": query[:200],
                "result_snippet": result[:200],
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in vector DB for semantic search
            if self.memory:
                try:
                    pattern_text = f"Agent: {agent}. Query: {query[:100]}. Pattern: {', '.join(pattern.get('flags', []))}"
                    self.memory.index_texts(
                        namespace="hallucination_patterns",
                        texts=[pattern_text],
                        metadatas=[pattern_features]
                    )
                except Exception as e:
                    logger.debug(f"Could not store in vector DB: {e}")
            
            # Update statistics
            self.pattern_stats["total_patterns"] = self.pattern_stats.get("total_patterns", 0) + 1
            
            # Update agent-specific stats
            if agent not in self.pattern_stats["patterns_by_agent"]:
                self.pattern_stats["patterns_by_agent"][agent] = 0
            self.pattern_stats["patterns_by_agent"][agent] += 1
            
            # Update type-specific stats
            for flag in pattern.get("flags", []):
                if flag not in self.pattern_stats["patterns_by_type"]:
                    self.pattern_stats["patterns_by_type"][flag] = 0
                self.pattern_stats["patterns_by_type"][flag] += 1
            
            # Save patterns
            self._save_patterns()
            
            logger.debug(f"Learned pattern from agent {agent}: {len(pattern.get('flags', []))} flags")
            
        except Exception as e:
            logger.warning(f"Error learning pattern: {e}")
    
    def check_patterns(self, query: str, agent: str) -> Optional[List[Dict[str, Any]]]:
        """
        Check for known patterns matching the query.
        
        Args:
            query: User query
            agent: Agent name
            
        Returns:
            List of matching patterns or None
        """
        if not self.memory:
            return None
        
        try:
            # Search for similar patterns
            results = self.memory.search(
                namespace="hallucination_patterns",
                query=query,
                k=5
            )
            
            # Filter by agent if needed
            matching_patterns = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("agent") == agent or not agent:
                    matching_patterns.append({
                        "pattern": result.get("document", ""),
                        "metadata": metadata,
                        "similarity": 1.0 - result.get("distance", 1.0)
                    })
            
            return matching_patterns if matching_patterns else None
            
        except Exception as e:
            logger.debug(f"Could not check patterns: {e}")
            return None
    
    def get_pattern_stats(self) -> Dict[str, Any]:
        """Get pattern learning statistics."""
        return self.pattern_stats.copy()
    
    def get_agent_patterns(self, agent: str) -> Dict[str, Any]:
        """
        Get patterns for a specific agent.
        
        Args:
            agent: Agent name
            
        Returns:
            Agent pattern statistics
        """
        return {
            "agent": agent,
            "total_patterns": self.pattern_stats["patterns_by_agent"].get(agent, 0),
            "pattern_types": {
                k: v for k, v in self.pattern_stats["patterns_by_type"].items()
            }
        }

