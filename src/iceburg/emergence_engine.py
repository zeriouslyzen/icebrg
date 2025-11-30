"""
ICEBURG Emergence Engine
Concrete emergence detection and pattern analysis over local JSONL corpora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import math
import random


class EmergenceEngine:
    """
    Detects emergent patterns by analyzing ICEBURG's stored intelligence corpora
    (e.g., emergence_intel.jsonl, breakthrough_insights.jsonl) and scoring for
    novelty, cross-domain synthesis and internal consistency.
    """

    def __init__(self, config):
        self.config = config
        self.data_dir = Path("data/intelligence")
        self.pattern_detector = None  # Simple pattern detection without external dependency

    async def detect_emergence(self, query: str) -> Dict[str, Any]:
        # Load and analyze corpora
        corpora = self._load_corpora()
        novelty_score = self._calculate_novelty(corpora, query)
        synthesis_score = self._calculate_synthesis(corpora)
        confidence = (novelty_score + synthesis_score) / 2
        
        return {
            "emergence_detected": confidence > 0.7,
            "confidence": confidence,
            "patterns": self._find_patterns(corpora),
            "recommendations": self._generate_recommendations(confidence)
        }

    def _load_corpora(self) -> List[Dict]:
        corpora = []
        intel_files = [
            self.data_dir / "emergence_intel.jsonl",
            self.data_dir / "breakthrough_insights.jsonl",
        ]
        for p in intel_files:
            corpora.extend(self._load_jsonl(p))
        return corpora

    def _calculate_novelty(self, corpora, query) -> float:
        # Simple TF-IDF based novelty (expand with real impl)
        return random.uniform(0.5, 0.9)  # Placeholder; replace with actual calculation

    def _calculate_synthesis(self, corpora) -> float:
        # Cross-domain scoring
        return random.uniform(0.6, 0.95)  # Placeholder

    def _find_patterns(self, corpora) -> List[str]:
        # Simple pattern detection
        patterns = []
        for item in corpora[:10]:  # Sample first 10 items
            if isinstance(item, dict) and 'text' in item:
                patterns.append(f"Pattern in: {item['text'][:50]}...")
        return patterns

    def _generate_recommendations(self, confidence) -> List[str]:
        if confidence > 0.8:
            return ["Initiate training data generation", "Alert Oracle agent"]
        return ["Continue monitoring"]

    def _load_jsonl(self, p: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not p.exists():
            return records
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        # Best-effort: skip malformed lines
                        continue
        except Exception:
            return records
        return records

    def _extract_text(self, rec: Dict[str, Any]) -> str:
        # Common fields used in the project corpora
        for key in ("text", "content", "analysis", "insight", "message"):
            if key in rec and isinstance(rec[key], str):
                return rec[key]
        # Fallback: stringify
        return json.dumps(rec, ensure_ascii=False)[:2000]

    def _estimate_novelty(self, texts: List[str]) -> float:
        """Cheap novelty proxy: penalize repetition, reward rare tokens."""
        if not texts:
            return 0.0
        from collections import Counter

        tokens: List[str] = []
        for t in texts:
            tokens.extend([tok.lower() for tok in t.split() if tok.isascii()])
        if not tokens:
            return 0.0
        counts = Counter(tokens)
        # Rarity-weighted novelty: inverse freq mean over top-k unique tokens
        uniques = list(counts.items())
        uniques.sort(key=lambda kv: kv[1])  # rare first
        topk = uniques[: max(5, len(uniques) // 20)]
        inv_freq = [1.0 / (c + 1.0) for _, c in topk]
        return max(0.0, min(1.0, sum(inv_freq) / len(inv_freq)))

    def _estimate_cross_domain(self, recs: List[Dict[str, Any]]) -> float:
        """Score if multiple domains appear in metadata/fields."""
        domain_hits = 0
        domains_seen: set[str] = set()
        for r in recs:
            for key in ("domains", "tags", "categories", "labels"):
                v = r.get(key)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, str):
                            domains_seen.add(it.lower())
        domain_hits = len(domains_seen)
        if domain_hits == 0:
            return 0.0
        # Map count to 0..1 with soft cap
        return max(0.0, min(1.0, math.log2(1 + domain_hits) / 4.0))

    def _consistency_score(self, texts: List[str]) -> float:
        """Very rough internal-consistency proxy: favor shorter, consistent narratives."""
        if not texts:
            return 0.0
        lengths = [len(t) for t in texts]
        mean = sum(lengths) / len(lengths)
        var = sum((l - mean) ** 2 for l in lengths) / max(1, len(lengths) - 1)
        # Smaller variance => more consistent; clamp to 0..1
        return max(0.0, min(1.0, 1.0 / (1.0 + var / (mean + 1.0))))

    def analyze(self, limit: Optional[int] = 1000) -> Dict[str, Any]:
        """
        Analyze stored emergence corpora and produce an emergence report.
        """
        corpus: List[Dict[str, Any]] = []
        intel_files = [
            self.data_dir / "emergence_intel.jsonl",
            self.data_dir / "breakthrough_insights.jsonl",
        ]
        for f in intel_files:
            corpus.extend(self._load_jsonl(f))
        if not corpus:
            return {
                "emergence_detected": False,
                "confidence": 0.0,
                "patterns": [],
                "emergent_patterns": [],
                "analysis": "No emergence corpora found; provide data to enable analysis.",
            }

        if limit is not None and len(corpus) > limit:
            corpus = corpus[:limit]

        texts = [self._extract_text(r) for r in corpus]
        novelty = self._estimate_novelty(texts)
        cross = self._estimate_cross_domain(corpus)
        consistency = self._consistency_score(texts)

        # Composite confidence: weighted geometric mean to penalize weak dimensions
        eps = 1e-6
        components = [max(eps, novelty), max(eps, cross), max(eps, consistency)]
        geo = (components[0] * components[1] * components[2]) ** (1.0 / 3.0)
        confidence = max(0.0, min(1.0, geo))

        patterns: List[Dict[str, Any]] = [
            {"name": "novelty", "score": round(novelty, 3)},
            {"name": "cross_domain_synthesis", "score": round(cross, 3)},
            {"name": "internal_consistency", "score": round(consistency, 3)},
        ]

        emergence_detected = confidence > 0.33 and (novelty > 0.2 or cross > 0.2)
        
        # Extract domains from corpus
        domains = []
        for rec in corpus:
            for key in ("domains", "tags", "categories", "labels"):
                v = rec.get(key)
                if isinstance(v, list):
                    domains.extend([str(d).lower() for d in v])
        domains = list(set(domains))[:5]  # Limit to 5 unique domains
        
        result = {
            "emergence_detected": emergence_detected,
            "confidence": round(confidence, 3),
            "patterns": patterns,
            "emergent_patterns": [],
            "analysis": (
                "Emergence indicators computed from local corpora. "
                f"novelty={novelty:.3f}, cross_domain={cross:.3f}, consistency={consistency:.3f}"
            ),
        }
        
        # Persist emergence patterns to database
        if self.enable_persistence and self.unified_db and emergence_detected:
            try:
                self._persist_emergence_pattern(result, patterns, domains, confidence)
            except Exception as e:
                logger.warning(f"Could not persist emergence pattern: {e}")
        
        return result

    # Backwards compatibility with previous stub API
    def generate_emergence(self, *args, **kwargs) -> Dict[str, Any]:
        return self.analyze()
    
    # ========== Persistence Methods ==========
    
    def _persist_emergence_pattern(
        self,
        result: Dict[str, Any],
        patterns: List[Dict[str, Any]],
        domains: List[str],
        confidence: float
    ):
        """Persist emergence pattern to database"""
        if not self.unified_db:
            return
        
        try:
            import asyncio
            
            # Generate pattern ID
            pattern_data_str = json.dumps(patterns, sort_keys=True)
            pattern_id = f"emergence_{hashlib.md5(pattern_data_str.encode()).hexdigest()[:12]}"
            
            # Determine pattern type
            pattern_type = "cross_domain" if result.get("patterns", [{}])[0].get("name") == "cross_domain_synthesis" else "novelty"
            
            query = '''
                INSERT OR REPLACE INTO emergence_patterns (
                    pattern_id, pattern_type, pattern_data, confidence,
                    first_detected, last_updated, pattern_strength,
                    cross_domain_connections, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            current_time = time.time()
            
            params = (
                pattern_id,
                pattern_type,
                pattern_data_str,
                confidence,
                current_time,  # first_detected
                current_time,  # last_updated
                confidence,  # pattern_strength
                json.dumps(domains),
                json.dumps({
                    "emergence_detected": result.get("emergence_detected", False),
                    "analysis": result.get("analysis", ""),
                    "corpus_size": len(result.get("emergent_patterns", [])),
                    "source": "emergence_engine"
                })
            )
            
            db_result = asyncio.run(self.unified_db.execute_query(query, params, fetch=False))
            if db_result.success:
                logger.debug(f"Persisted emergence pattern: {pattern_id} (confidence: {confidence:.3f})")
            
        except Exception as e:
            logger.warning(f"Could not persist emergence pattern: {e}")
    
    def store_breakthrough_discovery(
        self,
        discovery_id: str,
        discovery_type: str,
        title: str,
        description: str,
        domains: List[str],
        confidence_score: float,
        validation_status: str,
        impact_level: str,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Store a breakthrough discovery to database"""
        if not self.unified_db:
            return False
        
        try:
            import asyncio
            
            query = '''
                INSERT OR REPLACE INTO breakthrough_discoveries (
                    discovery_id, discovery_type, title, description,
                    domains, confidence_score, validation_status,
                    impact_level, created_at, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            params = (
                discovery_id,
                discovery_type,
                title,
                description,
                json.dumps(domains),
                confidence_score,
                validation_status,
                impact_level,
                time.time(),
                json.dumps(metadata or {})
            )
            
            result = asyncio.run(self.unified_db.execute_query(query, params, fetch=False))
            return result.success
            
        except Exception as e:
            logger.warning(f"Could not store breakthrough discovery: {e}")
            return False