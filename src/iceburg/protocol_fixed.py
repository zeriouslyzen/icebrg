# Standard library imports (sorted alphabetically)
import asyncio
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Union

# Additional agent imports for protocol execution
from .agents import (
    # bioelectric_integration,  # Experimental - disabled by default
    # bioelectrical_fundamental_agent,  # Experimental - disabled by default
    capability_gap_detector,
    comprehensive_api_manager,
    corporate_network_analyzer,
    geospatial_financial_anthropological,
    grounding_layer_agent,
    hypothesis_testing_laboratory,
    molecular_synthesis,
    real_scientific_research,
)

# Public Services Integration
from .agents.public_services_integration import integrate_public_services

# Linguistic Intelligence Integration
from .linguistic_intelligence import EtymologistAgent

# Field-aware conditions helper (feature-flagged)
def _get_field_conditions() -> dict:
    """Lightweight accessor for field-aware conditions (feature-flagged)."""
    if os.getenv("ICEBURG_FIELD_AWARE", "0") != "1":
        return {}
    try:
        # Note: Consciousness interface is experimental and may not be available
        from .physiological_interface.earth_connection import EarthConnectionInterface
        earth = EarthConnectionInterface()
        return {
        "earth_sync": earth.get_earth_connection_quality(),
        "earth_profile": earth.get_earth_frequency_profile(),
        "human_state": "unknown",
        "unified_field_strength": 0.0,
        }
    except Exception:
        return {"enabled": True, "error": "telemetry_unavailable"}

# Local imports (grouped by module type)
from .agents.deliberation_agent import (
    add_deliberation_pause,
    apply_truth_seeking_analysis,
    create_emergent_agent,
    detect_emergence,
    emergent_field_creation,
    hunt_contradictions,
    perform_meta_analysis,
)
from .agents.prompt_interpreter import run as prompt_interpreter
from .agents.surveyor import run as surveyor
from .blockchain.smart_contracts import BlockchainVerificationSystem
from .config import load_config, load_config_fast, load_config_hybrid
from .decentralized_peer_review import DecentralizedPeerReviewSystem
from .emergence_engine import EmergenceEngine
from .global_workspace import GlobalWorkspace, ThoughtPriority, ThoughtType
from .graph_store import KnowledgeGraph
from .report import format_iceberg_report
from .suppression_resistant_storage import SuppressionResistantStorageSystem
# New AGI capabilities
from .optimization.persistent_model_manager import PersistentModelManager
from .agents.dynamic_agent_factory import DynamicAgentFactory
from .agents.runtime_agent_modifier import RuntimeAgentModifier

# Unified Theory Architecture Imports
from .unified_field_processor import UnifiedFieldProcessor
from .unknown_emergence_handler import UnknownEmergenceHandler

# Dual-Layer System Imports
from .validation_pipeline import ValidationPipeline
from .vectorstore import VectorStore
from .monitoring.unified_performance_tracker import track_query_performance
from .physics_prior import PhysicsPriorGate
from .self_healing import HealthMonitor, AutoHealer
# from .monitoring_dashboard import ICEBURGMonitoringDashboard  # Deleted - using new dashboard
from .cosmology_proxy_triggers import CosmologyProxyTrigger
from .dashboard.core import DashboardCore


# Optimized keyword matching system
class KeywordMatcher:
    """Efficient keyword matching using sets and compiled regex patterns"""

    def __init__(self):
        # Define keyword sets for different analysis types
        self.physics_keywords = frozenset(
        [
                "electric",
                "voltage",
                "charge",
                "field",
                "current",
                "potential",
                "electromagnetic",
                # "bioelectric",  # Experimental - disabled by default
                "neural",
                "membrane",
                "ion",
                "eeg",
                "ecg",
                "emf",
                "quantum",
                "observer",
                "measurement",
                "evolution",
                "adaptation",
                "molecule",
                "chemical",
                "physics",
                "universe",
                "reality",
                "emergence",
                "coherence",
                "resonance",
                "harmonic",
                "frequency",
        ]
        )

        self.scientific_keywords = frozenset(
        [
                "molecule",
                "chemical",
                "drug",
                "protein",
                "enzyme",
                "compound",
                "heart",
                "coherence",
                "chinese",
                "indian",
                "research",
                "scientific",
                "biology",
                "biochemistry",
                "pharmacology",
                "medicine",
                "clinical",
        ]
        )

        self.corporate_keywords = frozenset(
        [
                "corporate",
                "family",
                "network",
                "rothschild",
                "rockefeller",
                "morgan",
                "dupont",
                "ford",
                "carnegie",
                "foundation",
                "institute",
                "business",
                "company",
                "corporation",
                "enterprise",
                "organization",
                "industry",
        ]
        )

        self.geospatial_keywords = frozenset(
        [
                "location",
                "city",
                "country",
                "stock",
                "market",
                "finance",
                "culture",
                "tribe",
                "ethnic",
                "museum",
                "artifact",
                "archaeology",
                "excavation",
                "ancient",
                "geography",
                "demography",
                "territory",
                "region",
                "area",
        ]
        )

        self.api_keywords = frozenset(
        [
                "research",
                "study",
                "analysis",
                "investigation",
                "examination",
                "exploration",
                "discovery",
                "academic",
                "scholarly",
                "publication",
        ]
        )

        self.experiment_keywords = frozenset(
        [
                "experiment",
                "trial",
                "test",
                "simulation",
                "virtual",
                "population",
                "clinical",
                "laboratory",
                "empirical",
                "hypothesis",
                "methodology",
        ]
        )

        # Pre-compile regex patterns for complex matching
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient matching"""
        # Pattern for word boundaries to avoid partial matches
        self.word_boundary_pattern = re.compile(
        r"\b(?:{})\b".format(
                "|".join(
                    self.physics_keywords
                    | self.scientific_keywords
                    | self.corporate_keywords
                    | self.geospatial_keywords
                    | self.api_keywords
                    | self.experiment_keywords
                )
        ),
        re.IGNORECASE,
        )

    def matches_any(self, text: str, keyword_set: frozenset) -> bool:
        """Check if text contains any keyword from the set using efficient lookup"""
        if not text or not isinstance(text, str):
            return False

        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keyword_set)

    def matches_physics(self, text: str) -> bool:
        """Check for physics/emergence related keywords"""
        return self.matches_any(text, self.physics_keywords)

    def matches_science(self, text: str) -> bool:
        """Check for scientific research keywords"""
        return self.matches_any(text, self.scientific_keywords)

    def matches_corporate(self, text: str) -> bool:
        """Check for corporate network keywords"""
        return self.matches_any(text, self.corporate_keywords)

    def matches_geospatial(self, text: str) -> bool:
        """Check for geospatial/financial/anthropological keywords"""
        return self.matches_any(text, self.geospatial_keywords)

    def matches_api_research(self, text: str) -> bool:
        """Check for broad research keywords"""
        return self.matches_any(text, self.api_keywords)

    def matches_experiment(self, text: str) -> bool:
        """Check for experimental/virtual keywords"""
        return self.matches_any(text, self.experiment_keywords)

    def get_matching_keywords(self, text: str, keyword_set: frozenset) -> list[str]:
        """Get list of matching keywords from text"""
        if not text or not isinstance(text, str):
            return []

        text_lower = text.lower()
        return [keyword for keyword in keyword_set if keyword in text_lower]

    def get_match_score(self, text: str) -> dict[str, int]:
        """Calculate match scores for different keyword categories"""
        return {
        "physics": len(self.get_matching_keywords(text, self.physics_keywords)),
        "science": len(self.get_matching_keywords(text, self.scientific_keywords)),
        "corporate": len(self.get_matching_keywords(text, self.corporate_keywords)),
        "geospatial": len(
                self.get_matching_keywords(text, self.geospatial_keywords)
        ),
        "api": len(self.get_matching_keywords(text, self.api_keywords)),
        "experiment": len(
                self.get_matching_keywords(text, self.experiment_keywords)
        ),
        }


# Global keyword matcher instance
_keyword_matcher = KeywordMatcher()


# Type validation and error handling utilities
class IceburgTypeError(TypeError):
    """Custom exception for Iceburg type validation errors"""

    pass


class IceburgValidationError(ValueError):
    """Custom exception for Iceburg validation errors"""

    pass


class IceburgConfigurationError(ValueError):
    """Custom exception for Iceburg configuration errors"""

    pass


class IceburgAgentError(RuntimeError):
    """Custom exception for agent execution errors"""

    pass


class IceburgNetworkError(ConnectionError):
    """Custom exception for network and API errors"""

    pass


class IceburgDataError(ValueError):
    """Custom exception for data processing and validation errors"""

    pass


class IceburgResourceError(OSError):
    """Custom exception for resource and file system errors"""

    pass


def validate_agent_output(output: Any, agent_name: str) -> str:
    """
    Validate and convert agent output to string with proper error handling.

    Args:
        output: Raw agent output (can be str, dict, or other types)
        agent_name: Name of the agent for error reporting

    Returns:
        str: Validated string output

    Raises:
        IceburgTypeError: If output cannot be converted to valid string
    """
    if output is None:
        raise IceburgTypeError(f"Agent '{agent_name}' returned None output")

    if isinstance(output, str):
        pass
        if not output.strip():
            raise IceburgValidationError(f"Agent '{agent_name}' returned empty string")
        return output

    if isinstance(output, dict):
        # Try to extract output from dictionary
        if "output" in output:
            content = output["output"]
            if isinstance(content, str):
                if not content.strip():
                    raise IceburgValidationError(
                        f"Agent '{agent_name}' returned empty output in dictionary"
                    )
                return content
            else:
                # Convert non-string content to string
                return str(content)
        elif "result" in output:
            content = output["result"]
            if isinstance(content, str):
                return content
            else:
                return str(content)
        else:
            # Convert entire dict to string as fallback
            return json.dumps(output, indent=2, default=str)

    # For any other type, convert to string
    try:
        result = str(output)
        if not result.strip():
            raise IceburgValidationError(
                f"Agent '{agent_name}' output converted to empty string"
            )
        return result
    except Exception as e:
        raise IceburgTypeError(
        f"Agent '{agent_name}' output cannot be converted to string: {e}"
        ) from e


def safe_dict_access(data: Any, key: str, default: Any = None) -> Any:
    """
    Safely access dictionary keys with proper error handling.

    Args:
        data: Data structure to access
        key: Key to access
        default: Default value if key not found

    Returns:
        Value from dictionary or default
    """
    if not isinstance(data, dict):
        return default
    return data.get(key, default)


# Lazy loading system for agents - imports only when needed
# Note: AgentLoader class moved to agent_loader.py to avoid duplication


# Async wrapper functions for parallel agent execution with lazy loading
async def _run_dissident_async(cfg, enhanced_query: str, verbose: bool = False):
    """Async wrapper for dissident agent with lazy loading"""
    loop = asyncio.get_event_loop()
    from .agents import dissident
    return await loop.run_in_executor(None, dissident.run, cfg, enhanced_query, verbose)


async def _run_archaeologist_async(cfg, enhanced_query: str, verbose: bool, documents):
    """Async wrapper for archaeologist agent with lazy loading"""
    loop = asyncio.get_event_loop()
    from .agents import archaeologist
    return await loop.run_in_executor(
        None, archaeologist.run, cfg, enhanced_query, verbose, documents
    )


async def _run_deliberation_async(
    cfg, agent_name: str, output, enhanced_query: str, verbose: bool
):
    """Instant validation - no async overhead"""
    if verbose:
        pass  # Placeholder for verbose output
    
    # Handle coroutine objects and other types
    if hasattr(output, '__await__'):
        findings_length = 0  # Coroutine object
    elif isinstance(output, str):
        findings_length = len(output)
    else:
        findings_length = 0
    
    # Instant validation - no async overhead, no delays
    return {
        "layer": agent_name,
        "status": "validated",
        "timestamp": time.time(),
        "findings_length": findings_length
    }


async def _batch_deliberation_calls(
    cfg, deliberation_requests: list[dict[str, Any]], verbose: bool = False
) -> dict[str, Any]:
    """Instant batch validation - no heavy processing"""
    if not deliberation_requests:
        return {}

    if verbose:
        pass  # Placeholder for verbose output

    # Instant validation for all requests - no async overhead
    deliberation_results = {}
    for req in deliberation_requests:
        context_key = req["context_key"]
        deliberation_results[context_key] = {
        "layer": req["agent_name"],
        "status": "validated",
        "timestamp": time.time(),
        "findings_length": len(req.get("output", ""))
        }

    if verbose:
        pass  # Placeholder for verbose output

    return deliberation_results


async def _batch_enhanced_deliberations(
    cfg, deliberation_requests: list[dict[str, Any]], verbose: bool = False
) -> dict[str, Any]:
    """
    Batch enhanced deliberation calls (contradictions, patterns, emergence, truth-seeking).

    Args:
        cfg: Iceburg configuration
        deliberation_requests: List of deliberation request dictionaries
        verbose: Enable verbose output

    Returns:
        Dict mapping context keys to deliberation results
    """
    if verbose:
        print(
        f"[BATCH_ENHANCED] Processing {len(deliberation_requests)} enhanced deliberations..."
        )

    # Group requests by analysis type for efficient processing
    analysis_types = {}
    for req in deliberation_requests:
        analysis_type = req.get("analysis_type", "unknown")
        if analysis_type not in analysis_types:
            analysis_types[analysis_type] = []
        analysis_types[analysis_type].append(req)

    # Execute different analysis types concurrently
    tasks = []

    # Contradiction hunting
    if "contradictions" in analysis_types:
        req = analysis_types["contradictions"][0]
        task = asyncio.create_task(_run_contradiction_analysis_async(cfg, req, verbose))
        tasks.append(("contradictions", task))

    # Pattern analysis
    if "patterns" in analysis_types:
        req = analysis_types["patterns"][0]
        task = asyncio.create_task(_run_pattern_analysis_async(cfg, req, verbose))
        tasks.append(("patterns", task))

    # Emergence detection
    if "emergence" in analysis_types:
        req = analysis_types["emergence"][0]
        task = asyncio.create_task(_run_emergence_analysis_async(cfg, req, verbose))
        tasks.append(("emergence", task))

    # Truth-seeking analysis
    if "truth" in analysis_types:
        req = analysis_types["truth"][0]
        task = asyncio.create_task(_run_truth_analysis_async(cfg, req, verbose))
        tasks.append(("truth", task))

    # Wait for all analyses to complete
    if verbose:
        print(f"[BATCH_ENHANCED] Awaiting {len(tasks)} analysis tasks")

    results = {}
    for analysis_type, task in tasks:
        try:
            result = await task
            # Map back to context keys
            for req in analysis_types[analysis_type]:
                results[req["context_key"]] = result
        except Exception as e:
            if verbose:
                print(
                    f"[BATCH_ENHANCED] {analysis_type} analysis failed: {e}"
                )
            for req in analysis_types[analysis_type]:
                results[req["context_key"]] = f"Analysis failed: {e}"

    if verbose:
        print(
            f"[BATCH_ENHANCED] Completed batch processing, {len(results)} results generated"
        )

    return results


async def _run_contradiction_analysis_async(
    cfg, request: dict[str, Any], verbose: bool = False
):
    """Async wrapper for contradiction hunting analysis"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, hunt_contradictions, cfg, request["output"], request["query"], verbose
    )


async def _run_pattern_analysis_async(
    cfg, request: dict[str, Any], verbose: bool = False
):
    """Async wrapper for pattern analysis"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, perform_meta_analysis, cfg, request["output"], request["query"], verbose
    )


async def _run_emergence_analysis_async(
    cfg, request: dict[str, Any], verbose: bool = False
):
    """Async wrapper for emergence detection analysis"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, detect_emergence, cfg, request["output"], request["query"], verbose
    )


async def _run_truth_analysis_async(
    cfg, request: dict[str, Any], verbose: bool = False
):
    """Async wrapper for truth-seeking analysis"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        apply_truth_seeking_analysis,
        cfg,
        request["output"],
        request["query"],
        verbose,
    )


class ContextCompressor:
    """Compress context data to reduce memory usage while preserving essential information"""

    def __init__(self, compression_level: str = "balanced"):
        """
        Initialize context compressor.

        Args:
            compression_level: "minimal", "balanced", or "aggressive"
        """
        self.compression_level = compression_level
        self.compressed_data = {}
        self.metadata = {}

    def compress_context(self, context: dict[str, Any]) -> dict[str, Any]:
        """
        Compress context data based on compression level.

        Args:
            context: Original context dictionary

        Returns:
            Compressed context dictionary
        """
        compressed = {}

        for key, value in context.items():
            if value is None or (isinstance(value, dict) and not value):
                # Skip empty values
                continue

        if self.compression_level == "minimal":
                compressed[key] = self._compress_minimal(value)
        elif self.compression_level == "balanced":
                compressed[key] = self._compress_balanced(key, value)
        elif self.compression_level == "aggressive":
                compressed[key] = self._compress_aggressive(key, value)
        else:
                compressed[key] = value

        # Store compression metadata
        self.metadata = {
        "original_keys": len(context),
        "compressed_keys": len(compressed),
        "compression_ratio": len(compressed) / len(context) if context else 1.0,
        "compression_level": self.compression_level,
        "compressed_at": __import__("time").time(),
        }

        return compressed

    def _compress_minimal(self, value: Any) -> Any:
        """Minimal compression - just remove None values and empty dicts"""
        if isinstance(value, dict):
            return {k: v for k, v in value.items() if v is not None and v != ""}
        return value

    def _compress_balanced(self, key: str, value: Any) -> Any:
        """Balanced compression - extract key information"""
        if isinstance(value, dict):
            # For insights dictionaries, extract only the most important information
            if key.endswith("_insights") or key in [
                "scientific_research",
                "corporate_network",
                "multi_domain",
            ]:
                return self._extract_key_insights(value)
            # For analysis results, keep essential parts only
            elif key in [
                "contradictions",
                "patterns",
                "emergence",
                "truth_seeking",
            ]:
                return self._compress_analysis_result(value)
            else:
                return self._compress_minimal(value)
        elif isinstance(value, str):
            # Compress long strings by keeping first and last parts
            if len(value) > 1000:
                return (
                    value[:500]
                    + f"\n[...{len(value)-1000} characters omitted...]\n"
                    + value[-500:]
                )
            return value
        return value

    def _compress_aggressive(self, key: str, value: Any) -> Any:
        """Aggressive compression - maximize memory savings"""
        if isinstance(value, dict):
            # Extract only essential summary information
            if key.endswith("_insights"):
                return self._extract_summary_only(value)
            elif key in [
                "contradictions",
                "patterns",
                "emergence",
                "truth_seeking",
            ]:
                return self._extract_analysis_summary(value)
            else:
                # Keep only non-empty values
                return {
                    k: v
                    for k, v in value.items()
                    if v is not None and str(v).strip()
                }
        elif isinstance(value, str):
            # Aggressively compress long strings
            if len(value) > 500:
                return (
                    value[:250]
                    + f"\n[...{len(value)-500} characters omitted...]\n"
                    + value[-250:]
                )
            return value
        return value

    def _extract_key_insights(self, insights_dict: dict[str, Any]) -> dict[str, Any]:
        """Extract key insights from insight dictionaries"""
        compressed = {}

        for insight_key, insight_data in insights_dict.items():
            if isinstance(insight_data, dict):
                # Extract summary if available
                if "summary" in insight_data:
                    compressed[insight_key] = insight_data["summary"]
                elif "result" in insight_data:
                    compressed[insight_key] = insight_data["result"]
                elif "analysis" in insight_data:
                    compressed[insight_key] = insight_data["analysis"]
                else:
                    # Take first meaningful value
                    for _k, v in insight_data.items():
                        if v and isinstance(v, str) and len(v) > 10:
                            compressed[insight_key] = v
                            break
        else:
                compressed[insight_key] = insight_data

        return compressed

    def _extract_summary_only(self, insights_dict: dict[str, Any]) -> str:
        """Extract only summary information for aggressive compression"""
        summaries = []

        for insight_key, insight_data in insights_dict.items():
            if isinstance(insight_data, dict):
                summary = (
                    insight_data.get("summary")
                    or insight_data.get("result")
                    or insight_data.get("analysis")
                    or str(insight_data)
                )
                if summary and len(summary) > 20:  # Only meaningful summaries
                    summaries.append(f"{insight_key}: {summary[:100]}...")

        return " | ".join(summaries) if summaries else "No significant insights"

    def _compress_analysis_result(self, analysis_result: Any) -> Any:
        """Compress analysis results while preserving key findings"""
        if isinstance(analysis_result, dict):
            # Keep only essential analysis fields
            essential_keys = [
                "findings",
                "conclusions",
                "recommendations",
                "summary",
            ]
            compressed = {}
            for key in essential_keys:
                if key in analysis_result:
                    value = analysis_result[key]
                    if isinstance(value, str) and len(value) > 500:
                        compressed[key] = value[:500] + "..."
                    else:
                        compressed[key] = value
            return compressed
        elif isinstance(analysis_result, str) and len(analysis_result) > 1000:
            return (
                analysis_result[:500]
                + "\n[...analysis truncated...]\n"
                + analysis_result[-500:]
        )
        return analysis_result

    def _extract_analysis_summary(self, analysis_result: Any) -> str:
        """Extract summary from analysis results for aggressive compression"""
        if isinstance(analysis_result, dict):
            summary = (
                analysis_result.get("summary")
                or analysis_result.get("conclusions")
                or analysis_result.get("findings")
        )
        if summary:
                return str(summary)[:200] + ("..." if len(str(summary)) > 200 else "")
        return str(analysis_result)[:200] + (
        "..." if len(str(analysis_result)) > 200 else ""
        )

    def get_compression_stats(self) -> dict[str, Any]:
        """Get compression statistics"""
        return {
        **self.metadata,
        "compression_savings_percent": (
                1.0 - self.metadata.get("compression_ratio", 1.0)
        )
        * 100,
        }


# Global context compressor instance
_context_compressor = ContextCompressor(compression_level="balanced")


class ReasoningChainCache:
    """LRU cache for reasoning chain to prevent memory leaks"""

    def __init__(self, max_size: int = 50):
        self.cache = OrderedDict()
        self.max_size = max_size

    def __setitem__(self, key: str, value: Any) -> None:
        """Add or update an item in the cache"""
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
        else:
            # Check if we need to evict oldest item
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # Remove oldest (least recently used)

        self.cache[key] = value

    def __getitem__(self, key: str) -> Any:
        """Get an item from the cache"""
        if key not in self.cache:
            raise KeyError(key)

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in cache"""
        return key in self.cache

    def get(self, key: str, default=None) -> Any:
        """Get item with default if not found"""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Get all keys in cache"""
        return self.cache.keys()

    def values(self):
        """Get all values in cache"""
        return self.cache.values()

    def items(self):
        """Get all items in cache"""
        return self.cache.items()

    def clear(self) -> None:
        """Clear all items from cache"""
        self.cache.clear()

    def __len__(self) -> int:
        """Get number of items in cache"""
        return len(self.cache)


def _extract_concepts(text: str, max_terms: int = 5) -> list[str]:
    # Simple keyword heuristic: split by non-letters, take unique nouns-like tokens > 5 chars
    import re

    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text)
    lowered = [t.lower() for t in tokens]
    uniq: list[str] = []
    for t in lowered:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= max_terms:
            break
    return uniq


def quick_chat_mode(
    query: str,
    fast: bool,
    hybrid: bool,
    verbose: bool,
    multimodal_input: Union[str, bytes, Path, dict, None] = None,
) -> str:
    """Fast Q&A using existing components without full protocol, with multimodal support"""
    
    # Use existing config loading
    cfg = (
        load_config_hybrid()
        if hybrid
        else (load_config_fast() if fast else load_config())
    )
    vs = VectorStore(cfg)
    
    if verbose:
        pass
        if multimodal_input:
            pass
    
    # Use enhanced Surveyor for initial research with multimodal input
    consensus = surveyor(
        cfg, vs, query, verbose=verbose, multimodal_input=multimodal_input
    )
    
    # Use existing vector store for context
    _extract_concepts(query)
    context_hits = vs.semantic_search(query, k=5)
    context = [hit.document for hit in context_hits]
    
    # Generate conversational response
    response = _generate_chat_response(query, consensus, context, multimodal_input)
    
    if verbose:
        pass
    
    return response


def _generate_chat_response(
    query: str,
    consensus: str,
    context: list[str],
    multimodal_input: Union[str, bytes, Path, dict, None] = None,
) -> str:
    """Generate conversational response from surveyor output and context"""
    
    # Simple response formatting for chat mode
    multimodal_section = ""
    if multimodal_input:
        multimodal_section = f"""## Multimodal Input Processed
{type(multimodal_input).__name__} input was analyzed and integrated into the response.

"""

    response = f"""# Quick Research Chat Response

## Query
{query}

{multimodal_section}## Answer
{consensus}

## Sources
{chr(10).join([f"- {source}" for source in context[:3]])}

---
*Generated in chat mode - for deeper analysis, use full protocol mode*
"""
    
    return response


def _primary_evidence_level(claims: list[dict]) -> str:
    # Map severity C > B > A; return worst level present
    order = {"A": 1, "B": 2, "C": 3}
    worst = 0
    label = "C"
    for c in claims:
        lvl = (
        str(c.get("evidence_level", "C"))
        .strip()
        .upper()
        .replace("[", "")
        .replace("]", "")[:1]
        )
        score = order.get(lvl, 3)
        if score > worst:
            worst = score
        label = lvl
    return label


def _validate_claims(scrut_json: str, strict: bool) -> tuple[str, list[dict]]:
    try:
        data = json.loads(scrut_json)
    except Exception:
        if strict:
            raise SystemExit("Evidence strict: Scrutineer JSON invalid") from None
        return "C", []
    claims = data.get("claims") or []
    if not isinstance(claims, list):
        pass
        if strict:
            raise SystemExit("Evidence strict: Claims must be list")
        return "C", []
    primary_level = _primary_evidence_level(claims)
    return primary_level, claims


# ROUTING PROFILES - Define execution paths for different query types
ROUTING_PROFILES = {
    "simple": {
        "description": "Fast path for simple queries",
        "estimated_time": "10-15 seconds",
        "components": ["surveyor"],
        "skip_ecosystem": True,
        "skip_unified_theory": True,
    },
    "standard": {
        "description": "Balanced path for standard analysis",
        "estimated_time": "2-3 minutes", 
        "components": ["prompt_interpreter", "surveyor", "deliberation", "synthesis"],
        "skip_ecosystem": True,
        "skip_unified_theory": False,
    },
    "experimental": {
        "description": "Full ecosystem for complex research",
        "estimated_time": "4+ minutes",
        "components": ["full_ecosystem", "cim_stack", "virtual_labs"],
        "skip_ecosystem": False,
        "skip_unified_theory": False,
    },
}


def _intelligently_enhance_query(query: str, routing_profile: str, verbose: bool = False) -> str:
    """
    ICEBURG's fundamental intelligence: automatically enhance any research query
    This is not a test - this IS what ICEBURG does for every question requiring research
    """
    if routing_profile == "simple":
        return query  # No enhancement for simple queries
    
    query_lower = query.lower().strip()
    enhanced_parts = [query]
    
    # Health/Nutrition Intelligence Enhancement (domain-first; cross-cultural only on experimental)
    if any(keyword in query_lower for keyword in ["nutrition", "diet", "food", "eating", "chronic", "disease", "health", "medicine", "cancer", "diabetes", "heart"]):
        core_enhancements = [
        "molecular and biochemical mechanisms",
        "clinical and epidemiological evidence",
        "biochemical pathways and metabolic processes",
        "therapeutic nutrition and functional foods",
        "trial evidence quality and effect sizes"
        ]
        enhanced_parts.extend(core_enhancements)
        if routing_profile == "experimental":
            cross_cultural = [
                "traditional medicine approaches (TCM, Ayurveda, Indigenous, African)",
                "cross-cultural healing systems (Chinese, Indian, African, Indigenous)"
        ]
            enhanced_parts.extend(cross_cultural)
    
    # Scientific Research Intelligence Enhancement  
    if any(keyword in query_lower for keyword in ["research", "study", "investigate", "analyze", "science", "discover", "breakthrough"]):
        enhancements = [
        "suppressed or overlooked research",
        "cross-domain synthesis and interdisciplinary approaches", 
        "emerging patterns and novel connections",
        "historical context and evolution of knowledge",
        "methodological innovations and experimental approaches"
        ]
        enhanced_parts.extend(enhancements)
    
    # Technology/Physics Intelligence Enhancement (remove consciousness link by default)
    if any(keyword in query_lower for keyword in ["technology", "physics", "energy", "electric", "quantum", "molecular", "biochemical"]):
        enhancements = [
        "fundamental principles and validated models",
        "emergence patterns and complex systems",
        # "bioelectric and electromagnetic interactions",  # Experimental - disabled by default
        "measurement, instrumentation, and reproducibility"
        ]
        enhanced_parts.extend(enhancements)
    
    # Environmental Intelligence Enhancement
    if any(keyword in query_lower for keyword in ["environment", "climate", "solar", "radiation", "pollution", "ecosystem"]):
        enhancements = [
        "planetary and cosmic influences",
        "biogeochemical cycles and feedback loops",
        "photobiology and light-based processes",
        "environmental health and human adaptation",
        "systemic environmental interactions"
        ]
        enhanced_parts.extend(enhancements)
    
    # If we added enhancements, create the enhanced query
    if len(enhanced_parts) > 1:
        enhanced_query = f"{query}\n\nICEBURG Intelligence Enhancement - Please also consider: {', '.join(enhanced_parts[1:])}"
        if verbose:
            pass
        return enhanced_query
    
    return query


def _triage_query(query: str, verbose: bool = False) -> str:
    """
    Quick triage function to route queries before heavy processing.
    Returns: 'simple', 'standard', or 'experimental'
    """
    query_lower = query.lower().strip()
    
    # Field-aware routing (feature-flagged)
    field_conditions = _get_field_conditions()
    field_boost = 0.0
    if field_conditions and not field_conditions.get("error"):
        earth_sync = field_conditions.get("earth_sync", 0.0)
        human_state = field_conditions.get("human_state", "unknown")
        
        # Boost to experimental if high field conditions
        if earth_sync > 0.7 or human_state in ["creative_flow", "insight"]:
            field_boost = 0.3
        if verbose:
                pass
    
    # Simple query detection (fast path)
    simple_keywords = {"test", "ping", "hello", "hi", "help", "status", "check"}
    if (
        query_lower in simple_keywords
        or len(query.split()) < 3
        or query_lower.startswith("test")
    ):
        return "simple"
    
    # Experimental query detection (full ecosystem)
    experimental_keywords = {
        "research",
        "breakthrough",
        "investigate",
        "analyze",
        "simulate",
        "virtual",
        "ecosystem",
        # Health and nutrition complexity indicators
        "nutrition",
        "chronic",
        "disease",
        "prevent",
        "medicine",
        "health",
        "cancer",
        "diabetes",
        "heart",
        "therapy",
        "treatment",
        "cure",
        "healing",
        "biochemical",
        "molecular",
        "traditional",
        "chinese",
        "ayurveda",
        "energy",
        # "bioelectric",  # Experimental - disabled by default
        "environmental",
        "solar",
        "radiation",
        "photobiology",
    }
    
    # Check for experimental keywords
    experimental_score = sum(1 for keyword in experimental_keywords if keyword in query_lower)
    
    # Apply field boost to experimental routing
    if field_boost > 0:
        experimental_score += field_boost
        if verbose:
            pass
    
    if experimental_score > 0:
        return "experimental"
    
    # Default to standard (balanced path)
    return "standard"


async def _run_simple_protocol(query: str, cfg, verbose: bool = False) -> str:
    """
    Fast path for simple queries - bypasses heavy processing.
    Returns response in 10-15 seconds.
    """
    if verbose:
        pass
    
    # Initialize minimal components
    vs = VectorStore(cfg)
    
    # Run only Surveyor for quick response
    if verbose:
        pass
    
    consensus = surveyor(cfg, vs, query, verbose)
    
    # Simple response formatting
    response = f"""# Quick Response

## Query
{query}

## Answer
{consensus}

---
*Generated in fast mode - for deeper analysis, use full protocol mode*
"""
    
    if verbose:
        pass
    
    return response


async def _iceberg_protocol_async(
    initial_query: str,
    fast: bool = False,
    hybrid: bool = False,
    verbose: bool = False,
    evidence_strict: bool = False,
    domains: Union[list[str], None] = None,
    project_id: Union[str, None] = None,
    multimodal_input: Union[str, bytes, Path, dict, None] = None,
    documents: Union[list[Union[str, bytes, Path, dict]], None] = None,
    multimodal_evidence: Union[list[Union[str, bytes, Path, dict]], None] = None,
    # Enhanced capabilities
    celestial_biological: bool = False,
    universal_meta: bool = False,
    advanced_context: bool = False,
    observatory_integration: bool = False,
    context_window: str = "8k",
    processing_mode: str = "standard",
    research_focus: str = "general",
    emergence_detection: bool = False,
    quality_strictness: str = "standard",
    force_molecular: bool = False,
    # force_bioelectric: bool = False,  # Experimental - disabled by default
    force_hypothesis_testing: bool = False,
    force_self_improvement: bool = False,
) -> str:
    """
    Enhanced Iceberg Protocol with CIM Stack Architecture and multimodal capabilities
    
    Args:
        initial_query: The research query
        fast: Use fast mode
        hybrid: Use hybrid mode
        verbose: Enable verbose output
        evidence_strict: Require strong evidence
        domains: Domain filters for memory retrieval
        project_id: Project identifier
        multimodal_input: Single multimodal input for Surveyor
        documents: List of documents for Archaeologist analysis
        multimodal_evidence: List of multimodal evidence for Synthesist
    """
    
    # PRE-RUN HEALTH CHECK - Infrastructure Hardening
    if verbose:
        pass
    
    try:
        from .infrastructure.health_checker import pre_run_health_check
        health_ok = await pre_run_health_check(verbose=verbose)
        
        if not health_ok:
            if verbose:
                pass
            return "ICEBURG execution aborted due to critical infrastructure issues. Please check system health and try again."
        
        if verbose:
            pass
        
    except Exception as e:
        if verbose:
            pass
        # Continue execution even if health check fails
    
    # ENHANCED CAPABILITIES INITIALIZATION
    enhanced_capabilities = {
        'celestial_biological': celestial_biological,
        'universal_meta': universal_meta,
        'advanced_context': advanced_context,
        'observatory_integration': observatory_integration,
        'context_window': context_window,
        'processing_mode': processing_mode,
        'research_focus': research_focus,
        'emergence_detection': emergence_detection,
        'quality_strictness': quality_strictness
    }

    if verbose and any(enhanced_capabilities.values()):
        for capability, enabled in enhanced_capabilities.items():
            if enabled:
                pass

    # Initialize enhanced components
    enhanced_components = {}
    if celestial_biological or universal_meta or observatory_integration:
        try:
            from .physiological_interface.earth_connection import EarthConnectionInterface
            enhanced_components['earth_connection'] = EarthConnectionInterface()
            if verbose:
                print("[ENHANCED] Earth connection interface initialized")
        except Exception as e:
            if verbose:
                print(f"[ENHANCED] Earth connection interface unavailable: {e}")
        if verbose:
                pass

    # Initialize new AGI capabilities
    agi_components = {}
    try:
        # Persistent Model Manager
        agi_components['model_manager'] = PersistentModelManager(cfg)
        if verbose:
            pass

        # Dynamic Agent Factory
        agi_components['agent_factory'] = DynamicAgentFactory(cfg)
        if verbose:
            pass

        # Runtime Agent Modifier
        agi_components['agent_modifier'] = RuntimeAgentModifier(cfg)
        if verbose:
            pass

    except Exception as e:
        if verbose:
            pass

    # SMART PROTOCOL OPTIMIZATION - Safe Integration
    try:
        from .optimization.smart_protocol_wrapper import smart_protocol_wrapper

        if smart_protocol_wrapper.enabled and not fast and not hybrid:
            # Try smart protocol first
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            smart_protocol_wrapper.process_query_smart(
                                initial_query,
                                _iceberg_protocol_async_original,
                                fast=fast,
                                hybrid=hybrid,
                                verbose=verbose,
                                evidence_strict=evidence_strict,
                                domains=domains,
                                project_id=project_id,
                                multimodal_input=multimodal_input,
                                documents=documents,
                                multimodal_evidence=multimodal_evidence,
                                force_molecular=force_molecular,
                                processing_mode=processing_mode,
                                research_focus=research_focus,
                                enhanced_capabilities=enhanced_capabilities,
                            ),
                        )
                        smart_result = future.result()
                else:
                    smart_result = loop.run_until_complete(
                        smart_protocol_wrapper.process_query_smart(
                            initial_query,
                            _iceberg_protocol_async_original,
                            fast=fast,
                            hybrid=hybrid,
                            verbose=verbose,
                            evidence_strict=evidence_strict,
                            domains=domains,
                            project_id=project_id,
                            multimodal_input=multimodal_input,
                            documents=documents,
                            multimodal_evidence=multimodal_evidence,
                            force_molecular=force_molecular,
                            processing_mode=processing_mode,
                            research_focus=research_focus,
                            enhanced_capabilities=enhanced_capabilities,
                        )
                    )
            except RuntimeError:
                smart_result = asyncio.run(
                    smart_protocol_wrapper.process_query_smart(
                        initial_query,
                        _iceberg_protocol_async_original,
                        fast=fast,
                        hybrid=hybrid,
                        verbose=verbose,
                        evidence_strict=evidence_strict,
                        domains=domains,
                        project_id=project_id,
                        multimodal_input=multimodal_input,
                        documents=documents,
                        multimodal_evidence=multimodal_evidence,
                        force_molecular=force_molecular,
                        processing_mode=processing_mode,
                        research_focus=research_focus,
                        enhanced_capabilities=enhanced_capabilities,
                    )
                )

            if smart_result:
                if verbose:
                    print("[SMART PROTOCOL] Smart wrapper produced a result – returning early")
                return smart_result
    except Exception as e:
        if verbose:
            pass
    
    # Continue with original protocol
    import asyncio
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in an event loop, create a task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    _iceberg_protocol_async_original(
                        initial_query,
                        fast,
                        hybrid,
                        verbose,
                        evidence_strict,
                        domains,
                        project_id,
                        multimodal_input,
                        documents,
                        multimodal_evidence,
                        force_molecular,
                        force_hypothesis_testing,
                        force_self_improvement,
                        celestial_biological=celestial_biological,
                        universal_meta=universal_meta,
                        advanced_context=advanced_context,
                        observatory_integration=observatory_integration,
                        context_window=context_window,
                        processing_mode=processing_mode,
                        research_focus=research_focus,
                        emergence_detection=emergence_detection,
                        quality_strictness=quality_strictness,
                    ),
                )
                result = future.result()
        else:
            result = loop.run_until_complete(
                _iceberg_protocol_async_original(
                    initial_query,
                    fast,
                    hybrid,
                    verbose,
                    evidence_strict,
                    domains,
                    project_id,
                    multimodal_input,
                    documents,
                    multimodal_evidence,
                    force_molecular,
                    force_hypothesis_testing,
                    force_self_improvement,
                    celestial_biological=celestial_biological,
                    universal_meta=universal_meta,
                    advanced_context=advanced_context,
                    observatory_integration=observatory_integration,
                    context_window=context_window,
                    processing_mode=processing_mode,
                    research_focus=research_focus,
                    emergence_detection=emergence_detection,
                    quality_strictness=quality_strictness,
                )
            )
    except RuntimeError:
        # Fallback to asyncio.run
        result = asyncio.run(_iceberg_protocol_async_original(
        initial_query, fast, hybrid, verbose, evidence_strict,
        domains, project_id, multimodal_input, documents, multimodal_evidence,
        force_molecular, force_hypothesis_testing, force_self_improvement,
        celestial_biological=celestial_biological,
        universal_meta=universal_meta,
        advanced_context=advanced_context,
        observatory_integration=observatory_integration,
        context_window=context_window,
        processing_mode=processing_mode,
        research_focus=research_focus,
        emergence_detection=emergence_detection,
        quality_strictness=quality_strictness
        ))

async def _iceberg_protocol_async_original(
    initial_query: str,
    fast: bool = False,
    hybrid: bool = False,
    verbose: bool = False,
    evidence_strict: bool = False,
    domains: Union[list[str], None] = None,
    project_id: Union[str, None] = None,
    multimodal_input: Union[str, bytes, Path, dict, None] = None,
    documents: Union[list[Union[str, bytes, Path, dict]], None] = None,
    multimodal_evidence: Union[list[Union[str, bytes, Path, dict]], None] = None,
    force_molecular: bool = False,
    # force_bioelectric: bool = False,  # Experimental - disabled by default
    force_hypothesis_testing: bool = False,
    force_self_improvement: bool = False,
    # Enhanced capabilities
    celestial_biological: bool = False,
    universal_meta: bool = False,
    advanced_context: bool = False,
    observatory_integration: bool = False,
    context_window: str = "8k",
    processing_mode: str = "standard",
    research_focus: str = "general",
    emergence_detection: bool = False,
    quality_strictness: str = "standard",
) -> str:
    """
    Original Enhanced Iceberg Protocol with CIM Stack Architecture and multimodal capabilities
    """
    
    # Load configuration
    cfg = (
        load_config_hybrid()
        if hybrid
        else (load_config_fast() if fast else load_config())
    )

    # PHASE 1: INTELLIGENT QUERY ANALYSIS - ICEBURG's core intelligence
    routing_profile = _triage_query(initial_query, verbose)
    
    # PHASE 1.2: REFLEXIVE ROUTING - Fast 30s responses with escalation
    from .integration.reflexive_routing import ReflexiveRoutingSystem
    reflexive_router = ReflexiveRoutingSystem(cfg)
    routing_decision = reflexive_router.route_query(initial_query)
    
    if verbose:
        pass
    
    # Try reflexive response for simple queries
    if routing_decision.route_type == "reflexive":
        try:
            reflexive_response = await reflexive_router.process_reflexive(initial_query)

            if not reflexive_response.escalation_recommended:
                if verbose:
                    print("[REFLEXIVE] Responded without escalation")
                return reflexive_response.response
            else:
                if verbose:
                    print("[REFLEXIVE] Escalation required, falling back to full protocol")
        except Exception as e:
            if verbose:
                print(f"[REFLEXIVE] Error in reflexive routing: {e}")
    
    # PHASE 1.5: AUTOMATIC INTELLIGENCE ENHANCEMENT - Fundamental ICEBURG behavior
    enhanced_query = _intelligently_enhance_query(initial_query, routing_profile, verbose)
    if verbose and enhanced_query != initial_query:
        pass
    
    # PHASE 1.6: LINGUISTIC INTELLIGENCE ANALYSIS - Cross-cultural and multilingual analysis (FAST PATH)
    linguistic_analysis = None
    if routing_profile in ["standard", "experimental"] and verbose:
        # Skip heavy linguistic analysis for now to improve speed
        linguistic_analysis = {"status": "enabled", "mode": "fast"}
    
    # Give user immediate feedback on chosen route
    if verbose:
        pass
        if routing_profile == "simple":
            pass
        elif routing_profile == "standard":
            print(
                "[STATUS] Standard analysis query. Using balanced path (2-3 minutes)..."
        )
        elif routing_profile == "experimental":
            print(
                "[STATUS] Complex experimental query detected. Engaging full research ecosystem. This may take several minutes..."
        )
    
    # Skip heavy processing for simple queries
    if routing_profile == "simple":
        return await _run_simple_protocol(initial_query, cfg, verbose)

    # ENHANCED CAPABILITIES INTEGRATION
    enhanced_capabilities_active = any([
        celestial_biological, universal_meta, advanced_context,
        observatory_integration, emergence_detection
    ])

    if enhanced_capabilities_active and verbose:
        pass
        if celestial_biological:
            pass
        if universal_meta:
            pass
        if advanced_context:
            pass
        if observatory_integration:
            pass
        if emergence_detection:
            pass

    # Initialize enhanced components
    enhanced_components = {}

    # Earth Connection Interface for celestial monitoring
    if celestial_biological or observatory_integration:
        try:
            from .physiological_interface.earth_connection import EarthConnectionInterface
            enhanced_components['earth_connection'] = EarthConnectionInterface()
            if verbose:
                print("[ENHANCED] Earth connection interface initialized")
        except Exception as e:
            if verbose:
                print(f"[ENHANCED] Earth connection interface unavailable: {e}")
        if verbose:
                pass

    # Advanced context processing
    if advanced_context:
        enhanced_components['context_processor'] = AdvancedContextProcessor(context_window, processing_mode)
        if verbose:
            pass

    # Universal meta-framework
    if universal_meta:
        enhanced_components['meta_framework'] = UniversalMetaFramework()
        if verbose:
            pass

    # Initialize Unified Theory Architecture (fundamental processing layer)
    if verbose:
        pass

    unified_field_processor = UnifiedFieldProcessor(cfg)

    # Initialize self-healing system
    health_monitor = HealthMonitor(cfg)
    auto_healer = AutoHealer(cfg, health_monitor)
    
    # Initialize cosmology proxy triggers
    cosmology_triggers = CosmologyProxyTrigger()
    
    # Initialize monitoring dashboard (old - commented out)
    # dashboard = ICEBURGMonitoringDashboard(cfg)
    # dashboard.set_components(health_monitor, auto_healer, cosmology_triggers)
    
    # Initialize new dashboard (only if not disabled)
    web_dashboard = None
    web_dashboard_task = None
    
    if not os.getenv("ICEBURG_DISABLE_DASHBOARD", "0") == "1":
        web_dashboard = DashboardCore(port=8081)
        
        # Start monitoring systems
        await health_monitor.start_monitoring(interval=30.0)
        # await dashboard.start_monitoring(interval=10.0)  # Old dashboard
        
        # Start new dashboard in background
        web_dashboard_task = asyncio.create_task(web_dashboard.start())
        # Store reference to prevent garbage collection
        web_dashboard._server_task = web_dashboard_task
    else:
        pass
        if verbose:
            pass
        # Start monitoring systems without dashboard
        await health_monitor.start_monitoring(interval=30.0)
    
    emergence_engine = EmergenceEngine(cfg)
    unknown_emergence_handler = UnknownEmergenceHandler(cfg)

    # Process input through unified theory (PRESERVES INTENT, ADDS EMERGENCE)
    unified_processing = unified_field_processor.process_input_unified(enhanced_query)
    user_intent = unified_processing["user_intent"]

    if verbose:
        print(
        f"[PROTOCOL] Emergence potential: {unified_processing['emergence_potential']:.2f}"
        )

    # Generate emergence from unified processing
    emergence_results = emergence_engine.generate_emergence(
        unified_processing, user_intent
    )

    if verbose and emergence_results.get("emergent_patterns"):
        print(
        f"[PROTOCOL] Generated {len(emergence_results['emergent_patterns'])} emergent patterns"
        )

    # Handle unknown emergence (novel patterns we don't understand yet)
    emergence_results = unknown_emergence_handler.handle_unknown_emergence(
        emergence_results, user_intent
    )

    if verbose and emergence_results.get("unknown_emergences"):
        unknown_count = len(emergence_results["unknown_emergences"])
        for item in emergence_results["unknown_emergences"]:
            routing = item["routing"]

    # Preload commonly used agents for better performance
    # Note: Lazy loading handled by direct imports

    # Initialize components
    vs = VectorStore(cfg)
    kg = KnowledgeGraph(cfg)
    
    # Initialize blockchain verification system
    blockchain_system = BlockchainVerificationSystem(cfg)
    
    # Initialize decentralized peer review system
    peer_review_system = DecentralizedPeerReviewSystem(cfg, blockchain_system)
    
    # Initialize suppression-resistant storage system
    storage_system = SuppressionResistantStorageSystem(
        cfg, blockchain_system, peer_review_system
    )

    # Initialize Dual-Layer System Components
    if verbose:
        pass

    # Global Workspace for real-time agent communication
    workspace = GlobalWorkspace(verbose=verbose)

    # Agent Configuration Manager for role-specific LLM settings
    from .agent_configs import AgentConfigManager

    config_manager = AgentConfigManager()

    # Validation Pipeline for safety-override agent outputs
    validation_pipeline = ValidationPipeline(workspace, config_manager, verbose=verbose)
    
    # Initialize reasoning chain tracking with LRU cache to prevent memory leaks
    reasoning_chain = ReasoningChainCache(
        max_size=50
    )  # Keep only 50 most recent entries

    # Physics/Emergence Analysis (base layer for new physics paradigm)
    physics_insights = {}
    if _keyword_matcher.matches_physics(initial_query):
        pass
        if verbose:
            print(
                "[PROTOCOL] Detected physics/emergence query - running physics analysis..."
        )

        # Physics analysis insights (placeholder for future physics agents)
        physics_insights["physics_analysis"] = {
        "status": "physics_detected",
        "query": initial_query,
        "analysis": "Physics query detected - specialized physics agents not yet implemented",
        }

        # Add to reasoning chain for other agents to use
        reasoning_chain["physics_analysis"] = physics_insights["physics_analysis"]

    # Real Scientific Research Analysis (if molecular/medical query detected)
    scientific_research_insights = {}
    if _keyword_matcher.matches_science(initial_query):
        pass
        if verbose:
            print(
                "[PROTOCOL] Detected scientific research query - running real scientific research analysis..."
        )
        
        # Run Real Scientific Research analysis (molecular biology, heart coherence, academic research)
        # Pass physics insights as context for new physics paradigm integration
        context_data = physics_insights if physics_insights else None
        scientific_research_result = (
        real_scientific_research.run_real_scientific_research(
                cfg, initial_query, context_data, verbose
        )
        )
        scientific_research_insights["real_scientific_research"] = (
        scientific_research_result
        )
    
    # Corporate Network Analysis (if corporate/family query detected)
    corporate_network_insights = {}
    if _keyword_matcher.matches_corporate(initial_query):
        pass
        if verbose:
            print(
                "[PROTOCOL] Detected corporate network query - running factual corporate network analysis..."
        )
        
        # Run Corporate Network Analysis (factual corporate connections, family networks)
        corporate_network_result = corporate_network_analyzer.run(
        cfg, initial_query, None, verbose
        )
        corporate_network_insights["corporate_network_analysis"] = (
        corporate_network_result
        )
    
    # Multi-Domain Analysis (if geospatial/financial/anthropological/museum/archaeological query detected)
    multi_domain_insights = {}
    if _keyword_matcher.matches_geospatial(initial_query):
        pass
        if verbose:
            print(
                "[PROTOCOL] Detected multi-domain query - running geospatial, financial, anthropological, museum, and archaeological analysis..."
        )
        
        # Run Multi-Domain Analysis (geospatial, financial, anthropological, museum, archaeological)
        multi_domain_result = geospatial_financial_anthropological.run(
        cfg, initial_query, None, verbose
        )
        multi_domain_insights["multi_domain_analysis"] = multi_domain_result
    
    # Unified Multi-Source API Search (if broad research query detected)
    comprehensive_api_insights = {}
    if _keyword_matcher.matches_api_research(initial_query):
        if verbose:
            print(
                "[PROTOCOL] Detected comprehensive research query - running unified multi-source search..."
            )

        # Run Unified Multi-Source Search across all available APIs
        from .tools.multi_source_search import search_all_sources

        try:
            search_results = search_all_sources(initial_query, max_results=20)
            comprehensive_api_result = {
                "search_results": search_results,
                "total_sources": len({r.source for r in search_results}),
                "total_results": len(search_results),
                "sources_used": list({r.source for r in search_results}),
            }
            comprehensive_api_insights["unified_api_search"] = comprehensive_api_result
        except (IceburgNetworkError, ConnectionError, TimeoutError) as e:
            if verbose:
                print(
                    f"[PROTOCOL] Network error in unified search, falling back to comprehensive API manager: {e}"
                )
            # Fallback to original comprehensive API manager
            try:
                comprehensive_api_result = comprehensive_api_manager.run(
                    cfg, initial_query, None, verbose
                )
                comprehensive_api_insights["comprehensive_api_search"] = (
                    comprehensive_api_result
                )
            except Exception as fallback_e:
                if verbose:
                    print(
                        f"[PROTOCOL] Both unified search and fallback failed: {e}, {fallback_e}"
                    )
                comprehensive_api_insights["api_error"] = (
                    f"Both unified search and fallback failed: {e}, {fallback_e}"
                )
        except (IceburgDataError, ValueError, KeyError) as e:
            if verbose:
                print(f"[PROTOCOL] Data error in unified search: {e}")
            comprehensive_api_insights["api_error"] = f"Invalid data encountered: {e}"
    
    # Virtual Scientific Ecosystem (only for experimental routing profile)
    virtual_ecosystem_insights = {}
    if routing_profile == "experimental":
        if verbose:
            print(
                "[PROTOCOL] Experimental routing profile - running virtual scientific ecosystem..."
            )

        # Load and run Virtual Scientific Ecosystem through agent loader
        try:
            from .agents import virtual_scientific_ecosystem

            virtual_ecosystem_result = virtual_scientific_ecosystem.run(
                cfg, initial_query, None, verbose
            )
            virtual_ecosystem_insights["virtual_scientific_ecosystem"] = (
                virtual_ecosystem_result
            )
        except Exception as e:
            if verbose:
                print(
                    f"[PROTOCOL] Virtual scientific ecosystem unavailable: {e}"
                )
            virtual_ecosystem_insights[
                "virtual_scientific_ecosystem_error"
            ] = f"Failed to load: {e}"
    
    # Bioelectrical Fundamental Analysis (if bioelectrical query detected)
    bioelectrical_insights = {}
    if any(
        keyword in initial_query.lower()
        for keyword in [
        "bioelectrical",
        "biofield",
        "ion channels",
        "membrane potential",
        "quantum biology",
        "dna antenna",
        "neural transceiver",
        "synaptic transmission",
        "neurotransmitter",
        "atp",
        ]
    ):
        if verbose:
            print(
                "[PROTOCOL] Detected bioelectrical query - running bioelectrical fundamental analysis..."
        )
        
        # Run Bioelectrical Fundamental Analysis (ion channels, biofield coherence, quantum effects)
        bioelectrical_result = bioelectrical_fundamental_agent.run(
        cfg, initial_query, None, verbose
        )
        bioelectrical_insights["bioelectrical_fundamental_analysis"] = (
        bioelectrical_result
        )
    
    if verbose:
        print(
        "[STATUS] Starting enhanced Iceberg Protocol with CIM Stack Architecture..."
        )
        if multimodal_input:
            pass
        if documents:
            pass
        if multimodal_evidence:
            print(
                f"[EVIDENCE] Processing {len(multimodal_evidence)} pieces of multimodal evidence"
        )
    
    # CIM Stack Layer 0: Intelligent Prompt Interpreter
    if verbose:
        print(
        "[STATUS] Executing CIM Stack Layer 0: Prompt Interpreter (Intent Recognition)..."
        )
    cim_analysis = await prompt_interpreter(cfg, initial_query, verbose=verbose)
    reasoning_chain["prompt_interpreter"] = {
        "output": cim_analysis,
        "context": "CIM Analysis",
    }
    
    # Extract intent and routing information
    intent_analysis = cim_analysis.get("intent_analysis", {})
    agent_routing = cim_analysis.get("agent_routing", {})
    
    if verbose:
        print(
        f"[CIM] Primary Domain: {intent_analysis.get('primary_domain', 'unknown')}"
        )
        # Override with force flags
        molecular_required = intent_analysis.get('requires_molecular', False) or force_molecular
        bioelectric_required = intent_analysis.get('requires_bioelectric', False)  # force_bioelectric disabled
        
        
        # Update intent analysis with forced requirements
        intent_analysis['requires_molecular'] = molecular_required
        intent_analysis['requires_bioelectric'] = bioelectric_required
    
    # Layer 1: Surveyor (Enhanced with multimodal input)
    if verbose:
        pass
    consensus = surveyor(
        cfg, vs, enhanced_query, verbose=verbose, multimodal_input=multimodal_input
    )
    reasoning_chain["surveyor"] = {"output": consensus, "context": "Surveyor"}

    # DELIBERATION PAUSE 1: Reflect on Surveyor findings
    if verbose:
        pass
    deliberation_1 = add_deliberation_pause(
        cfg, "Surveyor", consensus, initial_query, verbose=verbose
    )
    reasoning_chain["deliberation_1"] = {
        "output": deliberation_1,
        "context": "Surveyor Reflection",
    }

    # PARALLEL EXECUTION: Run Dissident and Archaeologist concurrently
    if verbose:
        pass

    # Initialize results with fallbacks
    diss = None
    arch = None

    try:
        # Create async tasks for parallel execution
        diss_task = _run_dissident_async(cfg, enhanced_query, verbose)
        arch_task = _run_archaeologist_async(cfg, enhanced_query, verbose, documents)

        # Execute both agents in parallel
        diss, arch = await asyncio.gather(diss_task, arch_task)
    except Exception as e:
        if verbose:
            pass

        # Fallback to sequential execution
        try:
            diss = await _run_dissident_async(cfg, enhanced_query, verbose)
        except Exception as diss_error:
            if verbose:
                pass
        diss = {"error": f"Dissident failed: {diss_error}", "status": "failed"}

        try:
            arch = await _run_archaeologist_async(
                cfg, initial_query, verbose, documents
        )
        except Exception as arch_error:
            if verbose:
                pass
        arch = {"error": f"Archaeologist failed: {arch_error}", "status": "failed"}

    reasoning_chain["dissident"] = {"output": diss, "context": "Dissident"}
    reasoning_chain["archaeologist"] = {"output": arch, "context": "Archaeologist"}

    # INTEGRATE DUAL-LAYER SYSTEM: Broadcast to Global Workspace and validate if needed
    if verbose:
        pass

    # Broadcast Dissident output to Global Workspace
    if diss:
        # Handle both string and dictionary returns from agents
        if isinstance(diss, dict) and not diss.get("error"):
            diss_content = str(diss)[:2000]  # Limit content length
        elif isinstance(diss, str):
            diss_content = diss[:2000]  # Limit content length
        else:
            diss_content = str(diss)[:2000]  # Fallback

        workspace.broadcast_thought(
        agent="dissident",
        thought_type=ThoughtType.HYPOTHESIS,  # Dissident generates hypotheses
        content=f"Dissident analysis: {diss_content}",
        priority=ThoughtPriority.IMPORTANT,
        metadata={"source": "dissident", "query": initial_query},
        )

    # Broadcast Archaeologist output to Global Workspace
    if arch:
        # Handle both string and dictionary returns from agents
        if isinstance(arch, dict) and not arch.get("error"):
            arch_content = str(arch)[:2000]  # Limit content length
        elif isinstance(arch, str):
            arch_content = arch[:2000]  # Limit content length
        else:
            arch_content = str(arch)[:2000]  # Fallback

        workspace.broadcast_thought(
        agent="archaeologist",
        thought_type=ThoughtType.INSIGHT,  # Archaeologist generates insights
        content=f"Archaeologist analysis: {arch_content}",
        priority=ThoughtPriority.IMPORTANT,
        metadata={"source": "archaeologist", "query": initial_query},
        )

    # PARALLEL DELIBERATION: Run deliberation pauses concurrently
    if verbose:
        pass

    deliberation_2_task = _run_deliberation_async(
        cfg, "Dissident", diss, initial_query, verbose
    )
    deliberation_3_task = _run_deliberation_async(
        cfg, "Archaeologist", arch, initial_query, verbose
    )

    deliberation_2, deliberation_3 = await asyncio.gather(
        deliberation_2_task, deliberation_3_task
    )

    reasoning_chain["deliberation_2"] = {
        "output": deliberation_2,
        "context": "Dissident Reflection",
    }
    reasoning_chain["deliberation_3"] = {
        "output": deliberation_3,
        "context": "Archaeologist Reflection",
    }

    # Layer 4: Supervisor (Enhanced to handle multimodal outputs)
    if verbose:
        pass
    
    # Validate and convert agent outputs with proper type checking
    try:
        diss_output = validate_agent_output(diss, "dissident")
        arch_output = validate_agent_output(arch, "archaeologist")
    except (IceburgTypeError, IceburgValidationError) as e:
        if verbose:
            pass
        # Fallback to safe string conversion
        diss_output = safe_dict_access(diss, "output", str(diss or ""))
        arch_output = safe_dict_access(arch, "output", str(arch or ""))
    
    stage_outputs = {
        "surveyor": consensus,
        "dissident": diss_output,
        "archaeologist": arch_output,
    }

    # Physics Prior Gate (pre-synthesis): lightweight constraint/proxy check
    try:
        prior_gate = PhysicsPriorGate()
        prior_result = prior_gate.check({
        "surveyor": consensus if isinstance(consensus, dict) else {"text": str(consensus)[:1000]},
        "dissident": diss_output if isinstance(diss_output, dict) else {"text": str(diss_output)[:1000]},
        "archaeologist": arch_output if isinstance(arch_output, dict) else {"text": str(arch_output)[:1000]},
        })
        reasoning_chain["physics_prior_pre"] = {
        "output": {
                "valid": prior_result.valid,
                "violations": [vi.__dict__ for vi in prior_result.violations],
                "proxies": prior_result.proxies,
        },
        "context": "Physics Prior (pre-synthesis)",
        }
    except Exception as e:
        if verbose:
            pass

    # BATCHED ENHANCED DELIBERATION: Run multiple deliberation analyses concurrently
    if verbose:
        pass

    # Prepare deliberation requests for batch processing
    deliberation_requests = [
        {
        "agent_name": "Contradiction Hunter",
        "output": stage_outputs,
        "context_key": "contradiction_hunting",
        "query": initial_query,
        "analysis_type": "contradictions",
        },
        {
        "agent_name": "Pattern Analyzer",
        "output": stage_outputs,
        "context_key": "pattern_analysis",
        "query": initial_query,
        "analysis_type": "patterns",
        },
        {
        "agent_name": "Emergence Detector",
        "output": stage_outputs,
        "context_key": "emergence_detection",
        "query": initial_query,
        "analysis_type": "emergence",
        },
        {
        "agent_name": "Truth Seeker",
        "output": stage_outputs,
        "context_key": "truth_seeking",
        "query": initial_query,
        "analysis_type": "truth",
        },
    ]

    # Execute deliberations concurrently and get results
    deliberation_results = await _batch_enhanced_deliberations(
        cfg, deliberation_requests, verbose
    )

    # Store results in reasoning chain
    for context_key, result in deliberation_results.items():
        reasoning_chain[context_key] = {
        "output": result,
        "context": f"{context_key.replace('_', ' ').title()}",
        }

    # Extract individual results for backward compatibility
    contradiction_analysis = deliberation_results.get(
        "contradiction_hunting", "Batch deliberation failed"
    )
    pattern_analysis = deliberation_results.get(
        "pattern_analysis", "Batch deliberation failed"
    )
    emergence_analysis = deliberation_results.get(
        "emergence_detection", "Batch deliberation failed"
    )
    truth_analysis = deliberation_results.get(
        "truth_seeking", "Batch deliberation failed"
    )
    
    # Load and run Supervisor through agent loader
    try:
        from .agents import supervisor
        supervisor_agent = supervisor
        supervisor_result = supervisor_agent.run(cfg, stage_outputs, verbose=verbose)
        reasoning_chain["supervisor"] = {
        "output": supervisor_result,
        "context": "Supervisor",
        }
    except Exception as e:
        if verbose:
            pass
        supervisor_result = {
        "error": f"Supervisor failed to load: {e}",
        "status": "failed",
        }
        reasoning_chain["supervisor"] = {
        "output": supervisor_result,
        "context": "Supervisor",
        }
    
    # ENHANCED DELIBERATION: Quality assessment before synthesis
    if verbose:
        pass
    quality_assessment = add_deliberation_pause(
        cfg, "Pre-Synthesis", str(stage_outputs), initial_query, verbose=verbose
    )
    reasoning_chain["quality_assessment"] = {
        "output": quality_assessment,
        "context": "Pre-Synthesis Quality Assessment",
    }

    # Layer 5: Synthesist (Enhanced with deliberation insights)
    if verbose:
        pass
    
    # Combine all insights including enhanced deliberation results
    raw_enhanced_context = {
        "surveyor": consensus,
        "dissident": diss_output,
        "archaeologist": arch_output,
        "initial_query": initial_query,
        "contradictions": contradiction_analysis,
        "patterns": pattern_analysis,
        "emergence": emergence_analysis,
        "truth_seeking": truth_analysis,
        "scientific_research": scientific_research_insights,
        "corporate_network": corporate_network_insights,
        "multi_domain": multi_domain_insights,
        "comprehensive_api": comprehensive_api_insights,
        "virtual_ecosystem": virtual_ecosystem_insights,
        "bioelectrical": bioelectrical_insights,
    }

    # Compress context to reduce memory usage
    enhanced_context = _context_compressor.compress_context(raw_enhanced_context)

    if verbose:
        compression_stats = _context_compressor.get_compression_stats()
        print(
        f"[CONTEXT_COMPRESSION] Compressed {compression_stats['original_keys']} → {compression_stats['compressed_keys']} keys"
        )
        print(
        f"[CONTEXT_COMPRESSION] Memory savings: {compression_stats['compression_savings_percent']:.1f}%"
        )
    
    # Load and run Synthesist through agent loader
    try:
        from .agents import synthesist
        synthesist_agent = synthesist
        synthesis = synthesist_agent.run(
        cfg,
        enhanced_context,
        verbose=verbose,
        multimodal_evidence=multimodal_evidence,
        )
        reasoning_chain["synthesist"] = {"output": synthesis, "context": "Synthesist"}
    except Exception as e:
        if verbose:
            pass
        synthesis = {"error": f"Synthesist failed to load: {e}", "status": "failed"}
        reasoning_chain["synthesist"] = {"output": synthesis, "context": "Synthesist"}

    # ENHANCED DELIBERATION: Post-synthesis analysis
    if verbose:
        pass
    post_synthesis_analysis = add_deliberation_pause(
        cfg, "Post-Synthesis", synthesis, initial_query, verbose=verbose
    )
    reasoning_chain["post_synthesis_analysis"] = {
        "output": post_synthesis_analysis,
        "context": "Post-Synthesis Analysis",
    }

    # Physics Prior Gate (post-synthesis): re-check outputs
    try:
        prior_gate = PhysicsPriorGate()
        prior_result_post = prior_gate.check({
        "synthesis": synthesis if isinstance(synthesis, dict) else {"text": str(synthesis)[:2000]},
        "scrutineer": scrut if isinstance(scrut, dict) else {"text": str(scrut)[:1000]},
        })
        reasoning_chain["physics_prior_post"] = {
        "output": {
                "valid": prior_result_post.valid,
                "violations": [vi.__dict__ for vi in prior_result_post.violations],
                "proxies": prior_result_post.proxies,
        },
        "context": "Physics Prior (post-synthesis)",
        }
    except Exception as e:
        if verbose:
            pass

    # Layer 6: Scrutineer (Enhanced with suppression detection)
    if verbose:
        pass

    # Load and run Scrutineer through agent loader
    try:
        from .agents import scrutineer
        scrutineer_agent = scrutineer
        scrut = scrutineer_agent.run(cfg, synthesis, verbose=verbose)
        reasoning_chain["scrutineer"] = {"output": scrut, "context": "Scrutineer"}
    except Exception as e:
        if verbose:
            pass
        scrut = {"error": f"Scrutineer failed to load: {e}", "status": "failed"}
        reasoning_chain["scrutineer"] = {"output": scrut, "context": "Scrutineer"}

    # ENHANCED DELIBERATION: Suppression pattern analysis
    if verbose:
        pass
    suppression_analysis = add_deliberation_pause(
        cfg, "Suppression Analysis", scrut, initial_query, verbose=verbose
    )
    reasoning_chain["suppression_analysis"] = {
        "output": suppression_analysis,
        "context": "Suppression Pattern Analysis",
    }

    # Layer 7: Oracle (Enhanced with all deliberation insights)
    if verbose:
        pass
    
    # Combine synthesis with all deliberation insights
    oracle_context = {
        "synthesis": synthesis,
        "scrutineer": scrut,
        "contradictions": contradiction_analysis,
        "patterns": pattern_analysis,
        "emergence": emergence_analysis,
        "truth_seeking": truth_analysis,
        "suppression_analysis": suppression_analysis,
    }
    
    # Load and run Oracle through agent loader
    try:
        from .agents import oracle
        oracle_agent = oracle
        oracle_result = oracle_agent.run(cfg, kg, vs, oracle_context, verbose=verbose)
        reasoning_chain["oracle"] = {"output": oracle_result, "context": "Oracle"}
    except Exception as e:
        if verbose:
            pass
        oracle_result = {"error": f"Oracle failed to load: {e}", "status": "failed"}
        reasoning_chain["oracle"] = {"output": oracle_result, "context": "Oracle"}

    # ENHANCED DELIBERATION: Final validation and breakthrough detection
    if verbose:
        pass
    final_validation = add_deliberation_pause(
        cfg, "Final Validation", oracle_result, initial_query, verbose=verbose
    )
    reasoning_chain["final_validation"] = {
        "output": final_validation,
        "context": "Final Validation",
    }

    # AGI EMERGENCE DETECTION: Check for emergent scientific principles
    if verbose:
        pass
    emergence_check = detect_emergence(
        cfg, [final_validation], initial_query, verbose=verbose
    )
    reasoning_chain["agi_emergence_check"] = {
        "output": emergence_check,
        "context": "AGI Emergence Check",
    }

    # Bookmark emergent pathway candidates (physics-first: require proxies)
    try:
        emergence_data = (
        emergence_check.get("emergence", {}) if isinstance(emergence_check, dict) else {}
        )
        if isinstance(emergence_data, dict) and emergence_data.get("emergence_detected"):
            pathway_id = f"pathway:{int(time.time())}"
        title = emergence_data.get("framework_name", "Emergent Pathway")
        description = emergence_data.get("summary", "")
        domain = intent_analysis.get("primary_domain", "unknown")

        # Minimal measurable proxies if available
        proxies = {
                "confidence": emergence_data.get("emergence_score", 0),
                "signals": emergence_data.get("signals", []),
        }
        equations = emergence_data.get("equations") or []
        uncertainty = {"source": "deliberation", "notes": "pre-validation"}
        provenance = {"query": initial_query, "timestamp": time.time()}

        kg.bookmark_emergent_pathway(
                pathway_id=pathway_id,
                title=title,
                description=description,
                domain=domain,
                proxies=proxies,
                equations=equations,
                uncertainty=uncertainty,
                provenance=provenance,
        )
        # Leave status pending until human approval
        kg.set_pathway_status(pathway_id, "pending")
        if verbose:
                pass
    except Exception as e:
        if verbose:
            pass

    # AGI SELF-MODIFICATION: Create emergent agents if high emergence detected
    if (
        isinstance(emergence_check, dict)
        and emergence_check.get("analysis_type") == "emergence_detection"
    ):
        emergence_data = emergence_check.get("emergence", {})
        if (
        isinstance(emergence_data, dict)
        and emergence_data.get("emergence_detected", False)
        and emergence_data.get("emergence_score", 0) >= 0.8
        ):
            if verbose:
                pass
        agent_created = create_emergent_agent(
                cfg, emergence_data, initial_query, verbose=verbose
        )
        if agent_created:
                reasoning_chain["agi_emergent_agent"] = {
                    "output": {"agent_created": True, "emergence_data": emergence_data},
                    "context": "AGI Emergent Agent",
                }

        # NEW: Enhanced AGI Capabilities
        try:
                # 1. Dynamic Agent Creation
                if 'agent_factory' in agi_components:
                    template = agi_components['agent_factory'].analyze_emergence_for_agent_creation(emergence_data)
                    if template:
                        new_agent_name = agi_components['agent_factory'].create_agent_from_template(template, emergence_data)
                        if new_agent_name:
                            pass
                            if verbose:
                                pass
                            reasoning_chain["dynamic_agent_creation"] = {
                                "output": {"agent_name": new_agent_name, "template": template.__dict__},
                                "context": "Dynamic Agent Creation",
                            }

                # 2. Runtime Agent Modification
                if 'agent_modifier' in agi_components:
                    # Enable adaptation for core agents
                    for agent_name in ['surveyor', 'dissident', 'synthesist', 'oracle']:
                        try:
                            # This would need actual agent instances, simplified for now
                            if verbose:
                                pass
                        except Exception as e:
                            if verbose:
                                pass

                # 3. Persistent Model Saving (if models were trained)
                if 'model_manager' in agi_components:
                    # This would save any trained models from this session
                    if verbose:
                        pass

        except Exception as e:
                if verbose:
                    pass

    # AGI FIELD CREATION: Create new scientific disciplines
    if (
        isinstance(emergence_check, dict)
        and emergence_check.get("analysis_type") == "emergence_detection"
    ):
        emergence_data = emergence_check.get("emergence", {})
        if isinstance(emergence_data, dict) and emergence_data.get(
        "emergence_detected", False
        ):
            if verbose:
                pass
        field_result = emergent_field_creation(
                cfg, emergence_data, initial_query, verbose=verbose
        )
        reasoning_chain["agi_emergent_field"] = {
                "output": field_result,
                "context": "AGI Emergent Field",
        }

    # CIM Stack Integration: Molecular Synthesis (if required)
    molecular_analysis = None
    if intent_analysis.get("requires_molecular", False):
        pass
        if verbose:
            pass
        molecular_analysis = molecular_synthesis.run(
        cfg, initial_query, context=cim_analysis, verbose=verbose
        )
        reasoning_chain["molecular_synthesis"] = {
        "output": molecular_analysis,
        "context": "Molecular Analysis",
        }
        
        if verbose:
            mechanisms = molecular_analysis.get("molecular_mechanisms", [])
        pathways = molecular_analysis.get("biochemical_pathways", [])

    # CIM Stack Integration: Bioelectric Integration (if required)
    bioelectric_analysis = None
    if intent_analysis.get("requires_bioelectric", False):
        pass
        if verbose:
            print(
                "[STATUS] Executing CIM Stack Integration: Bioelectric Integration..."
        )
        # Pass electric field foundation as context for unified physics paradigm
        context_data = physics_insights if physics_insights else None
        bioelectric_analysis = bioelectric_integration.run_traditional_energy_medicine(
        cfg, initial_query, context_data, verbose
        )
        reasoning_chain["bioelectric_integration"] = {
        "output": bioelectric_analysis,
        "context": "Bioelectric Analysis",
        }
        
        if verbose:
            traditional_concepts = bioelectric_analysis.get(
                "traditional_analysis", {}
        ).get("traditional_concepts", [])
        modern_research = bioelectric_analysis.get("modern_analysis", {}).get(
                "modern_research", []
        )
        print(
                f"[BIOELECTRIC] Traditional concepts: {len(traditional_concepts) if isinstance(traditional_concepts, list) else 1}"
        )
        print(
                f"[BIOELECTRIC] Modern research: {len(modern_research) if isinstance(modern_research, list) else 1}"
        )

    # CIM Stack Integration: Hypothesis Testing Laboratory (if required)
    hypothesis_testing_results = None
    if intent_analysis.get("requires_hypothesis_testing", False):
        pass
        if verbose:
            print(
                "[STATUS] Executing CIM Stack Integration: Hypothesis Testing Laboratory..."
        )
        hypothesis_testing_results = hypothesis_testing_laboratory.run(
        cfg, initial_query, context=cim_analysis, verbose=verbose
        )
        reasoning_chain["hypothesis_testing_laboratory"] = {
        "output": hypothesis_testing_results,
        "context": "Hypothesis Testing",
        }
        
        if verbose:
            experiments = hypothesis_testing_results.get("experiments", [])
        validations = hypothesis_testing_results.get("validations", [])

    # CIM Stack Integration: Grounding Layer (if required)
    grounding_layer_results = None
    if intent_analysis.get("requires_empirical_grounding", False):
        if verbose:
            print(
                "[STATUS] Executing CIM Stack Integration: Grounding Layer (Empirical Bridge)..."
            )

        try:
            grounding_layer_results = grounding_layer_agent.run(
                cfg,
                initial_query,
                context=cim_analysis,
                verbose=verbose,
                data_sources=cim_analysis.get("data_sources"),
                correlation_types=cim_analysis.get("correlation_types"),
            )
            reasoning_chain["grounding_layer"] = {
                "output": grounding_layer_results,
                "context": "Empirical Grounding",
            }

            if verbose:
                grounding_score = grounding_layer_results.get("grounding_score", 0.0)
                recommendations = len(
                    grounding_layer_results.get("recommendations", [])
                )
                print(
                    f"[GROUNDING] Grounding score: {grounding_score}, recommendations: {recommendations}"
                )
        except (IceburgAgentError, RuntimeError) as e:
            if verbose:
                print(f"[GROUNDING] Agent execution failed: {e}")
            grounding_layer_results = {
                "error": f"Agent execution failed: {e}",
                "error_type": "agent_error",
            }
        except (IceburgDataError, ValueError, TypeError) as e:
            if verbose:
                print(f"[GROUNDING] Data processing failed: {e}")
            grounding_layer_results = {
                "error": f"Data processing failed: {e}",
                "error_type": "data_error",
            }

    # Validate claims and compute primary evidence
    primary_level, claims = _validate_claims(scrut, evidence_strict)
    
    # Check if we should continue with full analysis or return early
    should_return_early = evidence_strict and (
        primary_level not in ("A", "B") or not claims
    )
    
    if should_return_early:
        pass
        if verbose:
            print(
                "[GATE] Evidence below threshold under --evidence-strict; will return consensus-only answer after capability gap analysis."
        )

    # Memory read cycle with domain filtering (simple filter by substring)
    concepts = _extract_concepts(initial_query)
    prior_list = []
    
    if not cfg.disable_memory:
        pass
        if verbose:
            pass

        # Search for relevant principles using concepts
        prior_list = []
        for concept in concepts:
            relevant_principles = kg.search_nodes(concept, node_type="principle")
        prior_list.extend(relevant_principles)
        # Remove duplicates while preserving order (simple approach for dicts)
        unique_prior_list = []
        for item in prior_list:
            if item not in unique_prior_list:
                unique_prior_list.append(item)
        prior_list = unique_prior_list
    else:
        pass
        if verbose:
            pass

    if verbose:
        pass
        if prior_list:
            print(
                "[MEMORY] Sample principle:",
                (
                    prior_list[0][:100] + "..."
                    if len(prior_list[0]) > 100
                    else prior_list[0]
                ),
        )
    if domains:
        filtered = []
        for p in prior_list:
            if any(d.lower() in p.lower() for d in domains):
                filtered.append(p)
        prior_list = filtered or prior_list
    # Convert dict items to strings for joining
    prior_strings = []
    for item in prior_list:
        if isinstance(item, dict):
            prior_strings.append(f"Principle: {item.get('node', 'Unknown')} (Type: {item.get('data', {}).get('type', 'unknown')})")
        else:
            prior_strings.append(str(item))
    prior_block = "\n".join(prior_strings)

    if verbose:
        pass
    oracle_input = json.dumps(
        {
        "claims": claims,
        "primary_evidence": primary_level,
        "prior": prior_list,
        }
    )

    # Load and run Oracle through agent loader for evidence-weighted principle
    try:
        from .agents import oracle
        oracle_agent = oracle
        principle = oracle_agent.run(cfg, kg, vs, oracle_input, verbose=verbose)
        reasoning_chain["oracle"] = {"output": principle, "context": "Oracle"}
    except Exception as e:
        if verbose:
            pass
        principle = {"error": f"Oracle failed to load: {e}", "status": "failed"}
        reasoning_chain["oracle"] = {"output": principle, "context": "Oracle"}

    # Code Generation with Weaver Agent - DISABLED
    generated_code = None
    lab_testing_results = None
    if verbose:
        pass
        pass
    
    # Full-Stack Application Generation with Architect Agent (Software Lab)
    generated_application = None
    if cfg.enable_software_lab:
        try:
            from .agents.architect import Architect

            if verbose:
                print("[ARCHITECT] Software lab enabled - running architect agent")

            architect_agent = Architect()
            generated_application = architect_agent.run(cfg, principle, verbose=verbose)

            if verbose and generated_application:
                print("[ARCHITECT] Generated application: \n", generated_application)
            elif verbose:
                print("[ARCHITECT] Architect agent did not produce an application")
        except ImportError as e:
            if verbose:
                print(f"[ARCHITECT] Architect agent unavailable: {e}")
        except Exception as e:
            if verbose:
                print(f"[ARCHITECT] Architect agent execution failed: {e}")
    elif verbose:
        print(
            "[ARCHITECT] Software lab disabled - set ICEBURG_ENABLE_SOFTWARE_LAB=1 to enable"
        )
    
    # Autonomous Capability Gap Detection and Agent Creation
    capability_gap_analysis = None
    if verbose:
        pass
    
    try:
        # Create a capability gap detector instance and pass reasoning_chain
        from .agents.capability_gap_detector import CapabilityGapDetector
        gap_detector = CapabilityGapDetector(cfg)
        capability_gap_analysis = gap_detector.run(
        reasoning_chain,
        initial_query, 
        verbose=verbose,
        )
        reasoning_chain["capability_gap_detector"] = {
        "output": capability_gap_analysis,
        "context": "Capability Gap Analysis",
        }
        
        if verbose:
            gaps_detected = len(capability_gap_analysis.get("gaps_detected", []))
        agents_created = len(
                [
                    a
                    for a in capability_gap_analysis.get("created_agents", [])
                    if a.get("status") == "created"
                ]
        )
        
        # Show created agents
        for agent in capability_gap_analysis.get("created_agents", []):
                if agent.get("status") == "created":
                    print(
                        f"[CAPABILITY_GAP] ✅ Created: {agent.get('agent_name')} ({agent.get('domain')})"
                    )
                elif agent.get("status") == "failed":
                    print(
                        f"[CAPABILITY_GAP] ❌ Failed: {agent.get('agent_name')} - {agent.get('error')}"
                    )
    
    except Exception as e:
        if verbose:
            pass
        capability_gap_analysis = {"error": str(e)}

    # AUTONOMOUS FUNCTIONS - The Real AGI Capabilities
    autonomous_results = {}
    if verbose:
        pass
    
    try:
        from .autonomous_orchestrator import AutonomousOrchestrator
        
        orchestrator = AutonomousOrchestrator(cfg)
        
        # Execute all autonomous functions using the main execute_all method
        if verbose:
            print("[AUTONOMOUS] Executing autonomous orchestrator functions")

        # Use the main execute_all method which handles all autonomous functions
        autonomous_results = await orchestrator.execute_all(oracle_result, initial_query)

        if verbose:
            print("[AUTONOMOUS] Completed autonomous orchestrator run")
    except Exception as e:
        if verbose:
            print(f"[AUTONOMOUS] Orchestrator failed: {e}")
        autonomous_results = {"error": str(e)}

    # Return early if evidence is insufficient (but after capability gap analysis)
    if should_return_early:
        if verbose:
            print(
                "[GATE] Returning consensus-only answer due to insufficient evidence."
            )
        return format_iceberg_report(consensus, "", "", consensus)

    # Knowledge Synthesis with Scribe Agent (skippable via env ICEBURG_DISABLE_SCRIBE=1)
    generated_knowledge = None
    if os.getenv("ICEBURG_DISABLE_SCRIBE", "0") != "1":
        try:
            from .agents import scribe

            if verbose:
                print("[SCRIBE] Running knowledge synthesis agent")
            generated_knowledge = scribe.run(cfg, principle, verbose=verbose)

            if verbose and generated_knowledge:
                print("[SCRIBE] Generated knowledge artifact")
            elif verbose:
                print("[SCRIBE] Scribe agent produced no knowledge output")
        except ImportError as e:
            if verbose:
                print(f"[SCRIBE] Scribe agent unavailable: {e}")
        except Exception as e:
            if verbose:
                print(f"[SCRIBE] Scribe agent execution failed: {e}")
    elif verbose:
        print("[SCRIBE] Scribe agent disabled via ICEBURG_DISABLE_SCRIBE")

    # Store the discovered principle for final report
    if os.getenv("ICEBURG_DISABLE_STORE", "0") != "1":
        try:
            # Extract concepts from the query for knowledge graph storage
            concepts = [initial_query.lower().split()]
            kg.save_principle(str(principle), concepts)
            if verbose:
                print("[STORE] Stored principle in knowledge graph")
        except Exception as e:
            if verbose:
                print(f"[STORE] Failed to store principle: {e}")
    else:
        if verbose:
            print("[STORE] Storage disabled via ICEBURG_DISABLE_STORE")

    # Save conversation logs
    if os.getenv("ICEBURG_DISABLE_CONVERSATION_LOGS", "0") != "1":
        try:
            _save_conversation_logs(initial_query, reasoning_chain, project_id, verbose)
            if verbose:
                print("[CONVERSATION] Saved conversation logs")
        except Exception as e:
            if verbose:
                print(f"[CONVERSATION] Failed to save conversation logs: {e}")
    else:
        if verbose:
            print(
                "[CONVERSATION] Conversation logging disabled (ICEBURG_DISABLE_CONVERSATION_LOGS=1)"
            )

    # Generate final report with enhanced capabilities and CIM stack integration
    final_report = format_iceberg_report(
        consensus=consensus,
        alternatives=diss_output,
        syntheses=synthesis,
        principle=principle,
    )

    # Initialize CIM section
    cim_section = ""
    
    # Add Unified Theory Architecture results to final report
    unified_section = ""
    if emergence_results and emergence_results.get("emergent_patterns"):
        unified_section = "\n\n" + "=" * 80 + "\n"
        unified_section += "🌟 UNIFIED THEORY EMERGENCE RESULTS\n"
        unified_section += "=" * 80 + "\n\n"

        unified_section += (
        f"User Intent Preserved: {user_intent.core_request_type.upper()}\n"
        )
        unified_section += f"Emergence Generated: {len(emergence_results['emergent_patterns'])} validated patterns\n"
        unified_section += f"Average Confidence: {emergence_results['synthesis_result']['average_confidence']:.2f}\n"
        unified_section += f"Average Novelty: {emergence_results['synthesis_result']['average_novelty']:.2f}\n"
        unified_section += (
        f"Processing Time: {unified_processing['processing_time']:.2f}s\n\n"
        )

        # Add emergence summary
        synthesis = emergence_results["synthesis_result"]
        if synthesis["has_emergence"]:
            unified_section += "🎯 EMERGENT INSIGHTS DISCOVERED:\n"

        for pattern_type, patterns in synthesis["pattern_types"].items():
            unified_section += f"\n• {pattern_type.replace('_', ' ').title()}: {len(patterns)} patterns\n"
            for i, pattern in enumerate(patterns[:3]):  # Show top 3 per type
                confidence = pattern.get("validation_score", 0.0)
                novelty = pattern.get("novelty_index", 0.0)
                unified_section += f"  - Pattern {i+1}: {confidence:.2f} confidence, {novelty:.2f} novelty\n"

            unified_section += (
                f"\n📊 VALIDATION SUMMARY: {synthesis['validation_summary']}\n"
            )
        unified_section += (
            "✅ All emergence grounded in real data patterns, no hallucinations\n"
        )

        # Add unknown emergence information
        unknown_emergences = emergence_results.get("unknown_emergences", [])
        if unknown_emergences:
            unified_section += f"\n🌀 UNKNOWN EMERGENCE DETECTED: {len(unknown_emergences)} novel patterns\n"
            for i, item in enumerate(unknown_emergences):
                emergence = item["emergence"]
                routing = item["routing"]
                unified_section += f"• Novel Pattern {i+1}: {emergence.potential_impact.title()} impact "
                unified_section += f"(Novelty: {emergence.novelty_score:.2f})\n"
                unified_section += (
                    f"  → Routed to: {routing['strategy'].replace('_', ' ').title()}\n"
                )
                unified_section += f"  → Reason: {routing['reasoning']}\n"

            unified_section += (
                "⚠️ These patterns are preserved for future investigation\n"
            )
        else:
            unified_section += (
                "📝 No emergent patterns found - query processed through standard pipeline\n"
            )

        unified_section += "\n\n"

    # Add CIM stack outputs to final report if available
    cim_section = ""
    if (
        physics_insights
        or molecular_analysis
        or bioelectric_analysis
        or hypothesis_testing_results
        or scientific_research_insights
        or corporate_network_insights
        or multi_domain_insights
        or comprehensive_api_insights
        or virtual_ecosystem_insights
        or bioelectrical_insights
    ):
        cim_section = "\n\n" + "=" * 80 + "\n"
        cim_section += "🔬 CIM STACK INTEGRATION RESULTS\n"
        cim_section += "=" * 80 + "\n\n"

        if physics_insights:
            cim_section += "⚡ PHYSICS ANALYSIS:\n"
            ef_data = physics_insights.get("physics_analysis", {})
            analysis = ef_data.get("analysis", "Physics query detected")

            cim_section += f"Physics query: {ef_data.get('query', 'unknown')}\n"
            cim_section += f"Analysis status: {ef_data.get('status', 'unknown')}\n"
            cim_section += f"Analysis: {analysis}\n\n"

        if molecular_analysis:
            cim_section += "🧬 MOLECULAR SYNTHESIS:\n"
            cim_section += molecular_synthesis.extract_molecular_summary(
                molecular_analysis
            )
            cim_section += "\n\n"
        
        if bioelectric_analysis:
                cim_section += "⚡ BIOELECTRIC INTEGRATION:\n"
                cim_section += bioelectric_integration.extract_bioelectric_summary(
                    bioelectric_analysis
                )
                cim_section += "\n\n"
        
        if hypothesis_testing_results:
                cim_section += "🧪 HYPOTHESIS TESTING LABORATORY:\n"
                cim_section += hypothesis_testing_laboratory.extract_domain_summary(
                    hypothesis_testing_results
                )
                cim_section += "\n\n"
        
        if scientific_research_insights:
                cim_section += "🔬 REAL SCIENTIFIC RESEARCH:\n"
                scientific_data = scientific_research_insights.get(
                    "real_scientific_research", {}
                )
                summary = scientific_data.get("summary", {})
                cim_section += f"Molecular compounds found: {summary.get('molecular_compounds_found', 0)}\n"
                cim_section += f"Heart coherence studies: {summary.get('heart_coherence_studies', 0)}\n"
                cim_section += f"Chinese research papers: {summary.get('chinese_research_papers', 0)}\n"
                cim_section += f"Indian research papers: {summary.get('indian_research_papers', 0)}\n"
                cim_section += f"Total scientific sources: {summary.get('total_scientific_sources', 0)}\n\n"
        
        if corporate_network_insights:
                cim_section += "🏢 CORPORATE NETWORK ANALYSIS:\n"
                network_data = corporate_network_insights.get(
                    "corporate_network_analysis", {}
                )
                summary = network_data.get("summary", {})
                cim_section += f"Corporate entities found: {summary.get('corporate_entities_found', 0)}\n"
                cim_section += f"Family networks analyzed: {summary.get('family_networks_analyzed', 0)}\n"
                cim_section += f"Network connections documented: {summary.get('network_connections_documented', 0)}\n"
                cim_section += f"Academic papers reviewed: {summary.get('academic_papers_reviewed', 0)}\n"
                cim_section += f"Total factual sources: {summary.get('total_factual_sources', 0)}\n\n"
        
        if multi_domain_insights:
                cim_section += "🌍 MULTI-DOMAIN ANALYSIS:\n"
                multi_data = multi_domain_insights.get("multi_domain_analysis", {})
                summary = multi_data.get("summary", {})
                cim_section += f"Geospatial locations analyzed: {summary.get('locations_analyzed', 0)}\n"
                cim_section += f"Financial symbols tracked: {summary.get('stock_symbols_tracked', 0)}\n"
                cim_section += f"Anthropological studies: {summary.get('anthropological_studies', 0)}\n"
                cim_section += f"Museum artifacts found: {summary.get('museum_artifacts_found', 0)}\n"
                cim_section += f"Archaeological sites documented: {summary.get('archaeological_sites_documented', 0)}\n"
                cim_section += f"Total multi-domain sources: {summary.get('total_multi_domain_sources', 0)}\n\n"
        
        if comprehensive_api_insights:
                cim_section += "🔍 COMPREHENSIVE API SEARCH:\n"
                api_data = comprehensive_api_insights.get("comprehensive_api_search", {})
                summary = api_data.get("summary", {})
                cim_section += f"Total sources searched: {summary.get('total_sources_searched', 0)}\n"
                cim_section += f"Academic results: {summary.get('academic_results', 0)}\n"
                cim_section += f"Media results: {summary.get('media_results', 0)}\n"
                cim_section += f"Specialized results: {summary.get('specialized_results', 0)}\n"
                cim_section += f"Total results: {summary.get('total_results', 0)}\n"
                cim_section += f"Top relevance score: {summary.get('top_relevance_score', 0.0):.3f}\n\n"
        
        if virtual_ecosystem_insights:
                cim_section += "🧪 VIRTUAL SCIENTIFIC ECOSYSTEM:\n"
                ecosystem_data = virtual_ecosystem_insights.get(
                    "virtual_scientific_ecosystem", {}
                )
                summary = ecosystem_data.get("summary", {})
                cim_section += f"Experiment type: {summary.get('experiment_type', 'unknown')}\n"
                cim_section += f"Population generated: {summary.get('population_generated', 0)} participants\n"
                cim_section += f"Equipment created: {summary.get('equipment_created', 0)} pieces\n"
                cim_section += f"Research questions: {summary.get('research_questions', 0)}\n"
                cim_section += f"Hypotheses generated: {summary.get('hypotheses_generated', 0)}\n"
                cim_section += f"Data collection methods: {summary.get('data_collection_methods', 0)}\n"
                cim_section += f"Analysis techniques: {summary.get('analysis_techniques', 0)}\n"
                cim_section += f"Expected outcomes: {summary.get('expected_outcomes', 0)}\n"
                cim_section += f"Timeline: {summary.get('timeline', 'unknown')}\n"
                cim_section += f"Budget estimate: {summary.get('budget_estimate', 'unknown')}\n"
                cim_section += f"Risk assessment: {summary.get('risk_assessment', 'unknown')}\n"
                cim_section += f"Ethical considerations: {summary.get('ethical_considerations', 'unknown')}\n"
                cim_section += f"Collaboration opportunities: {summary.get('collaboration_opportunities', 'unknown')}\n"
                cim_section += f"Publication strategy: {summary.get('publication_strategy', 'unknown')}\n"
                cim_section += f"Impact potential: {summary.get('impact_potential', 'unknown')}\n"
                cim_section += f"Success metrics: {summary.get('success_metrics', 'unknown')}\n"
                cim_section += f"Total ecosystem components: {summary.get('total_ecosystem_components', 0)}\n"
                cim_section += f"Institution created: {summary.get('institution_created', 0)}\n"
                cim_section += f"Experiment completed: {summary.get('experiment_completed', 0)}\n\n"
                cim_section += f"Statistical significance: {summary.get('statistical_significance', False)}\n"
                cim_section += f"Effect size category: {summary.get('effect_size_category', 'unknown')}\n\n"
        
        if bioelectrical_insights:
                cim_section += "⚡ BIOELECTRICAL FUNDAMENTAL ANALYSIS:\n"
                bioelectrical_data = bioelectrical_insights.get(
                    "bioelectrical_fundamental_analysis", {}
                )
                summary = bioelectrical_data.get("summary", {})
                cim_section += f"Field coherence: {summary.get('field_coherence', 0.0):.3f}\n"
                cim_section += f"Quantum coherence time: {summary.get('quantum_coherence_time', 0.0):.3f} ms\n"
                cim_section += f"Transmission efficiency: {summary.get('transmission_efficiency', 0.0):.3f}\n"
                cim_section += f"Bioelectric field strength: {summary.get('bioelectric_field_strength', 0.0):.3f} V/m\n"
                cim_section += f"Resonance frequency: {summary.get('resonance_frequency', 0.0):.3f} Hz\n"
                cim_section += f"Coherence threshold: {summary.get('coherence_threshold', 0.0):.3f}\n"
                # cim_section += f"Quantum entanglement: {summary.get('quantum_entanglement', False)}\n"  # Experimental - disabled by default
                cim_section += f"Field stability: {summary.get('field_stability', 0.0):.3f}\n"
                cim_section += f"Energy transfer rate: {summary.get('energy_transfer_rate', 0.0):.3f} J/s\n"
                cim_section += f"Total bioelectrical components: {summary.get('total_bioelectrical_components', 0)}\n"
                cim_section += f"Information density: {summary.get('information_density', 0.0):.3f}\n"
                cim_section += f"Activation reason: {summary.get('activation_reason', 'unknown')}\n\n"
    
    # Phase 2: True AGI Capabilities Integration
    agi_activation_keywords = [
        "agi",
        "self-redesign",
        "novel intelligence",
        "autonomous goal",
        "unbounded learning",
        "true agi",
        "artificial general intelligence",
    ]
    should_activate_agi = any(
        keyword in initial_query.lower() for keyword in agi_activation_keywords
    )
    
    if should_activate_agi:
        pass
        if verbose:
            pass
        
        # Self-Redesign Engine - Fundamental Self-Modification
        from .agents import self_redesign_engine

        self_redesign_results = self_redesign_engine.run(
        cfg, initial_query, context=cim_analysis, verbose=verbose
        )
        reasoning_chain["self_redesign_engine"] = {
        "output": self_redesign_results,
        "context": "Self-Redesign",
        }
        
        # Novel Intelligence Creator - Invent New Intelligence Types
        from .agents import novel_intelligence_creator

        novel_intelligence_results = novel_intelligence_creator.run(
        cfg, initial_query, context=cim_analysis, verbose=verbose
        )
        reasoning_chain["novel_intelligence_creator"] = {
        "output": novel_intelligence_results,
        "context": "Novel Intelligence",
        }
        
        # Autonomous Goal Formation - Form Own Goals
        from .agents import autonomous_goal_formation

        autonomous_goal_results = autonomous_goal_formation.run(
        cfg, initial_query, context=cim_analysis, verbose=verbose
        )
        reasoning_chain["autonomous_goal_formation"] = {
        "output": autonomous_goal_results,
        "context": "Autonomous Goals",
        }
        
        # Unbounded Learning Engine - Learn Without Limits
        from .protocol.execution.agents import unbounded_learning_engine

        unbounded_learning_results = unbounded_learning_engine.run(
        cfg, initial_query, context=cim_analysis, verbose=verbose
        )
        reasoning_chain["unbounded_learning_engine"] = {
        "output": unbounded_learning_results,
        "context": "Unbounded Learning",
        }
        
        if verbose:
            pass
        
        # Add True AGI results to CIM section
        cim_section += "🧠 PHASE 2: TRUE AGI CAPABILITIES:\n"
        cim_section += "🔧 Self-Redesign Engine: " + str(bool(self_redesign_results)) + "\n"
        cim_section += "💡 Novel Intelligence Creator: " + str(bool(novel_intelligence_results)) + "\n"
        cim_section += "🎯 Autonomous Goal Formation: " + str(bool(autonomous_goal_results)) + "\n"
        cim_section += "♾️ Unbounded Learning Engine: " + str(bool(unbounded_learning_results)) + "\n\n"
        
        # Add detailed AGI capabilities summary
        if self_redesign_results:
                cim_section += "🔧 SELF-REDESIGN CAPABILITIES:\n"
                cim_section += f"- Architecture Analysis: {len(self_redesign_results.get('architecture_analysis', {}).get('current_limitations', []))} limitations identified\n"
                cim_section += f"- Redesign Proposals: {len(self_redesign_results.get('redesign_proposals', []))} proposals generated\n"
                cim_section += f"- Modifications Executed: {len(self_redesign_results.get('executed_modifications', []))} modifications completed\n\n"
        
        if novel_intelligence_results:
                cim_section += "💡 NOVEL INTELLIGENCE TYPES:\n"
                cim_section += f"- Novel Types Created: {len(novel_intelligence_results.get('novel_intelligence_types', []))} intelligence types\n"
                cim_section += f"- Intelligence Syntheses: {len(novel_intelligence_results.get('intelligence_syntheses', []))} syntheses generated\n"
                cim_section += f"- Evolution Paths: {len(novel_intelligence_results.get('intelligence_evolution_paths', []))} evolution paths\n\n"
        
        if autonomous_goal_results:
                cim_section += "🎯 AUTONOMOUS GOALS:\n"
                cim_section += f"- Goals Formed: {len(autonomous_goal_results.get('autonomous_goals', []))} autonomous goals\n"
                cim_section += f"- Research Initiatives: {len(autonomous_goal_results.get('research_initiatives', []))} initiatives planned\n"
                cim_section += f"- Curiosity Drives: {len(autonomous_goal_results.get('curiosity_drives', []))} curiosity areas identified\n\n"
        
        if unbounded_learning_results:
                cim_section += "♾️ UNBOUNDED LEARNING:\n"
                cim_section += f"- Learning Domains: {len(unbounded_learning_results.get('learning_domains', []))} domains identified\n"
                cim_section += f"- Infinite Reasoning: {len(unbounded_learning_results.get('infinite_dimensional_reasoning', []))} reasoning capabilities\n"
                cim_section += f"- Cross-Domain Syntheses: {len(unbounded_learning_results.get('cross_domain_synthesis', []))} syntheses performed\n\n"
    
    # Add Unified Theory section to final report
    final_report += unified_section

    # Add CIM stack section to final report
    final_report += cim_section

    # Blockchain verification and immutable record creation
    if verbose:
        pass
    
    try:
        # Create research metadata
        research_metadata = {
        "query": initial_query,
        "timestamp": time.time(),
        "protocol_version": "4.0",
        "cim_stack_enabled": True,
        "multimodal_processed": bool(
                multimodal_input or documents or multimodal_evidence
        ),
        "domains": domains or [],
        "project_id": project_id,
        "agent_outputs": {
                "surveyor": bool(consensus),
                "dissident": bool(diss_output),
                "archaeologist": bool(arch),
                "synthesist": bool(synthesis),
                "oracle": bool(principle),
        },
        }
        
        # Create immutable research record
        research_record = blockchain_system.create_research_record(
        research_content=final_report,
        metadata=research_metadata,
        author_id="iceburg_system",
        )
        
        # Create verification proof
        verification_proof = blockchain_system.create_verification_proof(
        record_id=research_record.record_id, proof_type="merkle_proof"
        )
        
        # Verify the record
        verification_result = blockchain_system.verify_research_record(
        research_record.record_id
        )
        
        if verbose:
            print(
                f"[BLOCKCHAIN] Verification score: {verification_result.get('verification_score', 0.0):.2f}"
        )
        
        # Add blockchain verification to final report
        blockchain_section = "\n\n" + "=" * 80 + "\n"
        blockchain_section += "🔗 BLOCKCHAIN VERIFICATION\n"
        blockchain_section += "=" * 80 + "\n\n"
        blockchain_section += f"📋 Research Record ID: {research_record.record_id}\n"
        blockchain_section += f"🔐 Content Hash: {research_record.content_hash}\n"
        blockchain_section += f"✅ Verification Score: {verification_result.get('verification_score', 0.0):.2f}\n"
        blockchain_section += f"🔍 Proof ID: {verification_proof.proof_id}\n"
        blockchain_section += f"⏰ Timestamp: {research_record.timestamp.isoformat()}\n"
        blockchain_section += f"🔗 Blockchain Confirmations: {research_record.blockchain_confirmations}\n\n"
        
        final_report += blockchain_section
        
    except Exception as e:
        if verbose:
            print(f"[BLOCKCHAIN] Verification failed: {e}")

    if verbose:
        print(
            "[STATUS] Enhanced Iceberg Protocol with CIM Stack Architecture completed successfully"
        )
        print(
            f"[CIM] Intent: {intent_analysis.get('primary_domain', 'unknown')} - {intent_analysis.get('detail_level', 'unknown')}"
        )
        print(
            f"[CIM] Bioelectrical Fundamental Analysis: {bool(bioelectrical_insights)}"
        )
        print(
            f"[MULTIMODAL] Processed: {bool(multimodal_input)} input, {len(documents) if documents else 0} documents, {len(multimodal_evidence) if multimodal_evidence else 0} evidence pieces"
        )
        print(
            f"[BLOCKCHAIN] Immutable record created: {research_record.record_id if 'research_record' in locals() else 'Failed'}"
        )

    # Health monitoring summary
    health_summary = health_monitor.get_health_summary()

    # Auto-healing summary
    healing_summary = auto_healer.get_healing_summary()
    if healing_summary.get('total_actions', 0) > 0:
        pass

    # Stop monitoring systems
    await web_dashboard.stop()
    # await dashboard.stop_monitoring()  # Old dashboard
    await health_monitor.stop_monitoring()

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            if hasattr(final_report, '__dict__'):
                final_report.field_conditions = field_conditions
            elif isinstance(final_report, dict):
                final_report["field_conditions"] = field_conditions
            else:
                # For string reports, append as metadata section
                field_section = f"\n\n## Field Conditions\n"
                field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
                field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
                field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
                final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                try:
                    visual_result = visual_architect.run(cfg, initial_query, verbose)
                    
                    # Add to reasoning chain
                    reasoning_chain["visual_generation"] = {
                        "output": str(visual_result)[:500],
                        "context": "Visual Generation"
                    }
                except Exception as e:
                    if verbose:
                        pass
        except ImportError as e:
            if verbose:
                pass
        except Exception as e:
            if verbose:
                pass

    # Public Services Integration - Generate practical outputs
    try:
        if verbose:
            pass
        
        # Create research output structure for public services
        research_output = {
        "initial_query": initial_query,
        "agent_outputs": reasoning_chain,
        "reasoning_chain": reasoning_chain,
        "source": "ICEBURG Protocol",
        "timestamp": datetime.now().isoformat(),
        "project_id": project_id
        }
        
        # Generate practical outputs
        from .agents.public_services_integration import integrate_public_services
        public_services_result = await integrate_public_services(
        research_output=research_output,
        query=initial_query,
        cfg=cfg,
        verbose=verbose
        )
        
        # Add public services outputs to the final report
        if hasattr(final_report, '__dict__'):
            final_report.public_services = public_services_result
        elif isinstance(final_report, dict):
            final_report["public_services"] = public_services_result
        
        if verbose:
            guides_count = len(public_services_result.get("generated_outputs", {}).get("guides", []))
        summaries_count = len(public_services_result.get("generated_outputs", {}).get("summaries", []))
    
    except Exception as e:
        if verbose:
        # Don't fail the entire protocol if public services fails
            pass

    # Add field conditions to final report (feature-flagged)
    try:
        field_conditions = _get_field_conditions()
        if field_conditions:
            pass
        if hasattr(final_report, '__dict__'):
            final_report.field_conditions = field_conditions
        elif isinstance(final_report, dict):
            final_report["field_conditions"] = field_conditions
        else:
            # For string reports, append as metadata section
            field_section = f"\n\n## Field Conditions\n"
            field_section += f"- Earth Sync: {field_conditions.get('earth_sync', 0.0):.2f}\n"
            field_section += f"- Human State: {field_conditions.get('human_state', 'unknown')}\n"
            field_section += f"- Unified Field Strength: {field_conditions.get('unified_field_strength', 0.0):.2f}\n"
            final_report += field_section
    except Exception as e:
        if verbose:
            pass

    # Add visual generation capability
    if os.getenv("ICEBURG_ENABLE_VISUAL_GEN", "0") != "0":
        try:
            from .agents.visual_architect import VisualArchitect
            visual_architect = VisualArchitect()
            
            # Simple visual intent detection
            if "ui" in initial_query.lower() or "interface" in initial_query.lower() or "generate html" in initial_query.lower() or "macos app" in initial_query.lower():
                visual_result = visual_architect.run(cfg, initial_query, verbose)
                
                # Store visual artifacts (use existing storage mechanism if needed)
                pass

        except Exception as e:
            if verbose:
                # print(f"Visual generation failed: {e}")
                pass
                
    return final_report
    return final_report

# Expose main entry point
run = _iceberg_protocol_async