import hashlib
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Dict, List, Union

from .config import load_config
from .providers.factory import provider_factory


def _generate_fallback_response(prompt: str, messages: list) -> str:
    """Generate a basic fallback response when LLM provider is unavailable"""
    # Extract the user's query from the messages
    user_query = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break
    
    # Provide a helpful response indicating the system is in fallback mode
    provider = os.getenv("ICEBURG_LLM_PROVIDER", "ollama").upper()
    if "research" in user_query.lower():
        return f"ICEBURG is currently running in fallback mode. For research queries, please ensure {provider} is properly configured. The system can still process structured data and perform basic analysis."
    elif "analysis" in user_query.lower():
        return f"ICEBURG is in fallback mode. Analysis capabilities are limited without the full LLM backend. Please configure {provider} for complete functionality."
    else:
        return f"ICEBURG fallback mode: Unable to process '{user_query[:50]}...' without {provider}. Please configure the LLM provider for full AI capabilities."


# LLM Response Caching System
class LLMCache:
    """In-memory cache for LLM responses with TTL support"""

    def __init__(
        self, max_size: int = 1000, default_ttl: int = 3600
    ):  # 1 hour default TTL
        self.cache: dict[str, dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = Lock()
        # Track cache hits and misses
        self.hits = 0
        self.misses = 0

    def _generate_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a unique cache key for the request"""
        # Create a deterministic string from the request parameters
        key_data = {"model": model, "messages": messages, "options": options or {}}
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Get cached response if it exists and hasn't expired"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if datetime.now() < entry["expires_at"]:
                    self.hits += 1
                    return entry["response"]
                else:
                    # Remove expired entry
                    del self.cache[key]
                    self.misses += 1
            else:
                self.misses += 1
            return None

    def set(self, key: str, response: str, ttl: Optional[int] = None) -> None:
        """Cache a response with TTL"""
        with self.lock:
            # Remove oldest entries if cache is full
            if len(self.cache) >= self.max_size:
                # Remove expired entries first
                current_time = datetime.now()
                expired_keys = [
                    k for k, v in self.cache.items() if current_time >= v["expires_at"]
                ]
                for k in expired_keys:
                    del self.cache[k]

                # If still full, remove oldest entry
                if len(self.cache) >= self.max_size:
                    oldest_key = min(
                        self.cache.keys(), key=lambda k: self.cache[k]["created_at"]
                    )
                    del self.cache[oldest_key]

            # Add new entry
            ttl_seconds = ttl or self.default_ttl
            self.cache[key] = {
                "response": response,
                "created_at": datetime.now(),
                "expires_at": datetime.now() + timedelta(seconds=ttl_seconds),
            }

    def clear(self) -> None:
        """Clear all cached responses"""
        with self.lock:
            self.cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            current_time = datetime.now()
            valid_entries = sum(
                1 for v in self.cache.values() if current_time < v["expires_at"]
            )
            expired_entries = len(self.cache) - valid_entries
            
            # Calculate hit rate
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0

            return {
                "total_entries": len(self.cache),
                "valid_entries": valid_entries,
                "expired_entries": expired_entries,
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total_requests,
                "hit_rate": hit_rate,
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
            }


# Global cache instance
_llm_cache = LLMCache()


def _redact_text(text: str, max_len: int = 512) -> str:
    try:
        # Basic PII redaction heuristics: emails, phone-like, api keys-like
        import re

        redacted = re.sub(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[REDACTED_EMAIL]", text
        )
        redacted = re.sub(r"\b\+?\d[\d\-\(\)\s]{7,}\d\b", "[REDACTED_PHONE]", redacted)
        redacted = re.sub(r"\b[A-Za-z0-9_\-]{24,}\b", "[REDACTED_TOKEN]", redacted)
        # Truncate to avoid log bloat
        if len(redacted) > max_len:
            redacted = redacted[:max_len] + "…"
        return redacted
    except Exception:
        return text[:max_len]


def _log_prompt_event(record: dict[str, Any]) -> None:
    try:
        # Opt-in logging only
        if os.getenv("ICEBURG_LOG_PROMPTS", "0") != "1":
            return
        data_dir = Path(os.getenv("ICEBURG_DATA_DIR", "./data")).expanduser().resolve()
        log_dir = data_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "prompts.jsonl"
        # Avoid orjson to minimize deps coupling here
        import json as _json

        # Redact sensitive fields
        rec = dict(record)
        try:
            if isinstance(rec.get("input"), dict):
                rec["input"] = {
                    "system": _redact_text(str(rec["input"].get("system", "")), 512),
                    "prompt": _redact_text(str(rec["input"].get("prompt", "")), 512),
                }
            if isinstance(rec.get("output"), dict):
                rec["output"] = {
                    "text": _redact_text(str(rec["output"].get("text", "")), 512),
                }
        except Exception:
            pass
        with log_file.open("a", encoding="utf-8") as f:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def chat_complete(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    temperature: float = 0.2,
    options: Optional[dict[str, Any]] = None,
    context_tag: Optional[str] = None,
    images: Optional[list[str]] = None,
) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    
    # Handle multimodal messages with images
    if images:
        content_parts = [{"type": "text", "text": prompt}]
        for image_path in images:
            content_parts.append({"type": "image_url", "image_url": {"url": f"file://{image_path}"}})
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": prompt})
    merged_opts: dict[str, Any] = {"temperature": temperature}
    if options:
        merged_opts.update(options)
        if "temperature" not in options:
            merged_opts["temperature"] = temperature

    # Global fast-mode clamps via env flag
    if os.getenv("ICEBURG_FORCE_FAST_LIMITS", "0") == "1":
        # Clamp context and prediction budget aggressively
        max_ctx = 1024
        max_pred = 64
        merged_opts["num_ctx"] = min(int(merged_opts.get("num_ctx", max_ctx)), max_ctx)
        merged_opts["num_predict"] = min(int(merged_opts.get("num_predict", max_pred)), max_pred)

    # Check cache first
    cache_key = _llm_cache._generate_cache_key(model, messages, merged_opts)
    cached_response = _llm_cache.get(cache_key)
    if cached_response:
        return cached_response

    ts_start = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    t0 = time.time()

    cfg = load_config()
    provider = provider_factory(cfg)
    
    # Use primary provider (default: Google/Gemini) - no fallback
    try:
        content = provider.chat_complete(
            model=model,
            prompt=prompt,
            system=system,
            temperature=merged_opts.get("temperature", temperature),
            options=merged_opts,
            images=images,
        )
    except Exception as e:
        # No fallback - raise the error directly
        primary_provider = getattr(cfg, "llm_provider", None) or os.getenv("ICEBURG_LLM_PROVIDER", "ollama")
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Primary provider ({primary_provider}) failed: {e}. No fallback available.")
        raise

    # Cache the response
    _llm_cache.set(cache_key, content)

    t1 = time.time()
    ts_end = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    duration_ms = int((t1 - t0) * 1000)
    
    # Record performance metric for bottleneck detection
    # Note: Performance tracking is handled by UnifiedPerformanceTracker
    # which is already integrated into the system

    # Log JSONL record
    _log_prompt_event(
        {
            "ts_start": ts_start,
            "ts_end": ts_end,
            "duration_ms": duration_ms,
            "model": model,
            "context": context_tag or "",
            "options": merged_opts,
            "input": {
                "system": (system or "")[:2000],
                "prompt": prompt[:4000],
            },
            "output": {
                "text": content[:8000],
            },
        }
    )
    
    # Log full conversation for fine-tuning (not truncated)
    try:
        from .data_collection.fine_tuning_logger import FineTuningLogger
        logger = FineTuningLogger()
        
        # Build full messages (not truncated)
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.append({"role": "user", "content": prompt})
        full_messages.append({"role": "assistant", "content": content})
        
        metadata = {
            "model": model,
            "context_tag": context_tag or "",
            "temperature": temperature,
            "options": merged_opts,
            "duration_ms": duration_ms,
            "cached": False
        }
        
        # Note: Quality score not available here, will be logged separately if available
        logger.log_conversation(full_messages, metadata, quality_score=None)
    except Exception as e:
        # Don't fail if fine-tuning logging fails
        pass

    return content


def embed_texts(model: str, texts: list[str]) -> list[list[float]]:
    """Embed texts with negative caching and fast-mode local fallback.

    Behavior:
    - If ICEBURG_DISABLE_REMOTE_EMBED_FAST=1, use local hash embeddings (no HTTP).
    - On provider errors (e.g., 404), cache the failure for a short TTL and fall back to local embeddings.
    """
    # Local deterministic embedding fallback
    def _local_hash_embed(batch: list[str], dim: int = 384) -> list[list[float]]:
        out: list[list[float]] = []
        for t in batch:
            vec = [0.0] * dim
            b = t.encode("utf-8") if t else b""
            for i, ch in enumerate(b):
                vec[i % dim] += float(ch) / 255.0
            # Normalize
            s = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / s for v in vec])
        return out

    # Session-scoped negative cache for failed models
    global _failed_embed_models
    try:
        _failed_embed_models
    except NameError:
        _failed_embed_models = {}  # type: ignore[var-annotated]

    now = time.time()
    # Honor fast-mode disable flag
    if os.getenv("ICEBURG_DISABLE_REMOTE_EMBED_FAST", "0") == "1":
        return _local_hash_embed(texts)

    # Respect recent failures (TTL = 600s)
    fail_rec = _failed_embed_models.get(model)
    if isinstance(fail_rec, dict) and now - fail_rec.get("ts", 0) < 600:
        return _local_hash_embed(texts)

    # Try provider; on error, record failure and fall back
    try:
        cfg = load_config()
        provider = provider_factory(cfg)
        return provider.embed_texts(model, texts)
    except Exception:
        _failed_embed_models[model] = {"ts": now}
        return _local_hash_embed(texts)


# Cache Management Functions
def clear_llm_cache() -> None:
    """Clear all cached LLM responses"""
    _llm_cache.clear()


def get_llm_cache_stats() -> dict[str, Any]:
    """Get LLM cache statistics"""
    return _llm_cache.stats()


def set_llm_cache_config(max_size: int = 1000, default_ttl: int = 3600) -> None:
    """Configure LLM cache settings"""
    global _llm_cache
    _llm_cache = LLMCache(max_size=max_size, default_ttl=default_ttl)


# Vision-Language Model Support
class VisionModelManager:
    """Manages vision-language models for drone applications"""
    
    def __init__(self):
        self.vision_models = {
            "llava": "llava:latest",
            "bakllava": "bakllava:latest", 
            "moondream": "moondream:latest",
            "llava-phi3": "llava-phi3:latest"
        }
        self.edge_models = {
            "llava-7b": "llava:7b",
            "llava-13b": "llava:13b"
        }
    
    def get_vision_model(self, model_name: str = "llava") -> str:
        """Get vision model name, with fallback to available models"""
        if model_name in self.vision_models:
            return self.vision_models[model_name]
        return self.vision_models["llava"]  # Default fallback
    
    def get_edge_model(self, model_name: str = "llava-7b") -> str:
        """Get edge-optimized model for onboard processing"""
        if model_name in self.edge_models:
            return self.edge_models[model_name]
        return self.edge_models["llava-7b"]  # Default fallback
    
    def is_vision_model(self, model: str) -> bool:
        """Check if model supports vision capabilities"""
        return any(vision_model in model.lower() for vision_model in ["llava", "bakllava", "moondream"])


# Global vision model manager
_vision_manager = VisionModelManager()


def vision_chat_complete(
    prompt: str,
    images: list[str],
    model: str = "llava",
    system: Optional[str] = None,
    temperature: float = 0.2,
    context_tag: Optional[str] = None,
) -> str:
    """Enhanced chat completion with vision support for drone applications"""
    vision_model = _vision_manager.get_vision_model(model)
    
    # Default system prompt for drone vision analysis
    if not system:
        system = """You are an advanced AI vision system for autonomous drones. Analyze the provided images and respond with:
            1. Object detection and classification
2. Environmental assessment (weather, lighting, obstacles)
3. Navigation recommendations
4. Safety considerations
5. Mission-specific insights

Be precise, technical, and focus on actionable intelligence for drone operations."""
    
    return chat_complete(
        model=vision_model,
        prompt=prompt,
        system=system,
        temperature=temperature,
        context_tag=context_tag or "DRONE_VISION",
        images=images
    )


def edge_vision_complete(
    prompt: str,
    images: list[str],
    model: str = "llava-7b",
    system: Optional[str] = None,
    temperature: float = 0.1,
) -> str:
    """Lightweight vision processing for edge devices (onboard drones)"""
    edge_model = _vision_manager.get_edge_model(model)
    
    if not system:
        system = """You are a lightweight AI vision system for onboard drone processing. Provide concise, actionable analysis:
            1. Critical objects/obstacles
2. Navigation status
3. Safety alerts
4. Mission progress

Keep responses brief and focused on immediate operational needs."""
    
    return chat_complete(
        model=edge_model,
        prompt=prompt,
        system=system,
        temperature=temperature,
        context_tag="EDGE_VISION",
        images=images
    )
