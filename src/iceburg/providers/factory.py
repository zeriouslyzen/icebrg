from __future__ import annotations
from typing import Any, Optional, Union
import os

from .base import LLMProvider



_PROVIDER_SINGLETON: Optional[LLMProvider] = None
_PROVIDER_TYPE: Optional[str] = None  # Track which provider type is cached


def clear_provider_singleton():
    """Clear the cached provider singleton. Call when switching providers."""
    global _PROVIDER_SINGLETON, _PROVIDER_TYPE
    _PROVIDER_SINGLETON = None
    _PROVIDER_TYPE = None


def provider_factory(cfg: Any) -> LLMProvider:
    global _PROVIDER_SINGLETON, _PROVIDER_TYPE
    
    # Determine requested provider
    requested_provider = (getattr(cfg, "llm_provider", None) or os.getenv("ICEBURG_LLM_PROVIDER") or "ollama").lower()
    
    # If singleton exists BUT is different provider type, clear it
    if _PROVIDER_SINGLETON is not None:
        if _PROVIDER_TYPE != requested_provider:
            # Provider type changed - clear singleton
            _PROVIDER_SINGLETON = None
            _PROVIDER_TYPE = None
        else:
            return _PROVIDER_SINGLETON
    
    # Default to Ollama (no fallback)
    provider = requested_provider
    prefer_vllm = os.getenv("ICEBURG_PREFER_VLLM", "0") == "1"
    host = getattr(cfg, "provider_host", None) or os.getenv("HOST", "localhost")
    port = getattr(cfg, "provider_port", None) or os.getenv("OLLAMA_PORT", "11434")
    url = getattr(cfg, "provider_url", None) or f"http://{host}:{port}"
    timeout_s = getattr(cfg, "timeout_s", 60)

    if provider == "llama_cpp":
        from .llama_cpp_provider import LlamaCppProvider

        _PROVIDER_SINGLETON = LlamaCppProvider(base_url=url, timeout_s=timeout_s)
        _PROVIDER_TYPE = provider
        return _PROVIDER_SINGLETON

    if provider == "vllm" or (prefer_vllm and provider == "ollama"):
        from .vllm_provider import VLLMProvider

        _PROVIDER_SINGLETON = VLLMProvider(base_url=url, timeout_s=timeout_s)
        _PROVIDER_TYPE = "vllm"
        return _PROVIDER_SINGLETON

    # Optional: direct Ollama provider (no fallback)
    if provider == "ollama":
        from .ollama_provider import OllamaProvider

        _PROVIDER_SINGLETON = OllamaProvider(base_url=url, timeout_s=timeout_s)
        _PROVIDER_TYPE = "ollama"
        return _PROVIDER_SINGLETON

    # Try Google/Gemini specifically (NOT for "auto" - auto should use Ollama)
    if provider == "google" or provider == "gemini":
        try:
            from .google_provider import GoogleProvider
            _PROVIDER_SINGLETON = GoogleProvider(timeout_s=timeout_s)
            _PROVIDER_TYPE = "google"
            return _PROVIDER_SINGLETON
        except (ImportError, ValueError) as e:
            # No fallback - fail if Google not available
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Google/Gemini provider not available: {e}. Please configure GOOGLE_API_KEY.")
            raise

    if provider == "anthropic" or provider == "claude":
        try:
            from .anthropic_provider import AnthropicProvider
            _PROVIDER_SINGLETON = AnthropicProvider(timeout_s=timeout_s)
            _PROVIDER_TYPE = "anthropic"
            return _PROVIDER_SINGLETON
        except (ImportError, ValueError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Anthropic provider not available: {e}. Please configure ANTHROPIC_API_KEY.")
            raise

    if provider == "openai" or provider == "gpt":
        try:
            from .openai_provider import OpenAIProvider
            _PROVIDER_SINGLETON = OpenAIProvider(timeout_s=timeout_s)
            _PROVIDER_TYPE = "openai"
            return _PROVIDER_SINGLETON
        except (ImportError, ValueError) as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"OpenAI provider not available: {e}. Please configure OPENAI_API_KEY.")
            raise
    
    # Handle "auto" - default to Ollama for local operation
    if provider == "auto":
        from .ollama_provider import OllamaProvider
        _PROVIDER_SINGLETON = OllamaProvider(base_url=url, timeout_s=timeout_s)
        _PROVIDER_TYPE = "ollama"
        return _PROVIDER_SINGLETON

    # No automatic fallback - fail if provider not recognized
    raise ValueError(f"Unknown LLM provider: {provider}. Supported providers: google, anthropic, openai, ollama, llama_cpp, vllm")


