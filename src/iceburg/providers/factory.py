from __future__ import annotations
from typing import Any
import os

from .base import LLMProvider


_PROVIDER_SINGLETON: LLMProvider | None = None


def provider_factory(cfg: Any) -> LLMProvider:
    global _PROVIDER_SINGLETON
    if _PROVIDER_SINGLETON is not None:
        return _PROVIDER_SINGLETON
    # Respect config, but allow env to prefer high-capacity backends in server contexts
    provider = (getattr(cfg, "llm_provider", None) or os.getenv("ICEBURG_LLM_PROVIDER") or "ollama").lower()
    prefer_vllm = os.getenv("ICEBURG_PREFER_VLLM", "0") == "1"
    host = getattr(cfg, "provider_host", None) or os.getenv("HOST", "localhost")
    port = getattr(cfg, "provider_port", None) or os.getenv("OLLAMA_PORT", "11434")
    url = getattr(cfg, "provider_url", None) or f"http://{host}:{port}"
    timeout_s = getattr(cfg, "timeout_s", 60)

    if provider == "llama_cpp":
        from .llama_cpp_provider import LlamaCppProvider

        _PROVIDER_SINGLETON = LlamaCppProvider(base_url=url, timeout_s=timeout_s)
        return _PROVIDER_SINGLETON

    if provider == "vllm" or (prefer_vllm and provider == "ollama"):
        from .vllm_provider import VLLMProvider

        _PROVIDER_SINGLETON = VLLMProvider(base_url=url, timeout_s=timeout_s)
        return _PROVIDER_SINGLETON

    if provider == "anthropic" or provider == "claude":
        try:
            from .anthropic_provider import AnthropicProvider
            _PROVIDER_SINGLETON = AnthropicProvider(timeout_s=timeout_s)
            return _PROVIDER_SINGLETON
        except (ImportError, ValueError) as e:
            # Fallback to Ollama if Anthropic not available
            pass

    if provider == "openai" or provider == "gpt":
        try:
            from .openai_provider import OpenAIProvider
            _PROVIDER_SINGLETON = OpenAIProvider(timeout_s=timeout_s)
            return _PROVIDER_SINGLETON
        except (ImportError, ValueError) as e:
            # Fallback to Ollama if OpenAI not available
            pass

    if provider == "google" or provider == "gemini":
        try:
            from .google_provider import GoogleProvider
            _PROVIDER_SINGLETON = GoogleProvider(timeout_s=timeout_s)
            return _PROVIDER_SINGLETON
        except (ImportError, ValueError) as e:
            # Fallback to Ollama if Google not available
            pass

    from .ollama_provider import OllamaProvider

    _PROVIDER_SINGLETON = OllamaProvider(base_url=url, timeout_s=timeout_s)
    return _PROVIDER_SINGLETON


