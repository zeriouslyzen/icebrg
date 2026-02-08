"""
Unified LLM Provider
=====================
Single interface for all LLM backends with automatic fallback and routing.

This provider abstracts away the complexity of multiple LLM backends,
providing a simple interface for the ICEBURG unified endpoint.

Usage:
    provider = UnifiedLLMProvider()
    
    # Fast mode - uses cloud API (Claude/Gemini)
    response = await provider.complete("hello", mode="fast")
    
    # Research mode - uses configured research provider
    response = await provider.complete("deep analysis query", mode="research")
"""

from __future__ import annotations
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Optional

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Supported LLM provider types"""
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"
    OLLAMA = "ollama"
    XAI = "xai"


@dataclass
class ProviderConfig:
    """Configuration for a single provider"""
    type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    default_model: Optional[str] = None
    timeout_s: int = 60
    
    @classmethod
    def from_env(cls, provider_type: ProviderType) -> "ProviderConfig":
        """Create config from environment variables"""
        configs = {
            ProviderType.ANTHROPIC: cls(
                type=ProviderType.ANTHROPIC,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                default_model=os.getenv("ICEBURG_ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            ),
            ProviderType.GOOGLE: cls(
                type=ProviderType.GOOGLE,
                api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                default_model=os.getenv("ICEBURG_GOOGLE_MODEL", "gemini-2.0-flash-exp"),
            ),
            ProviderType.OPENAI: cls(
                type=ProviderType.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                default_model=os.getenv("ICEBURG_OPENAI_MODEL", "gpt-4o"),
            ),
            ProviderType.OLLAMA: cls(
                type=ProviderType.OLLAMA,
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                default_model=os.getenv("ICEBURG_OLLAMA_MODEL", "llama3.1:8b"),
            ),
            ProviderType.XAI: cls(
                type=ProviderType.XAI,
                api_key=os.getenv("XAI_API_KEY"),
                default_model=os.getenv("ICEBURG_XAI_MODEL", "grok-2"),
            ),
        }
        return configs.get(provider_type, cls(type=provider_type))


class UnifiedLLMProvider:
    """
    Unified LLM provider with automatic fallback chain.
    
    Modes:
    - fast: Use cloud API for instant responses (Claude > Gemini > OpenAI > Ollama)
    - research: Use configured research provider (typically more powerful model)
    - local: Force use of local Ollama
    
    Features:
    - Automatic fallback when primary provider fails
    - Streaming support for real-time responses
    - Cost tracking per provider
    - Health checking for all providers
    """
    
    def __init__(
        self,
        fast_provider: Optional[str] = None,
        research_provider: Optional[str] = None,
        fallback_enabled: bool = True,
    ):
        """
        Initialize unified provider.
        
        Args:
            fast_provider: Provider for fast mode (default: from env or anthropic)
            research_provider: Provider for research mode (default: from env or same as fast)
            fallback_enabled: Whether to fall back to other providers on failure
        """
        self.fast_provider_name = (
            fast_provider 
            or os.getenv("ICEBURG_FAST_PROVIDER", "anthropic")
        ).lower()
        
        self.research_provider_name = (
            research_provider 
            or os.getenv("ICEBURG_RESEARCH_PROVIDER") 
            or self.fast_provider_name
        ).lower()
        
        self.fallback_enabled = fallback_enabled
        self._providers: dict[str, Any] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}
        self._cost_tracker: dict[str, float] = {}
        
        # Define fallback chain order
        self.fallback_chain = ["anthropic", "google", "openai", "xai", "ollama"]
        
        # Initialize available providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available providers based on environment"""
        provider_types = {
            "anthropic": ProviderType.ANTHROPIC,
            "google": ProviderType.GOOGLE,
            "gemini": ProviderType.GOOGLE,
            "openai": ProviderType.OPENAI,
            "ollama": ProviderType.OLLAMA,
            "xai": ProviderType.XAI,
            "grok": ProviderType.XAI,
        }
        
        for name, ptype in provider_types.items():
            config = ProviderConfig.from_env(ptype)
            # Only add if we have required credentials (or it's Ollama)
            if config.api_key or ptype == ProviderType.OLLAMA:
                self._provider_configs[name] = config
                logger.info(f"✅ Provider '{name}' available (model: {config.default_model})")
            else:
                logger.debug(f"⚠️ Provider '{name}' not configured (missing API key)")
    
    def _get_provider(self, name: str):
        """Lazy-load a specific provider"""
        if name in self._providers:
            return self._providers[name]
        
        if name not in self._provider_configs:
            return None
        
        config = self._provider_configs[name]
        provider = None
        
        try:
            if config.type == ProviderType.ANTHROPIC:
                from .anthropic_provider import AnthropicProvider
                provider = AnthropicProvider(api_key=config.api_key, timeout_s=config.timeout_s)
            
            elif config.type == ProviderType.GOOGLE:
                from .google_provider import GoogleProvider
                provider = GoogleProvider(timeout_s=config.timeout_s)
            
            elif config.type == ProviderType.OPENAI:
                from .openai_provider import OpenAIProvider
                provider = OpenAIProvider(timeout_s=config.timeout_s)
            
            elif config.type == ProviderType.OLLAMA:
                from .ollama_provider import OllamaProvider
                provider = OllamaProvider(base_url=config.base_url, timeout_s=config.timeout_s)
            
            elif config.type == ProviderType.XAI:
                from .xai_provider import XAIProvider
                provider = XAIProvider(timeout_s=config.timeout_s)
            
            if provider:
                self._providers[name] = provider
                return provider
                
        except Exception as e:
            logger.warning(f"Failed to initialize provider '{name}': {e}")
            return None
        
        return None
    
    def _get_fallback_chain(self, starting_from: str) -> list[str]:
        """Get ordered fallback chain starting from a provider"""
        if starting_from in self.fallback_chain:
            idx = self.fallback_chain.index(starting_from)
            return self.fallback_chain[idx:] + self.fallback_chain[:idx]
        return self.fallback_chain
    
    async def complete(
        self,
        prompt: str,
        mode: str = "fast",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ) -> str:
        """
        Complete a prompt using the appropriate provider for the mode.
        
        Args:
            prompt: User prompt
            mode: "fast", "research", or "local"
            system: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Override model selection
            
        Returns:
            Generated response text
        """
        # Determine which provider to use
        if mode == "local":
            provider_name = "ollama"
        elif mode == "research":
            provider_name = self.research_provider_name
        else:  # fast mode
            provider_name = self.fast_provider_name
        
        # Build fallback chain
        if self.fallback_enabled:
            chain = self._get_fallback_chain(provider_name)
        else:
            chain = [provider_name]
        
        last_error = None
        
        for pname in chain:
            provider = self._get_provider(pname)
            if not provider:
                continue
            
            config = self._provider_configs.get(pname)
            use_model = model or (config.default_model if config else None)
            
            try:
                start_time = time.time()
                
                # Run sync provider in thread pool
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: provider.chat_complete(
                        model=use_model,
                        prompt=prompt,
                        system=system,
                        temperature=temperature,
                        options={"max_tokens": max_tokens},
                    )
                )
                
                elapsed = time.time() - start_time
                logger.info(f"✅ [{pname}] Response in {elapsed:.2f}s ({len(response)} chars)")
                
                # Track cost (approximate)
                self._track_cost(pname, len(prompt), len(response))
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"❌ [{pname}] Failed: {e}")
                if not self.fallback_enabled:
                    raise
                continue
        
        # All providers failed
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    async def stream(
        self,
        prompt: str,
        mode: str = "fast",
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """
        Stream a response token by token.
        
        Note: Currently wraps non-streaming call. 
        TODO: Implement true streaming for each provider.
        """
        # For now, get full response and yield in chunks
        response = await self.complete(
            prompt=prompt,
            mode=mode,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        
        # Simulate streaming by yielding chunks
        chunk_size = 10
        for i in range(0, len(response), chunk_size):
            yield response[i:i + chunk_size]
            await asyncio.sleep(0.01)  # Small delay for realistic streaming
    
    def _track_cost(self, provider: str, input_tokens: int, output_tokens: int):
        """Track approximate cost per provider"""
        # Rough cost estimates per 1K tokens (as of 2026)
        cost_per_1k = {
            "anthropic": 0.003,  # Claude 3.5 Sonnet
            "google": 0.00035,  # Gemini Flash
            "openai": 0.005,    # GPT-4o
            "xai": 0.002,       # Grok
            "ollama": 0.0,      # Local
        }
        
        rate = cost_per_1k.get(provider, 0.0)
        total_tokens = (input_tokens + output_tokens) / 4  # Rough char to token
        cost = (total_tokens / 1000) * rate
        
        if provider not in self._cost_tracker:
            self._cost_tracker[provider] = 0.0
        self._cost_tracker[provider] += cost
    
    def get_cost_summary(self) -> dict[str, float]:
        """Get cost summary per provider"""
        return self._cost_tracker.copy()
    
    def get_available_providers(self) -> list[str]:
        """Get list of available providers"""
        return list(self._provider_configs.keys())
    
    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers"""
        results = {}
        
        for name in self._provider_configs:
            try:
                provider = self._get_provider(name)
                if provider:
                    # Quick test with minimal prompt
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(
                        None,
                        lambda: provider.chat_complete(
                            model=self._provider_configs[name].default_model,
                            prompt="test",
                            options={"max_tokens": 5},
                        )
                    )
                    results[name] = True
                else:
                    results[name] = False
            except Exception:
                results[name] = False
        
        return results


# Singleton instance
_unified_provider: Optional[UnifiedLLMProvider] = None


def get_unified_provider() -> UnifiedLLMProvider:
    """Get singleton unified provider instance"""
    global _unified_provider
    if _unified_provider is None:
        _unified_provider = UnifiedLLMProvider()
    return _unified_provider


def reset_unified_provider():
    """Reset singleton (for testing)"""
    global _unified_provider
    _unified_provider = None
