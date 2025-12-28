"""
Multi-Provider Router
Routes requests to appropriate AI provider with runtime API key support.

Supports: OpenAI (GPT), Anthropic (Claude), Google (Gemini), xAI (Grok), DeepSeek
"""

from typing import Any, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging

logger = logging.getLogger(__name__)


class Provider(Enum):
    """Supported AI providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    XAI = "xai"
    DEEPSEEK = "deepseek"


@dataclass
class ProviderConfig:
    """Configuration for a provider"""
    api_key: str
    model: Optional[str] = None
    timeout_s: int = 60


@dataclass
class ChatRequest:
    """Unified chat request format"""
    message: str
    system: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 4096
    images: Optional[List[str]] = None


@dataclass
class ChatResponse:
    """Unified chat response format"""
    content: str
    provider: str
    model: str
    success: bool
    error: Optional[str] = None
    latency_ms: Optional[float] = None


class MultiProviderRouter:
    """
    Routes chat requests to multiple AI providers.
    
    Supports:
    - Runtime API key injection (not from env vars)
    - Failover to next provider on error
    - Hybrid mode (parallel queries to multiple providers)
    - Provider-specific model defaults
    """
    
    # Default models for each provider
    DEFAULT_MODELS = {
        Provider.OPENAI: "gpt-4o",
        Provider.ANTHROPIC: "claude-3-5-sonnet-20241022",
        Provider.GOOGLE: "gemini-1.5-flash",
        Provider.XAI: "grok-beta",
        Provider.DEEPSEEK: "deepseek-chat",
    }
    
    def __init__(self):
        self._provider_cache: Dict[str, Any] = {}
    
    def _get_provider_instance(self, provider: Provider, config: ProviderConfig):
        """Get or create a provider instance with the given API key."""
        cache_key = f"{provider.value}:{config.api_key[:8]}"
        
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key]
        
        if provider == Provider.OPENAI:
            from .openai_provider import OpenAIProvider
            instance = OpenAIProvider(api_key=config.api_key, timeout_s=config.timeout_s)
        elif provider == Provider.ANTHROPIC:
            from .anthropic_provider import AnthropicProvider
            instance = AnthropicProvider(api_key=config.api_key, timeout_s=config.timeout_s)
        elif provider == Provider.GOOGLE:
            from .google_provider import GoogleProvider
            instance = GoogleProvider(api_key=config.api_key, timeout_s=config.timeout_s)
        elif provider == Provider.XAI:
            from .xai_provider import XAIProvider
            instance = XAIProvider(api_key=config.api_key, timeout_s=config.timeout_s)
        elif provider == Provider.DEEPSEEK:
            from .deepseek_provider import DeepSeekProvider
            instance = DeepSeekProvider(api_key=config.api_key, timeout_s=config.timeout_s)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        self._provider_cache[cache_key] = instance
        return instance
    
    def chat(
        self,
        provider: Provider,
        config: ProviderConfig,
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Send a chat request to a specific provider.
        
        Args:
            provider: The provider to use
            config: Provider configuration including API key
            request: The chat request
            
        Returns:
            ChatResponse with the result
        """
        import time
        start = time.time()
        
        try:
            instance = self._get_provider_instance(provider, config)
            model = config.model or self.DEFAULT_MODELS.get(provider, "")
            
            content = instance.chat_complete(
                model=model,
                prompt=request.message,
                system=request.system,
                temperature=request.temperature,
                options={"max_tokens": request.max_tokens},
                images=request.images,
            )
            
            latency = (time.time() - start) * 1000
            
            return ChatResponse(
                content=content,
                provider=provider.value,
                model=model,
                success=True,
                latency_ms=latency,
            )
            
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Provider {provider.value} failed: {e}")
            
            return ChatResponse(
                content="",
                provider=provider.value,
                model=config.model or "",
                success=False,
                error=str(e),
                latency_ms=latency,
            )
    
    def chat_with_failover(
        self,
        providers: List[tuple[Provider, ProviderConfig]],
        request: ChatRequest,
    ) -> ChatResponse:
        """
        Try providers in order, falling back on failure.
        
        Args:
            providers: List of (provider, config) tuples to try in order
            request: The chat request
            
        Returns:
            First successful response, or last failure
        """
        last_response = None
        
        for provider, config in providers:
            response = self.chat(provider, config, request)
            
            if response.success:
                return response
            
            last_response = response
            logger.warning(f"Failover: {provider.value} failed, trying next...")
        
        return last_response or ChatResponse(
            content="",
            provider="none",
            model="",
            success=False,
            error="All providers failed",
        )
    
    async def chat_hybrid(
        self,
        providers: List[tuple[Provider, ProviderConfig]],
        request: ChatRequest,
    ) -> List[ChatResponse]:
        """
        Query multiple providers in parallel (hybrid mode).
        
        Args:
            providers: List of (provider, config) tuples to query
            request: The chat request
            
        Returns:
            List of responses from all providers
        """
        async def query_provider(provider: Provider, config: ProviderConfig):
            # Run synchronous chat in thread pool
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.chat(provider, config, request)
            )
        
        tasks = [query_provider(p, c) for p, c in providers]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                provider, _ = providers[i]
                results.append(ChatResponse(
                    content="",
                    provider=provider.value,
                    model="",
                    success=False,
                    error=str(response),
                ))
            else:
                results.append(response)
        
        return results


# Global router instance
_router: Optional[MultiProviderRouter] = None


def get_router() -> MultiProviderRouter:
    """Get or create global router instance."""
    global _router
    if _router is None:
        _router = MultiProviderRouter()
    return _router


def quick_chat(
    provider: str,
    api_key: str,
    message: str,
    model: Optional[str] = None,
    system: Optional[str] = None,
) -> str:
    """
    Quick convenience function for single-provider chat.
    
    Example:
        response = quick_chat("claude", "sk-ant-...", "Hello!")
        print(response)
    """
    router = get_router()
    
    provider_enum = Provider(provider.lower())
    config = ProviderConfig(api_key=api_key, model=model)
    request = ChatRequest(message=message, system=system)
    
    response = router.chat(provider_enum, config, request)
    
    if response.success:
        return response.content
    else:
        raise RuntimeError(f"Chat failed: {response.error}")
