"""
Google Provider
Gemini API integration with token tracking and cost management
"""

from typing import Any, Optional, List, Dict, Tuple
import os
import base64
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


# Gemini API Pricing (per 1M tokens, as of 2025)
GEMINI_PRICING = {
    "gemini-2.0-flash-exp": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    "default": {"input": 0.075, "output": 0.30},  # Default to Flash pricing
}

# Embedding pricing (per 1K characters)
EMBEDDING_PRICING = {
    "text-embedding-004": 0.0001,
    "default": 0.0001,
}


class GoogleProvider:
    """Google Gemini API provider with token tracking and cost management"""
    
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 60):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.timeout_s = timeout_s
        self.client = None
        self.total_tokens_used = 0
        self.total_cost = 0.0
        
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. Install with: pip install google-generativeai>=0.3.0"
                )
        else:
            raise ValueError("GOOGLE_API_KEY not set")
    
    def _extract_usage_metadata(self, response: Any) -> Dict[str, Any]:
        """Extract token usage metadata from Gemini API response"""
        usage_metadata = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        
        try:
            if hasattr(response, "usage_metadata"):
                metadata = response.usage_metadata
                usage_metadata["input_tokens"] = getattr(metadata, "prompt_token_count", 0)
                usage_metadata["output_tokens"] = getattr(metadata, "candidates_token_count", 0)
                usage_metadata["total_tokens"] = getattr(metadata, "total_token_count", 0)
        except Exception as e:
            logger.warning(f"Failed to extract usage metadata: {e}")
        
        return usage_metadata
    
    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate API cost based on token usage"""
        pricing = GEMINI_PRICING.get(model, GEMINI_PRICING["default"])
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost
    
    def _log_usage(self, model: str, usage_metadata: Dict[str, Any], cost: float, cached: bool = False):
        """Log token usage for monitoring and cost tracking"""
        try:
            from ..telemetry.advanced_telemetry import AdvancedTelemetry, PromptMetrics
            telemetry = AdvancedTelemetry()
            
            # Track prompt metrics
            telemetry.track_prompt(
                PromptMetrics(
                    prompt_id=f"gemini_{datetime.now().isoformat()}",
                    prompt_text="",  # Don't log full prompt for privacy
                    response_time=0.0,  # Will be set by caller
                    token_count=usage_metadata["total_tokens"],
                    model_used=model,
                    success=True,
                    quality_score=1.0,
                )
            )
        except Exception as e:
            logger.debug(f"Failed to log usage to telemetry: {e}")
        
        # Update internal counters
        self.total_tokens_used += usage_metadata["total_tokens"]
        self.total_cost += cost
        
        # Log usage
        logger.info(
            f"Gemini API usage - Model: {model}, "
            f"Input: {usage_metadata['input_tokens']}, "
            f"Output: {usage_metadata['output_tokens']}, "
            f"Total: {usage_metadata['total_tokens']}, "
            f"Cost: ${cost:.6f}, "
            f"Cached: {cached}"
        )
    
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Complete chat with Gemini, returning response and tracking token usage"""
        if not self.client:
            raise RuntimeError("Google client not initialized")
        
        try:
            model_name = model or "gemini-2.0-flash-exp"
            
            # Create model with system instruction if provided
            if system:
                gen_model = self.client.GenerativeModel(
                    model_name,
                    system_instruction=system
                )
            else:
                gen_model = self.client.GenerativeModel(model_name)
            
            # Prepare content
            content_parts = [prompt]
            
            # Add images if provided
            if images:
                for img_path in images:
                    import PIL.Image
                    img = PIL.Image.open(img_path)
                    content_parts.append(img)
            
            # Generate content
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": options.get("max_tokens", 4096) if options else 4096,
            }
            
            response = gen_model.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            # Extract usage metadata
            usage_metadata = self._extract_usage_metadata(response)
            
            # Calculate cost
            cost = self._calculate_cost(
                model_name,
                usage_metadata["input_tokens"],
                usage_metadata["output_tokens"]
            )
            
            # Log usage
            self._log_usage(model_name, usage_metadata, cost, cached=False)
            
            # Record usage in tracker
            try:
                from ..monitoring.token_usage_tracker import get_token_tracker
                tracker = get_token_tracker()
                tracker.record_usage(
                    provider="google",
                    model=model_name,
                    input_tokens=usage_metadata["input_tokens"],
                    output_tokens=usage_metadata["output_tokens"],
                    cost_usd=cost,
                    cached=False,
                )
            except Exception as e:
                logger.debug(f"Failed to record usage in tracker: {e}")
            
            # Store usage metadata in response object for later retrieval
            if hasattr(response, "__dict__"):
                response._iceburg_usage = usage_metadata
                response._iceburg_cost = cost
            
            if response.text:
                return response.text
            return ""
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Google API error: {error_msg}")
            
            # Provide more context in error message
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise RuntimeError(
                    f"Google API rate limit/quota exceeded: {error_msg}. "
                    "Consider using local models or increasing quota."
                )
            elif "api key" in error_msg.lower() or "authentication" in error_msg.lower():
                raise RuntimeError(
                    f"Google API authentication failed: {error_msg}. "
                    "Check GOOGLE_API_KEY environment variable."
                )
            else:
                raise RuntimeError(f"Google API error: {error_msg}")
    
    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini, tracking usage"""
        if not self.client:
            raise RuntimeError("Google client not initialized")
        
        try:
            # Use embedding model
            embedding_model_name = model or "models/text-embedding-004"
            response = self.client.embed_content(
                model=embedding_model_name,
                content=texts
            )
            
            embeddings = response.get("embeddings", [])
            
            # Calculate embedding cost (per 1K characters)
            total_chars = sum(len(text) for text in texts)
            embedding_cost = (total_chars / 1_000) * EMBEDDING_PRICING.get(
                embedding_model_name.replace("models/", ""),
                EMBEDDING_PRICING["default"]
            )
            
            # Log embedding usage
            logger.info(
                f"Gemini Embedding usage - Model: {embedding_model_name}, "
                f"Texts: {len(texts)}, "
                f"Characters: {total_chars}, "
                f"Cost: ${embedding_cost:.6f}"
            )
            
            self.total_cost += embedding_cost
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Google embedding error: {e}")
            raise RuntimeError(f"Google embedding error: {e}")
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get cumulative usage statistics"""
        return {
            "total_tokens": self.total_tokens_used,
            "total_cost_usd": self.total_cost,
            "provider": "google",
        }

