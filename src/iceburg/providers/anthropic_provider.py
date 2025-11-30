"""
Anthropic Provider
Claude API integration (optional dependency)
"""

from typing import Any, Optional, List
import os


class AnthropicProvider:
    """Anthropic Claude API provider"""
    
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 60):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.timeout_s = timeout_s
        self.client = None
        
        if self.api_key:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout_s)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install anthropic>=0.18.0"
                )
        else:
            raise ValueError("ANTHROPIC_API_KEY not set")
    
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Complete chat with Claude"""
        if not self.client:
            raise RuntimeError("Anthropic client not initialized")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            
            # Add images if provided
            if images:
                content = [{"type": "text", "text": prompt}]
                for img_path in images:
                    # Read image and convert to base64
                    import base64
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    })
                messages[0]["content"] = content
            
            response = self.client.messages.create(
                model=model or "claude-3-5-sonnet-20241022",
                max_tokens=options.get("max_tokens", 4096) if options else 4096,
                system=system,
                messages=messages,
                temperature=temperature
            )
            
            # Extract text from response
            if response.content:
                return response.content[0].text
            return ""
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")
    
    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embed texts using Claude (if supported)"""
        # Claude doesn't have a separate embedding API
        # Return empty embeddings or use alternative
        return [[] for _ in texts]

