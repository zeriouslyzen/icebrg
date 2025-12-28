"""
xAI Provider
Grok API integration (OpenAI-compatible)

Grok uses OpenAI-compatible API format at https://api.x.ai/v1
"""

from typing import Any, Optional, List
import os


class XAIProvider:
    """xAI Grok API provider (OpenAI-compatible)"""
    
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 60):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.timeout_s = timeout_s
        self.client = None
        self.base_url = "https://api.x.ai/v1"
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout_s
                )
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai>=1.12.0"
                )
        else:
            raise ValueError("XAI_API_KEY not set")
    
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Complete chat with Grok"""
        if not self.client:
            raise RuntimeError("xAI client not initialized")
        
        try:
            messages = []
            
            if system:
                messages.append({"role": "system", "content": system})
            
            # Grok supports vision like GPT-4o
            if images:
                import base64
                content = [{"type": "text", "text": prompt}]
                for img_path in images:
                    with open(img_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_data}"
                        }
                    })
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=model or "grok-beta",
                messages=messages,
                temperature=temperature,
                max_tokens=options.get("max_tokens", 4096) if options else 4096
            )
            
            if response.choices:
                return response.choices[0].message.content or ""
            return ""
            
        except Exception as e:
            raise RuntimeError(f"xAI API error: {e}")
    
    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embed texts using xAI (if supported)"""
        # xAI may not have embedding API yet, return empty
        return [[] for _ in texts]
