"""
OpenAI Provider
GPT-4o API integration (optional dependency)
"""

from typing import Any, Optional, List
import os
import base64


class OpenAIProvider:
    """OpenAI GPT-4o API provider"""
    
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 60):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.timeout_s = timeout_s
        self.client = None
        
        if self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key, timeout=self.timeout_s)
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install openai>=1.12.0"
                )
        else:
            raise ValueError("OPENAI_API_KEY not set")
    
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Complete chat with GPT-4o"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            messages = []
            
            if system:
                messages.append({"role": "system", "content": system})
            
            # Prepare user message
            if images:
                content = [{"type": "text", "text": prompt}]
                for img_path in images:
                    # Read image and convert to base64
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
                model=model or "gpt-4o",
                messages=messages,
                temperature=temperature,
                max_tokens=options.get("max_tokens", 4096) if options else 4096
            )
            
            if response.choices:
                return response.choices[0].message.content or ""
            return ""
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
    
    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embed texts using OpenAI"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")
        
        try:
            response = self.client.embeddings.create(
                model=model or "text-embedding-3-small",
                input=texts
            )
            
            return [item.embedding for item in response.data]
            
        except Exception as e:
            raise RuntimeError(f"OpenAI embedding error: {e}")

