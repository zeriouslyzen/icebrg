"""
Google Provider
Gemini API integration (optional dependency)
"""

from typing import Any, Optional, List
import os
import base64


class GoogleProvider:
    """Google Gemini API provider"""
    
    def __init__(self, api_key: Optional[str] = None, timeout_s: int = 60):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.timeout_s = timeout_s
        self.client = None
        
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
    
    def chat_complete(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        options: Optional[dict[str, Any]] = None,
        images: Optional[List[str]] = None,
    ) -> str:
        """Complete chat with Gemini"""
        if not self.client:
            raise RuntimeError("Google client not initialized")
        
        try:
            model_name = model or "gemini-2.0-flash-exp"
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
            
            if system:
                generation_config["system_instruction"] = system
            
            response = gen_model.generate_content(
                content_parts,
                generation_config=generation_config
            )
            
            if response.text:
                return response.text
            return ""
            
        except Exception as e:
            raise RuntimeError(f"Google API error: {e}")
    
    def embed_texts(self, model: str, texts: List[str]) -> List[List[float]]:
        """Embed texts using Gemini"""
        if not self.client:
            raise RuntimeError("Google client not initialized")
        
        try:
            # Use embedding model
            embedding_model = self.client.embed_content(
                model=model or "models/text-embedding-004",
                content=texts
            )
            
            return embedding_model.get("embeddings", [])
            
        except Exception as e:
            raise RuntimeError(f"Google embedding error: {e}")

