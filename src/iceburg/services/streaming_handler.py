"""
Streaming Handler
Server-Sent Events (SSE) streaming for real-time responses
"""

from typing import Any, Dict, Optional, List, AsyncGenerator
import json
import asyncio
from fastapi.responses import StreamingResponse


class StreamingHandler:
    """Handles streaming responses"""
    
    def __init__(self):
        self.active_streams: Dict[str, bool] = {}
    
    async def stream_response(
        self,
        generator: AsyncGenerator[str, None],
        stream_id: Optional[str] = None
    ) -> StreamingResponse:
        """Stream response using SSE"""
        if stream_id:
            self.active_streams[stream_id] = True
        
        async def event_generator():
            try:
                async for chunk in generator:
                    if stream_id and not self.active_streams.get(stream_id, False):
                        break
                    
                    # Format as SSE
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
                    # Small delay to prevent overwhelming client
                    await asyncio.sleep(0.01)
                
                # Send completion event
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                if stream_id:
                    self.active_streams.pop(stream_id, None)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    async def stream_tokens(
        self,
        text: str,
        chunk_size: int = 10
    ) -> AsyncGenerator[str, None]:
        """Stream text as tokens"""
        words = text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            yield chunk
            await asyncio.sleep(0.05)  # Small delay between chunks
    
    async def stream_structured(
        self,
        data: Dict[str, Any],
        fields: Optional[List[str]] = None
    ) -> AsyncGenerator[str, None]:
        """Stream structured data"""
        if fields:
            for field in fields:
                if field in data:
                    yield json.dumps({field: data[field]})
                    await asyncio.sleep(0.1)
        else:
            for key, value in data.items():
                yield json.dumps({key: value})
                await asyncio.sleep(0.1)
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id] = False
            return True
        return False
    
    def get_active_streams(self) -> List[str]:
        """Get list of active streams"""
        return [sid for sid, active in self.active_streams.items() if active]

