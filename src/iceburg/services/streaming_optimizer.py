"""
Streaming Optimizer
Efficient response streaming with progressive rendering
"""

from typing import Any, Dict, Optional, List, AsyncGenerator
import asyncio
import time
import json
from collections import deque


class StreamingOptimizer:
    """Optimizes response streaming"""
    
    def __init__(self, chunk_size: int = 10, buffer_size: int = 100):
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.buffer: deque = deque(maxlen=buffer_size)
        self.active_streams: Dict[str, bool] = {}
        self.stats = {
            "streams_created": 0,
            "chunks_sent": 0,
            "bytes_sent": 0,
            "average_latency": 0.0
        }
    
    async def stream_response(
        self,
        text: str,
        stream_id: Optional[str] = None,
        progressive: bool = True
    ) -> AsyncGenerator[str, None]:
        """Stream response with progressive rendering"""
        if stream_id:
            self.active_streams[stream_id] = True
            self.stats["streams_created"] += 1
        
        try:
            words = text.split()
            total_words = len(words)
            
            for i in range(0, total_words, self.chunk_size):
                if stream_id and not self.active_streams.get(stream_id, False):
                    break
                
                chunk_words = words[i:i+self.chunk_size]
                chunk = " ".join(chunk_words)
                
                if progressive:
                    # Add progressive rendering metadata
                    progress = (i + len(chunk_words)) / total_words
                    chunk_data = {
                        "chunk": chunk,
                        "progress": progress,
                        "total_words": total_words,
                        "words_sent": i + len(chunk_words)
                    }
                    yield json.dumps(chunk_data)
                else:
                    yield chunk
                
                self.stats["chunks_sent"] += 1
                self.stats["bytes_sent"] += len(chunk.encode('utf-8'))
                
                # Small delay to prevent overwhelming client
                await asyncio.sleep(0.01)
            
            # Send completion
            if stream_id:
                yield json.dumps({"done": True, "stream_id": stream_id})
        except Exception as e:
            yield json.dumps({"error": str(e)})
        finally:
            if stream_id:
                self.active_streams.pop(stream_id, None)
    
    async def stream_structured(
        self,
        data: Dict[str, Any],
        fields: Optional[List[str]] = None,
        stream_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Stream structured data"""
        if stream_id:
            self.active_streams[stream_id] = True
            self.stats["streams_created"] += 1
        
        try:
            if fields:
                for field in fields:
                    if stream_id and not self.active_streams.get(stream_id, False):
                        break
                    
                    if field in data:
                        chunk_data = {
                            "field": field,
                            "value": data[field],
                            "stream_id": stream_id
                        }
                        yield json.dumps(chunk_data)
                        self.stats["chunks_sent"] += 1
                        await asyncio.sleep(0.05)
            else:
                for key, value in data.items():
                    if stream_id and not self.active_streams.get(stream_id, False):
                        break
                    
                    chunk_data = {
                        "field": key,
                        "value": value,
                        "stream_id": stream_id
                    }
                    yield json.dumps(chunk_data)
                    self.stats["chunks_sent"] += 1
                    await asyncio.sleep(0.05)
            
            # Send completion
            if stream_id:
                yield json.dumps({"done": True, "stream_id": stream_id})
        except Exception as e:
            yield json.dumps({"error": str(e)})
        finally:
            if stream_id:
                self.active_streams.pop(stream_id, None)
    
    def stop_stream(self, stream_id: str) -> bool:
        """Stop a stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id] = False
            return True
        return False
    
    def get_active_streams(self) -> List[str]:
        """Get list of active streams"""
        return [sid for sid, active in self.active_streams.items() if active]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        return {
            **self.stats,
            "active_streams": len(self.active_streams),
            "buffer_size": len(self.buffer),
            "chunk_size": self.chunk_size
        }
    
    def optimize_bandwidth(self, text: str, max_bandwidth: int = 1000) -> str:
        """Optimize bandwidth usage"""
        # Simple bandwidth optimization
        # In production, use more sophisticated compression
        
        if len(text.encode('utf-8')) > max_bandwidth:
            # Truncate if too large
            words = text.split()
            truncated = " ".join(words[:max_bandwidth // 10])
            return truncated + "..."
        
        return text

