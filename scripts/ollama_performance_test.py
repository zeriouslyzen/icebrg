
import time
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.iceburg.config import load_config
from src.iceburg.llm import chat_complete

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("performance_test")

async def benchmark_prompt():
    cfg = load_config()
    model = cfg.surveyor_model # Using surveyor model for test
    
    query = "Explain the principle of resonance in energy harvesting."
    system_prompt = "You are an elite research agent. Be direct and technical."
    
    logger.info(f"🚀 Starting Benchmarking for model: {model}")
    
    start_time = time.time()
    
    response = chat_complete(
        model,
        query,
        system=system_prompt,
        temperature=0.0
    )
    
    end_time = time.time()
    duration = end_time - start_time
    
    if response:
        char_count = len(response)
        logger.info(f"✅ Response received in {duration:.2f} seconds.")
        logger.info(f"📝 Length: {char_count} characters.")
        logger.info(f"⚡ Approx throughput: {char_count / duration:.2f} chars/sec")
    else:
        logger.error("❌ Failed to get response.")

if __name__ == "__main__":
    asyncio.run(benchmark_prompt())
