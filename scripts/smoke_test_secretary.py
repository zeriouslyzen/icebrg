
import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.iceburg.config import load_config
from src.iceburg.agents.secretary import run as secretary_run
from src.iceburg.providers.factory import provider_factory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smoke_test")

async def run_smoke_test():
    logger.info("Starting ICEBURG Smoke Test...")
    
    # 1. Load Configuration
    try:
        cfg = load_config()
        logger.info(f"✅ Configuration loaded. Provider: {cfg.llm_provider}")
        logger.info(f"   Surveyor Model: {cfg.surveyor_model}")
    except Exception as e:
        logger.error(f"❌ Failed to load configuration: {e}")
        return False

    # 2. Check Provider Accessibility
    try:
        provider = provider_factory(cfg)
        logger.info(f"✅ Provider factory initialized: {type(provider).__name__}")
        
        # Simple health check if provider supports it
        if hasattr(provider, 'check_health'):
            health = await provider.check_health()
            logger.info(f"✅ Provider health: {health}")
    except Exception as e:
        logger.error(f"❌ Provider initialization failed: {e}")
        return False

    # 3. Test Secretary Agent (Synchronous but we run in thread if needed)
    try:
        query = "What is ICEBURG?"
        logger.info(f"Testing Secretary Agent with query: '{query}'")
        
        # Secretary run is synchronous
        response = await asyncio.to_thread(secretary_run, cfg, query)
        
        if response and len(response) > 0:
            logger.info(f"✅ Secretary Agent responded successfully!")
            logger.info(f"   Response preview: {response[:100]}...")
        else:
            logger.error("❌ Secretary Agent returned empty response")
            return False
            
    except Exception as e:
        logger.error(f"❌ Secretary Agent execution failed: {e}")
        return False

    logger.info("🚀 Smoke test completed successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(run_smoke_test())
    sys.exit(0 if success else 1)
