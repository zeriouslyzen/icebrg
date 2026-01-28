
import asyncio
import logging
import sys
import os
import time
import requests
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("api_metric_test")

BASE_URL = "http://localhost:8000/api"

def test_api_metrics():
    logger.info("🚀 Starting API Metric Test...")
    
    # Wait for API to be ready
    for i in range(30):
        try:
            requests.get(f"http://localhost:8000/docs")
            logger.info("✅ API is online.")
            break
        except requests.exceptions.ConnectionError:
            logger.info(f"⏳ Waiting for API... ({i+1}/10)")
            time.sleep(2)
    else:
        logger.error("❌ API failed to start.")
        return

    # Payload for Secretay Agent
    payload = {
        "query": "What is the capital of France?",  # Simple query
        "mode": "chat",
        "model": "secretary",
        "stream": True  # Enable streaming to hit the fast path
    }
    
    logger.info(f"📤 Sending query: '{payload['query']}'")
    start_time = time.time()
    
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload, stream=True)
        
        full_response = ""
        logger.info("⬇️ Receving stream...")
        
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "text" in data:
                            full_response += data["text"]
                            print(data["text"], end="", flush=True)
                    except json.JSONDecodeError:
                        pass
                        
        end_time = time.time()
        print("\n")
        logger.info(f"✅ Response complete in {end_time - start_time:.2f}s")
        logger.info(f"📝 Length: {len(full_response)} chars")
        
    except Exception as e:
        logger.error(f"❌ detailed error: {e}")

if __name__ == "__main__":
    test_api_metrics()
