
import asyncio
import logging
import sys
import os
import time
import requests
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("fast_mode_test")

BASE_URL = "http://localhost:8000/api"

def test_fast_mode():
    logger.info("🚀 Starting Secretary Fast Mode Batch Test (API)...")
    
    # Wait for API to be ready
    for i in range(5):
        try:
            requests.get(f"http://localhost:8000/docs")
            break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            
    queries = [
        "Who are you?",
        "What can ICEBURG do?",
        "How do I start a research task?",
        "What is the difference between Surveyor and Dissident agents?",
        "Can you help me design a software application?",
        "Tell me about the AGI civilization simulation.",
        "What is the current status of the system?"
    ]
    
    results = []
    
    for i, query in enumerate(queries):
        logger.info(f"\n[{i+1}/{len(queries)}] Query: '{query}'")
        
        # Payload for API
        payload = {
            "query": query,
            "mode": "chat",
            "model": "secretary",
            "stream": True
        }
        
        start_time = time.time()
        full_response = ""
        
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload, stream=True)
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "text" in data: # Standard chunk
                                full_response += data["text"]
                            elif "content" in data: # Thinking/chunk
                                full_response += data["content"]
                        except json.JSONDecodeError:
                            pass
                            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"✅ Received ({duration:.2f}s)")
            results.append({
                "query": query,
                "full_response": full_response,
                "duration": duration,
                "length": len(full_response)
            })
            
        except Exception as e:
            logger.error(f"❌ Failed: {e}")
            results.append({
                "query": query,
                "full_response": f"ERROR: {e}",
                "duration": 0,
                "length": 0
            })

    # Summary Table
    print("\n" + "="*80)
    print(f"{'QUERY':<40} | {'TIME':<8} | {'CHARS':<6}")
    print("-" * 80)
    for res in results:
        print(f"{res['query'][:40]:<40} | {res['duration']:>6.2f}s | {res['length']:>5}")
    print("="*80)
    
    avg_time = sum(r['duration'] for r in results) / len(results)
    print(f"Average Response Time: {avg_time:.2f}s")

    # Write transcript
    transcript_path = "fast_mode_transcript.md"
    with open(transcript_path, "w") as f:
        f.write("# Secretary Fast Mode Transcript (API)\n\n")
        f.write(f"**Average Time:** {avg_time:.2f}s\n\n")
        for res in results:
            f.write(f"## Query: {res['query']}\n")
            f.write(f"**Time:** {res['duration']:.2f}s\n\n")
            f.write(f"### Response:\n{res['full_response']}\n\n")
            f.write("---\n\n")
    logger.info(f"Transcript saved to {transcript_path}")

if __name__ == "__main__":
    test_fast_mode()
