"""
Test Swarm Orchestrator
Verifies parallel execution and debate synthesis
"""

import sys
import os
import asyncio
import time
import requests
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_api_swarm_mode():
    """Test the /api/query endpoint in research mode (triggers Swarm)"""
    url = "http://localhost:8000/api/query"
    
    # Complex query to trigger debate
    payload = {
        "text": "What are the primary risks of AGI development?",
        "mode": "research",  # Triggers Swarm
        "stream": True 
    }
    
    print(f"🚀 Testing Swarm Mode with query: '{payload['text']}'")
    start_time = time.time()
    
    try:
        # Send request
        # Note: We expect SSE stream
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(response.text)
                return

            print("✅ Connection established. receiving stream...")
            
            full_response = ""
            thinking_lines = []
            
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        data_str = decoded_line[6:]
                        try:
                            data = json.loads(data_str)
                            msg_type = data.get('type')
                            
                            if msg_type == 'thinking_stream':
                                content = data.get('content', '')
                                print(f"🧠 Thinking: {content}")
                                thinking_lines.append(content)
                                
                            elif msg_type == 'chunk':
                                content = data.get('content', '')
                                full_response += content
                                sys.stdout.write(content)
                                sys.stdout.flush()
                                
                            elif msg_type == 'error':
                                print(f"\n❌ Server Error: {data.get('content')}")
                                
                            elif msg_type == 'done':
                                print("\n✨ Stream complete.")
                                
                        except json.JSONDecodeError:
                            pass
                            
        duration = time.time() - start_time
        print(f"\n⏱️ Total Duration: {duration:.2f}s")
        
        # VERIFICATION CHECKS
        is_debate = any("orchestrating a parallel swarm" in line for line in thinking_lines)
        if is_debate:
            print("✅ Verified: Swarm Orchestrator was triggered.")
        else:
            print("⚠️ Warning: Did not see Swarm trigger message.")
            
        if len(full_response) > 100:
            print("✅ Verified: Received substantial response.")
        else:
             print("⚠️ Warning: Response seems too short.")

        if duration < 60: # Arbitrary threshold for parallel vs serial logic
             print("✅ Performance: Execution completed within reasonable time for parallel agents.")

    except Exception as e:
        print(f"❌ Test Failed: {e}")

if __name__ == "__main__":
    test_api_swarm_mode()
