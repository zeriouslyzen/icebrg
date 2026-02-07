import asyncio
import websockets
import json
import sys
import uuid
import time

async def send_prompt(prompt, uri="ws://localhost:8000/ws"):
    # The server does not expect a client_id in the URL path
    full_uri = uri
    
    
    # Check if this is a defense audit request
    defense_mode = False
    if len(sys.argv) > 2 and sys.argv[2] == "--defense":
        defense_mode = True
        print("ACTIVATING RESEARCH DEFENCE PROTOCOL...")

    try:
        async with websockets.connect(full_uri) as websocket:
            print("Connected!")
            
            # Send prompt
            message = {
                "query": prompt,
                "mode": "research", 
                "agent": "dissident" if defense_mode else "auto", # Target dissident specifically
                "id": str(uuid.uuid4()),
                "auto_mode": True,
                "defense_mode": defense_mode # Custom flag for server to pick up
            }
            print(f"Sending: {json.dumps(message, indent=2)}")
            await websocket.send(json.dumps(message))
            
            # Listen for responses
            while True:
                try:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    msg_type = data.get("type")
                    content = data.get("content")
                    
                    if msg_type == "ping":
                        continue
                        
                    if msg_type == "chunk":
                        print(content, end="", flush=True)
                    elif msg_type == "thinking":
                        print(f"\n[THINKING]: {content}")
                    elif msg_type == "agent_thinking":
                        agent = data.get("agent", "unknown")
                        print(f"\n[{agent.upper()}]: {content}")
                    elif msg_type == "done":
                        print("\n\n[DONE] Processing complete.")
                        break
                    elif msg_type == "error":
                        print(f"\n[ERROR]: {data.get('message')}")
                        break
                    else:
                        print(f"\n[MSG: {msg_type}] {data}")
                        
                except websockets.exceptions.ConnectionClosed:
                    print("\nConnection closed.")
                    break
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prompt_cli.py \"Your prompt here\"")
        sys.exit(1)
    
    prompt = sys.argv[1]
    asyncio.run(send_prompt(prompt))
