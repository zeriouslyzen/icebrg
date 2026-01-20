import requests
import json
import time

def test_model(model_name, prompt):
    print(f"\n🧪 Testing {model_name}...")
    start = time.time()
    
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        duration = time.time() - start
        
        print(f"✅ Success! ({duration:.2f}s)")
        print(f"📝 Response preview: {result['response'][:100]}...")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

print("🧊 Checking Project Silicon ICEBURG Ingredients...")

# Test 8B (Fast Agent)
ok_8b = test_model("llama3.1:8b", "What is 2+2? Answer briefly.")

# Test 70B (Deep Research Agent)
ok_70b = test_model("llama3.1:70b", "Explain the concept of 'emergence' in one sentence.")

if ok_8b and ok_70b:
    print("\n🚀 All Systems Go! Your M4 Mac is ready to run the civilization.")
else:
    print("\n⚠️  Some models failed. Check if Ollama is running.")
