#!/usr/bin/env python3
"""
Test LLM response times on M4 chip
Measures actual response times for different models and configurations
"""
import time
import ollama
import sys

def test_llm_speed(model: str, num_predict: int = 10, num_ctx: int = 512):
    """Test LLM response time"""
    try:
        start = time.time()
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': 'hi'}],
            options={'num_predict': num_predict, 'num_ctx': num_ctx}
        )
        end = time.time()
        response_time_ms = (end - start) * 1000
        return response_time_ms, True
    except Exception as e:
        return None, False

def main():
    print("Testing LLM Response Times on M4 Chip")
    print("=" * 60)
    
    # Get available models
    try:
        available = ollama.list().get('models', [])
        available_models = [m.get('name', '') for m in available]
        print(f"Available models: {len(available_models)}")
    except Exception as e:
        print(f"Error listing models: {e}")
        return
    
    # Test configurations
    configs = [
        {"num_predict": 10, "num_ctx": 512, "desc": "Ultra-fast (10 tokens, 512 ctx)"},
        {"num_predict": 50, "num_ctx": 1024, "desc": "Fast (50 tokens, 1024 ctx)"},
        {"num_predict": 100, "num_ctx": 2048, "desc": "Standard (100 tokens, 2048 ctx)"},
    ]
    
    # Test each available model
    for model in available_models[:3]:  # Test first 3 models
        print(f"\nModel: {model}")
        print("-" * 60)
        
        for config in configs:
            response_time, success = test_llm_speed(
                model, 
                num_predict=config["num_predict"],
                num_ctx=config["num_ctx"]
            )
            
            if success:
                tokens_per_sec = config["num_predict"] / (response_time / 1000) if response_time > 0 else 0
                print(f"  {config['desc']}: {response_time:.2f}ms ({tokens_per_sec:.1f} tokens/sec)")
            else:
                print(f"  {config['desc']}: FAILED")
            
            time.sleep(0.5)  # Small delay between tests

if __name__ == "__main__":
    main()

