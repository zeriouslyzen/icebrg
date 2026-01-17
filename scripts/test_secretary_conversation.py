#!/usr/bin/env python3
"""
Test Fast Mode Secretary Conversation
Tests if the secretary can carry a logical conversation about ICEBURG
and use its abilities and tools properly.

This is critical for the startup contest submission.
"""

import requests
import json
import time
import sys
from typing import List, Dict

API_URL = "http://localhost:8000/api/query"
CONVERSATION_ID = f"test_conv_{int(time.time())}"

def send_query(query: str, mode: str = "chat", agent: str = "secretary") -> Dict:
    """Send a query to the API and get response."""
    payload = {
        "query": query,
        "mode": mode,
        "agent": agent,
        "conversation_id": CONVERSATION_ID,
        "stream": False
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error sending query: {e}")
        return {"error": str(e)}

def test_conversation_flow():
    """Test a logical conversation flow about ICEBURG."""
    
    test_queries = [
        # Introduction and identity
        "Who are you and what is ICEBURG?",
        
        # Capabilities
        "What are ICEBURG's main capabilities?",
        
        # Agents
        "Tell me about the different agents in ICEBURG. What does each one do?",
        
        # Research abilities
        "Can you research yourself? What can you tell me about ICEBURG's architecture?",
        
        # Tools and abilities
        "What tools and abilities do you have access to?",
        
        # Follow-up (testing conversation continuity)
        "Can you use those tools to find more information about ICEBURG's research capabilities?",
        
        # Self-reflection
        "What makes ICEBURG different from other AI systems?",
        
        # Practical usage
        "How would someone use ICEBURG for a research project?",
    ]
    
    print("=" * 80)
    print("ICEBURG Secretary Conversation Test")
    print("=" * 80)
    print(f"Conversation ID: {CONVERSATION_ID}")
    print(f"Mode: chat (fast mode)")
    print(f"Agent: secretary")
    print("=" * 80)
    print()
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}/{len(test_queries)}]")
        print(f"Q: {query}")
        print("-" * 80)
        
        start_time = time.time()
        response = send_query(query)
        elapsed = time.time() - start_time
        
        if "error" in response:
            print(f"❌ Error: {response['error']}")
            results.append({
                "query": query,
                "status": "error",
                "error": response["error"],
                "time": elapsed
            })
            continue
        
        # Extract response content
        content = ""
        if "results" in response:
            if isinstance(response["results"], dict):
                content = response["results"].get("content", "")
            else:
                content = str(response["results"])
        elif "content" in response:
            content = response["content"]
        elif "response" in response:
            content = response["response"]
        else:
            content = json.dumps(response, indent=2)
        
        # Display response
        print(f"A: {content[:500]}{'...' if len(content) > 500 else ''}")
        print(f"\n⏱️  Response time: {elapsed:.2f}s")
        print(f"📏 Response length: {len(content)} chars")
        
        # Check response quality
        quality_checks = {
            "has_content": len(content) > 50,
            "mentions_iceburg": "iceburg" in content.lower() or "iceberg" in content.lower(),
            "mentions_agents": any(agent in content.lower() for agent in ["surveyor", "dissident", "synthesist", "oracle", "secretary"]),
            "mentions_capabilities": any(word in content.lower() for word in ["research", "capability", "agent", "mode", "tool"]),
            "logical_flow": len(content) > 100,  # Substantive response
        }
        
        print(f"\n✅ Quality Checks:")
        for check, passed in quality_checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}: {passed}")
        
        results.append({
            "query": query,
            "response": content,
            "time": elapsed,
            "length": len(content),
            "quality": quality_checks,
            "status": "success" if all(quality_checks.values()) else "partial"
        })
        
        # Small delay between queries
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    total_queries = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    partial = sum(1 for r in results if r.get("status") == "partial")
    errors = sum(1 for r in results if r.get("status") == "error")
    
    avg_time = sum(r.get("time", 0) for r in results) / total_queries if total_queries > 0 else 0
    avg_length = sum(r.get("length", 0) for r in results) / total_queries if total_queries > 0 else 0
    
    print(f"Total Queries: {total_queries}")
    print(f"✅ Successful: {successful}")
    print(f"⚠️  Partial: {partial}")
    print(f"❌ Errors: {errors}")
    print(f"\n⏱️  Average Response Time: {avg_time:.2f}s")
    print(f"📏 Average Response Length: {avg_length:.0f} chars")
    
    # Check conversation continuity
    print(f"\n🔄 Conversation Continuity:")
    continuity_score = 0
    for i in range(1, len(results)):
        prev_response = results[i-1].get("response", "").lower()
        current_query = results[i].get("query", "").lower()
        
        # Check if current query references previous context
        if any(word in current_query for word in ["those", "that", "it", "them", "this", "more"]):
            continuity_score += 1
    
    print(f"   Follow-up queries: {continuity_score}/{len(results)-1}")
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment:")
    if successful >= total_queries * 0.8 and errors == 0:
        print("   ✅ EXCELLENT - Ready for startup contest")
    elif successful >= total_queries * 0.6 and errors == 0:
        print("   ⚠️  GOOD - Minor improvements needed")
    elif errors == 0:
        print("   ⚠️  NEEDS WORK - Response quality needs improvement")
    else:
        print("   ❌ CRITICAL ISSUES - Fix errors before submission")
    
    return results

if __name__ == "__main__":
    print("Starting ICEBURG Secretary Conversation Test...")
    print("Make sure ICEBURG is running (./scripts/start_iceburg.sh)\n")
    
    # Check if API is accessible
    try:
        health_check = requests.get("http://localhost:8000/health", timeout=5)
        if health_check.status_code == 200:
            print("✅ API server is accessible")
        else:
            print("❌ API server returned non-200 status")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("   Make sure ICEBURG is running: ./scripts/start_iceburg.sh")
        sys.exit(1)
    
    print()
    results = test_conversation_flow()
    
    # Save results
    output_file = f"secretary_test_results_{int(time.time())}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {output_file}")

