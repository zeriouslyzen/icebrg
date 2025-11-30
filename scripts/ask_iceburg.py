#!/usr/bin/env python3
"""
Simple script to query ICEBURG about itself
"""
import sys
sys.path.insert(0, 'src')
from iceburg.protocol.legacy.protocol_legacy import iceberg_protocol
import json

def ask_iceburg(question, fast=False):
    """Ask ICEBURG a question and return the response"""
    print(f"Query: {question}\n")
    print("Processing...\n")
    
    try:
        result = iceberg_protocol(question, fast=fast, verbose=False)
        
        # Extract the final output
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            # Try to get oracle output
            agent_outputs = result.get('agent_outputs', {})
            if 'oracle' in agent_outputs:
                oracle = agent_outputs['oracle']
                if isinstance(oracle, str):
                    try:
                        oracle_data = json.loads(oracle)
                        return oracle_data.get('one_sentence_summary', oracle)
                    except:
                        return oracle
                else:
                    return json.dumps(oracle, indent=2)
            elif 'final_output' in result:
                return result['final_output']
            else:
                return json.dumps(result, indent=2)
        else:
            return str(result)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        fast = "--fast" in sys.argv
        response = ask_iceburg(question, fast=fast)
        print("=" * 70)
        print("ICEBURG RESPONSE:")
        print("=" * 70)
        print(response)
    else:
        print("Usage: python ask_iceburg.py 'Your question here' [--fast]")
        print()
        print("Examples:")
        print("  python ask_iceburg.py 'What is ICEBURG?'")
        print("  python ask_iceburg.py 'How does ICEBURG understand itself?' --fast")
