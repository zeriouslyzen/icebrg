
import asyncio
import sys
import os
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from iceburg.search.web_search import get_web_search
from iceburg.search.search_answer_pipeline import answer_query

# Setup logging
logging.basicConfig(level=logging.INFO)

async def test_search():
    print("--- Testing Web Search Aggregator ---")
    search = get_web_search()
    query = "Why is the crypto market down today?"
    print(f"Query: {query}")
    
    results = search.search_for_current_events(query)
    print(f"Found {len(results)} results:")
    for i, res in enumerate(results[:3], 1):
        print(f"{i}. {res.title}")
        print(f"   URL: {res.url}")
        print(f"   Snippet: {res.snippet[:100]}...")
        print()

    print("--- Testing Answer Pipeline (Fallback Mode) ---")
    # This will use fallback mode since we don't pass an llm_client
    answer = answer_query(query)
    print("Result:")
    print(answer['answer'][:500] + "...")
    print("\nSources cited:", len(answer['sources']))

if __name__ == "__main__":
    asyncio.run(test_search())
