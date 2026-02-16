from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
import logging
import json
from pathlib import Path

# Initialize Router
router = APIRouter(prefix="/api/mobile", tags=["mobile"])
logger = logging.getLogger(__name__)

# Request Model
class RetrievalRequest(BaseModel):
    query: str
    limit: int = 3

# Helper to load Encyclopedia (cached)
_encyclopedia_cache = None
def get_encyclopedia_data():
    global _encyclopedia_cache
    if _encyclopedia_cache:
        return _encyclopedia_cache
    
    try:
        # Path relative to this file: .../src/iceburg/api/mobile_routes.py -> .../data/celestial_encyclopedia.json
        # Adjust as needed based on actual structure
        root_dir = Path(__file__).parent.parent.parent.parent
        data_path = root_dir / "data" / "celestial_encyclopedia.json"
        
        if data_path.exists():
            with open(data_path, 'r', encoding='utf-8') as f:
                _encyclopedia_cache = json.load(f)
            return _encyclopedia_cache
    except Exception as e:
        logger.error(f"Failed to load encyclopedia: {e}")
    return {}

@router.post("/context")
async def get_mobile_context(request: RetrievalRequest):
    """
    Retrieves context for the mobile agent.
    Combines 'Lost Knowledge' search and 'Encyclopedia' lookups.
    Returns a unified text block suitable for injection into LLM context.
    """
    context_parts = []
    
    # 1. Search Lost Knowledge (if available)
    try:
        from ..ingestion import HumanCuratedSubmission
        submitter = HumanCuratedSubmission()
        # Simple search
        results = submitter.search(query=request.query)
        # Take top N
        top_results = results[:request.limit]
        
        if top_results:
            context_parts.append(f"--- MAINFRAME SEARCH RESULTS ({len(top_results)}) ---")
            for i, res in enumerate(top_results):
                title = res.get('title', 'Unknown')
                desc = res.get('description', res.get('content', ''))[:200] + "..."
                context_parts.append(f"[{i+1}] {title}: {desc}")
    except ImportError:
        logger.warning("HumanCuratedSubmission module not found, skipping knowledge search")
    except Exception as e:
        logger.error(f"Error searching knowledge base: {e}")

    # 2. Search Encyclopedia (Simple substring match for now)
    enc_data = get_encyclopedia_data()
    if enc_data and "entries" in enc_data:
        matches = []
        q_lower = request.query.lower()
        
        for entry in enc_data["entries"]:
            if q_lower in entry.get("title", "").lower() or q_lower in entry.get("content", "").lower()[:100]:
                matches.append(entry)
                if len(matches) >= request.limit:
                    break
        
        if matches:
            context_parts.append(f"\n--- ENCYCLOPEDIA ENTRIES ({len(matches)}) ---")
            for entry in matches:
                title = entry.get("title", "Unknown")
                content = entry.get("content", "")[:200] + "..."
                context_parts.append(f"TITLE: {title}\nSUMMARY: {content}")

    # 3. Construct Final Context
    if not context_parts:
        return {
            "found": False,
            "context": "No relevant data found in local mainframe databases."
        }
    
    final_text = "\n".join(context_parts)
    return {
        "found": True,
        "context": final_text
    }
