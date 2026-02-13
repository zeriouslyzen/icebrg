from __future__ import annotations
from typing import List, Optional
from functools import lru_cache
import ollama


@lru_cache(maxsize=1)
def _available_model_names() -> List[str]:
    try:
        listing = ollama.list()
        models = listing.get("models", [])
        names: List[str] = []
        for m in models:
            name = m.get("model") or m.get("name")
            if name:
                names.append(str(name))
        return names
    except Exception:
        return []


def _matches(available: str, candidate: str) -> bool:
    a = available.lower()
    c = candidate.lower()
    if a == c:
        return True
    if a.startswith(c) or c.startswith(a):
        return True
    # Match by family to allow e.g. "mistral:latest" for "mistral:7b-instruct"
    a_family = a.split(":")[0]
    c_family = c.split(":")[0]
    if a_family == c_family:
        return True
    # Ignore "-instruct" suffix for loose matching
    if a.replace("-instruct", "") == c.replace("-instruct", ""):
        return True
    return False


def _first_present(candidates: List[str], available: List[str]) -> Optional[str]:
    for c in candidates:
        for a in available:
            if _matches(a, c):
                return a
    return None


def resolve_models(
    surveyor_pref: str,
    dissident_pref: str,
    synthesist_pref: str,
    oracle_pref: str,
    embed_pref: str,
) -> tuple:
    available = _available_model_names()

    surveyor = surveyor_pref if (surveyor_pref in available or not available) else None
    dissident = dissident_pref if (dissident_pref in available or not available) else None
    synthesist = synthesist_pref if (synthesist_pref in available or not available) else None
    oracle = oracle_pref if (oracle_pref in available or not available) else None
    embed = embed_pref if (embed_pref in available or not available) else None

    if not surveyor:
        surveyor = _first_present([
            "deepseek-r1:7b",
            "dolphin-mistral:latest",
            "llama3.1:8b-instruct",
            "llama3:8b-instruct",
            "qwen2.5:7b",
            "mistral:7b-instruct",
            "llama3:latest",
            "mistral:latest",
        ], available) or surveyor_pref

    if not dissident:
        dissident = _first_present([
            "dolphin-mistral:latest",
            "dolphin-mistral:7b",
            "hermes:latest",
            "openhermes:latest",
            "wizardlm:latest",
            "mistral-openorca:latest",
            "deepseek-r1:7b",
            "mistral:7b-instruct",
            "mistral:7b",
            "mistral:latest",
            "llama3:8b-instruct",
            "llama3:latest",
            "qwen2.5:7b",
            "qwen2.5:7b-instruct",
            "qwen2.5:latest",
        ], available) or dissident_pref

    if not synthesist:
        synthesist = _first_present([
            "deepseek-r1:7b",
            "dolphin-mistral:latest",
            "hermes:latest",
            "qwen2.5:7b",
            "llama3.1:8b",
            "llama3:latest",
            "mistral:7b",
            "mixtral:latest",
        ], available) or synthesist_pref

    if not oracle:
        oracle = _first_present([
            "deepseek-r1:7b",
            "dolphin-mistral:latest",
            "hermes:latest",
            "qwen2.5:7b",
            "llama3.1:8b",
            "llama3:latest",
            "mistral:7b",
            "mixtral:latest",
        ], available) or oracle_pref

    if not embed:
        embed = _first_present([
            "nomic-embed-text",
            "snowflake-arctic-embed",
            "all-minilm",
        ], available) or embed_pref

    # Enforce per-role minimum model capacity floors for quality
    # Synthesist and Oracle should not use <8B unless absolutely no alternatives
    def _enforce_floor(current: str, floor_candidates: List[str]) -> str:
        if not available:
            return current
        # If the current model looks tiny/small, try to upgrade to any floor candidate present
        small_markers = [":1b", ":2b", ":3b", "1.5b", "2.7b"]
        cl = (current or "").lower()
        if any(m in cl for m in small_markers):
            upgraded = _first_present(floor_candidates, available)
            return upgraded or current
        return current

    floor8b = [
        "llama3.1:8b",
        "llama3:8b",
        "mistral:7b-instruct",
        "qwen2.5:7b-instruct",
    ]
    floor_big = [
        "llama3:70b-instruct",
        "qwen2.5:32b-instruct",
        "mixtral:latest",
        "llama3.1:8b",
    ]

    synthesist = _enforce_floor(synthesist, floor_big)
    oracle = _enforce_floor(oracle, floor_big)

    return surveyor, dissident, synthesist, oracle, embed


def resolve_models_small() -> tuple:
    available = _available_model_names()
    tiny = [
        "llama3.2:1b",
        "phi3.5",
        "gemma2:2b",
    ]
    small = [
        "deepseek-r1:7b",
        "llama3.1:8b",
        "llama3:8b",
        "llama3:latest",
        "qwen2.5:7b",
        "qwen2.5:latest",
        "mistral:7b",
        "mistral:latest",
        "deepseek-coder:6.7b",
        "phi3:mini",
    ]

    surveyor = _first_present(tiny + small, available) or (available[0] if available else "llama3.2:1b")
    dissident = _first_present(tiny + small, available) or surveyor
    synthesist = _first_present(tiny + small, available) or surveyor
    oracle = _first_present(tiny + small, available) or synthesist
    embed = _first_present([
        "nomic-embed-text",
        "snowflake-arctic-embed",
        "all-minilm",
    ], available) or "nomic-embed-text"

    return surveyor, dissident, synthesist, oracle, embed


def resolve_models_hybrid() -> tuple:
    available = _available_model_names()
    surveyor_prefer = [
        "deepseek-r1:7b",
        "llama3.1:8b",
        "llama3:8b",
        "qwen2.5:7b",
        "llama3:latest",
        "mistral:7b",
        "mistral:latest",
        "qwen2.5:latest",
        "phi3:mini",
        "phi3.5",
        "llama3.2:1b",
    ]
    dissident_prefer = [
        "deepseek-r1:7b",
        "llama3.1:8b",
        "llama3:8b",
        "mistral:7b",
        "mistral:latest",
        "qwen2.5:7b",
        "qwen2.5:latest",
        "phi3:mini",
        "phi3.5",
        "llama3.2:1b",
    ]
    big_candidates = [
        "deepseek-r1:7b",
        "qwen2.5:7b",
        "llama3.1:8b",
        "llama3:latest",
        "mistral:7b",
        "llama3:70b-instruct",
        "qwen2.5:32b-instruct",
        "mixtral:latest",
    ]
    surveyor = _first_present(surveyor_prefer, available) or (available[0] if available else "llama3.1:8b")
    dissident = _first_present(dissident_prefer, available) or surveyor
    synthesist = _first_present(big_candidates, available) or surveyor
    oracle = _first_present(big_candidates, available) or synthesist
    embed = _first_present([
        "nomic-embed-text",
        "snowflake-arctic-embed",
        "all-minilm",
    ], available) or "nomic-embed-text"
    return surveyor, dissident, synthesist, oracle, embed
