from __future__ import annotations

import json
import os
from typing import Any

from ..config import IceburgConfig
from ..graph_store import KnowledgeGraph
from ..llm import chat_complete
from ..vectorstore import VectorStore

ORACLE_SYSTEM = (
    "ROLE: Meta-analyst who formulates principles weighted by evidence.\n"
    "TASK: Given evidence-weighted claims and prior principles, synthesize a single overarching principle reflecting certainty.\n"
    "CONSTRAINTS: If primary evidence is [A] => 'Conclusion', [B] => 'Theory', [C] => 'Hypothesis'. Explicitly mention varied evidence levels.\n"
    "OUTPUT_FORMAT: You MUST output valid JSON with these exact keys: principle_name, one_sentence_summary, derivation_logic, framing, domains, evidence_pairs, predictions, study_design, implications, prior_principles.\n"
    "CRITICAL: Your response must be ONLY valid JSON. No additional text before or after the JSON object.\n"
    "EXAMPLE FORMAT:\n"
    "{\n"
    '  "principle_name": "Example Principle",\n'
    '  "one_sentence_summary": "Brief summary",\n'
    '  "derivation_logic": "How derived",\n'
    '  "framing": "Conclusion|Theory|Hypothesis",\n'
    '  "domains": ["domain1", "domain2"],\n'
    '  "evidence_pairs": [["evidence1", "weight1"]],\n'
    '  "predictions": ["prediction1", "prediction2"],\n'
    '  "study_design": {"manipulation": "test", "measurement": "metric", "success_criteria": "criteria", "minimal_design_risk": "risk"},\n'
    '  "implications": ["implication1"],\n'
    '  "prior_principles": ["prior1"]\n'
    "}"
)


def _safe_parse_json(text: str) -> dict[str, Any]:
    try:
        # Clean the text first
        text = text.strip()
        
        # Try direct parsing first
        try:
            return json.loads(text)
        except:
            pass
        
        # Find JSON boundaries
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_text = text[start : end + 1]
            return json.loads(json_text)
        
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
                return json.loads(json_text)
        
        # Try to extract JSON from code blocks
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                json_text = text[start:end].strip()
                try:
                    return json.loads(json_text)
                except:
                    pass
        
        return {}
    except Exception:
        return {}


def _is_valid_schema(d: dict[str, Any]) -> bool:
    try:
        if not isinstance(d, dict):
            return False
        
        # More lenient validation - only require principle_name
        required = ["principle_name"]
        
        # Check if principle_name exists and is not empty
        principle_name = d.get("principle_name", "")
        if not principle_name or not str(principle_name).strip():
            return False
        
        # Optional fields with defaults
        if "one_sentence_summary" not in d:
            d["one_sentence_summary"] = str(principle_name)
        
        if "domains" not in d:
            d["domains"] = ["general"]
            
        if "evidence_pairs" not in d:
            d["evidence_pairs"] = []
            
        if "implications" not in d:
            d["implications"] = []
            
        if "prior_principles" not in d:
            d["prior_principles"] = []
        
        return True
    except Exception:
        return False


def _minimal_hypothesis(evidence_str: str) -> dict[str, Any]:
    return {
        "principle_name": "Hypothesis (Insufficient Evidence)",
        "one_sentence_summary": "Insufficient primary evidence; propose a conservative hypothesis and retrieval plan.",
        "derivation_logic": "No A/B-grade evidence available; applying truth-funnel fallback.",
        "framing": "Hypothesis",
        "domains": [],
        "evidence_pairs": [],
        "predictions": [],
        "study_design": {
            "manipulation": "N/A",
            "measurement": "N/A",
            "success_criteria": "N/A",
            "minimal_design_risk": "N/A",
        },
        "implications": [],
        "prior_principles": [],
    }


def run(
    cfg: IceburgConfig,
    kg: KnowledgeGraph,
    vs: VectorStore,
    evidence_weighted_input: str | dict[str, Any],
    verbose: bool = False,
) -> str:
    # Handle dict input by converting to string
    if isinstance(evidence_weighted_input, dict):
        evidence_str = json.dumps(evidence_weighted_input, indent=2)
    else:
        evidence_str = evidence_weighted_input

    tournament_enabled = os.getenv("ICEBURG_ORACLE_TOURNAMENT", "0") == "1"

    # Default prompt to ensure it is always defined for retries
    prompt = (
        "EVIDENCE INPUT (claims/prior):\n" + evidence_str + "\n\n"
        "Synthesize and output JSON per schema. Respect primary evidence for framing."
    )

    def _generate_once(prompt: str, tag: str) -> dict[str, Any]:
        raw_local = chat_complete(
            cfg.oracle_model,
            prompt,
            system=ORACLE_SYSTEM,
            temperature=0.2,
            options={"num_ctx": 4096, "num_predict": 900},
            context_tag=tag,
        )
        return _safe_parse_json(raw_local)

    data: dict[str, Any] = {}
    if tournament_enabled:
        gen_prompt = (
            "EVIDENCE INPUT (claims/prior):\n" + evidence_str + "\n\n"
            "Generate 3 candidate principles as a JSON object with key 'candidates' (list), each item following OUTPUT_FORMAT schema."
        )
        raw_gen = chat_complete(
            cfg.oracle_model,
            gen_prompt,
            system=ORACLE_SYSTEM,
            temperature=0.2,
            options={"num_ctx": 4096, "num_predict": 1200},
            context_tag="Oracle:generate",
        )
        gen_data = _safe_parse_json(raw_gen)
        candidates = gen_data.get("candidates") if isinstance(gen_data, dict) else None
        if isinstance(candidates, list) and candidates:
            crit_prompt = (
                "You are a rigorous reviewer. Score each candidate on: (1) framing correctness vs primary evidence, (2) clarity, (3) use of evidence_pairs.\n"
                "Return JSON with key 'scores' mapping index->score (0-100) and 'best_index'.\n\n"
                "PRIMARY INPUT:\n"
                + evidence_str
                + "\n\nCANDIDATES:\n"
                + json.dumps(candidates)
            )
            raw_crit = chat_complete(
                cfg.oracle_model,
                crit_prompt,
                system="Be strict and concise.",
                temperature=0.0,
                options={"num_ctx": 4096, "num_predict": 400},
                context_tag="Oracle:critique",
            )
            crit = _safe_parse_json(raw_crit)
            try:
                best_index = int(crit.get("best_index", 0))
            except Exception:
                best_index = 0
            try:
                data = candidates[best_index]
            except Exception:
                data = candidates[0]
        else:
            # Use default prompt defined above
            data = _generate_once(prompt, "Oracle")
    else:
        data = _generate_once(prompt, "Oracle")

    # Retry-and-validate loop
    max_retries = 2
    attempts = 0
    while not _is_valid_schema(data) and attempts < max_retries:
        attempts += 1
        if verbose:
            print(f"[ORACLE] Retry attempt {attempts}/{max_retries}")
        data = _generate_once(prompt, f"Oracle:retry:{attempts}")

    if not _is_valid_schema(data):
        # Fail-fast by default to avoid placeholder propagation
        if os.getenv("ICEBURG_FAIL_FAST", "1") == "1":
            raise RuntimeError("OracleFailFast: invalid schema after retries")
        # Fallback to minimal hypothesis
        try:
            data = _minimal_hypothesis(evidence_str)
        except Exception:
            # Ultimate fallback
            data = {"principle_name": "Unable to extract principle from evidence"}

    principle_name = str(data.get("principle_name") or "Emergent Principle").strip()
    summary = str(data.get("one_sentence_summary") or "").strip()
    domains = [str(d).strip() for d in (data.get("domains") or []) if str(d).strip()]
    evidence_pairs: list[tuple[str, str]] = []
    for pair in data.get("evidence_pairs") or []:
        try:
            a = pair.get("idea") or pair.get("prior_principle") or pair[0]
            b = pair.get("level") or pair.get("weight") or pair[1]
            evidence_pairs.append((str(a).strip(), str(b).strip()))
        except Exception:
            continue

    if principle_name:
        try:
            kg.add_synthesis(
                title=principle_name,
                domains=domains,
                principle=summary or principle_name,
                evidence=evidence_pairs,
            )
        except Exception:
            pass

    try:
        doc = f"Principle: {principle_name}\nSummary: {summary}\nDomains: {', '.join(domains)}"
        vs.add([doc], metadatas=[{"type": "principle", "name": principle_name}])
    except Exception:
        pass

    data['trade_signals'] = []

    return json.dumps(data)
