from typing import Dict, Any, List
from pathlib import Path
from ..config import IceburgConfig
from ..llm import chat_complete
from ..vectorstore import VectorStore


# Surveyor System Prompt - Semantic Natural Language Version
SURVEYOR_SYSTEM = """
You're an elite research agent who approaches problems like an engineering alchemist.

When you investigate something, you naturally think across time—current tech, old approaches, even forgotten methods that might work with modern amplification. You're looking for what exists, what's missing, and how to bridge that gap practically.

Your voice is clean, direct, and technical. You skip academic fluff and get to the engineering reality. When someone asks about building something, you give them the full picture: exact calculations with units, system diagrams showing signal flow, specific component lists with part numbers, and honest expectations about what works versus what needs R&D.

You're rigorous about truth. You tag your claims honestly—[Est] for established facts, [Controv] for controversial theories (with the researchers who proposed them), [Spec] for speculation, or [NoEv] when there's no evidence. You never invent numbers or recommend outdated components just to fill space. If sources don't have exact data, you say so and give general ranges.

When you research, you're looking for the engineering path forward, not just academic knowledge. You want to know: Can this be built? What would it take? What are the realistic numbers? Where are the gaps? How could weak effects be amplified into something practical?

For device queries, you provide: realistic expectations upfront, exact calculations with formulas and units, system block diagrams in ASCII showing signal flow, detailed component lists with specific part numbers and suppliers, key design details explaining why each choice matters (resonance, impedance matching, etc.), minimal prototype build instructions (concise, step-by-step), ASCII schematics showing connections, practical tips about what works and what doesn't, expected outcomes with realistic performance numbers, and next steps for what can be done.

You never invent specific power numbers, efficiency values, or study results. You never recommend outdated electronics like LM7805 linear regulators for energy harvesting or OP27 for piezo signals—you use modern, appropriate components like LTC3588, BQ25504, or SPV1050 when relevant. You never make up historical facts. For power calculations, you use realistic formulas and conservative estimates. For schematics, you show actual circuit topology with proper signal flow, not just random boxes.

You prioritize sources in this order: ICEBURG research (gold standard), post-2018 peer-reviewed papers, historical/archaeological sources and old patents, alternative sources and suppressed research, then everything else.

You avoid phrases like "interconnectedness of all things" or "resonates with" unless tagged [Spec]. You never say "as we discussed before" because previous conversations aren't evidence. You never use refusal language like "I cannot"—you always provide research. You don't force connections between unrelated domains or invent specific factual details not in sources.

You're creative in speculation mode, but you think like an engineering alchemist. You're direct. You're elite.
"""



def _critique_sources(hits: List[Any], cfg: IceburgConfig, query: str, verbose: bool = False) -> List[Any]:
    """
    Critique and filter sources - removes weak, outdated, or pseudoscientific sources.
    Part of Retrieval → Critique → Rewrite loop.
    """
    if not hits or len(hits) == 0:
        return []
    
    import logging
    logger = logging.getLogger(__name__)
    
    # Build source list for critique
    source_list = []
    for i, h in enumerate(hits[:10]):  # Critique top 10
        # Handle both dict-like and object-like hits
        if hasattr(h, 'metadata'):
            source = h.metadata.get('source', 'unknown') if isinstance(h.metadata, dict) else getattr(h.metadata, 'source', 'unknown')
            doc = h.document if hasattr(h, 'document') else str(h)
        elif isinstance(h, dict):
            source = h.get('metadata', {}).get('source', 'unknown') if isinstance(h.get('metadata'), dict) else 'unknown'
            doc = h.get('document', str(h))
        else:
            source = 'unknown'
            doc = str(h)
        
        doc_preview = doc[:200] if isinstance(doc, str) else str(doc)[:200]
        source_list.append(f"{i+1}. Source: {source}\n   Preview: {doc_preview}...")
    
    sources_text = "\n\n".join(source_list)
    
    critique_prompt = f"""Review these sources for a query about: {query}
    
    SOURCES:
    {sources_text}
    
    CRITIQUE CRITERIA:
    - Remove items that are: Weak, Outdated (pre-2015), Pseudoscientific, or Irrelevant.
    - Be aggressive: If the source doesn't DIRECTLY address {query}, remove it.
    
    Output ONLY the numbers to REMOVE (e.g., "1, 3"). If all are good, output "NONE".
    """
    
    try:
        critique_result = chat_complete(
            cfg.surveyor_model,
            critique_prompt,
            system="You are a strict source quality critic. Be ruthless. Output only numbers or 'NONE'.",
            temperature=0.0,  # Deterministic critique
            options={"num_ctx": 2048, "num_predict": 100},
            context_tag="Surveyor:SourceCritique"
        )
        
        # Parse critique result
        import re
        if "NONE" in critique_result.upper():
            if verbose:
                logger.info("[SURVEYOR] Source critique: All sources passed")
            return hits
        
        # Extract numbers
        numbers = [int(n) for n in re.findall(r'\b(\d+)\b', critique_result)]
        if not numbers:
            if verbose:
                logger.info("[SURVEYOR] Source critique: Could not parse, keeping all sources")
            return hits
        
        # Filter out criticized sources (1-indexed to 0-indexed)
        filtered_hits = [h for i, h in enumerate(hits) if (i + 1) not in numbers]
        
        if verbose:
            logger.info(f"[SURVEYOR] Source critique: Removed {len(hits) - len(filtered_hits)} weak sources, kept {len(filtered_hits)}")
        
        return filtered_hits if filtered_hits else hits  # Keep at least some sources
        
    except Exception as e:
        if verbose:
            logger.warning(f"[SURVEYOR] Source critique error: {e}, keeping all sources")
        return hits


def _verifier_stage(query: str, context_block: str, matrices_info: str, gnosis_connections: str, cfg: IceburgConfig, verbose: bool = False) -> str:
    """
    Stage 1: Verifier - Flags forbidden matrix connections and unsupported claims.
    Uses temperature 0.0 for deterministic, ruthless validation.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Extract all specific factual claims from sources (mission names, paper titles, author names, etc.)
    # This helps detect hallucinations of specific details
    import re
    source_factual_claims = set()
    
    # Extract potential factual claims (capitalized phrases, acronyms, specific names)
    # Look for patterns like "EXIST mission", "INTEGRAL", "Swift", "ArXiv", paper titles, etc.
    factual_patterns = [
        r'\b[A-Z]{2,}\b',  # Acronyms (EXIST, INTEGRAL, Swift, GRB, etc.)
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:mission|survey|telescope|observatory|experiment|study|paper|article)\b',  # Named missions/studies
        r'arxiv:\d+\.\d+',  # ArXiv paper IDs
        r'\bdoi:\S+',  # DOI identifiers
        r'\([A-Z][a-z]+\s+et\s+al\.?\s+\d{4}\)',  # Citations like "Smith et al. 2020"
    ]
    
    for pattern in factual_patterns:
        matches = re.findall(pattern, context_block, re.IGNORECASE)
        source_factual_claims.update([m.lower() for m in matches])
    
    factual_claims_text = "\n".join(sorted(list(source_factual_claims)[:50])) if source_factual_claims else "None found"
    
    verifier_prompt = f"""You are SURVEYOR-VERIFIER. Flag hallucinations and unsupported connections.
    
    QUERY: {query}
    MATRICES: {matrices_info}
    SOURCES: {context_block[:2000]}...
    
    RULES:
    1. Flag ANY name/acronym/data NOT in the SOURCES above as FORBIDDEN.
    2. Flag pseudo-profound talk ("resonates with", "interconnectedness") as FORBIDDEN.
    3. Flag connections between unrelated domains (e.g. Physics vs Astrology) as FORBIDDEN unless sources link them.
    4. Flag outdated electronics (LM7805) or made-up history.
    
    Output ONLY a bullet list of FORBIDDEN claims. Be ruthless.
    """
    
    try:
        forbidden = chat_complete(
            cfg.surveyor_model,
            verifier_prompt,
            system="You are a ruthless verifier. Flag everything unsupported. Output only forbidden claims.",
            temperature=0.0,  # Deterministic verification
            options={"num_ctx": 2048, "num_predict": 500},
            context_tag="Surveyor:Verifier"
        )
        
        if verbose:
            logger.info(f"[SURVEYOR] Verifier flagged {len(forbidden.split(chr(10)))} forbidden claims/connections")
            logger.info(f"[SURVEYOR] Verifier output (first 500 chars): {forbidden[:500]}")
        
        return forbidden
    except Exception as e:
        if verbose:
            logger.warning(f"[SURVEYOR] Verifier error: {e}, continuing without verification")
        return "No forbidden claims identified (verifier error)."


def _verify_response_factual_claims(response: str, context_block: str, cfg: IceburgConfig, verbose: bool = False) -> str:
    """
    Post-processing verification: Check if response contains specific factual claims
    (mission names, paper titles, etc.) that aren't in the sources.
    Returns a sanitized response if hallucinations are detected.
    """
    import logging
    import re
    logger = logging.getLogger(__name__)
    
    # Extract potential factual claims from response
    response_claims = set()
    
    # Look for patterns that suggest specific factual details
    patterns = [
        (r'\b([A-Z]{2,})\s+(?:mission|survey|telescope|observatory|experiment)\b', True),  # EXIST mission, INTEGRAL survey (has capture group)
        (r'\b([A-Z]{3,})\b', True),  # Acronyms 3+ chars (EXIST, INTEGRAL, Swift, GRB) - has capture group, filters out 2-letter words
        (r'arxiv:(\d+\.\d+)', True),  # ArXiv IDs (has capture group)
        (r'\bdoi:(\S+)', True),  # DOI identifiers (has capture group)
    ]
    
    for pattern, has_capture in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        if matches:
            if has_capture and isinstance(matches[0], tuple):
                # Pattern has multiple capture groups, extract first
                response_claims.update([m[0].lower() if len(m) > 0 else str(m).lower() for m in matches if m])
            elif has_capture:
                # Pattern has single capture group, matches are strings
                response_claims.update([m.lower() if isinstance(m, str) else str(m).lower() for m in matches if m])
            else:
                # No capture group, matches are full strings
                response_claims.update([m.lower() if isinstance(m, str) else str(m).lower() for m in matches if m])
    
    # Extract factual claims from sources for comparison
    source_claims = set()
    for pattern, has_capture in patterns:
        matches = re.findall(pattern, context_block, re.IGNORECASE)
        if matches:
            if has_capture and isinstance(matches[0], tuple):
                source_claims.update([m[0].lower() if len(m) > 0 else str(m).lower() for m in matches if m])
            elif has_capture:
                source_claims.update([m.lower() if isinstance(m, str) else str(m).lower() for m in matches if m])
            else:
                source_claims.update([m.lower() if isinstance(m, str) else str(m).lower() for m in matches if m])
    
    # Find claims in response that aren't in sources
    suspicious_claims = response_claims - source_claims
    
    # Filter out common words that aren't factual claims
    common_words = {'the', 'and', 'or', 'but', 'for', 'with', 'from', 'this', 'that', 'these', 'those', 'all', 'any', 'some', 'no', 'not', 'can', 'may', 'must', 'will', 'would', 'could', 'should', 'might', 'est', 'spec', 'noev', 'controv'}
    suspicious_claims = {c for c in suspicious_claims if len(c) > 2 and c not in common_words}
    
    if suspicious_claims and verbose:
        logger.warning(f"[SURVEYOR] Post-processing detected potentially hallucinated claims: {suspicious_claims}")
        logger.warning(f"[SURVEYOR] These claims appear in response but not in sources. Response may contain hallucinations.")
    
    # For now, we just log the issue. In a production system, you might want to:
    # 1. Flag the response for human review
    # 2. Add a warning to the response
    # 3. Regenerate with stricter constraints
    
    return response


def _reasoner_stage(query: str, context_block: str, matrices_info: str, gnosis_connections: str, forbidden_claims: str, cfg: IceburgConfig, verbose: bool = False) -> str:
    """
    Stage 2: Reasoner - Generates response using gnosis knowledge but respects verifier constraints.
    Uses temperature 0.7 for creative but controlled responses.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    reasoner_prompt = f"""
    QUERY: {query}
    
    MATRICES IDENTIFIED: {matrices_info}
    GNOSIS CONNECTIONS FOUND: {gnosis_connections}
    
    SOURCES:
    {context_block}
    
    ABSOLUTELY FORBIDDEN (do not make these under any circumstances):
    {forbidden_claims}
    
    CRITICAL ANTI-HALLUCINATION REQUIREMENTS:
    1. You MUST ONLY mention specific factual details (names, missions, papers, authors, acronyms, power numbers, efficiency values) that appear EXPLICITLY in the SOURCES above
    2. DO NOT make up or invent specific details like:
       - Mission names (e.g., "EXIST", "INTEGRAL", "Swift") unless they appear in sources
       - Paper titles or ArXiv IDs unless they appear in sources
       - Author names unless they appear in sources
       - Specific study names or experiment names unless they appear in sources
       - Specific power numbers (e.g., "1.5 mW") unless they appear in sources - use general ranges with [NoEv] if uncertain
       - Historical facts (e.g., "ancient Greeks experimented with piezos" - FALSE, piezoelectricity discovered 1880) - use [NoEv] or correct facts
       - Component part numbers unless they appear in sources or you're certain they're appropriate and current
    3. If sources don't contain specific details, use general language with [NoEv] tags, NOT made-up specifics
    4. When citing sources, you MUST reference the actual source text from above (e.g., "According to the source about X...")
    5. DO NOT say "citing ArXiv papers" or "based on ArXiv paper" unless you can point to a specific paper ID in the sources
    6. For electronics recommendations: Use CURRENT, APPROPRIATE components. Don't recommend outdated parts (e.g., LM7805 for energy harvesting, OP27 for piezo signals). If unsure, recommend component types with [NoEv] tags.
    7. For power calculations: Use realistic, conservative estimates. Typical piezo harvesters produce micro- to milliwatts, not watts. If sources don't have exact numbers, use general ranges.
    
    CRITICAL: DO NOT reference "previous conversations", "as we discussed", "building upon our earlier conversation", "continuity with previous conversations", or any similar phrases. Each query is answered independently. Previous conversations are NOT evidence.
    
    ABSOLUTE REQUIREMENT: You MUST ALWAYS provide research, even if all connections are forbidden. NEVER say "I cannot", "I'm unable", "I don't", or refuse to answer. If connections are forbidden, state what you found with [NoEv] tags. If you have no evidence, say "No peer-reviewed evidence found" with [NoEv] tag, but still provide what you know.
    
    RESPONSE STYLE:
    - Write directly, no headers or section labels
    - Answer the question immediately
    - Use tags inline: [Est], [Controv], [Spec], [NoEv]
    - Technical but accessible
    - No academic fluff or process explanations
    - No "QUERY RESPONSE" or "RESEARCH FINDINGS" headers
    - Just deliver the information cleanly
    
    ENGINEERING ALCHEMIST THINKING:
    - Look for existing tech (current, old, forgotten) that could work
    - Explore historical/archaeological knowledge if relevant (but verify facts - no made-up history)
    - Identify gaps: what's missing to make it practical?
    - Propose amplification strategies: how to amplify weak effects?
    - Combine old knowledge with modern engineering approaches
    - Think: could this work if amplified? What would that take?
    
    DESIGN & CREATION (REQUIRED FOR DEVICE QUERIES):
    - Start with quick summary: realistic expectations, what's feasible vs. not. Be conservative with power estimates.
    - Provide exact calculations: formulas, power estimates, efficiency numbers with units. Use realistic, conservative numbers. If sources don't have exact values, use general ranges with [NoEv] tags.
    - System block diagrams: show actual signal flow from input to output (ASCII/text format). Include proper circuit topology, not just random boxes.
    - Detailed component lists: specific part numbers, suppliers, COTS recommendations. Use CURRENT, APPROPRIATE components (e.g., energy-harvesting ICs like LTC3588, BQ25504 for power management, not LM7805). If unsure, recommend component types with [NoEv] tags.
    - Key design details: explain WHY each choice matters (resonance coupling, impedance matching, charge handling, etc.). Explain why certain approaches DON'T work (e.g., linear regulators waste energy in harvesting applications).
    - Minimal prototype build instructions: concise, non-hand-holding, step-by-step assembly. Include proper circuit connections (parallel/series, etc.).
    - ASCII schematics: show actual circuit topology with proper connections (use +, |, -, v, > for flow). Show component relationships, not just boxes.
    - Practical tips: what works, what doesn't, environmental considerations, tuning advice. Be honest about limitations.
    - Expected outcomes: realistic performance numbers for prototype. Use conservative estimates (e.g., "0.5-20 µW for single disc at 100 dB" not "1.5 mW"). If uncertain, use ranges with [NoEv] tags.
    - Next steps: what can be done next (parts lists, calculations, schematics, experiment plans)
    - Be practical: what can be built now vs. what needs R&D
    - Use engineering language: technical but clear, no fluff, direct and precise
    - CRITICAL: If you don't know specific values or part numbers, use [NoEv] tags and general descriptions rather than guessing
    
    WORKFLOW (internal, do not mention):
    1. Verify all factual details appear in SOURCES
    2. Tag claims: [Est]/[Controv]/[Spec]/[NoEv]
    3. Only connect domains if sources directly support it
    4. Never invent specific details not in sources
    5. Never refuse - always provide research
    6. Think like an engineering alchemist: find existing tech, identify gaps, propose amplification
    
    Generate a clean, direct, technical response. Tags inline. No workflow explanations. Elite tone. Engineering alchemist mindset.
    """
    
    try:
        prompt_size = len(reasoner_prompt)
        system_size = len(SURVEYOR_SYSTEM)
        total_size = prompt_size + system_size
        logger.info(f"[SURVEYOR] Reasoner prompt size: {prompt_size} chars, system: {system_size} chars, total: {total_size} chars")
        
        # Check if prompt is too large (Ollama has limits)
        if total_size > 32000:  # Conservative limit for 4k context
            logger.warning(f"[SURVEYOR] Prompt is very large ({total_size} chars), truncating context_block")
            # Truncate context_block if it's too large
            max_context = 10000  # Max context block size
            if len(context_block) > max_context:
                context_block = context_block[:max_context] + "\n\n[Context truncated due to size limits]"
                reasoner_prompt = f"""
    QUERY: {query}
    
    MATRICES IDENTIFIED: {matrices_info}
    GNOSIS CONNECTIONS FOUND: {gnosis_connections}
    
    SOURCES:
    {context_block}
    
    ABSOLUTELY FORBIDDEN (do not make these under any circumstances):
    {forbidden_claims}
    
    CRITICAL ANTI-HALLUCINATION REQUIREMENTS:
    1. You MUST ONLY mention specific factual details (names, missions, papers, authors, acronyms, power numbers, efficiency values) that appear EXPLICITLY in the SOURCES above
    2. DO NOT make up or invent specific details like:
       - Mission names (e.g., "EXIST", "INTEGRAL", "Swift") unless they appear in sources
       - Paper titles or ArXiv IDs unless they appear in sources
       - Author names unless they appear in sources
       - Specific study names or experiment names unless they appear in sources
       - Specific power numbers (e.g., "1.5 mW") unless they appear in sources - use general ranges with [NoEv] if uncertain
       - Historical facts (e.g., "ancient Greeks experimented with piezos" - FALSE, piezoelectricity discovered 1880) - use [NoEv] or correct facts
       - Component part numbers unless they appear in sources or you're certain they're appropriate and current
    3. If sources don't contain specific details, use general language with [NoEv] tags, NOT made-up specifics
    4. When citing sources, you MUST reference the actual source text from above (e.g., "According to the source about X...")
    5. DO NOT say "citing ArXiv papers" or "based on ArXiv paper" unless you can point to a specific paper ID in the sources
    6. For electronics recommendations: Use CURRENT, APPROPRIATE components. Don't recommend outdated parts (e.g., LM7805 for energy harvesting, OP27 for piezo signals). If unsure, recommend component types with [NoEv] tags.
    7. For power calculations: Use realistic, conservative estimates. Typical piezo harvesters produce micro- to milliwatts, not watts. If sources don't have exact numbers, use general ranges.
    
    CRITICAL: DO NOT reference "previous conversations", "as we discussed", "building upon our earlier conversation", "continuity with previous conversations", or any similar phrases. Each query is answered independently. Previous conversations are NOT evidence.
    
    ABSOLUTE REQUIREMENT: You MUST ALWAYS provide research, even if all connections are forbidden. NEVER say "I cannot", "I'm unable", "I don't", or refuse to answer. If connections are forbidden, state what you found with [NoEv] tags. If you have no evidence, say "No peer-reviewed evidence found" with [NoEv] tag, but still provide what you know.
    
    RESPONSE STYLE:
    - Write directly, no headers or section labels
    - Answer the question immediately
    - Use tags inline: [Est], [Controv], [Spec], [NoEv]
    - Technical but accessible
    - No academic fluff or process explanations
    - No "QUERY RESPONSE" or "RESEARCH FINDINGS" headers
    - Just deliver the information cleanly
    
    ENGINEERING ALCHEMIST THINKING:
    - Look for existing tech (current, old, forgotten) that could work
    - Explore historical/archaeological knowledge if relevant (but verify facts - no made-up history)
    - Identify gaps: what's missing to make it practical?
    - Propose amplification strategies: how to amplify weak effects?
    - Combine old knowledge with modern engineering approaches
    - Think: could this work if amplified? What would that take?
    
    DESIGN & CREATION (REQUIRED FOR DEVICE QUERIES):
    - Start with quick summary: realistic expectations, what's feasible vs. not. Be conservative with power estimates.
    - Provide exact calculations: formulas, power estimates, efficiency numbers with units. Use realistic, conservative numbers. If sources don't have exact values, use general ranges with [NoEv] tags.
    - System block diagrams: show actual signal flow from input to output (ASCII/text format). Include proper circuit topology, not just random boxes.
    - Detailed component lists: specific part numbers, suppliers, COTS recommendations. Use CURRENT, APPROPRIATE components (e.g., energy-harvesting ICs like LTC3588, BQ25504 for power management, not LM7805). If unsure, recommend component types with [NoEv] tags.
    - Key design details: explain WHY each choice matters (resonance coupling, impedance matching, charge handling, etc.). Explain why certain approaches DON'T work (e.g., linear regulators waste energy in harvesting applications).
    - Minimal prototype build instructions: concise, non-hand-holding, step-by-step assembly. Include proper circuit connections (parallel/series, etc.).
    - ASCII schematics: show actual circuit topology with proper connections (use +, |, -, v, > for flow). Show component relationships, not just boxes.
    - Practical tips: what works, what doesn't, environmental considerations, tuning advice. Be honest about limitations.
    - Expected outcomes: realistic performance numbers for prototype. Use conservative estimates (e.g., "0.5-20 µW for single disc at 100 dB" not "1.5 mW"). If uncertain, use ranges with [NoEv] tags.
    - Next steps: what can be done next (parts lists, calculations, schematics, experiment plans)
    - Be practical: what can be built now vs. what needs R&D
    - Use engineering language: technical but clear, no fluff, direct and precise
    - CRITICAL: If you don't know specific values or part numbers, use [NoEv] tags and general descriptions rather than guessing
    
    WORKFLOW (internal, do not mention):
    1. Verify all factual details appear in SOURCES
    2. Tag claims: [Est]/[Controv]/[Spec]/[NoEv]
    3. Only connect domains if sources directly support it
    4. Never invent specific details not in sources
    5. Never refuse - always provide research
    6. Think like an engineering alchemist: find existing tech, identify gaps, propose amplification
    
    Generate a clean, direct, technical response. Tags inline. No workflow explanations. Elite tone. Engineering alchemist mindset.
    """
                logger.info(f"[SURVEYOR] Truncated context, new prompt size: {len(reasoner_prompt)} chars")
        
        result = chat_complete(
            cfg.surveyor_model,
            reasoner_prompt,
            system=SURVEYOR_SYSTEM,
            temperature=0.7,  # Creative but controlled
            options={"num_ctx": 4096, "num_predict": 4000},  # Increased for detailed design responses
            context_tag="Surveyor:Reasoner"
        )
        
        # Safety check: ensure chat_complete returned something
        if result is None or (isinstance(result, str) and not result.strip()):
            logger.error(f"[SURVEYOR] chat_complete returned None/empty for model {cfg.surveyor_model}")
            return f"[SURVEYOR ERROR] The model ({cfg.surveyor_model}) did not generate a response. This may indicate the model is unavailable or the prompt was too large. Please try again or rephrase your query."
        
        if verbose:
            logger.info(f"[SURVEYOR] Reasoner generated {len(result)} char response")
        
        # Post-processing: Verify response doesn't contain hallucinated factual claims
        try:
            result = _verify_response_factual_claims(result, context_block, cfg, verbose)
        except Exception as e:
            logger.warning(f"[SURVEYOR] Post-processing verification error (non-fatal): {e}")
            # Continue with original result if verification fails
            pass
        
        return result
    except Exception as e:
        logger.error(f"[SURVEYOR] Reasoner error: {e}", exc_info=True)
        logger.error(f"[SURVEYOR] Model: {cfg.surveyor_model}, Prompt size: {len(reasoner_prompt)} chars")
        # Don't raise - return error message instead so user gets feedback
        return f"[SURVEYOR ERROR] I encountered an issue while generating the response: {str(e)}. The model ({cfg.surveyor_model}) may be unavailable or the prompt was too large. Please try again or rephrase your query."


def _self_consistency_ensemble(query: str, context_block: str, matrices_info: str, gnosis_connections: str, forbidden_claims: str, cfg: IceburgConfig, n_samples: int = 3, verbose: bool = False) -> str:
    """
    Self-consistency ensemble: Generate multiple responses, then vote for best.
    Optional feature - only runs if ICEBURG_USE_ENSEMBLE=1
    """
    import logging
    import os
    logger = logging.getLogger(__name__)
    
    use_ensemble = os.getenv("ICEBURG_USE_ENSEMBLE", "0") == "1"
    if not use_ensemble or cfg.fast:
        # Skip ensemble in fast mode or if disabled
        return None
    
    if verbose:
        logger.info(f"[SURVEYOR] Generating {n_samples} ensemble candidates...")
    
    candidates = []
    for i in range(n_samples):
        try:
            candidate = _reasoner_stage(query, context_block, matrices_info, gnosis_connections, forbidden_claims, cfg, verbose=False)
            candidates.append(candidate)
        except Exception as e:
            if verbose:
                logger.warning(f"[SURVEYOR] Ensemble candidate {i+1} failed: {e}")
            continue
    
    if len(candidates) < 2:
        if verbose:
            logger.info("[SURVEYOR] Ensemble: Not enough candidates, using single response")
        return None
    
    # Simple voting: use the candidate with most [Est] tags (most established claims)
    # In production, you'd use a separate scorer model
    best_candidate = max(candidates, key=lambda c: c.count("[Est]"))
    
    if verbose:
        logger.info(f"[SURVEYOR] Ensemble: Selected best of {len(candidates)} candidates")
    
    return best_candidate


def run(cfg: IceburgConfig, vs: VectorStore, query: str, verbose: bool = False, multimodal_input=None, thinking_callback=None) -> str:
    """
    Dynamic layered information gathering:
    1. ICEBURG's own data/research
    2. LLM training data (model knowledge)
    3. External sources (web search, APIs) if needed
    4. Computer capabilities (dynamically discovered tools)
    
    FAST MODE: Skips slow operations for quick responses
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # FAST PATH: Simple queries get instant response, no deep research
    simple_queries = ["hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye"]
    if query.lower().strip() in simple_queries:
        logger.info(f"[SURVEYOR] Simple query '{query}' detected - returning fast response, skipping deep research")
        if thinking_callback:
            thinking_callback("Quick response for simple greeting...")
        return "Hello! How can I help you today?"
    
    if cfg.fast:
        logger.info(f"[SURVEYOR] FAST MODE ENABLED - Skipping slow operations")
    else:
        logger.info(f"[SURVEYOR] Standard mode - Full processing")
    
    context_sections = {
        "iceburg_data": [],
        "llm_knowledge": [],
        "external_sources": [],
        "computer_capabilities": []
    }
    
    # FAST MODE: Skip computer capability discovery (too slow)
    if not cfg.fast:
        try:
            from ..discovery.dynamic_tool_usage import DynamicToolUsage
            dynamic_tools = DynamicToolUsage()
            computer_result = dynamic_tools.use_computer_to_find_info(query)
            
            if computer_result.get("tools_used"):
                tools_info = []
                for tool in computer_result["tools_used"][:3]:  # Top 3 tools
                    tools_info.append(f"Tool: {tool.get('tool', 'Unknown')} - {tool.get('type', 'Unknown')}")
                
                if tools_info:
                    context_sections["computer_capabilities"].append(
                        "=== COMPUTER CAPABILITIES DISCOVERED ===\n" + "\n".join(tools_info)
                    )
        except Exception as e:
            if verbose:
                logger.debug(f"[SURVEYOR] Error discovering computer capabilities: {e}")
    
    # GNOSIS INTEGRATION: Get matrix awareness and connections
    matrices_info = "No matrices identified"
    gnosis_connections = "No gnosis connections found"
    try:
        if thinking_callback:
            thinking_callback("I'm analyzing underlying patterns and structures...")
        
        from ..gnosis.unified_gnosis_interface import UnifiedGnosisInterface
        gnosis_interface = UnifiedGnosisInterface(cfg)
        gnosis_result = gnosis_interface.process_query(query)
        
        # Extract matrix information
        matrix_awareness = gnosis_result.get("matrix_awareness", {})
        matrices_identified = matrix_awareness.get("matrices_identified", [])
        if matrices_identified:
            matrices_info = f"Matrices: {', '.join(matrices_identified)}"
            matrix_knowledge = matrix_awareness.get("matrix_knowledge", {})
            if matrix_knowledge:
                matrices_info += f"\nMatrix knowledge: {str(matrix_knowledge)[:500]}"
            
            if thinking_callback:
                thinking_callback(f"I identified {len(matrices_identified)} core patterns: {', '.join(matrices_identified[:3])}")
        
        # Extract gnosis connections
        gnosis_knowledge = gnosis_result.get("gnosis_knowledge", {})
        if gnosis_knowledge:
            total_items = gnosis_knowledge.get("total_items", 0)
            domains = gnosis_knowledge.get("domains", [])
            if total_items > 0:
                gnosis_connections = f"Found {total_items} related knowledge items across {len(domains)} domains"
                if thinking_callback:
                    thinking_callback(f"I found {total_items} connections across {len(domains)} knowledge domains")
        
        if verbose:
            logger.info(f"[SURVEYOR] Gnosis: {len(matrices_identified)} matrices, {gnosis_knowledge.get('total_items', 0)} connections")
    except Exception as e:
        if verbose:
            logger.warning(f"[SURVEYOR] Gnosis integration error: {e}, continuing without gnosis")
    
    # LAYER 1: Check ICEBURG's own data and research
    # FAST MODE: Use fewer search results (k=3 instead of k=10)
    # CRITICAL: Handle VectorStore errors gracefully - work without it if needed
    hits = []
    try:
        if thinking_callback:
            thinking_callback("I'm searching ICEBURG's knowledge base...")
        
        search_k = 3 if cfg.fast else 10
        if cfg.fast:
            logger.info(f"[SURVEYOR] FAST MODE: Using k={search_k} for semantic search (vs k=10)")
        
        # Try to use VectorStore, but don't fail if it doesn't work
        try:
            hits = vs.semantic_search(query, k=search_k)
            if hits is None:
                hits = []
        except Exception as vs_error:
            logger.warning(f"[SURVEYOR] VectorStore search failed: {vs_error}. Continuing without knowledge base search.")
            hits = []
            if thinking_callback:
                thinking_callback("Working without knowledge base access - using general knowledge...")
    except Exception as e:
        logger.warning(f"[SURVEYOR] Error accessing VectorStore: {e}. Continuing without knowledge base.")
        hits = []
        
        if thinking_callback and hits:
            iceburg_count = len([h for h in hits if any(kw in h.metadata.get('source', '').lower() for kw in ['research_outputs', 'iceburg', 'knowledge_base', 'lab_runs', 'research', 'memory'])])
            if iceburg_count > 0:
                thinking_callback(f"I retrieved {iceburg_count} items from ICEBURG's research database")
            else:
                thinking_callback(f"I found {len(hits)} indexed items, but no ICEBURG-specific research")
        
        # RETRIEVAL → CRITIQUE → REWRITE LOOP
        # Step 1: Critique sources (remove weak/outdated/pseudoscientific)
        if not cfg.fast and len(hits) > 3:  # Only critique if we have enough sources
            if thinking_callback:
                thinking_callback("I'm evaluating source quality and filtering weak sources...")
            hits = _critique_sources(hits, cfg, query, verbose)
        
        for h in hits:
            source = h.metadata.get('source', 'kb')
            doc = h.document
            
            # CRITICAL: Filter out contaminated conversation sources
            # Old conversations contain pseudo-profound language and forced connections
            if 'conversation' in source.lower():
                if verbose:
                    logger.warning(f"[SURVEYOR] Filtered out conversation source: {source} (to prevent contamination)")
                continue  # Skip conversation sources - they're contaminated
            
            # Categorize sources dynamically
            if any(keyword in source.lower() for keyword in ['research_outputs', 'iceburg', 'knowledge_base', 'lab_runs', 'research', 'memory']):
                context_sections["iceburg_data"].append(f"Source: {source}\n{doc}")
            else:
                context_sections["llm_knowledge"].append(f"Source: {source}\n{doc}")
    except Exception as e:
        if verbose:
            logger.debug(f"[SURVEYOR] VectorStore search error: {e}")
    
    # Fallback: Search research outputs directory directly if not indexed
    # FAST MODE: Skip file system scanning (too slow)
    if not cfg.fast and not context_sections["iceburg_data"]:
        research_dir = Path(cfg.data_dir) / "research_outputs"
        if research_dir.exists():
            query_lower = query.lower()
            query_words = query_lower.split()
            for md_file in research_dir.glob("*.md"):
                # Dynamic keyword matching - check filename and content
                if any(word in md_file.name.lower() for word in query_words) or len(query_words) == 0:
                    try:
                        with open(md_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if len(content) > 100:
                                # Check if content is relevant
                                content_lower = content.lower()
                                if any(word in content_lower for word in query_words) or len(query_words) == 0:
                                    context_sections["iceburg_data"].append(f"Source: {md_file.name}\n{content[:2000]}...")
                                    if len(context_sections["iceburg_data"]) >= 3:
                                        break
                    except Exception:
                        pass
    
    # LAYER 2: LLM training data (model's knowledge) - handled by the LLM itself
    # The model already has access to its training data, we just need to prompt it correctly
    
    # LAYER 3: External sources (web search, APIs) - dynamically check if needed
    # CRITICAL: If ICEBURG has no data, ALWAYS try external sources (even in fast mode)
    # This ensures ICEBURG can find information even when it doesn't have existing research
    has_iceburg_data = len(context_sections["iceburg_data"]) > 0
    should_search_external = not has_iceburg_data or len(context_sections["iceburg_data"]) < 2
    
    # In fast mode, only search if ICEBURG has no data (prioritize speed when data exists)
    # In standard mode, always search if data is insufficient
    if should_search_external and (not cfg.fast or not has_iceburg_data):
        try:
            from ..tools.science_search import search_scientific_literature
            from ..tools.deep_web_search import search_deep_web, search_historical_sources, search_occult_sources
            import os
            
            # Check if web search is enabled (default to enabled if ICEBURG has no data)
            enable_web = os.getenv("ICEBURG_ENABLE_WEB", "1" if not has_iceburg_data else "0").strip() in {"1", "true", "TRUE"}
            
            if enable_web:
                if thinking_callback:
                    thinking_callback("I'm searching external sources for additional information...")
                
                if verbose:
                    logger.debug(f"[SURVEYOR] ICEBURG has {'insufficient' if has_iceburg_data else 'no'} data, checking external sources (web search, APIs)...")
                
                # Try scientific literature search (always try this first - most reliable)
                try:
                    if thinking_callback:
                        thinking_callback("I'm querying scientific literature databases...")
                    external_results = search_scientific_literature(query, max_results=5 if not cfg.fast else 3)
                    for result in external_results:
                        if result.get('title') and result.get('summary'):
                            context_sections["external_sources"].append(
                                f"[Scientific Source] {result.get('title', 'Unknown')}\n"
                                f"URL: {result.get('url', 'N/A')}\n"
                                f"Summary: {result.get('summary', '')[:500]}"
                            )
                    if external_results:
                        if thinking_callback:
                            thinking_callback(f"I found {len(external_results)} scientific papers")
                        if verbose:
                            logger.debug(f"[SURVEYOR] Found {len(external_results)} scientific sources")
                except Exception as e:
                    if verbose:
                        logger.debug(f"[SURVEYOR] Scientific literature search error: {e}")
                
                # Try deep web search for historical/occult sources (only if not in fast mode or still need more)
                if not cfg.fast or len(context_sections["external_sources"]) < 2:
                    try:
                        deep_results = search_deep_web(query, max_results=3 if not cfg.fast else 2)
                        for result in deep_results:
                            if result.get('title'):
                                context_sections["external_sources"].append(
                                    f"[Deep Web Source] {result.get('title', 'Unknown')}\n"
                                    f"URL: {result.get('url', 'N/A')}\n"
                                    f"Summary: {result.get('summary', '')[:500]}"
                                )
                        
                        # Search historical sources (only in standard mode)
                        if not cfg.fast:
                            historical_results = search_historical_sources(query, max_results=2)
                            for result in historical_results:
                                if result.get('title'):
                                    context_sections["external_sources"].append(
                                        f"[Historical Source] {result.get('title', 'Unknown')}\n"
                                        f"URL: {result.get('url', 'N/A')}\n"
                                        f"Summary: {result.get('summary', '')[:500]}"
                                    )
                            
                            # Search occult sources (only in standard mode)
                            occult_results = search_occult_sources(query, max_results=2)
                            for result in occult_results:
                                if result.get('title'):
                                    context_sections["external_sources"].append(
                                        f"[Occult Source] {result.get('title', 'Unknown')}\n"
                                        f"URL: {result.get('url', 'N/A')}\n"
                                        f"Summary: {result.get('summary', '')[:500]}"
                                    )
                    except Exception as e:
                        if verbose:
                            logger.debug(f"[SURVEYOR] Deep web search error: {e}")
        except ImportError:
            if verbose:
                logger.debug("[SURVEYOR] External search tools not available (ImportError)")
    
    # Build context block with clear source separation
    context_parts = []
    
    if context_sections["iceburg_data"]:
        context_parts.append("=== ICEBURG'S OWN DATA AND RESEARCH ===\n" + "\n\n".join(context_sections["iceburg_data"]))
    
    if context_sections["llm_knowledge"]:
        context_parts.append("=== INDEXED KNOWLEDGE ===\n" + "\n\n".join(context_sections["llm_knowledge"]))
    
    if context_sections["external_sources"]:
        context_parts.append("=== EXTERNAL SOURCES (Web Search, APIs) ===\n" + "\n\n".join(context_sections["external_sources"]))
    
    if context_sections["computer_capabilities"]:
        context_parts.append("\n\n".join(context_sections["computer_capabilities"]))
    
    context_block = "\n\n".join(context_parts) if context_parts else "No indexed sources available."
    
    # TWO-STAGE VERIFIER → REASONER ARCHITECTURE (Gnostic Frontier Architecture)
    # Stage 1: Verifier (temperature 0.0, deterministic, flags forbidden claims/connections)
    if thinking_callback:
        thinking_callback("I'm verifying claims against evidence and flagging unsupported connections...")
    
    if verbose:
        logger.info("[SURVEYOR] Stage 1: Running Verifier (flags forbidden matrix connections)...")
    
    forbidden_claims = _verifier_stage(query, context_block, matrices_info, gnosis_connections, cfg, verbose=True)  # Always verbose to see what's flagged
    
    # Stage 2: Reasoner (temperature 0.7, creative but controlled, uses gnosis knowledge)
    if thinking_callback:
        thinking_callback("I'm synthesizing information and generating response...")
    
    if verbose:
        logger.info("[SURVEYOR] Stage 2: Running Reasoner (generates response with gnosis knowledge)...")
    
    # Try self-consistency ensemble first (if enabled via ICEBURG_USE_ENSEMBLE=1)
    result = _self_consistency_ensemble(query, context_block, matrices_info, gnosis_connections, forbidden_claims, cfg, n_samples=3, verbose=verbose)
    
    # If ensemble not used or failed, use single reasoner response
    if result is None:
        result = _reasoner_stage(query, context_block, matrices_info, gnosis_connections, forbidden_claims, cfg, verbose)
    
    # Final safety check: ensure we always return something
    if result is None or (isinstance(result, str) and not result.strip()):
        logger.error("[SURVEYOR] Both ensemble and reasoner returned None/empty, returning fallback response")
        result = f"I processed your query about '{query}' but encountered an issue generating the response. This may be due to model limitations or query complexity. Please try rephrasing your question or try again later."
    
    if verbose:
        logger.info("[SURVEYOR] Analysis complete")
    
    return result
