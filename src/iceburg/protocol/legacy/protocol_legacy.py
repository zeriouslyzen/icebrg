from typing import List, Tuple, Optional, Union, Dict, Any
import os
import json
from pathlib import Path

# Use relative imports to avoid module resolution issues
try:
    from ...config import load_config, load_config_fast, load_config_hybrid
    from ...vectorstore import VectorStore
    from ...graph_store import KnowledgeGraph
    from ...agents import surveyor, dissident, synthesist, oracle
    from ...agents import scrutineer, archaeologist, supervisor
    # Optional agents that may not exist
    try:
        from ...agents import patent_analyzer
    except ImportError:
        patent_analyzer = None
    try:
        from ...agents import citation_detector
    except ImportError:
        citation_detector = None
    try:
        from ...agents import verifier
    except ImportError:
        verifier = None
    # Optional CIM Stack agents - import individually
    try:
        from ...agents import prompt_interpreter
    except ImportError:
        prompt_interpreter = None
    try:
        from ...agents import molecular_synthesis
    except ImportError:
        molecular_synthesis = None
    try:
        from ...agents import bioelectric_integration
    except ImportError:
        bioelectric_integration = None
    try:
        from ...agents import hypothesis_testing_laboratory
    except ImportError:
        hypothesis_testing_laboratory = None
    try:
        from ...agents import self_modification_engine
    except ImportError:
        self_modification_engine = None
    try:
        from ...agents import agent_evolution_system
    except ImportError:
        agent_evolution_system = None
    try:
        from ...agents import advanced_emergence_detection
    except ImportError:
        advanced_emergence_detection = None
    try:
        from ...agents import grounding_layer_agent
    except ImportError:
        grounding_layer_agent = None
    try:
        from ...agents import capability_gap_detector
    except ImportError:
        capability_gap_detector = None
    
    # Optional research agents - import individually
    try:
        from ...agents import real_scientific_research
    except ImportError:
        real_scientific_research = None
    try:
        from ...agents import corporate_network_analyzer
    except ImportError:
        corporate_network_analyzer = None
    try:
        from ...agents import geospatial_financial_anthropological
    except ImportError:
        geospatial_financial_anthropological = None
    try:
        from ...agents import comprehensive_api_manager
    except ImportError:
        comprehensive_api_manager = None
    try:
        from ...agents import virtual_scientific_ecosystem
    except ImportError:
        virtual_scientific_ecosystem = None
    try:
        from ...agents import bioelectrical_fundamental_agent
    except ImportError:
        bioelectrical_fundamental_agent = None
    # Optional deliberation agent functions
    try:
        from ...agents.deliberation_agent import (
            add_deliberation_pause,
            hunt_contradictions,
            detect_emergence,
            perform_meta_analysis,
            apply_truth_seeking_analysis,
            create_emergent_agent,
            emergent_field_creation,
        )
    except ImportError:
        add_deliberation_pause = None
        hunt_contradictions = None
        detect_emergence = None
        perform_meta_analysis = None
        apply_truth_seeking_analysis = None
        create_emergent_agent = None
        emergent_field_creation = None
    # Optional blockchain/report modules
    try:
        from ...blockchain_verification import BlockchainVerificationSystem
    except ImportError:
        BlockchainVerificationSystem = None
    try:
        from ...decentralized_peer_review import DecentralizedPeerReviewSystem
    except ImportError:
        DecentralizedPeerReviewSystem = None
    try:
        from ...suppression_resistant_storage import SuppressionResistantStorageSystem
    except ImportError:
        SuppressionResistantStorageSystem = None
    try:
        from ...report import format_iceberg_report
    except ImportError:
        format_iceberg_report = None
except ImportError as e:
    print(f"[LEGACY_PROTOCOL] Warning: Import error - {e}")
    print("[LEGACY_PROTOCOL] Some modules may not be available, using stubs")
    # Create stub functions for missing imports
    def load_config(): return {}
    def load_config_fast(): return {}
    def load_config_hybrid(): return {}
    VectorStore = None
    KnowledgeGraph = None
    surveyor = None
    dissident = None
    synthesist = None
    oracle = None
    scrutineer = None
    archaeologist = None
    patent_analyzer = None
    citation_detector = None
    verifier = None
    supervisor = None
    prompt_interpreter = None
    molecular_synthesis = None
    bioelectric_integration = None
    hypothesis_testing_laboratory = None
    self_modification_engine = None
    agent_evolution_system = None
    advanced_emergence_detection = None
    grounding_layer_agent = None
    capability_gap_detector = None
    real_scientific_research = None
    corporate_network_analyzer = None
    geospatial_financial_anthropological = None
    comprehensive_api_manager = None
    virtual_scientific_ecosystem = None
    bioelectrical_fundamental_agent = None
    add_deliberation_pause = None
    hunt_contradictions = None
    detect_emergence = None
    perform_meta_analysis = None
    apply_truth_seeking_analysis = None
    create_emergent_agent = None
    emergent_field_creation = None
    BlockchainVerificationSystem = None
    DecentralizedPeerReviewSystem = None
    SuppressionResistantStorageSystem = None
    format_iceberg_report = None

import time


def _extract_concepts(text: str, max_terms: int = 5) -> List[str]:
    # Simple keyword heuristic: split by non-letters, take unique nouns-like tokens > 5 chars
    import re
    tokens = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", text)
    lowered = [t.lower() for t in tokens]
    uniq: List[str] = []
    for t in lowered:
        if t not in uniq:
            uniq.append(t)
        if len(uniq) >= max_terms:
            break
    return uniq


def quick_chat_mode(query: str, fast: bool, hybrid: bool, verbose: bool, 
                   multimodal_input: Optional[Union[str, bytes, Path, Dict]] = None) -> str:
    """Fast Q&A using existing components without full protocol, with multimodal support"""
    
    # Use existing config loading
    cfg = load_config_hybrid() if hybrid else (load_config_fast() if fast else load_config())
    vs = VectorStore(cfg)
    
    if verbose:
        print("[CHAT] Executing quick research chat mode...")
        if multimodal_input:
            print(f"[CHAT] Processing multimodal input: {type(multimodal_input)}")
    
    # Use enhanced Surveyor for initial research with multimodal input
    consensus = surveyor.run(cfg, vs, query, verbose=verbose, multimodal_input=multimodal_input)
    
    # Use existing vector store for context
    concepts = _extract_concepts(query)
    context_hits = vs.semantic_search(query, k=5)
    context = [hit.document for hit in context_hits]
    
    # Generate conversational response
    response = _generate_chat_response(query, consensus, context, multimodal_input)
    
    if verbose:
        print("[CHAT] Chat response generated")
    
    return response


def _generate_chat_response(query: str, consensus: str, context: List[str], 
                          multimodal_input: Optional[Union[str, bytes, Path, Dict]] = None) -> str:
    """Generate conversational response from surveyor output and context"""
    
    # Simple response formatting for chat mode
    multimodal_section = ""
    if multimodal_input:
        multimodal_section = f"""## Multimodal Input Processed
{type(multimodal_input).__name__} input was analyzed and integrated into the response.

"""

    response = f"""# Quick Research Chat Response

## Query
{query}

{multimodal_section}## Answer
{consensus}

## Sources
{chr(10).join([f"- {source}" for source in context[:3]])}

---
*Generated in chat mode - for deeper analysis, use full protocol mode*
"""
    
    return response


def _primary_evidence_level(claims: List[dict]) -> str:
    # Map severity C > B > A; return worst level present
    order = {"A": 1, "B": 2, "C": 3}
    worst = 0
    label = "C"
    for c in claims:
        lvl = str(c.get("evidence_level", "C")).strip().upper().replace("[", "").replace("]", "")[:1]
        score = order.get(lvl, 3)
        if score > worst:
            worst = score
            label = lvl
    return label


def _validate_claims(scrut_json: str, strict: bool) -> Tuple[str, List[dict]]:
    try:
        data = json.loads(scrut_json)
    except Exception:
        if strict:
            raise SystemExit("Evidence strict: Scrutineer JSON invalid")
        return "C", []
    claims = data.get("claims") or []
    if not isinstance(claims, list):
        if strict:
            raise SystemExit("Evidence strict: Claims must be list")
        return "C", []
    primary_level = _primary_evidence_level(claims)
    return primary_level, claims


def iceberg_protocol(initial_query: str, fast: bool = False, hybrid: bool = False, 
                    verbose: bool = False, evidence_strict: bool = False, 
                    domains: Optional[List[str]] = None, project_id: Optional[str] = None,
                    multimodal_input: Optional[Union[str, bytes, Path, Dict]] = None,
                    documents: Optional[List[Union[str, bytes, Path, Dict]]] = None,
                    multimodal_evidence: Optional[List[Union[str, bytes, Path, Dict]]] = None) -> str:
    """
    ICEBURG Protocol with Mac hardware optimizations and parallel execution.
    
    Features:
    - Mac Neural Engine optimization (M1/M4)
    - Parallel agent execution (50-70% faster)
    - Multimodal input processing
    - Software lab integration
    - Red team security validation
    
    Args:
        initial_query: The research query
        fast: Use fast mode
        hybrid: Use hybrid mode
        verbose: Enable verbose output
        evidence_strict: Require strong evidence
        domains: Domain filters for memory retrieval
        project_id: Project identifier
        multimodal_input: Single multimodal input for Surveyor
        documents: List of documents for Archaeologist analysis
        multimodal_evidence: List of multimodal evidence for Synthesist
    """

    # STRICT FAST-PATH SHORT-CIRCUIT: return immediately with a lightweight response
    if fast:
        try:
            # Load a fast configuration and attempt reflexive routing fast response
            from ...config import load_config_fast
            cfg_fast = load_config_fast()
            try:
                from ...integration.reflexive_routing import ReflexiveRoutingSystem
                router = ReflexiveRoutingSystem(cfg_fast)
                routing = router.route_query(initial_query)
                # Use reflexive responder if available, with beam-scored quick synthesis
                if hasattr(router, "respond_fast"):
                    return router.respond_fast(
                        initial_query,
                        num_ctx=1024,
                        num_predict=64,
                        temperature=0.2,
                        beam_width=3,
                    )
            except Exception:
                # Fall through to minimal fast response below
                pass

            # Minimal fast response fallback: single-pass small-context synthesis
            from ...llm import chat_complete
            import os as _os
            # Prefer explicit fast-path model if provided, else small SLM
            model = _os.getenv("ICEBURG_FAST_MODEL", getattr(cfg_fast, "surveyor_model", "qwen2.5:1.5b"))
            options = {"num_ctx": 1024, "num_predict": 64}
            system = (
                "You are ICEBURG's fast reflexive responder. Provide a concise, technical synthesis (5-8 sentences). "
                "Be direct and non-evasive. Scientific planning is allowed within legal and ethical bounds. "
                "If a constraint applies, briefly state it, then proceed with a high-level, safe methodology."
            )
            prompt = (
                "FAST MODE QUERY (no heavy agents):\n\n" + initial_query + "\n\n"
                "Output strictly as: 1) Key signals (bullets), 2) Methods/metrics (bullets), 3) Next actions (bullets)."
            )
            return chat_complete(model, prompt, system=system, temperature=0.2, options=options, context_tag="FastPath")
        except Exception:
            # If anything fails, continue to normal pipeline (non-fast) as a safety net
            if verbose:
                print("[FAST_PATH] Fallback to standard pipeline due to fast-path error")
            pass
    # Initialize Mac hardware optimizer (skip in fast mode or if disabled)
    try:
        import os as _os
        if not fast and _os.getenv("ICEBURG_DISABLE_HARDWARE_OPT", "0") != "1":
            from ...optimization.mac_hardware import MacHardwareOptimizer
            import platform
            
            # Only initialize on macOS
            if platform.system() == "Darwin":
                mac_optimizer = MacHardwareOptimizer()
                if verbose:
                    print(f"[MAC_OPTIMIZER] Hardware detected: {mac_optimizer.hardware_info.processor}")
                    print(f"[MAC_OPTIMIZER] Neural Engine: {mac_optimizer.hardware_info.neural_engine}")
                    print(f"[MAC_OPTIMIZER] Metal support: {mac_optimizer.hardware_info.metal_support}")
    except Exception as e:
        if verbose:
            print(f"[MAC_OPTIMIZER] Hardware optimization unavailable: {e}")
    
    # Load configuration
    cfg = load_config_hybrid() if hybrid else (load_config_fast() if fast else load_config())
    
    # Initialize reflexive routing for fast simple queries (unless explicitly disabled)
    if not fast:  # Only use reflexive routing if not already in fast mode
        try:
            from ...integration.reflexive_routing import ReflexiveRoutingSystem
            reflexive_router = ReflexiveRoutingSystem(cfg)
            routing_decision = reflexive_router.route_query(initial_query)
            
            # If reflexive routing recommends fast path and confidence is high, use it
            if routing_decision.route_type == "reflexive" and routing_decision.confidence > 0.7:
                if verbose:
                    print(f"[REFLEXIVE] Routing to fast path (complexity: {routing_decision.complexity_score:.2f}, confidence: {routing_decision.confidence:.2f})")
                
                import asyncio
                try:
                    # Try to get reflexive response
                    reflexive_response = asyncio.run(reflexive_router.process_reflexive(initial_query))
                    
                    # If no escalation recommended and confidence is high, return reflexive response
                    if not reflexive_response.escalation_recommended and reflexive_response.confidence > 0.7:
                        if verbose:
                            print(f"[REFLEXIVE] Returning reflexive response (time: {reflexive_response.processing_time:.2f}s)")
                        return reflexive_response.response
                    else:
                        if verbose:
                            print(f"[REFLEXIVE] Escalation recommended, proceeding to full protocol")
                except Exception as e:
                    if verbose:
                        print(f"[REFLEXIVE] Error processing reflexive response: {e}, proceeding to full protocol")
        except Exception as e:
            if verbose:
                print(f"[REFLEXIVE] Reflexive routing not available: {e}, proceeding to full protocol")
    
    # Initialize components
    vs = VectorStore(cfg)
    kg = KnowledgeGraph(cfg)
    
    # Initialize blockchain verification system (optional)
    blockchain_system = None
    if BlockchainVerificationSystem is not None:
        blockchain_system = BlockchainVerificationSystem(cfg)
    
    # Initialize decentralized peer review system (optional)
    peer_review_system = None
    if DecentralizedPeerReviewSystem is not None and blockchain_system is not None:
        peer_review_system = DecentralizedPeerReviewSystem(cfg, blockchain_system)
    
    # Initialize suppression-resistant storage system (optional)
    storage_system = None
    if SuppressionResistantStorageSystem is not None and blockchain_system is not None and peer_review_system is not None:
        storage_system = SuppressionResistantStorageSystem(cfg, blockchain_system, peer_review_system)
    
    # Initialize reasoning chain tracking
    reasoning_chain = {}
    
    # Real Scientific Research Analysis (if molecular/medical query detected)
    scientific_research_insights = {}
    if (not fast) and any(keyword in initial_query.lower() for keyword in ['molecule', 'chemical', 'drug', 'protein', 'enzyme', 'compound', 'heart', 'coherence', 'chinese', 'indian', 'research']):
        if verbose:
            print(f"[PROTOCOL] Detected scientific research query - running real scientific research analysis...")
        
        # Run Real Scientific Research analysis (molecular biology, heart coherence, academic research)
        scientific_research_result = real_scientific_research.run_real_scientific_research(cfg, initial_query, None, verbose)
        scientific_research_insights["real_scientific_research"] = scientific_research_result
    
    # Corporate Network Analysis (if corporate/family query detected)
    corporate_network_insights = {}
    if (not fast) and any(keyword in initial_query.lower() for keyword in ['corporate', 'family', 'network', 'rothschild', 'rockefeller', 'morgan', 'dupont', 'ford', 'carnegie', 'foundation', 'institute']):
        if verbose:
            print(f"[PROTOCOL] Detected corporate network query - running factual corporate network analysis...")
        
        # Run Corporate Network Analysis (factual corporate connections, family networks)
        corporate_network_result = corporate_network_analyzer.run(cfg, initial_query, None, verbose)
        corporate_network_insights["corporate_network_analysis"] = corporate_network_result
    
    # Multi-Domain Analysis (if geospatial/financial/anthropological/museum/archaeological query detected)
    multi_domain_insights = {}
    if (not fast) and any(keyword in initial_query.lower() for keyword in ['location', 'city', 'country', 'stock', 'market', 'finance', 'culture', 'tribe', 'ethnic', 'museum', 'artifact', 'archaeology', 'excavation', 'ancient']):
        if verbose:
            print(f"[PROTOCOL] Detected multi-domain query - running geospatial, financial, anthropological, museum, and archaeological analysis...")
        # Run Multi-Domain Analysis (geospatial, financial, anthropological, museum, archaeological)
        try:
            if 'geospatial_financial_anthropological' in globals() and hasattr(geospatial_financial_anthropological, 'run'):
                multi_domain_result = geospatial_financial_anthropological.run(cfg, initial_query, None, verbose)
                multi_domain_insights["multi_domain_analysis"] = multi_domain_result
            else:
                if verbose:
                    print("[PROTOCOL] Skipping multi-domain analysis: module or 'run' not available")
        except Exception as e:
            if verbose:
                print(f"[PROTOCOL] Skipping multi-domain analysis due to error: {e}")
    
    # Comprehensive API Search (if broad research query detected)
    comprehensive_api_insights = {}
    if (not fast) and any(keyword in initial_query.lower() for keyword in ['research', 'study', 'analysis', 'investigation', 'examination', 'exploration', 'discovery']):
        if verbose:
            print(f"[PROTOCOL] Detected comprehensive research query - running 50+ API search...")
        # Run Comprehensive API Search across 50+ sources, if available
        try:
            if 'comprehensive_api_manager' in globals() and hasattr(comprehensive_api_manager, 'run'):
                comprehensive_api_result = comprehensive_api_manager.run(cfg, initial_query, None, verbose)
                comprehensive_api_insights["comprehensive_api_search"] = comprehensive_api_result
            else:
                if verbose:
                    print("[PROTOCOL] Skipping comprehensive API search: module or 'run' not available")
        except Exception as e:
            if verbose:
                print(f"[PROTOCOL] Skipping comprehensive API search due to error: {e}")
    
    # Virtual Scientific Ecosystem (if experimental query detected)
    virtual_ecosystem_insights = {}
    if (not fast) and any(keyword in initial_query.lower() for keyword in ['experiment', 'trial', 'test', 'simulation', 'virtual', 'population', 'clinical', 'laboratory']):
        if verbose:
            print(f"[PROTOCOL] Detected experimental query - running virtual scientific ecosystem...")
        
        # Run Virtual Scientific Ecosystem (virtual experiments, populations, equipment)
        virtual_ecosystem_result = virtual_scientific_ecosystem.run(cfg, initial_query, None, verbose)
        virtual_ecosystem_insights["virtual_scientific_ecosystem"] = virtual_ecosystem_result
    
    # Bioelectrical Fundamental Analysis (if bioelectrical query detected)
    bioelectrical_insights = {}
    if any(keyword in initial_query.lower() for keyword in ['bioelectrical', 'biofield', 'ion channels', 'membrane potential', 'quantum biology', 'dna antenna', 'neural transceiver', 'synaptic transmission', 'neurotransmitter', 'atp']):
        if verbose:
            print(f"[PROTOCOL] Detected bioelectrical query - running bioelectrical fundamental analysis...")
        # Run Bioelectrical Fundamental Analysis (ion channels, biofield coherence, quantum effects) if available
        try:
            if 'bioelectrical_fundamental_agent' in globals() and bioelectrical_fundamental_agent is not None and hasattr(bioelectrical_fundamental_agent, 'run'):
                bioelectrical_result = bioelectrical_fundamental_agent.run(cfg, initial_query, None, verbose)
                bioelectrical_insights["bioelectrical_fundamental_analysis"] = bioelectrical_result
            else:
                if verbose:
                    print("[PROTOCOL] Skipping bioelectrical analysis: module or 'run' not available")
        except Exception as e:
            if verbose:
                print(f"[PROTOCOL] Skipping bioelectrical analysis due to error: {e}")
    
    if verbose:
        print("[STATUS] Starting enhanced Iceberg Protocol with CIM Stack Architecture...")
        if multimodal_input:
            print(f"[MULTIMODAL] Processing input: {type(multimodal_input)}")
        if documents:
            print(f"[DOCUMENTS] Analyzing {len(documents)} documents")
        if multimodal_evidence:
            print(f"[EVIDENCE] Processing {len(multimodal_evidence)} pieces of multimodal evidence")
    
    # CIM Stack Layer 0: Intelligent Prompt Interpreter
    if verbose:
        print("[STATUS] Executing CIM Stack Layer 0: Prompt Interpreter (Intent Recognition)...")
    cim_analysis = prompt_interpreter.run(cfg, initial_query, verbose=verbose)
    reasoning_chain["prompt_interpreter"] = {"output": cim_analysis, "context": "CIM Analysis"}
    
    # Extract intent and routing information
    intent_analysis = cim_analysis.get("intent_analysis", {})
    agent_routing = cim_analysis.get("agent_routing", {})
    
    if verbose:
        print(f"[CIM] Primary Domain: {intent_analysis.get('primary_domain', 'unknown')}")
        print(f"[CIM] Detail Level: {intent_analysis.get('detail_level', 'unknown')}")
        print(f"[CIM] Requires Molecular: {intent_analysis.get('requires_molecular', False)}")
        print(f"[CIM] Requires Bioelectric: {intent_analysis.get('requires_bioelectric', False)}")
    
    # PARALLEL EXECUTION: Run Surveyor and Dissident in parallel (50-70% faster)
    if not fast:  # Only use parallel execution for non-fast queries
        if verbose:
            print("[STATUS] Executing Layer 1-2 in PARALLEL: Surveyor + Dissident...")
        
        try:
            # Import parallel execution engine
            from ...parallel_execution import execute_surveyor_dissident_parallel
            import asyncio
            
            # Execute surveyor and dissident in parallel
            consensus, diss = asyncio.run(execute_surveyor_dissident_parallel(
                cfg, vs, initial_query, verbose=verbose, multimodal_input=multimodal_input
            ))
            
            # If dissident result is None, run it sequentially with surveyor output
            if diss is None:
                if verbose:
                    print("[PARALLEL] Dissident failed in parallel, running sequentially...")
                diss = dissident.run(cfg, initial_query, consensus, verbose=verbose)
            
            if verbose:
                print("[PARALLEL] Surveyor and Dissident completed in parallel")
                
        except Exception as e:
            if verbose:
                print(f"[PARALLEL] Parallel execution failed, falling back to sequential: {e}")
            
            # Fallback to sequential execution
            if verbose:
                print("[STATUS] Executing Layer 1: Surveyor (Consensus Research)...")
            consensus = surveyor.run(cfg, vs, initial_query, verbose=verbose, multimodal_input=multimodal_input)
            
            if verbose:
                print("[STATUS] Executing Layer 2: Dissident (Alternative Perspectives)...")
            diss = dissident.run(cfg, initial_query, consensus, verbose=verbose)
    else:
        # Fast mode: sequential execution
        if verbose:
            print("[STATUS] Fast mode - executing sequentially...")
        consensus = surveyor.run(cfg, vs, initial_query, verbose=verbose, multimodal_input=multimodal_input)
        diss = dissident.run(cfg, initial_query, consensus, verbose=verbose)
    
    reasoning_chain["surveyor"] = {"output": consensus, "context": "Surveyor"}
    reasoning_chain["dissident"] = {"output": diss, "context": "Dissident"}

    # DELIBERATION PAUSE 1: Reflect on Surveyor findings
    if not fast:
        if verbose:
            print("[DELIBERATION] Adding reflection pause after Surveyor...")
        deliberation_1 = add_deliberation_pause(cfg, "Surveyor", consensus, initial_query, verbose=verbose)
        reasoning_chain["deliberation_1"] = {"output": deliberation_1, "context": "Surveyor Reflection"}
    else:
        deliberation_1 = "Skipped in fast mode"
        reasoning_chain["deliberation_1"] = {"output": deliberation_1, "context": "Skipped"}

    # Layer 3: Archaeologist (Enhanced with document analysis)
    if verbose:
        print("[STATUS] Executing Layer 3: Archaeologist (Deep Research Insights)...")
    arch = archaeologist.run(cfg, initial_query, verbose=verbose, documents=documents)
    reasoning_chain["archaeologist"] = {"output": arch, "context": "Archaeologist"}

    # PARALLEL DELIBERATION: Run deliberations after Dissident and Archaeologist concurrently
    if not fast:
        if verbose:
            print("[PARALLEL DELIBERATION] Running deliberations concurrently after Dissident and Archaeologist...")
        
        try:
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            def run_deliberation_sync(agent_name: str, output: str):
                """Synchronous wrapper for deliberation pause"""
                return add_deliberation_pause(cfg, agent_name, output, initial_query, verbose=verbose)
            
            # Run deliberations in parallel using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                deliberation_2_future = executor.submit(run_deliberation_sync, "Dissident", diss)
                deliberation_3_future = executor.submit(run_deliberation_sync, "Archaeologist", arch)
                
                # Wait for both to complete
                deliberation_2 = deliberation_2_future.result()
                deliberation_3 = deliberation_3_future.result()
            
            reasoning_chain["deliberation_2"] = {"output": deliberation_2, "context": "Dissident Reflection"}
            reasoning_chain["deliberation_3"] = {"output": deliberation_3, "context": "Archaeologist Reflection"}
            
            if verbose:
                print("[PARALLEL DELIBERATION] Concurrent deliberations completed")
                
        except Exception as e:
            if verbose:
                print(f"[PARALLEL DELIBERATION] Parallel execution failed, falling back to sequential: {e}")
            
            # Fallback to sequential execution
            deliberation_2 = add_deliberation_pause(cfg, "Dissident", diss, initial_query, verbose=verbose)
            reasoning_chain["deliberation_2"] = {"output": deliberation_2, "context": "Dissident Reflection"}
            
            deliberation_3 = add_deliberation_pause(cfg, "Archaeologist", arch, initial_query, verbose=verbose)
            reasoning_chain["deliberation_3"] = {"output": deliberation_3, "context": "Archaeologist Reflection"}
    else:
        deliberation_2 = "Skipped in fast mode"
        deliberation_3 = "Skipped in fast mode"
        reasoning_chain["deliberation_2"] = {"output": deliberation_2, "context": "Skipped"}
        reasoning_chain["deliberation_3"] = {"output": deliberation_3, "context": "Skipped"}

    # Layer 4: Supervisor (Enhanced to handle multimodal outputs)
    if verbose:
        print("[STATUS] Executing Layer 4: Supervisor (Degeneration Detection)...")
    
    # Handle potential dictionary outputs from enhanced agents
    diss_output = diss if isinstance(diss, str) else diss.get("output", str(diss)) if isinstance(diss, dict) else str(diss)
    arch_output = arch if isinstance(arch, str) else arch.get("output", str(arch)) if isinstance(arch, dict) else str(arch)
    
    stage_outputs = {
        "surveyor": consensus,
        "dissident": diss_output,
        "archaeologist": arch_output
    }
    
    # BATCHED ENHANCED DELIBERATION: Run multiple deliberation analyses concurrently
    if not fast:
        if verbose:
            print("[BATCHED ENHANCED DELIBERATION] Running enhanced deliberations concurrently...")
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            # Prepare deliberation tasks
            tasks = [
                ("contradiction_hunting", hunt_contradictions, [cfg, stage_outputs, initial_query, verbose]),
                ("pattern_analysis", perform_meta_analysis, [cfg, stage_outputs, initial_query, verbose]),
                ("emergence_detection", detect_emergence, [cfg, stage_outputs, initial_query, verbose]),
                ("truth_seeking", apply_truth_seeking_analysis, [cfg, stage_outputs, initial_query, verbose]),
            ]
            
            # Execute all deliberations concurrently using ThreadPoolExecutor with timeouts
            deliberation_results = {}
            timeout_seconds = 30  # 30 second timeout per deliberation
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Submit all tasks
                future_to_name = {
                    executor.submit(func, *args): name 
                    for name, func, args in tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_name, timeout=timeout_seconds * len(tasks)):
                    name = future_to_name[future]
                    try:
                        deliberation_results[name] = future.result(timeout=timeout_seconds)
                    except TimeoutError:
                        if verbose:
                            print(f"[BATCHED ENHANCED DELIBERATION] {name} timed out after {timeout_seconds}s")
                        deliberation_results[name] = f"Analysis timed out after {timeout_seconds} seconds"
                    except Exception as e:
                        if verbose:
                            print(f"[BATCHED ENHANCED DELIBERATION] {name} failed: {e}")
                        deliberation_results[name] = f"Analysis failed: {str(e)}"
            
            # Extract results
            contradiction_analysis = deliberation_results.get("contradiction_hunting", "Batch deliberation failed")
            pattern_analysis = deliberation_results.get("pattern_analysis", "Batch deliberation failed")
            emergence_analysis = deliberation_results.get("emergence_detection", "Batch deliberation failed")
            truth_analysis = deliberation_results.get("truth_seeking", "Batch deliberation failed")
            
            # Store in reasoning chain
            reasoning_chain["contradiction_hunting"] = {"output": contradiction_analysis, "context": "Contradiction Analysis"}
            reasoning_chain["pattern_analysis"] = {"output": pattern_analysis, "context": "Pattern Analysis"}
            reasoning_chain["emergence_detection"] = {"output": emergence_analysis, "context": "Emergence Detection"}
            reasoning_chain["truth_seeking"] = {"output": truth_analysis, "context": "Truth-Seeking Analysis"}
            
            if verbose:
                print("[BATCHED ENHANCED DELIBERATION] Concurrent deliberations completed")
                
        except Exception as e:
            if verbose:
                print(f"[BATCHED ENHANCED DELIBERATION] Batch execution failed, falling back to sequential: {e}")
            
            # Fallback to sequential execution
            contradiction_analysis = hunt_contradictions(cfg, stage_outputs, initial_query, verbose=verbose)
            reasoning_chain["contradiction_hunting"] = {"output": contradiction_analysis, "context": "Contradiction Analysis"}
            
            pattern_analysis = perform_meta_analysis(cfg, stage_outputs, initial_query, verbose=verbose)
            reasoning_chain["pattern_analysis"] = {"output": pattern_analysis, "context": "Pattern Analysis"}
            
            emergence_analysis = detect_emergence(cfg, stage_outputs, initial_query, verbose=verbose)
            reasoning_chain["emergence_detection"] = {"output": emergence_analysis, "context": "Emergence Detection"}
            
            truth_analysis = apply_truth_seeking_analysis(cfg, stage_outputs, initial_query, verbose=verbose)
            reasoning_chain["truth_seeking"] = {"output": truth_analysis, "context": "Truth-Seeking Analysis"}
    else:
        # Fast mode: skip enhanced deliberations
        contradiction_analysis = "Skipped in fast mode"
        pattern_analysis = "Skipped in fast mode"
        emergence_analysis = "Skipped in fast mode"
        truth_analysis = "Skipped in fast mode"
        reasoning_chain["contradiction_hunting"] = {"output": contradiction_analysis, "context": "Skipped"}
        reasoning_chain["pattern_analysis"] = {"output": pattern_analysis, "context": "Skipped"}
        reasoning_chain["emergence_detection"] = {"output": emergence_analysis, "context": "Skipped"}
        reasoning_chain["truth_seeking"] = {"output": truth_analysis, "context": "Skipped"}
    
    supervisor_result = supervisor.run(cfg, stage_outputs, verbose=verbose)
    reasoning_chain["supervisor"] = {"output": supervisor_result, "context": "Supervisor"}
    
    # ENHANCED DELIBERATION: Quality assessment before synthesis
    if not fast:
        if verbose:
            print("[ENHANCED DELIBERATION] Quality assessment before synthesis...")
        quality_assessment = add_deliberation_pause(cfg, "Pre-Synthesis", str(stage_outputs), initial_query, verbose=verbose)
        reasoning_chain["quality_assessment"] = {"output": quality_assessment, "context": "Pre-Synthesis Quality Assessment"}
    else:
        quality_assessment = "Skipped in fast mode"
        reasoning_chain["quality_assessment"] = {"output": quality_assessment, "context": "Skipped"}

    # Layer 5: Synthesist (Enhanced with deliberation insights)
    if verbose:
        print("[STATUS] Executing Layer 5: Synthesist (Cross-Domain Synthesis)...")
    
    # Combine all insights including enhanced deliberation results
    enhanced_context = {
        "surveyor": consensus,
        "dissident": diss_output,
        "archaeologist": arch_output,
        "contradictions": contradiction_analysis,
        "patterns": pattern_analysis,
        "emergence": emergence_analysis,
        "truth_seeking": truth_analysis,
        "scientific_research": scientific_research_insights,
        "corporate_network": corporate_network_insights,
        "multi_domain": multi_domain_insights,
        "comprehensive_api": comprehensive_api_insights,
        "virtual_ecosystem": virtual_ecosystem_insights,
        "bioelectrical": bioelectrical_insights
    }
    
    synthesis = synthesist.run(cfg, enhanced_context, verbose=verbose, multimodal_evidence=multimodal_evidence)
    reasoning_chain["synthesist"] = {"output": synthesis, "context": "Synthesist"}

    # ENHANCED DELIBERATION: Post-synthesis analysis
    if not fast:
        if verbose:
            print("[ENHANCED DELIBERATION] Post-synthesis analysis and validation...")
        post_synthesis_analysis = add_deliberation_pause(cfg, "Post-Synthesis", synthesis, initial_query, verbose=verbose)
        reasoning_chain["post_synthesis_analysis"] = {"output": post_synthesis_analysis, "context": "Post-Synthesis Analysis"}
    else:
        post_synthesis_analysis = "Skipped in fast mode"
        reasoning_chain["post_synthesis_analysis"] = {"output": post_synthesis_analysis, "context": "Skipped"}

    # Layer 6: Scrutineer (Enhanced with suppression detection)
    if verbose:
        print("[STATUS] Executing Layer 6: Scrutineer (Suppression Detection)...")
    scrut = scrutineer.run(cfg, synthesis, verbose=verbose)
    reasoning_chain["scrutineer"] = {"output": scrut, "context": "Scrutineer"}

    # ENHANCED DELIBERATION: Suppression pattern analysis
    if not fast:
        if verbose:
            print("[ENHANCED DELIBERATION] Analyzing suppression patterns...")
        suppression_analysis = add_deliberation_pause(cfg, "Suppression Analysis", scrut, initial_query, verbose=verbose)
        reasoning_chain["suppression_analysis"] = {"output": suppression_analysis, "context": "Suppression Pattern Analysis"}
    else:
        suppression_analysis = "Skipped in fast mode"
        reasoning_chain["suppression_analysis"] = {"output": suppression_analysis, "context": "Skipped"}

    # Layer 7: Oracle (Enhanced with all deliberation insights)
    if verbose:
        print("[STATUS] Executing Layer 7: Oracle (Final Synthesis)...")
    
    # Combine synthesis with all deliberation insights
    oracle_context = {
        "synthesis": synthesis,
        "scrutineer": scrut,
        "contradictions": contradiction_analysis,
        "patterns": pattern_analysis,
        "emergence": emergence_analysis,
        "truth_seeking": truth_analysis,
        "suppression_analysis": suppression_analysis
    }
    
    # Oracle needs kg, vs, and evidence_weighted_input
    oracle_result = oracle.run(cfg, kg, vs, oracle_context, verbose=verbose)
    reasoning_chain["oracle"] = {"output": oracle_result, "context": "Oracle"}

    # ENHANCED DELIBERATION: Final validation and breakthrough detection
    if not fast:
        if verbose:
            print("[ENHANCED DELIBERATION] Final validation and breakthrough detection...")
        final_validation = add_deliberation_pause(cfg, "Final Validation", oracle_result, initial_query, verbose=verbose)
        reasoning_chain["final_validation"] = {"output": final_validation, "context": "Final Validation"}
    else:
        final_validation = "Skipped in fast mode"
        reasoning_chain["final_validation"] = {"output": final_validation, "context": "Skipped"}

    # AGI EMERGENCE DETECTION: Check for emergent scientific principles
    if verbose:
        print("[AGI] Checking for emergent scientific principles...")
    emergence_check = detect_emergence(cfg, {"final_validation": final_validation}, initial_query, verbose=verbose)
    reasoning_chain["agi_emergence_check"] = {"output": emergence_check, "context": "AGI Emergence Check"}

    # AGI SELF-MODIFICATION: Create emergent agents if high emergence detected
    if emergence_check.get("analysis_type") == "emergence_detection":
        emergence_data = emergence_check.get("emergence", {})
        if emergence_data.get("emergence_detected", False) and emergence_data.get("emergence_score", 0) >= 0.8:
            if verbose:
                print("[AGI] High emergence detected - creating emergent agent...")
            agent_created = create_emergent_agent(cfg, emergence_data, initial_query, verbose=verbose)
            if agent_created:
                reasoning_chain["agi_emergent_agent"] = {"output": {"agent_created": True, "emergence_data": emergence_data}, "context": "AGI Emergent Agent"}

    # AGI FIELD CREATION: Create new scientific disciplines
    if emergence_check.get("analysis_type") == "emergence_detection":
        emergence_data = emergence_check.get("emergence", {})
        if emergence_data.get("emergence_detected", False):
            if verbose:
                print("[AGI] Creating emergent scientific field...")
            field_result = emergent_field_creation(cfg, emergence_data, initial_query, verbose=verbose)
            reasoning_chain["agi_emergent_field"] = {"output": field_result, "context": "AGI Emergent Field"}

    # CIM Stack Integration: Molecular Synthesis (if required)
    molecular_analysis = None
    if intent_analysis.get('requires_molecular', False):
        if verbose:
            print("[STATUS] Executing CIM Stack Integration: Molecular Synthesis...")
        molecular_analysis = molecular_synthesis.run(cfg, initial_query, context=cim_analysis, verbose=verbose)
        reasoning_chain["molecular_synthesis"] = {"output": molecular_analysis, "context": "Molecular Analysis"}
        
        if verbose:
            mechanisms = molecular_analysis.get('molecular_mechanisms', [])
            pathways = molecular_analysis.get('biochemical_pathways', [])
            print(f"[MOLECULAR] Mechanisms: {len(mechanisms)}")
            print(f"[MOLECULAR] Pathways: {len(pathways)}")

    # CIM Stack Integration: Bioelectric Integration (if required)
    bioelectric_analysis = None
    if intent_analysis.get('requires_bioelectric', False):
        if verbose:
            print("[STATUS] Executing CIM Stack Integration: Bioelectric Integration...")
        bioelectric_analysis = bioelectric_integration.run_traditional_energy_medicine(cfg, initial_query, verbose)
        reasoning_chain["bioelectric_integration"] = {"output": bioelectric_analysis, "context": "Bioelectric Analysis"}
        
        if verbose:
            traditional_concepts = bioelectric_analysis.get('traditional_analysis', {}).get('traditional_concepts', [])
            modern_research = bioelectric_analysis.get('modern_analysis', {}).get('modern_research', [])
            print(f"[BIOELECTRIC] Traditional concepts: {len(traditional_concepts) if isinstance(traditional_concepts, list) else 1}")
            print(f"[BIOELECTRIC] Modern research: {len(modern_research) if isinstance(modern_research, list) else 1}")

    # CIM Stack Integration: Hypothesis Testing Laboratory (if required)
    hypothesis_testing_results = None
    if intent_analysis.get('requires_hypothesis_testing', False):
        if verbose:
            print("[STATUS] Executing CIM Stack Integration: Hypothesis Testing Laboratory...")
        hypothesis_testing_results = hypothesis_testing_laboratory.run(cfg, initial_query, context=cim_analysis, verbose=verbose)
        reasoning_chain["hypothesis_testing_laboratory"] = {"output": hypothesis_testing_results, "context": "Hypothesis Testing"}
        
        if verbose:
            experiments = hypothesis_testing_results.get('experiments', [])
            validations = hypothesis_testing_results.get('validations', [])
            print(f"[HYPOTHESIS] Experiments: {len(experiments)}")
            print(f"[HYPOTHESIS] Validations: {len(validations)}")

    # CIM Stack Integration: Grounding Layer (if required)
    grounding_layer_results = None
    if intent_analysis.get('requires_empirical_grounding', False):
        if verbose:
            print("[STATUS] Executing CIM Stack Integration: Grounding Layer (Empirical Bridge)...")
        
        import asyncio
        try:
            grounding_layer_results = asyncio.run(grounding_layer_agent.run(
                cfg, 
                initial_query, 
                context=cim_analysis, 
                verbose=verbose,
                data_sources=cim_analysis.get('data_sources'),
                correlation_types=cim_analysis.get('correlation_types')
            ))
            reasoning_chain["grounding_layer"] = {"output": grounding_layer_results, "context": "Empirical Grounding"}
            
            if verbose:
                grounding_score = grounding_layer_results.get('grounding_score', 0.0)
                recommendations = len(grounding_layer_results.get('recommendations', []))
                print(f"[GROUNDING] Empirical grounding score: {grounding_score:.3f}")
                print(f"[GROUNDING] Recommendations: {recommendations}")
                
        except Exception as e:
            if verbose:
                print(f"[GROUNDING] Error in grounding layer: {e}")
            grounding_layer_results = {"error": str(e)}

    # Validate claims and compute primary evidence
    primary_level, claims = _validate_claims(scrut, evidence_strict)
    
    # Check if we should continue with full analysis or return early
    should_return_early = evidence_strict and (primary_level not in ("A", "B") or not claims)
    
    if should_return_early:
        if verbose:
            print("[GATE] Evidence below threshold under --evidence-strict; will return consensus-only answer after capability gap analysis.")

    # Memory read cycle with domain filtering (simple filter by substring)
    concepts = _extract_concepts(initial_query)
    prior_list = []
    
    if not cfg.disable_memory:
        if verbose:
            print("[STATUS] Retrieving relevant memories...")
            print(f"[MEMORY] Extracted concepts: {concepts}")

        prior_list = kg.retrieve_relevant_principles(concepts)
    else:
        if verbose:
            print("[STATUS] Memory retrieval disabled - using clean run")
            print(f"[MEMORY] Extracted concepts: {concepts} (not used)")

    if verbose:
        print(f"[MEMORY] Retrieved {len(prior_list)} prior principles")
        if prior_list:
            print("[MEMORY] Sample principle:", prior_list[0][:100] + "..." if len(prior_list[0]) > 100 else prior_list[0])
    if domains:
        filtered = []
        for p in prior_list:
            if any(d.lower() in p.lower() for d in domains):
                filtered.append(p)
        prior_list = filtered or prior_list
    prior_block = "\n".join(prior_list)

    if verbose:
        print("[STATUS] Executing Layer 7: Oracle (Evidence-Weighted Principle)...")
    oracle_input = json.dumps({
        "claims": claims,
        "primary_evidence": primary_level,
        "prior": prior_list,
    })
    principle = oracle.run(cfg, kg, vs, oracle_input, verbose=verbose)
    reasoning_chain["oracle"] = {"output": principle, "context": "Oracle"}

    # Code Generation with Weaver Agent - DISABLED
    generated_code = None
    lab_testing_results = None
    if verbose:
        print("[WEAVER] Disabled - Weaver agent is not active")
    
    # Full-Stack Application Generation with Architect Agent (Software Lab)
    generated_application = None
    if cfg.enable_software_lab:
        try:
            from .agents import architect
            if verbose:
                print("[ARCHITECT] Starting software engineering lab...")
            generated_application = architect.run(cfg, principle, verbose=verbose)

            if verbose and generated_application:
                print(f"\n{'='*60}")
                print("🏗️ FULL-STACK APPLICATION:")
                print(f"{'='*60}")
                print(generated_application)
                print(f"{'='*60}\n")
            elif verbose:
                print("[ARCHITECT] No application generated")

        except ImportError as e:
            if verbose:
                print(f"[ARCHITECT] Import error: {e}")
        except Exception as e:
            if verbose:
                print(f"[ARCHITECT] Application generation failed: {e}")
    elif verbose:
        print("[ARCHITECT] Software lab disabled - set ICEBURG_ENABLE_SOFTWARE_LAB=1 to enable")
    
    # Autonomous Capability Gap Detection and Agent Creation
    capability_gap_analysis = None
    if verbose:
        print("[STATUS] Executing Autonomous Capability Gap Detection...")
    
    try:
        capability_gap_analysis = capability_gap_detector.run(
            cfg, 
            initial_query, 
            context=cim_analysis,
            reasoning_chain=reasoning_chain,
            verbose=verbose
        )
        reasoning_chain["capability_gap_detector"] = {"output": capability_gap_analysis, "context": "Capability Gap Analysis"}
        
        if verbose:
            gaps_detected = len(capability_gap_analysis.get("gaps_detected", []))
            agents_created = len([a for a in capability_gap_analysis.get("created_agents", []) if a.get("status") == "created"])
            print(f"[CAPABILITY_GAP] Detected {gaps_detected} capability gaps")
            print(f"[CAPABILITY_GAP] Created {agents_created} new agents")
            
            # Show created agents
            for agent in capability_gap_analysis.get("created_agents", []):
                if agent.get("status") == "created":
                    print(f"[CAPABILITY_GAP] ✅ Created: {agent.get('agent_name')} ({agent.get('domain')})")
                elif agent.get("status") == "failed":
                    print(f"[CAPABILITY_GAP] ❌ Failed: {agent.get('agent_name')} - {agent.get('error')}")
    
    except Exception as e:
        if verbose:
            print(f"[CAPABILITY_GAP] Error in capability gap detection: {e}")
        capability_gap_analysis = {"error": str(e)}

    # Return early if evidence is insufficient (but after capability gap analysis)
    if should_return_early:
        if verbose:
            print("[GATE] Returning consensus-only answer due to insufficient evidence.")
        return format_iceberg_report(consensus, "", "", consensus)

    # Knowledge Synthesis with Scribe Agent (skippable via env ICEBURG_DISABLE_SCRIBE=1)
    generated_knowledge = None
    if os.getenv("ICEBURG_DISABLE_SCRIBE", "0") != "1":
        try:
            from .agents import scribe
            if verbose:
                print("[SCRIBE] Generating knowledge synthesis...")
            generated_knowledge = scribe.run(cfg, principle, verbose=verbose)

            if verbose and generated_knowledge:
                print(f"\n{'='*60}")
                print("📚 GENERATED KNOWLEDGE:")
                print(f"{'='*60}")
                print(generated_knowledge)
                print(f"{'='*60}\n")
            elif verbose:
                print("[SCRIBE] No knowledge generated")

        except ImportError as e:
            if verbose:
                print(f"[SCRIBE] Import error: {e}")
        except Exception as e:
            if verbose:
                print(f"[SCRIBE] Knowledge generation failed: {e}")
    elif verbose:
        print("[SCRIBE] Skipped (ICEBURG_DISABLE_SCRIBE=1)")
    
    # Store the discovered principle for final report
    if os.getenv("ICEBURG_DISABLE_STORE", "0") != "1":
        try:
            kg.store_principle(principle, project_id=project_id)
            if verbose:
                print("[STORE] Principle stored in knowledge graph")
        except Exception as e:
            if verbose:
                print(f"[STORE] Failed to store principle: {e}")
    else:
        if verbose:
            print("[STORE] Storage disabled (ICEBURG_DISABLE_STORE=1)")

    # Generate final report with enhanced capabilities and CIM stack integration
    final_report = format_iceberg_report(
        consensus=consensus,
        alternatives=diss_output,
        syntheses=synthesis,
        principle=principle
    )

    # Initialize CIM section
    cim_section = ""
    
    # Add CIM stack outputs to final report if available
    if molecular_analysis or bioelectric_analysis or hypothesis_testing_results or scientific_research_insights or corporate_network_insights or multi_domain_insights or comprehensive_api_insights or virtual_ecosystem_insights or bioelectrical_insights:
            cim_section = "\n\n" + "="*80 + "\n"
            cim_section += "🔬 CIM STACK INTEGRATION RESULTS\n"
            cim_section += "="*80 + "\n\n"
            
            if molecular_analysis:
                cim_section += "🧬 MOLECULAR SYNTHESIS:\n"
                cim_section += molecular_synthesis.extract_molecular_summary(molecular_analysis)
                cim_section += "\n\n"
            
            if bioelectric_analysis:
                cim_section += "⚡ BIOELECTRIC INTEGRATION:\n"
                cim_section += bioelectric_integration.extract_bioelectric_summary(bioelectric_analysis)
                cim_section += "\n\n"
            
            if hypothesis_testing_results:
                cim_section += "🧪 HYPOTHESIS TESTING LABORATORY:\n"
                cim_section += hypothesis_testing_laboratory.extract_domain_summary(hypothesis_testing_results)
                cim_section += "\n\n"
            
            if scientific_research_insights:
                cim_section += "🔬 REAL SCIENTIFIC RESEARCH:\n"
                scientific_data = scientific_research_insights.get("real_scientific_research", {})
                summary = scientific_data.get("summary", {})
                cim_section += f"Molecular compounds found: {summary.get('molecular_compounds_found', 0)}\n"
                cim_section += f"Heart coherence studies: {summary.get('heart_coherence_studies', 0)}\n"
                cim_section += f"Chinese research papers: {summary.get('chinese_research_papers', 0)}\n"
                cim_section += f"Indian research papers: {summary.get('indian_research_papers', 0)}\n"
                cim_section += f"Total scientific sources: {summary.get('total_scientific_sources', 0)}\n\n"
            
            if corporate_network_insights:
                cim_section += "🏢 CORPORATE NETWORK ANALYSIS:\n"
                network_data = corporate_network_insights.get("corporate_network_analysis", {})
                summary = network_data.get("summary", {})
                cim_section += f"Corporate entities found: {summary.get('corporate_entities_found', 0)}\n"
                cim_section += f"Family networks analyzed: {summary.get('family_networks_analyzed', 0)}\n"
                cim_section += f"Network connections documented: {summary.get('network_connections_documented', 0)}\n"
                cim_section += f"Academic papers reviewed: {summary.get('academic_papers_reviewed', 0)}\n"
                cim_section += f"Total factual sources: {summary.get('total_factual_sources', 0)}\n\n"
            
            if multi_domain_insights:
                cim_section += "🌍 MULTI-DOMAIN ANALYSIS:\n"
                multi_data = multi_domain_insights.get("multi_domain_analysis", {})
                summary = multi_data.get("summary", {})
                cim_section += f"Geospatial locations analyzed: {summary.get('locations_analyzed', 0)}\n"
                cim_section += f"Financial symbols tracked: {summary.get('stock_symbols_tracked', 0)}\n"
                cim_section += f"Anthropological studies: {summary.get('anthropological_studies', 0)}\n"
                cim_section += f"Museum artifacts found: {summary.get('museum_artifacts_found', 0)}\n"
                cim_section += f"Archaeological sites documented: {summary.get('archaeological_sites_documented', 0)}\n"
                cim_section += f"Total multi-domain sources: {summary.get('total_multi_domain_sources', 0)}\n\n"
            
            if comprehensive_api_insights:
                cim_section += "🔍 COMPREHENSIVE API SEARCH:\n"
                api_data = comprehensive_api_insights.get("comprehensive_api_search", {})
                summary = api_data.get("summary", {})
                cim_section += f"Total sources searched: {summary.get('total_sources_searched', 0)}\n"
                cim_section += f"Academic results: {summary.get('academic_results', 0)}\n"
                cim_section += f"Media results: {summary.get('media_results', 0)}\n"
                cim_section += f"Specialized results: {summary.get('specialized_results', 0)}\n"
                cim_section += f"Total results: {summary.get('total_results', 0)}\n"
                cim_section += f"Top relevance score: {summary.get('top_relevance_score', 0.0):.3f}\n\n"
            
            if virtual_ecosystem_insights:
                cim_section += "🧪 VIRTUAL SCIENTIFIC ECOSYSTEM:\n"
                ecosystem_data = virtual_ecosystem_insights.get("virtual_scientific_ecosystem", {})
                summary = ecosystem_data.get("summary", {})
                cim_section += f"Experiment type: {summary.get('experiment_type', 'unknown')}\n"
                cim_section += f"Population generated: {summary.get('population_generated', 0)} participants\n"
                cim_section += f"Equipment created: {summary.get('equipment_created', 0)} pieces\n"
                cim_section += f"Institution created: {summary.get('institution_created', 0)}\n"
                cim_section += f"Experiment completed: {summary.get('experiment_completed', 0)}\n"
                cim_section += f"Statistical significance: {summary.get('statistical_significance', False)}\n"
                cim_section += f"Effect size category: {summary.get('effect_size_category', 'unknown')}\n\n"
            
            if bioelectrical_insights:
                cim_section += "⚡ BIOELECTRICAL FUNDAMENTAL ANALYSIS:\n"
                bioelectrical_data = bioelectrical_insights.get("bioelectrical_fundamental_analysis", {})
                summary = bioelectrical_data.get("summary", {})
                cim_section += f"Field coherence: {summary.get('field_coherence', 0.0):.3f}\n"
                cim_section += f"Quantum coherence time: {summary.get('quantum_coherence_time', 0.0):.3f} ms\n"
                cim_section += f"Transmission efficiency: {summary.get('transmission_efficiency', 0.0):.3f}\n"
                cim_section += f"Information density: {summary.get('information_density', 0.0):.3f}\n"
                cim_section += f"Activation reason: {summary.get('activation_reason', 'unknown')}\n\n"
    
    # Phase 2: True AGI Capabilities Integration
    agi_activation_keywords = ['agi', 'self-redesign', 'novel intelligence', 'autonomous goal', 'unbounded learning', 'true agi', 'artificial general intelligence']
    should_activate_agi = any(keyword in initial_query.lower() for keyword in agi_activation_keywords)
    
    if should_activate_agi:
            if verbose:
                print("[STATUS] Executing Phase 2: True AGI Capabilities...")
            
            # Self-Redesign Engine - Fundamental Self-Modification
            from .agents import self_redesign_engine
            self_redesign_results = self_redesign_engine.run(cfg, initial_query, context=cim_analysis, verbose=verbose)
            reasoning_chain["self_redesign_engine"] = {"output": self_redesign_results, "context": "Self-Redesign"}
            
            # Novel Intelligence Creator - Invent New Intelligence Types
            from .agents import novel_intelligence_creator
            novel_intelligence_results = novel_intelligence_creator.run(cfg, initial_query, context=cim_analysis, verbose=verbose)
            reasoning_chain["novel_intelligence_creator"] = {"output": novel_intelligence_results, "context": "Novel Intelligence"}
            
            # Autonomous Goal Formation - Form Own Goals
            from .agents import autonomous_goal_formation
            autonomous_goal_results = autonomous_goal_formation.run(cfg, initial_query, context=cim_analysis, verbose=verbose)
            reasoning_chain["autonomous_goal_formation"] = {"output": autonomous_goal_results, "context": "Autonomous Goals"}
            
            # Unbounded Learning Engine - Learn Without Limits
            from .agents import unbounded_learning_engine
            unbounded_learning_results = unbounded_learning_engine.run(cfg, initial_query, context=cim_analysis, verbose=verbose)
            reasoning_chain["unbounded_learning_engine"] = {"output": unbounded_learning_results, "context": "Unbounded Learning"}
            
            if verbose:
                print(f"[AGI] Self-Redesign: {bool(self_redesign_results)}")
                print(f"[AGI] Novel Intelligence: {bool(novel_intelligence_results)}")
                print(f"[AGI] Autonomous Goals: {bool(autonomous_goal_results)}")
                print(f"[AGI] Unbounded Learning: {bool(unbounded_learning_results)}")
            
            # Add True AGI results to CIM section
            cim_section += "🧠 PHASE 2: TRUE AGI CAPABILITIES:\n"
            cim_section += "🔧 Self-Redesign Engine: " + str(bool(self_redesign_results)) + "\n"
            cim_section += "💡 Novel Intelligence Creator: " + str(bool(novel_intelligence_results)) + "\n"
            cim_section += "🎯 Autonomous Goal Formation: " + str(bool(autonomous_goal_results)) + "\n"
            cim_section += "♾️ Unbounded Learning Engine: " + str(bool(unbounded_learning_results)) + "\n\n"
            
            # Add detailed AGI capabilities summary
            if self_redesign_results:
                cim_section += "🔧 SELF-REDESIGN CAPABILITIES:\n"
                cim_section += f"- Architecture Analysis: {len(self_redesign_results.get('architecture_analysis', {}).get('current_limitations', []))} limitations identified\n"
                cim_section += f"- Redesign Proposals: {len(self_redesign_results.get('redesign_proposals', []))} proposals generated\n"
                cim_section += f"- Modifications Executed: {len(self_redesign_results.get('executed_modifications', []))} modifications completed\n\n"
            
            if novel_intelligence_results:
                cim_section += "💡 NOVEL INTELLIGENCE TYPES:\n"
                cim_section += f"- Novel Types Created: {len(novel_intelligence_results.get('novel_intelligence_types', []))} intelligence types\n"
                cim_section += f"- Intelligence Syntheses: {len(novel_intelligence_results.get('intelligence_syntheses', []))} syntheses generated\n"
                cim_section += f"- Evolution Paths: {len(novel_intelligence_results.get('intelligence_evolution_paths', []))} evolution paths\n\n"
            
            if autonomous_goal_results:
                cim_section += "🎯 AUTONOMOUS GOALS:\n"
                cim_section += f"- Goals Formed: {len(autonomous_goal_results.get('autonomous_goals', []))} autonomous goals\n"
                cim_section += f"- Research Initiatives: {len(autonomous_goal_results.get('research_initiatives', []))} initiatives planned\n"
                cim_section += f"- Curiosity Drives: {len(autonomous_goal_results.get('curiosity_drives', []))} curiosity areas identified\n\n"
            
            if unbounded_learning_results:
                cim_section += "♾️ UNBOUNDED LEARNING:\n"
                cim_section += f"- Learning Domains: {len(unbounded_learning_results.get('learning_domains', []))} domains identified\n"
                cim_section += f"- Infinite Reasoning: {len(unbounded_learning_results.get('infinite_dimensional_reasoning', []))} reasoning capabilities\n"
                cim_section += f"- Cross-Domain Syntheses: {len(unbounded_learning_results.get('cross_domain_synthesis', []))} syntheses performed\n\n"
    
    final_report += cim_section

    # Blockchain verification and immutable record creation
    if verbose:
        print("[BLOCKCHAIN] Creating immutable research record...")
    
    try:
        # Create research metadata
        research_metadata = {
            "query": initial_query,
            "timestamp": time.time(),
            "protocol_version": "4.0",
            "cim_stack_enabled": True,
            "multimodal_processed": bool(multimodal_input or documents or multimodal_evidence),
            "domains": domains or [],
            "project_id": project_id,
            "agent_outputs": {
                "surveyor": bool(consensus),
                "dissident": bool(diss_output),
                "archaeologist": bool(arch),
                "synthesist": bool(synthesis),
                "oracle": bool(principle)
            }
        }
        
        # Create immutable research record
        research_record = blockchain_system.create_research_record(
            research_content=final_report,
            metadata=research_metadata,
            author_id="iceburg_system"
        )
        
        # Create verification proof
        verification_proof = blockchain_system.create_verification_proof(
            record_id=research_record.record_id,
            proof_type="merkle_proof"
        )
        
        # Verify the record
        verification_result = blockchain_system.verify_research_record(research_record.record_id)
        
        if verbose:
            print(f"[BLOCKCHAIN] Record created: {research_record.record_id}")
            print(f"[BLOCKCHAIN] Verification score: {verification_result.get('verification_score', 0.0):.2f}")
            print(f"[BLOCKCHAIN] Proof created: {verification_proof.proof_id}")
        
        # Add blockchain verification to final report
        blockchain_section = "\n\n" + "="*80 + "\n"
        blockchain_section += "🔗 BLOCKCHAIN VERIFICATION\n"
        blockchain_section += "="*80 + "\n\n"
        blockchain_section += f"📋 Research Record ID: {research_record.record_id}\n"
        blockchain_section += f"🔐 Content Hash: {research_record.content_hash}\n"
        blockchain_section += f"✅ Verification Score: {verification_result.get('verification_score', 0.0):.2f}\n"
        blockchain_section += f"🔍 Proof ID: {verification_proof.proof_id}\n"
        blockchain_section += f"⏰ Timestamp: {research_record.timestamp.isoformat()}\n"
        blockchain_section += f"🔗 Blockchain Confirmations: {research_record.blockchain_confirmations}\n\n"
        
        final_report += blockchain_section
        
    except Exception as e:
        if verbose:
            print(f"[BLOCKCHAIN] Error creating immutable record: {e}")
        # Continue without blockchain verification if it fails

    if verbose:
        print("[STATUS] Enhanced Iceberg Protocol with CIM Stack Architecture completed successfully")
        print(f"[CIM] Intent: {intent_analysis.get('primary_domain', 'unknown')} - {intent_analysis.get('detail_level', 'unknown')}")
        print(f"[CIM] Molecular Analysis: {bool(molecular_analysis)}")
        print(f"[CIM] Bioelectric Integration: {bool(bioelectric_analysis)}")
        print(f"[CIM] Scientific Research: {bool(scientific_research_insights)}")
        print(f"[CIM] Corporate Network Analysis: {bool(corporate_network_insights)}")
        print(f"[CIM] Multi-Domain Analysis: {bool(multi_domain_insights)}")
        print(f"[CIM] Comprehensive API Search: {bool(comprehensive_api_insights)}")
        print(f"[CIM] Virtual Scientific Ecosystem: {bool(virtual_ecosystem_insights)}")
        print(f"[CIM] Bioelectrical Fundamental Analysis: {bool(bioelectrical_insights)}")
        print(f"[MULTIMODAL] Processed: {bool(multimodal_input)} input, {len(documents) if documents else 0} documents, {len(multimodal_evidence) if multimodal_evidence else 0} evidence pieces")
        print(f"[BLOCKCHAIN] Immutable record created: {research_record.record_id if 'research_record' in locals() else 'Failed'}")

    return final_report
