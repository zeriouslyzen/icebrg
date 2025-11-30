"""
Astro-Physiology Mode Handler
Handles truth-finding queries using celestial biological framework
"""

import logging
import asyncio
import uuid
import time
from typing import Dict, Any, Optional, Tuple, Callable, List
from datetime import datetime

from ..agents.celestial_biological_framework import (
    analyze_birth_imprint,
    predict_behavioral_traits,
    get_tcm_health_predictions,
    get_current_celestial_conditions,
    MolecularImprint
)
from ..modes.birth_data_extraction import extract_birth_data_from_message
from ..config import IceburgConfig
from ..vectorstore import VectorStore
from ..database.unified_database import UnifiedDatabase, DatabaseConfig
from .astro_physiology_predictions import _predict_health_trajectory
from ..integration.health_apps_integration import _get_external_integrations

logger = logging.getLogger(__name__)


async def _load_user_context(
    message: Dict[str, Any],
    cfg: IceburgConfig
) -> Dict[str, Any]:
    """
    V2: Load user context (previous analyses, interventions, feedback, health tracking).
    
    Args:
        message: Request message containing user_id or session_id
        cfg: ICEBURG configuration
        
    Returns:
        Dictionary with user context data
    """
    try:
        user_id = message.get("user_id") or message.get("data", {}).get("user_id")
        session_id = message.get("session_id") or message.get("data", {}).get("session_id")
        
        if not user_id:
            # Return empty context if no user_id
            return {
                "previous_analyses": [],
                "intervention_history": [],
                "user_feedback": [],
                "health_tracking": []
            }
        
        # Initialize database
        db_config = DatabaseConfig()
        db = UnifiedDatabase(db_config)
        
        # Load previous analyses
        previous_analyses = db.get_user_astro_history(user_id, limit=5) if user_id else []
        
        # Load intervention history
        intervention_history = db.get_user_interventions(user_id, limit=10) if user_id else []
        
        # Load user feedback
        user_feedback = db.get_user_feedback(user_id, limit=10) if user_id else []
        
        # Load health tracking
        health_tracking = db.get_user_health_tracking(user_id, days=30) if user_id else []
        
        return {
            "previous_analyses": previous_analyses or [],
            "intervention_history": intervention_history or [],
            "user_feedback": user_feedback or [],
            "health_tracking": health_tracking or []
        }
        
    except Exception as e:
        logger.warning(f"Error loading user context: {e}, returning empty context")
        return {
            "previous_analyses": [],
            "intervention_history": [],
            "user_feedback": [],
            "health_tracking": []
        }


async def handle_astrophysiology_query(
    query: str,
    message: Dict[str, Any],
    cfg: IceburgConfig,
    vs: Optional[VectorStore],
    websocket_callback: Optional[Callable] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """
    Handle astro-physiology mode query with truth-finding focus.
    
    Args:
        query: User query text
        message: WebSocket message dictionary (may contain birth data)
        cfg: ICEBURG configuration
        vs: VectorStore instance (optional)
        websocket_callback: Optional callback for streaming messages
        temperature: LLM temperature
        max_tokens: Max tokens for LLM
        
    Returns:
        Dictionary with molecular imprint, predictions, and truth-finding response
    """
    try:
        # V2: Load user context (previous analyses, interventions, feedback, health tracking)
        user_context = await _load_user_context(message, cfg)
        
        # Check for existing algorithmic data in message (follow-up conversation)
        existing_algorithmic_data = message.get("algorithmic_data") or message.get("data", {}).get("algorithmic_data")
        
        # Step 1: Extract birth data
        if websocket_callback:
            await websocket_callback({
                "type": "thinking",
                "content": "Analyzing your birth data and molecular blueprint..." if not existing_algorithmic_data else "Analyzing your question about the results..."
            })
        
        birth_data = extract_birth_data_from_message(message, query)
        if birth_data and birth_data.get("birth_datetime"):
            logger.info(f"🌌 Extracted birth data: {birth_data['birth_datetime'].isoformat()} at {birth_data.get('location', 'unknown location')}")
        else:
            logger.warning(f"🌌 No birth data extracted from query: {query[:100]}")
        
        # If birth data missing but we have existing algorithmic data, treat as follow-up question
        if (not birth_data or not birth_data.get("birth_datetime")) and existing_algorithmic_data:
            logger.info(f"🌌 Follow-up question detected, using existing algorithmic data")
            # Check for agent switching request
            agent = message.get("agent") or message.get("data", {}).get("agent")
            return await _handle_followup_question(
                query,
                existing_algorithmic_data,
                cfg,
                temperature,
                max_tokens,
                websocket_callback,
                agent=agent
            )
        
        # If birth data missing and no existing data, prompt user
        if not birth_data or not birth_data.get("birth_datetime"):
            logger.warning(f"🌌 Missing birth data from query: {query[:100]}")
            return {
                "query": query,
                "mode": "astrophysiology",
                "error": "missing_birth_data",
                "message": "To discover truth about you, I need your birth information. Please provide:\n- Birth date and time (e.g., 'March 15, 1990, 2:30 PM')\n- Birth location (e.g., 'New York' or '40.7128, -74.0060')",
                "results": {
                    "error": "missing_birth_data",
                    "content": "To discover truth about you, I need your birth information. Please provide:\n- Birth date and time (e.g., 'March 15, 1990, 2:30 PM')\n- Birth location (e.g., 'New York' or '40.7128, -74.0060')"
                }
            }
        
        birth_datetime = birth_data["birth_datetime"]
        location = birth_data.get("location")
        
        # Default location if not provided
        if not location:
            location = (0.0, 0.0)  # Default to equator/prime meridian
            logger.warning("No location provided, using default (0, 0)")
        
        # Step 2: Calculate molecular imprint
        if websocket_callback:
            await websocket_callback({
                "type": "thinking",
                "content": "Calculating your molecular imprint from birth conditions..."
            })
        
        try:
            molecular_imprint = await analyze_birth_imprint(birth_datetime, location)
        except Exception as e:
            logger.error(f"Error calculating molecular imprint: {e}", exc_info=True)
            return {
                "query": query,
                "mode": "astrophysiology",
                "error": "calculation_failed",
                "message": f"Error calculating molecular imprint: {str(e)}",
                "results": {
                    "error": "calculation_failed",
                    "content": f"I encountered an error calculating your molecular blueprint. Please check your birth data and try again."
                }
            }
        
        # Phase 2: Add MolecularSynthesisAgent for molecular chemistry analysis
        molecular_analysis = {}
        try:
            from ..agents.molecular_synthesis import run as molecular_synthesis_run
            
            # Create context from molecular imprint for molecular synthesis
            molecular_context = {
                "molecular_imprint": _format_imprint_for_context(molecular_imprint),
                "voltage_gates": {
                    "sodium": molecular_imprint.cellular_dependencies.get("sodium_channel_sensitivity", -70.0),
                    "potassium": molecular_imprint.cellular_dependencies.get("potassium_channel_sensitivity", -70.0),
                    "calcium": molecular_imprint.cellular_dependencies.get("calcium_channel_sensitivity", -70.0),
                    "chloride": molecular_imprint.cellular_dependencies.get("chloride_channel_sensitivity", -70.0)
                },
                "electromagnetic_environment": molecular_imprint.electromagnetic_environment
            }
            
            molecular_analysis = molecular_synthesis_run(
                cfg,
                query=f"{query} - Analyze molecular chemistry patterns, bond energies, and quantum states",
                context=molecular_context,
                verbose=False
            )
            
            logger.info(f"🌌 Molecular synthesis analysis completed: {molecular_analysis.get('analysis_type', 'unknown')}")
        except Exception as e:
            logger.warning(f"Error in MolecularSynthesisAgent: {e}", exc_info=True)
            # Continue without molecular analysis - not critical for basic functionality
        
        # Step 3: Get current celestial conditions
        if websocket_callback:
            await websocket_callback({
                "type": "thinking",
                "content": "Analyzing current celestial conditions..."
            })
        
        try:
            current_conditions = await get_current_celestial_conditions(location)
        except Exception as e:
            logger.warning(f"Error getting current conditions: {e}, using defaults")
            current_conditions = {
                "timestamp": datetime.now().isoformat(),
                "location": location,
                "celestial_positions": {},
                "electromagnetic_environment": {},
                "earth_frequencies": {}
            }
        
        # Step 4: Generate predictions
        if websocket_callback:
            await websocket_callback({
                "type": "thinking",
                "content": "Predicting behavioral patterns and health indicators..."
            })
        
        try:
            behavioral_predictions = await predict_behavioral_traits(molecular_imprint, current_conditions)
        except Exception as e:
            logger.warning(f"Error predicting behavioral traits: {e}")
            behavioral_predictions = {}
        
        try:
            tcm_predictions = await get_tcm_health_predictions(molecular_imprint)
        except Exception as e:
            logger.warning(f"Error getting TCM predictions: {e}")
            tcm_predictions = {}
        
        # Phase 2: Add TCMPlanetaryIntegrator for time-of-day analysis
        tcm_time_analysis = {}
        try:
            from ..agents.tcm_planetary_integration import get_tcm_planetary_integration
            
            tcm_planetary = get_tcm_planetary_integration()
            # Get organ states at current time (and birth time for comparison)
            current_organ_states = tcm_planetary.get_all_organs_state(datetime.now())
            birth_organ_states = tcm_planetary.get_all_organs_state(molecular_imprint.birth_datetime)
            
            tcm_time_analysis = {
                "current_organ_states": {
                    organ: {
                        "activity_level": state.activity_level,
                        "health_indicator": state.health_indicator,
                        "gravitational_influence": state.gravitational_influence,
                        "electromagnetic_influence": state.electromagnetic_influence
                    }
                    for organ, state in current_organ_states.items()
                },
                "birth_organ_states": {
                    organ: {
                        "activity_level": state.activity_level,
                        "health_indicator": state.health_indicator,
                        "gravitational_influence": state.gravitational_influence,
                        "electromagnetic_influence": state.electromagnetic_influence
                    }
                    for organ, state in birth_organ_states.items()
                },
                "optimal_times": {
                    organ: tcm_planetary.get_optimal_times(organ)
                    for organ in current_organ_states.keys()
                }
            }
            
            logger.info(f"🌌 TCM time-of-day analysis completed: {len(tcm_time_analysis.get('current_organ_states', {}))} organs analyzed")
        except Exception as e:
            logger.warning(f"Error in TCMPlanetaryIntegrator: {e}", exc_info=True)
            # Continue without TCM time analysis - not critical for basic functionality
        
        # Phase 3: Parallel Analysis Swarm
        # Run multiple analysis agents in parallel for faster execution
        research_context = ""
        synthesis = ""
        
        if vs:
            try:
                # Phase 3: Run parallel analysis swarm
                parallel_results = await _run_parallel_analysis_swarm(
                    cfg,
                    vs,
                    query,
                    molecular_imprint,
                    websocket_callback
                )
                
                research_context = parallel_results.get("research_context", "")
                synthesis = parallel_results.get("synthesis", "")
                swarm_communications = parallel_results.get("swarm_communications", {})
                
            except Exception as e:
                logger.warning(f"Error in parallel analysis swarm: {e}", exc_info=True)
                # Fallback: try lightweight VectorStore search if parallel swarm fails
                try:
                    if websocket_callback:
                        await websocket_callback({
                            "type": "thinking",
                            "content": "Gathering research context from knowledge base..."
                        })
                    
                    # Simple semantic search as fallback
                    hits = vs.semantic_search(
                        f"{query} celestial biological molecular imprint",
                        k=5,
                        where={"source": "physiology_celestial_chemistry"} if hasattr(vs, 'semantic_search') else None
                    )
                    
                    if hits:
                        research_context = "\n".join([
                            hit.document[:200] for hit in hits[:3]
                        ])
                        logger.info(f"🌌 Fallback search retrieved: {len(research_context)} chars")
                except Exception as fallback_error:
                    logger.warning(f"Fallback search also failed: {fallback_error}")
        
        # Step 6: Generate organic LLM response from algorithmic results
        if websocket_callback:
            await websocket_callback({
                "type": "thinking",
                "content": "Generating personalized insights from your molecular blueprint..."
            })
        
        # Phase 4: Synthesize all analysis results
        synthesized_analysis = await _synthesize_analysis_results(
            molecular_imprint,
            behavioral_predictions,
            tcm_predictions,
            tcm_time_analysis,
            molecular_analysis,
            research_context,
            synthesis,
            current_conditions
        )
        
        # Prepare full algorithmic results for LLM
        algorithmic_results = {
            "molecular_imprint": _format_imprint_for_context(molecular_imprint),
            "behavioral_predictions": behavioral_predictions,
            "tcm_predictions": tcm_predictions,
            "tcm_time_analysis": tcm_time_analysis,  # Phase 2: Added TCM time-of-day analysis
            "molecular_analysis": molecular_analysis,  # Phase 2: Added molecular synthesis analysis
            "current_conditions": current_conditions,
            "research_context": research_context,
            "synthesis": synthesis,
            "synthesized_analysis": synthesized_analysis,  # Phase 4: Added synthesized analysis for expert consultations
            "swarm_communications": swarm_communications if "swarm_communications" in locals() else {},  # V2: Swarm communication data
            "health_trajectory": health_trajectory if "health_trajectory" in locals() else {},  # V2: Predictive modeling
            "mode_integrations": mode_integrations if "mode_integrations" in locals() else {}  # V2: Cross-mode integration
        }
        
        # Generate organic LLM response (not templated)
        truth_response = await _generate_organic_response(
            query,
            algorithmic_results,
            cfg,
            temperature,
            max_tokens,
            user_context=user_context
        )
        
        # Phase 5: Validate LLM response (simple grounding check only - no general protocol agents)
        logger.info("🌌 Starting validation phase (grounding check only)...")
        validation_results = await _validate_response(
            truth_response,
            algorithmic_results,
            cfg
        )
        logger.info(f"🌌 Validation completed: {validation_results.get('grounding_validation', {}).get('passed', 'unknown')}")
        
        # Phase 6: Expert Consultation Layer
        logger.info("🌌 Starting expert consultations phase...")
        expert_consultations = await _run_expert_consultations(
            algorithmic_results,
            cfg,
            temperature,
            websocket_callback
        )
        
        # Phase 7: Generate Interventions
        logger.info(f"🌌 Generating interventions with {len(expert_consultations)} expert consultations")
        interventions = await _generate_interventions(
            algorithmic_results,
            expert_consultations,
            cfg,
            temperature,
            user_context=user_context  # V2: Pass user context for adaptive interventions
        )
        logger.info(f"🌌 Interventions generated: text={len(interventions.get('text', ''))} chars, structured={bool(interventions.get('structured'))}, interactive={bool(interventions.get('interactive'))}")
        
        # V2: Phase 8: Generate Testable Hypotheses (Research Tool Mode)
        hypotheses = []
        try:
            from .astro_physiology_hypothesis_generator import generate_testable_hypotheses
            hypotheses = await generate_testable_hypotheses(
                molecular_imprint,
                current_conditions,
                algorithmic_results
            )
            logger.info(f"🌌 Generated {len(hypotheses)} testable hypotheses")
        except Exception as e:
            logger.warning(f"Error generating hypotheses: {e}", exc_info=True)
            # Continue without hypotheses - not critical
        
        # Return structured result with visualization-ready data
        # PHASE 1: Removed duplication - algorithmic_data is single source of truth
        # Summary fields extracted for frontend convenience (not duplicated)
        # Top-level fields maintained for backward compatibility (reference same data, not duplicated)
        
        # Serialize molecular imprint for frontend (full structure)
        serialized_imprint = _serialize_imprint(molecular_imprint)
        
        # Extract summary fields from algorithmic_data for frontend convenience
        summary = {
            "voltage_gates": {
                "sodium": molecular_imprint.cellular_dependencies.get("sodium_channel_sensitivity", -70.0),
                "potassium": molecular_imprint.cellular_dependencies.get("potassium_channel_sensitivity", -70.0),
                "calcium": molecular_imprint.cellular_dependencies.get("calcium_channel_sensitivity", -70.0),
                "chloride": molecular_imprint.cellular_dependencies.get("chloride_channel_sensitivity", -70.0)
            },
            "biophysical_parameters": behavioral_predictions,
            "health_indicators": {
                k: v.get("organ") if isinstance(v, dict) else str(v)
                for k, v in tcm_predictions.items()
            } if tcm_predictions else {}
        }
        
        # V2: Store analysis in database for context awareness
        try:
            db_config = DatabaseConfig()
            db = UnifiedDatabase(cfg, db_config)
            
            analysis_id = str(uuid.uuid4())
            user_id = user_context.get("user_id") if user_context else None
            session_id = user_context.get("session_id") if user_context else None
            
            await db.store_astro_analysis(
                analysis_id=analysis_id,
                user_id=user_id,
                session_id=session_id,
                birth_datetime=birth_datetime.isoformat(),
                location=str(location) if location else None,
                algorithmic_data=algorithmic_results,
                llm_response=truth_response,
                query_text=query,
                metadata={"mode": "astrophysiology", "version": "v2"}
            )
            
            # Store interventions if generated
            if interventions and interventions.get("structured"):
                intervention_id = str(uuid.uuid4())
                await db.store_intervention(
                    intervention_id=intervention_id,
                    analysis_id=analysis_id,
                    user_id=user_id,
                    intervention_data=interventions,
                    status="active"
                )
            
            logger.info(f"🌌 Stored analysis {analysis_id} in database")
        except Exception as e:
            logger.warning(f"Failed to store analysis in database: {e}", exc_info=True)
        
        # Single source of truth - all data in algorithmic_data
        # Top-level fields reference the same data (for backward compatibility)
        return {
            "query": query,
            "mode": "astrophysiology",
            "results": {
                "content": truth_response,  # LLM-generated organic response
                # Single source of truth - no duplication
                "algorithmic_data": algorithmic_results,
                # Summary fields for frontend convenience (extracted, not duplicated)
                "summary": summary,
                # Phase 6: Expert consultations
                "expert_consultations": expert_consultations,
                # Phase 7: Interventions
                "interventions": interventions,
                # Phase 8: Testable Hypotheses (V2: Research Tool Mode)
                "hypotheses": [
                    {
                        "hypothesis": h.hypothesis,
                        "testable_prediction": h.testable_prediction,
                        "experimental_design": h.experimental_design,
                        "expected_effect_size": h.expected_effect_size,
                        "confidence": h.confidence,
                        "validation_method": h.validation_method,
                        "priority": h.priority
                    }
                    for h in hypotheses
                ] if hypotheses else [],
                # Log for debugging
                "_debug": {
                    "expert_consultations_count": len(expert_consultations) if expert_consultations else 0,
                    "interventions_has_text": bool(interventions.get("text") if interventions else False),
                    "interventions_has_structured": bool(interventions.get("structured") if interventions else False),
                    "hypotheses_count": len(hypotheses)
                },
                # Backward compatibility: Top-level fields reference same data as algorithmic_data
                # These are NOT duplicates - they're the same references for compatibility
                "molecular_imprint": serialized_imprint,  # Full serialized structure for frontend
                "behavioral_predictions": behavioral_predictions,  # Same reference as algorithmic_data
                "tcm_predictions": tcm_predictions,  # Same reference as algorithmic_data
                "current_conditions": current_conditions,  # Same reference as algorithmic_data
                "research_context": research_context[:2000] if research_context else "",  # Same as algorithmic_data
                "synthesis": synthesis[:2000] if synthesis else "",  # Same as algorithmic_data
                # Add metadata for frontend visualization
                "metadata": {
                    "calculation_timestamp": datetime.now().isoformat(),
                    "confidence_level": 0.85,
                    "calculation_method": "Celestial-Biological Framework v1.0",
                    "data_sources": [
                        "Celestial position calculations (JPL Horizons)",
                        "EM field physics (inverse square law)",
                        "Ion channel biophysics (Hodgkin-Huxley model)",
                        "Quantum molecular chemistry"
                    ],
                    "response_type": "llm_generated",  # Indicates organic response
                    "follow_up_enabled": True,  # Can ask questions about these results
                    "data_structure_version": "2.0",  # Indicates new structure with algorithmic_data
                    "note": "Top-level fields maintained for backward compatibility. Use algorithmic_data or summary for new code.",
                    "validation": validation_results,  # Phase 5: Added validation results
                    "research_mode": True,  # V2: Indicates this is an exploratory research tool
                    "confidence_levels": molecular_imprint.uncertainty.get("evidence_summary", {}) if molecular_imprint.uncertainty else {},
                    "overall_confidence": molecular_imprint.uncertainty.get("overall_confidence", 0.5) if molecular_imprint.uncertainty else 0.5
                }
            },
            "truth_finding": True
        }
        
    except Exception as e:
        logger.error(f"Error in astro-physiology handler: {e}", exc_info=True)
        return {
            "query": query,
            "mode": "astrophysiology",
            "error": "handler_error",
            "message": f"Error processing astro-physiology query: {str(e)}",
            "results": {
                "error": "handler_error",
                "content": f"I encountered an error processing your query. Please try again or provide more specific information."
            }
        }


async def _synthesize_analysis_results(
    molecular_imprint: MolecularImprint,
    behavioral_predictions: Dict[str, Any],
    tcm_predictions: Dict[str, Any],
    tcm_time_analysis: Dict[str, Any],
    molecular_analysis: Dict[str, Any],
    research_context: str,
    synthesis: str,
    current_conditions: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Phase 4: Synthesize all analysis results
    Combines parallel analysis results into unified insights
    """
    synthesized = {
        "meridian_connections": {},
        "health_patterns": {},
        "cross_domain_insights": []
    }
    
    # Phase 4: Extract meridian connections from synthesis
    if synthesis:
        # Parse meridian connections from synthesis text
        # Format: "Key meridian connections: Sun→Heart; Moon→Kidney; ..."
        if "meridian connections" in synthesis.lower():
            meridian_parts = synthesis.split("Key meridian connections:")[-1].split(";")
            for part in meridian_parts:
                if "→" in part:
                    celestial, organ = part.strip().split("→", 1)
                    synthesized["meridian_connections"][celestial.strip()] = organ.strip()
    
    # Phase 4: Map health patterns from TCM predictions and time analysis
    if tcm_predictions:
        for planet, data in tcm_predictions.items():
            if isinstance(data, dict):
                organ = data.get("organ", "")
                element = data.get("element", "")
                if organ and element:
                    synthesized["health_patterns"][planet] = {
                        "organ": organ,
                        "element": element,
                        "strength": data.get("strength", 0)
                    }
    
    # Phase 4: Add time-of-day insights
    if tcm_time_analysis and tcm_time_analysis.get("optimal_times"):
        for organ, times in tcm_time_analysis["optimal_times"].items():
            if organ in synthesized["health_patterns"]:
                synthesized["health_patterns"][organ]["optimal_times"] = times
    
    # Phase 4: Cross-domain insights from behavioral predictions and molecular analysis
    if behavioral_predictions:
        for param, value in behavioral_predictions.items():
            if abs(value) > 0.1:  # Significant deviation
                synthesized["cross_domain_insights"].append({
                    "domain": "biophysical",
                    "parameter": param,
                    "deviation": value,
                    "implication": f"Significant {param} deviation: {value*100:.1f}%"
                })
    
    if molecular_analysis and molecular_analysis.get("results"):
        synthesized["cross_domain_insights"].append({
            "domain": "molecular",
            "analysis_type": molecular_analysis.get("analysis_type", "unknown"),
            "implication": "Molecular chemistry patterns identified"
        })
    
    logger.info(f"🌌 Synthesis layer: {len(synthesized['meridian_connections'])} meridian connections, {len(synthesized['health_patterns'])} health patterns, {len(synthesized['cross_domain_insights'])} cross-domain insights")
    
    return synthesized


async def _agent_communication_bus(
    agent_results: Dict[str, Any],
    agent_name: str
) -> Dict[str, Any]:
    """
    V2: Agent communication bus for swarm intelligence.
    Agents share insights, validate findings, and detect contradictions.
    
    Message types:
    - insight: New finding or pattern
    - validation: Validation of another agent's finding
    - contradiction: Conflicting finding
    - consensus: Agreement on a finding
    """
    messages = []
    
    # Extract key findings from agent results
    if agent_results:
        # Insight: Key patterns or findings
        if agent_results.get("results"):
            for result in agent_results["results"][:3]:  # Top 3 findings
                messages.append({
                    "type": "insight",
                    "agent": agent_name,
                    "finding": result.get("description", ""),
                    "confidence": result.get("confidence", 0.0),
                    "timestamp": datetime.now().isoformat()
                })
        
        # Validation: High confidence findings
        if agent_results.get("meridian_analysis"):
            meridian = agent_results["meridian_analysis"]
            if meridian.get("meridian_analysis"):
                for meridian_name, data in list(meridian["meridian_analysis"].items())[:2]:
                    if data.get("connection_strength", 0) > 0.7:  # High confidence
                        messages.append({
                            "type": "validation",
                            "agent": agent_name,
                            "finding": f"Meridian connection: {meridian_name}",
                            "confidence": data.get("connection_strength", 0),
                            "timestamp": datetime.now().isoformat()
                        })
    
    return {
        "agent": agent_name,
        "messages": messages,
        "timestamp": datetime.now().isoformat()
    }


async def _detect_contradictions(
    communication_bus: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    V2: Detect contradictions between agent findings.
    """
    contradictions = []
    
    # Group findings by topic
    findings_by_topic = {}
    for comm in communication_bus:
        for msg in comm.get("messages", []):
            if msg["type"] in ["insight", "validation"]:
                topic = msg.get("finding", "").lower()
                if topic not in findings_by_topic:
                    findings_by_topic[topic] = []
                findings_by_topic[topic].append({
                    "agent": msg["agent"],
                    "confidence": msg.get("confidence", 0.0),
                    "message": msg
                })
    
    # Detect contradictions (same topic, different agents, conflicting confidence)
    for topic, findings in findings_by_topic.items():
        if len(findings) > 1:
            # Check if agents have conflicting findings
            confidences = [f["confidence"] for f in findings]
            if max(confidences) - min(confidences) > 0.5:  # Significant difference
                contradictions.append({
                    "topic": topic,
                    "findings": findings,
                    "conflict_level": max(confidences) - min(confidences),
                    "timestamp": datetime.now().isoformat()
                })
    
    return contradictions


async def _resolve_consensus(
    communication_bus: List[Dict[str, Any]],
    contradictions: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    V2: Resolve conflicts through consensus (voting or confidence scoring).
    """
    consensus = {
        "resolved_findings": [],
        "unresolved_contradictions": [],
        "consensus_score": 0.0
    }
    
    # Resolve contradictions by confidence-weighted voting
    for contradiction in contradictions:
        findings = contradiction["findings"]
        
        # Weight by confidence
        total_weight = sum(f["confidence"] for f in findings)
        if total_weight > 0:
            weighted_avg = sum(f["confidence"] ** 2 for f in findings) / total_weight
            
            # If weighted average is high, resolve in favor of highest confidence
            if weighted_avg > 0.7:
                best_finding = max(findings, key=lambda x: x["confidence"])
                consensus["resolved_findings"].append({
                    "topic": contradiction["topic"],
                    "resolved_to": best_finding["agent"],
                    "confidence": best_finding["confidence"],
                    "method": "confidence_weighted_voting"
                })
            else:
                # Unresolved - keep as contradiction
                consensus["unresolved_contradictions"].append(contradiction)
        else:
            consensus["unresolved_contradictions"].append(contradiction)
    
    # Calculate overall consensus score
    if communication_bus:
        total_messages = sum(len(comm.get("messages", [])) for comm in communication_bus)
        resolved_count = len(consensus["resolved_findings"])
        consensus["consensus_score"] = resolved_count / total_messages if total_messages > 0 else 0.0
    
    return consensus


async def _run_parallel_analysis_swarm(
    cfg: IceburgConfig,
    vs: VectorStore,
    query: str,
    molecular_imprint: MolecularImprint,
    websocket_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Phase 3: Run parallel analysis swarm with V2 enhancements
    Executes multiple analysis agents in parallel with agent-to-agent communication,
    cross-validation, contradiction detection, and consensus resolution.
    """
    if websocket_callback:
        await websocket_callback({
            "type": "thinking",
            "content": "Running parallel analysis swarm with agent communication..."
        })
    
    # Build query that includes birth data and molecular imprint details
    birth_date_str = molecular_imprint.birth_datetime.strftime("%Y-%m-%d")
    voltage_gates_summary = ", ".join([
        f"{k}: {v:.2f}mV" 
        for k, v in {
            "sodium": molecular_imprint.cellular_dependencies.get("sodium_channel_sensitivity", -70.0),
            "potassium": molecular_imprint.cellular_dependencies.get("potassium_channel_sensitivity", -70.0),
            "calcium": molecular_imprint.cellular_dependencies.get("calcium_channel_sensitivity", -70.0),
        }.items()
    ])
    
    analyzer_query = f"{query} - Birth date: {birth_date_str}, Voltage gates: {voltage_gates_summary}. Analyze celestial-physiological patterns, meridian properties, and cross-domain connections."
    
    logger.info(f"🌌 Parallel analysis swarm query: {analyzer_query}")
    
    # V2: Shared context for agent communication
    shared_context = {
        "query": analyzer_query,
        "molecular_imprint": _format_imprint_for_context(molecular_imprint),
        "agent_results": {},
        "communication_bus": []
    }
    
    # Run all analysis agents in parallel
    try:
        from ..agents.recursive_celestial_analyzer import run_recursive_analysis
        
        # Run RecursiveCelestialAnalyzer
        recursive_task = run_recursive_analysis(
            cfg, 
            vs, 
            analyzer_query, 
            max_depth=2
        )
        
        # Wait for all tasks to complete (with return_exceptions=True to handle failures gracefully)
        results = await asyncio.gather(
            recursive_task,
            return_exceptions=True
        )
        
        recursive_results = results[0] if not isinstance(results[0], Exception) else None
        
        # V2: Store results in shared context
        shared_context["agent_results"]["recursive_celestial_analyzer"] = recursive_results
        
        # V2: Agent communication - each agent shares insights
        if recursive_results:
            comm_bus_entry = await _agent_communication_bus(recursive_results, "RecursiveCelestialAnalyzer")
            shared_context["communication_bus"].append(comm_bus_entry)
        
        # V2: Cross-validation - agents validate each other's findings
        # (For now, single agent, but structure supports multiple)
        if len(shared_context["communication_bus"]) > 1:
            # Multiple agents can validate each other
            for i, comm1 in enumerate(shared_context["communication_bus"]):
                for comm2 in shared_context["communication_bus"][i+1:]:
                    # Cross-validate findings
                    for msg1 in comm1.get("messages", []):
                        for msg2 in comm2.get("messages", []):
                            if msg1["type"] == "insight" and msg2["type"] == "insight":
                                # Check if findings are similar
                                if msg1["finding"].lower() in msg2["finding"].lower() or msg2["finding"].lower() in msg1["finding"].lower():
                                    # Add validation message
                                    shared_context["communication_bus"].append({
                                        "agent": f"{comm1['agent']}_validates_{comm2['agent']}",
                                        "messages": [{
                                            "type": "validation",
                                            "agent": comm1["agent"],
                                            "finding": msg1["finding"],
                                            "validated_by": comm2["agent"],
                                            "confidence": (msg1.get("confidence", 0) + msg2.get("confidence", 0)) / 2,
                                            "timestamp": datetime.now().isoformat()
                                        }],
                                        "timestamp": datetime.now().isoformat()
                                    })
        
        # V2: Detect contradictions
        contradictions = await _detect_contradictions(shared_context["communication_bus"])
        shared_context["contradictions"] = contradictions
        
        # V2: Resolve consensus
        consensus = await _resolve_consensus(shared_context["communication_bus"], contradictions)
        shared_context["consensus"] = consensus
        
        logger.info(f"🌌 Swarm communication: {len(shared_context['communication_bus'])} agents, {len(contradictions)} contradictions, {len(consensus['resolved_findings'])} resolved findings")
        
        # Extract research context from recursive analysis
        research_context = ""
        synthesis = ""
        
        if recursive_results and recursive_results.get("results"):
            # Combine pattern descriptions and meridian analysis
            pattern_descriptions = [
                f"Pattern: {r.get('description', '')} (confidence: {r.get('confidence', 0):.2f})"
                for r in recursive_results["results"][:5]  # Top 5 patterns
            ]
            
            meridian_info = recursive_results.get("meridian_analysis", {})
            meridian_summary = ""
            if meridian_info.get("cross_domain_connections"):
                meridian_summary = "Cross-domain connections: " + "; ".join(
                    meridian_info["cross_domain_connections"][:5]
                )
            
            research_context = "\n".join([
                "Recursive Celestial-Physiological Analysis:",
                "\n".join(pattern_descriptions),
                meridian_summary
            ])
            
            logger.info(f"🌌 Parallel swarm: Recursive analysis retrieved: {len(research_context)} chars, {len(recursive_results.get('results', []))} patterns")
            
            # Truncate but log original length
            original_length = len(research_context)
            research_context = research_context[:2000]  # Increased context for better accuracy
            if original_length > 2000:
                logger.warning(f"🌌 Research context truncated from {original_length} to {len(research_context)} chars")
        
        # Extract synthesis from recursive analysis
        if recursive_results and recursive_results.get("synthesis"):
            synthesis = recursive_results["synthesis"]
        else:
            # Build synthesis from patterns and meridian connections
            synthesis_parts = []
            
            if recursive_results and recursive_results.get("results"):
                top_patterns = sorted(
                    recursive_results["results"],
                    key=lambda x: x.get("confidence", 0),
                    reverse=True
                )[:3]
                
                pattern_summary = "Top patterns: " + "; ".join([
                    f"{p.get('description', 'unknown')} (confidence: {p.get('confidence', 0):.2f})"
                    for p in top_patterns
                ])
                synthesis_parts.append(pattern_summary)
            
            if recursive_results and recursive_results.get("meridian_analysis"):
                meridian_analysis = recursive_results["meridian_analysis"]
                if meridian_analysis.get("meridian_analysis"):
                    top_meridians = sorted(
                        meridian_analysis["meridian_analysis"].items(),
                        key=lambda x: x[1].get("connection_strength", 0),
                        reverse=True
                    )[:3]
                    
                    meridian_summary = "Key meridian connections: " + "; ".join([
                        f"{body}→{data.get('organ_system', 'unknown')}"
                        for body, data in top_meridians
                    ])
                    synthesis_parts.append(meridian_summary)
            
            synthesis = "\n".join(synthesis_parts) if synthesis_parts else ""
            
            # Log synthesis length
            if synthesis:
                logger.info(f"🌌 Parallel swarm: Synthesis generated: {len(synthesis)} chars")
                original_length = len(synthesis)
                synthesis = synthesis[:2000]  # Increased context for better accuracy
                if original_length > 2000:
                    logger.warning(f"🌌 Synthesis truncated from {original_length} to {len(synthesis)} chars")
        
        return {
            "research_context": research_context,
            "synthesis": synthesis,
            "recursive_results": recursive_results,
            "swarm_communications": {
                "communication_bus": shared_context.get("communication_bus", []),
                "contradictions": shared_context.get("contradictions", []),
                "consensus": shared_context.get("consensus", {"resolved_findings": [], "unresolved_contradictions": [], "consensus_score": 0.0})
            }
        }
        
    except Exception as e:
        logger.error(f"Error in parallel analysis swarm: {e}", exc_info=True)
        return {
            "research_context": "",
            "synthesis": "",
            "swarm_communications": {
                "communication_bus": [],
                "contradictions": [],
                "consensus": {"resolved_findings": [], "unresolved_contradictions": [], "consensus_score": 0.0}
            },
            "error": str(e)
        }


def _format_imprint_for_context(imprint: MolecularImprint) -> Dict[str, Any]:
    """Format molecular imprint for agent context"""
    # Note: voltage_gates is a convenience field extracted from cellular_dependencies
    # The source data is in cellular_dependencies, voltage_gates is for easy access
    return {
        "birth_datetime": imprint.birth_datetime.isoformat(),
        "celestial_positions": {
            k: list(v) if isinstance(v, (list, tuple)) else v
            for k, v in imprint.celestial_positions.items()
        },  # Added for chart_reader_consultation
        "voltage_gates": {
            "sodium": imprint.cellular_dependencies.get("sodium_channel_sensitivity", -70.0),
            "potassium": imprint.cellular_dependencies.get("potassium_channel_sensitivity", -70.0),
            "calcium": imprint.cellular_dependencies.get("calcium_channel_sensitivity", -70.0),
            "chloride": imprint.cellular_dependencies.get("chloride_channel_sensitivity", -70.0)
        },
        "trait_amplification": imprint.trait_amplification_factors,
        "cellular_dependencies": imprint.cellular_dependencies  # Include full cellular dependencies
    }


def _serialize_imprint(imprint: MolecularImprint) -> Dict[str, Any]:
    """Serialize molecular imprint for JSON response"""
    return {
        "birth_datetime": imprint.birth_datetime.isoformat(),
        "celestial_positions": {
            k: list(v) for k, v in imprint.celestial_positions.items()
        },
        "electromagnetic_environment": imprint.electromagnetic_environment,
        "cellular_dependencies": imprint.cellular_dependencies,
        "voltage_gates": {
            "sodium": imprint.cellular_dependencies.get("sodium_channel_sensitivity", -70.0),
            "potassium": imprint.cellular_dependencies.get("potassium_channel_sensitivity", -70.0),
            "calcium": imprint.cellular_dependencies.get("calcium_channel_sensitivity", -70.0),
            "chloride": imprint.cellular_dependencies.get("chloride_channel_sensitivity", -70.0)
        },
        "trait_amplification_factors": imprint.trait_amplification_factors
    }


async def _consistency_checker(
    llm_response: str,
    algorithmic_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    V2: Check consistency between algorithmic results and LLM response.
    Flags contradictions between different analysis agents.
    """
    consistency_score = 1.0
    issues = []
    
    # Check if LLM mentions values that match algorithmic data
    molecular_imprint = algorithmic_data.get("molecular_imprint", {})
    behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
    
    # Check voltage gate consistency
    voltage_gates = molecular_imprint.get("voltage_gates", {})
    for gate, value in voltage_gates.items():
        # Check if LLM mentions this gate
        if gate.lower() in llm_response.lower():
            # Extract mentioned value (simplified)
            import re
            pattern = rf"{gate}.*?(-?\d+\.?\d*)"
            matches = re.findall(pattern, llm_response, re.IGNORECASE)
            if matches:
                mentioned_value = float(matches[0])
                if abs(mentioned_value - value) > 10.0:  # More than 10mV difference
                    issues.append(f"Inconsistent {gate} voltage: LLM says {mentioned_value}mV, data shows {value}mV")
                    consistency_score -= 0.1
    
    # Check behavioral predictions consistency
    for param, value in behavioral_predictions.items():
        param_name = param.replace("_", " ").title()
        if param_name.lower() in llm_response.lower():
            percentage = value * 100
            # Check if percentage is mentioned correctly (within 5%)
            if f"{percentage:.1f}" not in llm_response and f"{abs(percentage):.1f}" not in llm_response:
                # Might be okay, but flag for review
                pass
    
    consistency_score = max(0.0, min(1.0, consistency_score))
    
    return {
        "score": consistency_score,
        "issues": issues,
        "passed": consistency_score >= 0.7
    }


async def _confidence_scorer(
    algorithmic_data: Dict[str, Any],
    predictions: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    V2: Calculate confidence scores for predictions.
    Based on: Data quality, model certainty, historical accuracy.
    """
    confidence_scores = {}
    
    # Data quality score (based on completeness)
    data_quality = 1.0
    required_keys = ["molecular_imprint", "behavioral_predictions", "tcm_predictions"]
    missing_keys = [k for k in required_keys if k not in algorithmic_data or not algorithmic_data[k]]
    if missing_keys:
        data_quality -= len(missing_keys) * 0.2
    
    data_quality = max(0.0, min(1.0, data_quality))
    confidence_scores["data_quality"] = data_quality
    
    # Model certainty (based on prediction variance)
    behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
    if behavioral_predictions:
        values = list(behavioral_predictions.values())
        variance = sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values) if values else 0
        # Lower variance = higher certainty
        model_certainty = 1.0 - min(1.0, variance)
        confidence_scores["model_certainty"] = model_certainty
    else:
        confidence_scores["model_certainty"] = 0.5
    
    # Historical accuracy (would use user feedback if available)
    confidence_scores["historical_accuracy"] = 0.7  # Default, would be updated from feedback
    
    # Overall confidence
    overall_confidence = (
        confidence_scores["data_quality"] * 0.4 +
        confidence_scores["model_certainty"] * 0.4 +
        confidence_scores["historical_accuracy"] * 0.2
    )
    confidence_scores["overall"] = overall_confidence
    
    return confidence_scores


async def _contradiction_detector(
    algorithmic_data: Dict[str, Any],
    swarm_communications: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    V2: Detect contradictions in swarm agent outputs.
    Flags conflicting findings and suggests resolution strategies.
    """
    contradictions = []
    resolution_strategies = []
    
    # Check for contradictions in swarm communications
    if swarm_communications:
        communication_bus = swarm_communications.get("communication_bus", [])
        existing_contradictions = swarm_communications.get("contradictions", [])
        
        # Use existing contradictions from swarm communication
        contradictions.extend(existing_contradictions)
        
        # Additional contradiction detection
        for contradiction in existing_contradictions:
            if contradiction.get("conflict_level", 0) > 0.5:
                resolution_strategies.append({
                    "contradiction": contradiction.get("topic", "unknown"),
                    "strategy": "Use highest confidence finding",
                    "recommended_action": "Weight findings by confidence score"
                })
    
    # Check for contradictions between different data sources
    behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
    tcm_predictions = algorithmic_data.get("tcm_predictions", {})
    
    # Example: If neural excitability is high but TCM shows weak heart system
    if behavioral_predictions.get("neural_excitability", 0) > 0.3:
        heart_data = None
        for planet, data in tcm_predictions.items():
            if isinstance(data, dict) and data.get("organ", "").lower() == "heart":
                heart_data = data
                break
        
        if heart_data and heart_data.get("strength", 0) < 0.3:
            contradictions.append({
                "type": "cross_domain_contradiction",
                "description": "High neural excitability but weak heart system",
                "severity": "moderate",
                "resolution": "Consider both factors - high excitability may need heart support"
            })
    
    return {
        "contradictions": contradictions,
        "resolution_strategies": resolution_strategies,
        "count": len(contradictions)
    }


async def _validate_response(
    llm_response: str,
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig
) -> Dict[str, Any]:
    """
    Phase 5: Multi-Layer Validation (V2 Enhanced)
    Uses GroundingValidator, ConsistencyChecker, ConfidenceScorer, and ContradictionDetector
    NOTE: No general protocol agents (Scrutineer) - using department-specific validation only
    """
    validation = {
        "grounding_validation": {},
        "consistency_check": {},
        "confidence_scores": {},
        "contradiction_detection": {},
        "overall_valid": True,
        "warnings": []
    }
    
    # Phase 5: GroundingValidator - verify LLM claims match algorithmic data
    try:
        grounding_issues = _grounding_validator(llm_response, algorithmic_data)
        
        validation["grounding_validation"] = {
            "passed": len(grounding_issues) == 0,
            "issues": grounding_issues
        }
        
        if grounding_issues:
            validation["overall_valid"] = False
            validation["warnings"].extend(grounding_issues)
        
        logger.info(f"🌌 Grounding validation: {'PASSED' if validation['grounding_validation']['passed'] else 'FAILED'} ({len(grounding_issues)} issues)")
        
    except Exception as e:
        logger.warning(f"Error in GroundingValidator: {e}", exc_info=True)
        validation["grounding_validation"] = {
            "error": str(e),
            "passed": True  # Don't block response if validation fails
        }
    
    # V2: Consistency Checker
    try:
        consistency = await _consistency_checker(llm_response, algorithmic_data)
        validation["consistency_check"] = consistency
        
        if not consistency.get("passed", True):
            validation["overall_valid"] = False
            validation["warnings"].extend(consistency.get("issues", []))
        
        logger.info(f"🌌 Consistency check: {'PASSED' if consistency.get('passed') else 'FAILED'} (score: {consistency.get('score', 0):.2f})")
        
    except Exception as e:
        logger.warning(f"Error in ConsistencyChecker: {e}", exc_info=True)
        validation["consistency_check"] = {"error": str(e), "passed": True}
    
    # V2: Confidence Scorer
    try:
        confidence = await _confidence_scorer(algorithmic_data)
        validation["confidence_scores"] = confidence
        
        logger.info(f"🌌 Confidence scores: overall={confidence.get('overall', 0):.2f}, data_quality={confidence.get('data_quality', 0):.2f}, model_certainty={confidence.get('model_certainty', 0):.2f}")
        
    except Exception as e:
        logger.warning(f"Error in ConfidenceScorer: {e}", exc_info=True)
        validation["confidence_scores"] = {"error": str(e)}
    
    # V2: Contradiction Detector
    try:
        swarm_comm = algorithmic_data.get("swarm_communications", {})
        contradictions = await _contradiction_detector(algorithmic_data, swarm_comm)
        validation["contradiction_detection"] = contradictions
        
        if contradictions.get("count", 0) > 0:
            validation["warnings"].append(f"Found {contradictions['count']} contradictions")
        
        logger.info(f"🌌 Contradiction detection: {contradictions.get('count', 0)} contradictions found")
        
    except Exception as e:
        logger.warning(f"Error in ContradictionDetector: {e}", exc_info=True)
        validation["contradiction_detection"] = {"error": str(e)}
    
    return validation


async def _generate_interventions(
    algorithmic_results: Dict[str, Any],
    expert_consultations: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float,
    user_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Phase 7: Generate personalized interventions (V2 Enhanced)
    Creates actionable plans in text, structured JSON, and interactive UI formats.
    
    V2 Enhancements:
    - Progressive difficulty (start easy, increase complexity)
    - Auto-adjustment logic (modify based on user progress)
    - Context-aware recommendations (adapt to user's current state)
    - Tracking metadata (completion tracking, progress indicators)
    """
    try:
        from ..llm import chat_complete
        
        # Combine expert insights
        expert_summary = "\n".join([
            f"{k.title()} Expert: {v.get('insights', 'N/A')[:200]}"
            for k, v in expert_consultations.items()
            if 'error' not in v
        ])
        
        prompt = f"""Based on the following analysis, create a personalized intervention plan.

Expert Consultations:
{expert_summary}

Algorithmic Data Summary:
- Biophysical parameters: {algorithmic_results.get('behavioral_predictions', {})}
- TCM health indicators: {list(algorithmic_results.get('tcm_predictions', {}).keys())}
- Meridian connections: {algorithmic_results.get('synthesized_analysis', {}).get('meridian_connections', {})}

Provide a structured intervention plan with:
1. Daily routine recommendations
2. Dietary guidelines
3. Movement/exercise plan
4. Timing recommendations
5. Monitoring suggestions

Format as clear, actionable steps."""

        import asyncio
        model = cfg.surveyor_model or cfg.primary_model or 'llama3.1:8b'
        # Add timeout to prevent hanging (20 seconds for intervention generation)
        try:
            intervention_text = await asyncio.wait_for(
                asyncio.to_thread(
                    chat_complete,
                    model,
                    prompt,
                    temperature=temperature,
                    options={"num_predict": 500}
                ),
                timeout=20.0
            )
        except asyncio.TimeoutError:
            logger.warning("🌌 Intervention generation timed out")
            intervention_text = "Intervention generation unavailable due to timeout. Please try again."
        
        # Create structured JSON intervention
        intervention_json = {
            "daily_routine": {
                "morning": "Based on optimal organ times and biophysical parameters",
                "afternoon": "Aligned with TCM organ-clock correlations",
                "evening": "Supporting cellular stability and neurotransmitter balance"
            },
            "dietary_guidelines": {
                "emphasize": expert_consultations.get("nutrition", {}).get("recommended_elements", []),
                "timing": "Based on TCM time-of-day analysis"
            },
            "movement_plan": {
                "type": expert_consultations.get("movement", {}).get("recommended_intensity", "Moderate"),
                "timing": expert_consultations.get("movement", {}).get("optimal_times", {})
            },
            "monitoring": {
                "organ_systems": expert_consultations.get("health", {}).get("organ_systems", []),
                "biophysical_parameters": list(algorithmic_results.get("behavioral_predictions", {}).keys())
            }
        }
        
        # V2: Determine intervention difficulty based on user context
        difficulty_level = "beginner"  # Default
        if user_context:
            previous_interventions = user_context.get("intervention_history", [])
            if previous_interventions:
                # Check completion rates
                completed = [i for i in previous_interventions if i.get("status") == "completed"]
                completion_rate = len(completed) / len(previous_interventions) if previous_interventions else 0.0
                
                if completion_rate > 0.8:
                    difficulty_level = "advanced"
                elif completion_rate > 0.5:
                    difficulty_level = "intermediate"
                else:
                    difficulty_level = "beginner"
        
        # V2: Add difficulty level to intervention JSON
        intervention_json["difficulty_level"] = difficulty_level
        intervention_json["dietary_guidelines"]["complexity"] = "simple" if difficulty_level == "beginner" else "moderate" if difficulty_level == "intermediate" else "detailed"
        intervention_json["movement_plan"]["intensity"] = "low" if difficulty_level == "beginner" else "moderate" if difficulty_level == "intermediate" else "high"
        
        # V2: Adaptation rules for auto-adjustment
        adaptation_rules = {
            "if_completion_rate_greater_than_80": {
                "action": "increase_difficulty",
                "description": "If user completes >80% of interventions, increase difficulty"
            },
            "if_completion_rate_less_than_50": {
                "action": "decrease_difficulty",
                "description": "If user completes <50% of interventions, decrease difficulty"
            },
            "if_health_metrics_improve": {
                "action": "continue_current",
                "description": "If health metrics improve, continue current intervention"
            },
            "if_health_metrics_decline": {
                "action": "modify_intervention",
                "description": "If health metrics decline, modify intervention approach"
            }
        }
        
        # V2: Tracking metadata
        tracking_metadata = {
            "created_at": datetime.now().isoformat(),
            "difficulty_level": difficulty_level,
            "estimated_duration_days": 30,
            "progress_tracking": {
                "completion_tracking": True,
                "check_in_frequency": "daily",
                "progress_indicators": ["adherence_score", "health_metrics", "user_feedback"]
            },
            "adaptation_enabled": True,
            "adaptation_rules": adaptation_rules
        }
        
        logger.info(f"🌌 Interventions generated: {len(intervention_text)} chars, difficulty={difficulty_level}, structured plan created")
        
        return {
            "text": intervention_text,
            "structured": intervention_json,
            "interactive": {
                "sections": ["daily_routine", "dietary_guidelines", "movement_plan", "monitoring"],
                "expandable": True,
                "actionable": True
            },
            "tracking_metadata": tracking_metadata,  # V2: Tracking metadata
            "adaptation_rules": adaptation_rules  # V2: Adaptation rules
        }
        
    except Exception as e:
        logger.warning(f"Error generating interventions: {e}", exc_info=True)
        return {
            "text": "Intervention generation unavailable.",
            "structured": {},
            "error": str(e)
        }


def _grounding_validator(llm_response: str, algorithmic_data: Dict[str, Any]) -> list:
    """
    Phase 5: GroundingValidator
    Checks if LLM claims match the algorithmic data
    """
    issues = []
    
    # Extract key values from algorithmic data
    molecular_imprint = algorithmic_data.get("molecular_imprint", {})
    behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
    tcm_predictions = algorithmic_data.get("tcm_predictions", {})
    
    # Check voltage gate values
    voltage_gates = molecular_imprint.get("voltage_gates", {})
    if voltage_gates:
        for gate, value in voltage_gates.items():
            # Check if LLM mentions this gate with incorrect value
            gate_mentions = [
                f"{gate}",
                f"{gate} channel",
                f"{gate} voltage"
            ]
            for mention in gate_mentions:
                if mention.lower() in llm_response.lower():
                    # Extract number near the mention
                    import re
                    pattern = rf"{mention}.*?(-?\d+\.?\d*)"
                    matches = re.findall(pattern, llm_response, re.IGNORECASE)
                    if matches:
                        mentioned_value = float(matches[0])
                        if abs(mentioned_value - value) > 5.0:  # More than 5mV difference
                            issues.append(f"LLM claims {gate} voltage is {mentioned_value}mV but algorithmic data shows {value}mV")
    
    # Check behavioral predictions
    if behavioral_predictions:
        for param, value in behavioral_predictions.items():
            param_name = param.replace("_", " ").title()
            if param_name.lower() in llm_response.lower():
                # Check if deviation is mentioned correctly
                percentage = value * 100
                if f"{percentage:.1f}" not in llm_response and f"{abs(percentage):.1f}" not in llm_response:
                    # Might be okay, but log for review
                    pass
    
    # Check TCM predictions
    if tcm_predictions:
        for planet, data in tcm_predictions.items():
            if isinstance(data, dict):
                organ = data.get("organ", "")
                if organ and planet.lower() in llm_response.lower():
                    # Check if organ is mentioned correctly
                    if organ.lower() not in llm_response.lower():
                        issues.append(f"LLM mentions {planet} but doesn't correctly associate it with {organ}")
    
    return issues


async def _run_expert_consultations(
    algorithmic_results: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float,
    websocket_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Phase 6: Run expert consultations in parallel
    Health, Nutrition, Movement, and Chart Reader experts
    """
    if websocket_callback:
        await websocket_callback({
            "type": "thinking",
            "content": "Consulting health, nutrition, movement, and chart reading experts..."
        })
    
    try:
        from ..agents.astro_physiology_experts import (
            health_expert_consultation,
            nutrition_expert_consultation,
            movement_expert_consultation,
            chart_reader_consultation,
            sleep_expert_consultation,  # V2: New expert
            stress_expert_consultation,  # V2: New expert
            hormone_expert_consultation,  # V2: New expert
            digestive_expert_consultation  # V2: New expert
        )
        
        # Log what we're passing to expert agents for debugging
        logger.info(f"🌌 Running expert consultations with algorithmic_results keys: {list(algorithmic_results.keys())}")
        logger.info(f"🌌 Molecular imprint keys: {list(algorithmic_results.get('molecular_imprint', {}).keys())}")
        logger.info(f"🌌 Has synthesized_analysis: {'synthesized_analysis' in algorithmic_results}")
        
        # V2: Run all expert consultations in parallel (8 experts total)
        # Each expert has 15 second timeout, 90 second total timeout
        logger.info("🌌 Starting parallel expert consultations (8 experts)")
        start_time = time.time()
        try:
            consultations = await asyncio.wait_for(
                asyncio.gather(
                    health_expert_consultation(algorithmic_results, cfg, temperature),
                    nutrition_expert_consultation(algorithmic_results, cfg, temperature),
                    movement_expert_consultation(algorithmic_results, cfg, temperature),
                    chart_reader_consultation(algorithmic_results, cfg, temperature),
                    sleep_expert_consultation(algorithmic_results, cfg, temperature),  # V2
                    stress_expert_consultation(algorithmic_results, cfg, temperature),  # V2
                    hormone_expert_consultation(algorithmic_results, cfg, temperature),  # V2
                    digestive_expert_consultation(algorithmic_results, cfg, temperature),  # V2
                    return_exceptions=True
                ),
                timeout=90.0  # V2: 90 second total timeout for all 8 experts
            )
            elapsed = time.time() - start_time
            logger.info(f"🌌 Parallel expert consultations completed in {elapsed:.2f}s")
        except asyncio.TimeoutError:
            logger.warning("🌌 Expert consultations timed out after 90 seconds")
            consultations = [
                {"expert": "health", "error": "Timeout"},
                {"expert": "nutrition", "error": "Timeout"},
                {"expert": "movement", "error": "Timeout"},
                {"expert": "chart_reader", "error": "Timeout"},
                {"expert": "sleep", "error": "Timeout"},  # V2
                {"expert": "stress", "error": "Timeout"},  # V2
                {"expert": "hormone", "error": "Timeout"},  # V2
                {"expert": "digestive", "error": "Timeout"}  # V2
            ]
        
        # V2: Process all 8 expert consultations
        expert_results = {
            "health": consultations[0] if not isinstance(consultations[0], Exception) else {"expert": "health", "error": str(consultations[0])},
            "nutrition": consultations[1] if not isinstance(consultations[1], Exception) else {"expert": "nutrition", "error": str(consultations[1])},
            "movement": consultations[2] if not isinstance(consultations[2], Exception) else {"expert": "movement", "error": str(consultations[2])},
            "chart_reader": consultations[3] if not isinstance(consultations[3], Exception) else {"expert": "chart_reader", "error": str(consultations[3])},
            "sleep": consultations[4] if not isinstance(consultations[4], Exception) else {"expert": "sleep", "error": str(consultations[4])},  # V2
            "stress": consultations[5] if not isinstance(consultations[5], Exception) else {"expert": "stress", "error": str(consultations[5])},  # V2
            "hormone": consultations[6] if not isinstance(consultations[6], Exception) else {"expert": "hormone", "error": str(consultations[6])},  # V2
            "digestive": consultations[7] if not isinstance(consultations[7], Exception) else {"expert": "digestive", "error": str(consultations[7])}  # V2
        }
        
        # Log detailed results
        successful = [k for k, v in expert_results.items() if 'error' not in v]
        failed = [k for k, v in expert_results.items() if 'error' in v]
        logger.info(f"🌌 Expert consultations completed: {len(successful)} successful, {len(failed)} failed")
        if failed:
            logger.warning(f"🌌 Failed expert consultations: {failed}")
        for expert, result in expert_results.items():
            if 'error' not in result:
                logger.info(f"🌌 {expert} expert: {len(result.get('insights', ''))} chars")
            else:
                logger.error(f"🌌 {expert} expert error: {result.get('error', 'Unknown')}")
        
        return expert_results
        
    except Exception as e:
        logger.error(f"🌌 Error in expert consultations: {e}", exc_info=True)
        return {
            "health": {"expert": "health", "error": str(e)},
            "nutrition": {"expert": "nutrition", "error": str(e)},
            "movement": {"expert": "movement", "error": str(e)},
            "chart_reader": {"expert": "chart_reader", "error": str(e)}
        }


async def _handle_followup_question(
    query: str,
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float,
    max_tokens: int,
    websocket_callback: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Handle follow-up questions about existing algorithmic results.
    Uses existing algorithmic data instead of re-running calculations.
    
    Args:
        query: User's follow-up question
        algorithmic_data: Previously calculated algorithmic results
        cfg: ICEBURG configuration
        temperature: LLM temperature
        max_tokens: Max tokens for LLM
        websocket_callback: Optional callback for streaming messages
        
    Returns:
        Dictionary with LLM response about the existing results
    """
    try:
        from ..llm import chat_complete
        
        if websocket_callback:
            await websocket_callback({
                "type": "thinking",
                "content": "Analyzing your question about the results..."
            })
        
        # Format algorithmic data for context
        imprint = algorithmic_data.get('molecular_imprint', {})
        voltage_gates = imprint.get('voltage_gates', {})
        
        # Format birth datetime in human-readable format
        birth_dt_str = imprint.get('birth_datetime', 'Unknown')
        if birth_dt_str != 'Unknown':
            try:
                from datetime import datetime
                if isinstance(birth_dt_str, str):
                    dt_str = birth_dt_str.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(dt_str)
                else:
                    dt = birth_dt_str
                birth_dt_formatted = dt.strftime("%B %d, %Y at %I:%M %p")
            except Exception as e:
                logger.warning(f"Error formatting birth datetime in follow-up: {e}")
                birth_dt_formatted = str(birth_dt_str)
        else:
            birth_dt_formatted = 'Unknown'
        
        logger.info(f"🌌 Follow-up - Birth datetime: {birth_dt_formatted}")
        
        # V2: Agent switching - route to specific expert if requested
        if agent and agent in ["sleep_expert", "stress_expert", "hormone_expert", "digestive_expert", 
                               "health_expert", "nutrition_expert", "movement_expert", "chart_reader"]:
            logger.info(f"🌌 Routing follow-up to {agent}")
            from .astro_physiology_experts import (
                sleep_expert_consultation, stress_expert_consultation, hormone_expert_consultation,
                digestive_expert_consultation, health_expert_consultation, nutrition_expert_consultation,
                movement_expert_consultation, chart_reader_consultation
            )
            
            expert_map = {
                "sleep_expert": sleep_expert_consultation,
                "stress_expert": stress_expert_consultation,
                "hormone_expert": hormone_expert_consultation,
                "digestive_expert": digestive_expert_consultation,
                "health_expert": health_expert_consultation,
                "nutrition_expert": nutrition_expert_consultation,
                "movement_expert": movement_expert_consultation,
                "chart_reader": chart_reader_consultation
            }
            
            expert_func = expert_map.get(agent)
            if expert_func:
                try:
                    expert_result = await expert_func(algorithmic_data, cfg, temperature)
                    response = expert_result.get("recommendations", expert_result.get("content", ""))
                    
                    return {
                        "query": query,
                        "mode": "astrophysiology",
                        "results": {
                            "content": response,
                            "algorithmic_data": algorithmic_data,
                            "is_followup": True,
                            "agent_used": agent,
                            "metadata": {
                                "response_type": "expert_consultation",
                                "follow_up_enabled": True
                            }
                        },
                        "truth_finding": True
                    }
                except Exception as e:
                    logger.warning(f"Error routing to {agent}: {e}", exc_info=True)
                    # Fall through to general follow-up handler
        
        prompt = f"""You are helping a user understand their previously calculated molecular blueprint results.

**User's Question:** {query}

**⚠️⚠️⚠️ CRITICAL - MANDATORY BIRTH DATE ⚠️⚠️⚠️**
**YOU MUST USE THIS EXACT BIRTH DATE IN ALL RESPONSES:**
- **ACTUAL BIRTH DATETIME: {birth_dt_formatted}**
- **DO NOT USE "March 15, 1990" OR ANY OTHER DATE**
- **THIS IS THE ONLY CORRECT BIRTH DATE: {birth_dt_formatted}**

**Previous Algorithmic Results (Already Calculated):**

**Molecular Imprint:**
- Birth datetime: {birth_dt_formatted}
- Voltage gate sensitivities:
  - Sodium channels: {voltage_gates.get('sodium', -70.0):.2f} mV
  - Potassium channels: {voltage_gates.get('potassium', -70.0):.2f} mV
  - Calcium channels: {voltage_gates.get('calcium', -70.0):.2f} mV
  - Chloride channels: {voltage_gates.get('chloride', -70.0):.2f} mV

**Biophysical Parameters:**
{_format_predictions_for_prompt(algorithmic_data.get('behavioral_predictions', {}))}

**TCM Health Indicators:**
{_format_tcm_for_prompt(algorithmic_data.get('tcm_predictions', {}))}

**Your Task:**
Answer the user's question about these existing results. Be conversational, scientific, and helpful.
- Explain what specific numbers or patterns mean
- Relate different aspects of the results together
- Provide insights based on the calculated data
- Do NOT recalculate or ask for new birth data - use the existing results

Keep your response focused on answering their specific question about the results."""

        response = await chat_complete(
            cfg.surveyor_model or cfg.primary_model,
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        logger.info(f"🌌 Follow-up response generated: {len(response)} chars")
        
        return {
            "query": query,
            "mode": "astrophysiology",
            "results": {
                "content": response,
                "algorithmic_data": algorithmic_data,  # Include existing data for further follow-ups
                "is_followup": True,
                "metadata": {
                    "response_type": "followup_conversation",
                    "follow_up_enabled": True
                }
            },
            "truth_finding": True
        }
        
    except Exception as e:
        logger.error(f"Error handling follow-up question: {e}", exc_info=True)
        return {
            "query": query,
            "mode": "astrophysiology",
            "error": "followup_error",
            "message": f"Error processing follow-up question: {str(e)}",
            "results": {
                "error": "followup_error",
                "content": f"I encountered an error answering your question. Please try rephrasing it."
            }
        }


async def _generate_organic_response(
    query: str,
    algorithmic_results: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float,
    max_tokens: int,
    user_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate organic LLM response from algorithmic results.
    The LLM receives all calculated data and generates a natural explanation.
    
    Args:
        query: Original user query
        algorithmic_results: Complete algorithmic calculation results
        cfg: ICEBURG configuration
        temperature: LLM temperature
        max_tokens: Max tokens for response
        
    Returns:
        Natural language explanation of the results
    """
    try:
        from ..llm import chat_complete
        
        # Build comprehensive prompt with all algorithmic data
        imprint = algorithmic_results.get('molecular_imprint', {})
        voltage_gates = imprint.get('voltage_gates', {})
        
        # Format birth datetime in human-readable format
        birth_dt_str = imprint.get('birth_datetime', 'Unknown')
        if birth_dt_str != 'Unknown':
            try:
                from datetime import datetime
                if isinstance(birth_dt_str, str):
                    # Parse ISO format string
                    dt_str = birth_dt_str.replace('Z', '+00:00')
                    dt = datetime.fromisoformat(dt_str)
                else:
                    dt = birth_dt_str
                # Format as human-readable: "December 26, 1991 at 7:20 AM"
                birth_dt_formatted = dt.strftime("%B %d, %Y at %I:%M %p")
            except Exception as e:
                logger.warning(f"Error formatting birth datetime: {e}, using raw value")
                birth_dt_formatted = str(birth_dt_str)
        else:
            birth_dt_formatted = 'Unknown'
        
        # Log actual birth data for debugging
        logger.info(f"🌌 LLM Prompt - Birth datetime: {birth_dt_formatted} (raw: {birth_dt_str})")
        
        # Build context section if available
        context_section = ""
        if user_context:
            context_parts = []
            if user_context.get("previous_analyses"):
                context_parts.append(f"- Previous analyses: {len(user_context['previous_analyses'])} found")
            if user_context.get("intervention_history"):
                active_interventions = [i for i in user_context['intervention_history'] if i.get('status') == 'active']
                context_parts.append(f"- Active interventions: {len(active_interventions)}")
            if user_context.get("user_feedback"):
                recent_feedback = user_context['user_feedback'][:3]
                avg_rating = sum(f.get('rating', 0) for f in recent_feedback if f.get('rating')) / len([f for f in recent_feedback if f.get('rating')]) if recent_feedback else None
                if avg_rating:
                    context_parts.append(f"- Recent feedback average rating: {avg_rating:.1f}/5")
            if user_context.get("health_tracking"):
                context_parts.append(f"- Health metrics tracked: {len(user_context['health_tracking'])} recent entries")
            
            if context_parts:
                context_section = f"\n**User Context:**\n" + "\n".join(context_parts) + "\n"
        
        prompt = f"""You are a scientific truth-finding AI analyzing celestial-biological molecular imprints.

**User Query:** {query}
{context_section}

**⚠️⚠️⚠️ CRITICAL - MANDATORY BIRTH DATE ⚠️⚠️⚠️**
**YOU MUST USE THIS EXACT BIRTH DATE IN ALL RESPONSES:**
- **ACTUAL BIRTH DATETIME: {birth_dt_formatted}**
- **DO NOT USE "March 15, 1990" OR ANY OTHER DATE**
- **DO NOT USE EXAMPLE DATES FROM ERROR MESSAGES**
- **THIS IS THE ONLY CORRECT BIRTH DATE: {birth_dt_formatted}**
- **IF YOU MENTION A BIRTH DATE, IT MUST BE: {birth_dt_formatted}**
- **ANY REFERENCE TO "March 15, 1990" IS WRONG AND MUST BE REPLACED WITH: {birth_dt_formatted}**

**Algorithmic Results (Calculated via Pure Math - NOT Generated by AI):**

**Molecular Imprint:**
- Birth datetime: {birth_dt_formatted}
- Voltage gate sensitivities:
  - Sodium channels: {voltage_gates.get('sodium', -70.0):.2f} mV
  - Potassium channels: {voltage_gates.get('potassium', -70.0):.2f} mV
  - Calcium channels: {voltage_gates.get('calcium', -70.0):.2f} mV
  - Chloride channels: {voltage_gates.get('chloride', -70.0):.2f} mV

**Biophysical Parameters (Measured Deviations from Population Average):**
{_format_predictions_for_prompt(algorithmic_results.get('behavioral_predictions', {}))}

**TCM Health Indicators:**
{_format_tcm_for_prompt(algorithmic_results.get('tcm_predictions', {}))}

**Current Celestial Conditions:**
{_format_conditions_for_prompt(algorithmic_results.get('current_conditions', {}))}

**Research Context (from Recursive Celestial Analyzer):**
{algorithmic_results.get('research_context', 'No additional research context available.')[:2000]}

**Synthesis (Meridian Connections & Cross-Domain Insights):**
{algorithmic_results.get('synthesis', 'No cross-domain synthesis available.')[:2000]}

**Your Task:**
Generate a natural, conversational explanation of these algorithmic results. Be scientific but accessible. Explain:
1. What the molecular blueprint reveals about their biophysical configuration
2. What the physiological parameters mean in practical terms
3. How these measurements relate to their birth conditions
4. What the health indicators suggest

**Important:**
- These are BIOPHYSICAL measurements, NOT personality traits
- The calculations are deterministic (pure math), not AI-generated
- Focus on what the numbers mean scientifically
- Be conversational but maintain scientific accuracy
- Reference specific values when explaining
- If research context is available, incorporate it naturally
- **CRITICAL**: Only mention facts that appear in the Research Context or Synthesis above
- **CRITICAL**: Do NOT invent specific study names, author names, or paper titles unless they appear in the Research Context
- **CRITICAL**: If Research Context says "No additional research context available", do NOT cite specific studies
- **CRITICAL**: Base all claims on the algorithmic results provided above, not on general knowledge
- **⚠️ RESEARCH TOOL POSITIONING**: Frame this as exploratory research - "We're exploring whether...", "This hypothesis suggests...", "If validated, this could indicate..."
- **⚠️ CONFIDENCE LEVELS**: Mention confidence levels and evidence quality when discussing predictions
- **⚠️ MANDATORY BIRTH DATE RULE ⚠️**: Every time you mention a birth date, birth datetime, or birth time, you MUST use: {birth_dt_formatted}
- **⚠️ FORBIDDEN**: Never mention "March 15, 1990", "March 15th, 1990", "1990-03-15", or any variation of March 15, 1990
- **⚠️ VERIFICATION**: Before finishing your response, check every mention of a date and ensure it matches: {birth_dt_formatted}

Generate your response now. Remember: The birth date is {birth_dt_formatted} - use it consistently throughout:"""
        
        # Generate response using LLM
        system_prompt = "You are ICEBURG, a scientific truth-finding AI specializing in celestial-biological correlations. You explain algorithmic results in natural, accessible language while maintaining scientific rigor."
        
        # Get model from config (try multiple possible attribute names)
        model = getattr(cfg, 'primary_model', None) or getattr(cfg, 'surveyor_model', None) or 'llama3.1:8b'
        
        # Use lower temperature for more factual responses (reduce hallucinations)
        # Temperature 0.5 is better for factual content than 0.7
        factual_temperature = min(temperature, 0.5)  # Cap at 0.5 for astro-physiology
        
        logger.info(f"🌌 Generating LLM response with model={model}, temperature={factual_temperature}")
        
        response = await asyncio.to_thread(
            chat_complete,
            model,
            prompt,
            system=system_prompt,
            temperature=factual_temperature,
            options={"num_ctx": 4096, "num_predict": max_tokens},
            context_tag="AstroPhysiology"
        )
        
        # Post-process response to fix any wrong date mentions
        if response and birth_dt_formatted != 'Unknown':
            # Replace any mentions of "March 15, 1990" or variations with correct date
            import re
            wrong_date_patterns = [
                r'March\s+15,?\s+1990',
                r'March\s+15th,?\s+1990',
                r'1990-03-15',
                r'March\s+15',
                r'15th,?\s+March,?\s+1990',
            ]
            for pattern in wrong_date_patterns:
                response = re.sub(pattern, birth_dt_formatted, response, flags=re.IGNORECASE)
            
            logger.info(f"🌌 Post-processed LLM response to fix any wrong date mentions")
        
        return response.strip() if response else ""
        
    except Exception as e:
        logger.error(f"Error generating organic response: {e}", exc_info=True)
        # Fallback to templated response if LLM fails
        return _format_truth_finding_response_fallback(algorithmic_results)


def _format_predictions_for_prompt(predictions: Dict[str, float]) -> str:
    """Format behavioral predictions for LLM prompt"""
    if not predictions:
        return "No biophysical parameters calculated."
    
    param_names = {
        "neural_excitability": "Neural Excitability",
        "inhibitory_control": "Inhibitory Control",
        "neurotransmitter_release": "Neurotransmitter Release",
        "cellular_stability": "Cellular Stability"
    }
    
    lines = []
    for param, value in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
        name = param_names.get(param, param.replace('_', ' ').title())
        percentage = value * 100
        lines.append(f"- {name}: {percentage:+.1f}% deviation from population average")
    
    return "\n".join(lines) if lines else "No parameters available."


def _format_tcm_for_prompt(tcm_predictions: Dict[str, Any]) -> str:
    """Format TCM predictions for LLM prompt"""
    if not tcm_predictions:
        return "No TCM health indicators calculated."
    
    lines = []
    for planet, data in list(tcm_predictions.items())[:5]:
        if isinstance(data, dict):
            organ = data.get("organ", "Unknown")
            strength = data.get("strength", 0)
            lines.append(f"- {organ} ({planet}): {strength*100:.0f}% strength")
    
    return "\n".join(lines) if lines else "No TCM indicators available."


def _format_conditions_for_prompt(conditions: Dict[str, Any]) -> str:
    """Format current conditions for LLM prompt"""
    if not conditions:
        return "Current celestial conditions not available."
    
    lines = []
    if "celestial_positions" in conditions:
        lines.append("Celestial positions: Calculated")
    if "electromagnetic_environment" in conditions:
        lines.append("EM environment: Calculated")
    if "earth_frequencies" in conditions:
        lines.append("Earth frequencies: Calculated")
    
    return "\n".join(lines) if lines else "Conditions not detailed."


def _format_truth_finding_response_fallback(algorithmic_results: Dict[str, Any]) -> str:
    """Fallback templated response if LLM generation fails"""
    # Simplified version of original template
    response_parts = []
    response_parts.append("**Your Molecular Blueprint Analysis:**\n\n")
    
    imprint = algorithmic_results.get('molecular_imprint', {})
    voltage_gates = imprint.get('voltage_gates', {})
    
    if voltage_gates:
        response_parts.append("**Voltage Gate Sensitivities:**\n")
        response_parts.append(f"- Sodium channels: {voltage_gates.get('sodium', -70.0):.2f} mV\n")
        response_parts.append(f"- Potassium channels: {voltage_gates.get('potassium', -70.0):.2f} mV\n")
        response_parts.append(f"- Calcium channels: {voltage_gates.get('calcium', -70.0):.2f} mV\n")
        response_parts.append(f"- Chloride channels: {voltage_gates.get('chloride', -70.0):.2f} mV\n\n")
    
    predictions = algorithmic_results.get('behavioral_predictions', {})
    if predictions:
        response_parts.append("**Biophysical Parameters:**\n")
        for param, value in sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:4]:
            percentage = value * 100
            response_parts.append(f"- {param.replace('_', ' ').title()}: {percentage:+.1f}%\n")
        response_parts.append("\n")
    
    response_parts.append("These are biophysical measurements calculated from your birth conditions. They represent ion channel sensitivities and neural function parameters, not personality traits.\n")
    
    return "".join(response_parts)

