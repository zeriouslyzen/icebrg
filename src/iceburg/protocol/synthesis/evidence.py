from __future__ import annotations

from typing import Dict, List

from ..config import ProtocolConfig
from ..models import AgentResult, EvidenceBundle


def _extract_payload(results: List[AgentResult], agent_name: str) -> str | Dict:
    for result in results:
        if result.agent == agent_name:
            return result.payload
    return ""


def synthesize(results: List[AgentResult], config: ProtocolConfig) -> EvidenceBundle:
    # Core agent outputs
    consensus = _extract_payload(results, "surveyor")
    dissident_view = _extract_payload(results, "dissident")
    archaeologist_output = _extract_payload(results, "archaeologist")
    synthesis = _extract_payload(results, "synthesist")
    oracle = _extract_payload(results, "oracle")
    
    # CIM Stack outputs
    molecular_analysis = _extract_payload(results, "molecular_synthesis")
    bioelectric_analysis = _extract_payload(results, "bioelectric_integration")
    hypothesis_testing_results = _extract_payload(results, "hypothesis_testing_laboratory")
    grounding_layer_results = _extract_payload(results, "grounding_layer_agent")
    
    # AGI Capabilities outputs
    self_redesign_results = _extract_payload(results, "self_redesign_engine")
    novel_intelligence_results = _extract_payload(results, "novel_intelligence_creator")
    autonomous_goal_results = _extract_payload(results, "autonomous_goal_formation")
    unbounded_learning_results = _extract_payload(results, "unbounded_learning_engine")
    
    # Deliberation analysis outputs
    deliberation_1 = _extract_payload(results, "deliberation_pause")
    contradictions = _extract_payload(results, "hunt_contradictions")
    emergence = _extract_payload(results, "detect_emergence")
    meta_analysis = _extract_payload(results, "perform_meta_analysis")
    truth_seeking = _extract_payload(results, "apply_truth_seeking")
    
    # Quality control output
    supervisor_output = _extract_payload(results, "supervisor")
    
    # Blockchain verification outputs
    blockchain_verification_results = _extract_payload(results, "blockchain_verification")
    peer_review_results = _extract_payload(results, "decentralized_peer_review")
    suppression_resistant_results = _extract_payload(results, "suppression_resistant_storage")
    
    # Multimodal processing outputs
    multimodal_processing_results = _extract_payload(results, "multimodal_processor")
    visual_generation_results = _extract_payload(results, "visual_generator")

    diagnostics = {
        "consensus_present": bool(consensus),
        "dissident_present": bool(dissident_view),
        "archaeologist_present": bool(archaeologist_output),
        "synthesis_present": bool(synthesis),
        "oracle_present": bool(oracle),
        "molecular_present": bool(molecular_analysis),
        "bioelectric_present": bool(bioelectric_analysis),
        "hypothesis_testing_present": bool(hypothesis_testing_results),
        "grounding_layer_present": bool(grounding_layer_results),
        "self_redesign_present": bool(self_redesign_results),
        "novel_intelligence_present": bool(novel_intelligence_results),
        "autonomous_goal_present": bool(autonomous_goal_results),
        "unbounded_learning_present": bool(unbounded_learning_results),
        "deliberation_present": bool(deliberation_1),
        "contradictions_present": bool(contradictions),
        "emergence_present": bool(emergence),
        "meta_analysis_present": bool(meta_analysis),
        "truth_seeking_present": bool(truth_seeking),
        "supervisor_present": bool(supervisor_output),
        "blockchain_verification_present": bool(blockchain_verification_results),
        "peer_review_present": bool(peer_review_results),
        "suppression_resistant_present": bool(suppression_resistant_results),
        "multimodal_processing_present": bool(multimodal_processing_results),
        "visual_generation_present": bool(visual_generation_results),
        # Store outputs for synthesis
        "dissident_output": dissident_view,
        "archaeologist_output": archaeologist_output,
        "synthesist_output": synthesis,
        "oracle_output": oracle,
        "molecular_output": molecular_analysis,
        "bioelectric_output": bioelectric_analysis,
        "hypothesis_testing_output": hypothesis_testing_results,
        "grounding_layer_output": grounding_layer_results,
        "self_redesign_output": self_redesign_results,
        "novel_intelligence_output": novel_intelligence_results,
        "autonomous_goal_output": autonomous_goal_results,
        "unbounded_learning_output": unbounded_learning_results,
        "deliberation_output": deliberation_1,
        "contradictions_output": contradictions,
        "emergence_output": emergence,
        "meta_analysis_output": meta_analysis,
        "truth_seeking_output": truth_seeking,
        "supervisor_output": supervisor_output,
        "blockchain_verification_output": blockchain_verification_results,
        "peer_review_output": peer_review_results,
        "suppression_resistant_output": suppression_resistant_results,
        "multimodal_processing_output": multimodal_processing_results,
        "visual_generation_output": visual_generation_results,
    }

    return EvidenceBundle(
        results=results,
        consensus=consensus if isinstance(consensus, str) else str(consensus),
        diagnostics=diagnostics,
    )
