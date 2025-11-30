from __future__ import annotations

from typing import Any, Dict

from ..config import ProtocolConfig
from ..models import EvidenceBundle, ProtocolReport


def format_report(evidence: EvidenceBundle, config: ProtocolConfig) -> ProtocolReport:
    sections: Dict[str, Any] = {
        "consensus": evidence.consensus,
        "counterpoints": evidence.diagnostics.get("dissident_output"),
        "archaeologist": evidence.diagnostics.get("archaeologist_output"),
        "synthesis": evidence.diagnostics.get("synthesist_output"),
        "oracle": evidence.diagnostics.get("oracle_output"),
        "agents": evidence.diagnostics.get("agents"),
    }
    
    # Add CIM Stack sections if present
    if evidence.diagnostics.get("molecular_present"):
        sections["molecular_analysis"] = evidence.diagnostics.get("molecular_output")
    
    if evidence.diagnostics.get("bioelectric_present"):
        sections["bioelectric_analysis"] = evidence.diagnostics.get("bioelectric_output")
    
    if evidence.diagnostics.get("hypothesis_testing_present"):
        sections["hypothesis_testing"] = evidence.diagnostics.get("hypothesis_testing_output")
    
    if evidence.diagnostics.get("grounding_layer_present"):
        sections["grounding_layer"] = evidence.diagnostics.get("grounding_layer_output")
    
    # Add AGI Capabilities sections if present
    if evidence.diagnostics.get("self_redesign_present"):
        sections["self_redesign"] = evidence.diagnostics.get("self_redesign_output")
    
    if evidence.diagnostics.get("novel_intelligence_present"):
        sections["novel_intelligence"] = evidence.diagnostics.get("novel_intelligence_output")
    
    if evidence.diagnostics.get("autonomous_goal_present"):
        sections["autonomous_goals"] = evidence.diagnostics.get("autonomous_goal_output")
    
    if evidence.diagnostics.get("unbounded_learning_present"):
        sections["unbounded_learning"] = evidence.diagnostics.get("unbounded_learning_output")
    
    # Add deliberation analysis sections if present
    if evidence.diagnostics.get("deliberation_present"):
        sections["deliberation"] = evidence.diagnostics.get("deliberation_output")
    
    if evidence.diagnostics.get("contradictions_present"):
        sections["contradictions"] = evidence.diagnostics.get("contradictions_output")
    
    if evidence.diagnostics.get("emergence_present"):
        sections["emergence"] = evidence.diagnostics.get("emergence_output")
    
    if evidence.diagnostics.get("meta_analysis_present"):
        sections["meta_analysis"] = evidence.diagnostics.get("meta_analysis_output")
    
    if evidence.diagnostics.get("truth_seeking_present"):
        sections["truth_seeking"] = evidence.diagnostics.get("truth_seeking_output")
    
    if evidence.diagnostics.get("supervisor_present"):
        sections["supervisor"] = evidence.diagnostics.get("supervisor_output")
    
    # Add blockchain verification sections if present
    if evidence.diagnostics.get("blockchain_verification_present"):
        sections["blockchain_verification"] = evidence.diagnostics.get("blockchain_verification_output")
    
    if evidence.diagnostics.get("peer_review_present"):
        sections["peer_review"] = evidence.diagnostics.get("peer_review_output")
    
    if evidence.diagnostics.get("suppression_resistant_present"):
        sections["suppression_resistant_storage"] = evidence.diagnostics.get("suppression_resistant_output")
    
    # Add multimodal processing sections if present
    if evidence.diagnostics.get("multimodal_processing_present"):
        sections["multimodal_processing"] = evidence.diagnostics.get("multimodal_processing_output")
    
    if evidence.diagnostics.get("visual_generation_present"):
        sections["visual_generation"] = evidence.diagnostics.get("visual_generation_output")

    audit = {
        "agent_count": len(evidence.results),
        "confidence": evidence.confidence,
        "feature_flags": config.feature_flags,
        "cim_stack_enabled": any([
            evidence.diagnostics.get("molecular_present", False),
            evidence.diagnostics.get("bioelectric_present", False),
            evidence.diagnostics.get("hypothesis_testing_present", False),
            evidence.diagnostics.get("grounding_layer_present", False),
        ]),
        "agi_capabilities_enabled": any([
            evidence.diagnostics.get("self_redesign_present", False),
            evidence.diagnostics.get("novel_intelligence_present", False),
            evidence.diagnostics.get("autonomous_goal_present", False),
            evidence.diagnostics.get("unbounded_learning_present", False),
        ]),
        "blockchain_verification_enabled": any([
            evidence.diagnostics.get("blockchain_verification_present", False),
            evidence.diagnostics.get("peer_review_present", False),
            evidence.diagnostics.get("suppression_resistant_present", False),
        ]),
        "multimodal_processing_enabled": any([
            evidence.diagnostics.get("multimodal_processing_present", False),
            evidence.diagnostics.get("visual_generation_present", False),
        ]),
        "deliberation_enabled": evidence.diagnostics.get("deliberation_present", False),
        "quality_control_enabled": evidence.diagnostics.get("supervisor_present", False),
    }
    return ProtocolReport(sections=sections, audit=audit)
