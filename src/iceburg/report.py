def format_iceberg_report(
    consensus: str, alternatives: str, syntheses: str, principle: str
) -> str:
    sections: list[str] = []
    
    # Handle consensus
    consensus_text = str(consensus).strip() if consensus else "No consensus data available"
    sections.append(
        "### Layer 1 — Surveyor (Consensus Reality)\n\n" + consensus_text
    )
    
    # Handle alternatives
    alternatives_text = str(alternatives).strip() if alternatives else "No alternatives data available"
    sections.append(
        "### Layer 2 — Dissident (Assumptions & Alternatives)\n\n" + alternatives_text
    )
    
    # Handle syntheses - can be string or dict
    if isinstance(syntheses, dict):
        syntheses_text = str(syntheses)
    else:
        syntheses_text = str(syntheses).strip() if syntheses else "No synthesis data available"
    sections.append(
        "### Layer 3 — Synthesist (Cross-Domain Evidence)\n\n" + syntheses_text
    )
    
    # Handle principle - can be string or dict
    if isinstance(principle, dict):
        principle_text = str(principle)
    else:
        principle_text = str(principle).strip() if principle else "No principle data available"
    sections.append(
        "### Layer 4 — Oracle (Evidence-Weighted Principle)\n\n" + principle_text
    )
    
    return "\n\n---\n\n".join(sections) + "\n"
