from __future__ import annotations
from typing import Dict, Any, Optional, List
import json
import re
from datetime import datetime


def run(cfg, oracle_output: str, verbose: bool = False) -> str:
    """Generate linguistic, theoretical, and knowledge outputs from Oracle principles"""
    
    if verbose:
        print("[Scribe] Starting knowledge synthesis...")
    
    try:
        # Parse Oracle output
        oracle_data = _extract_principle_data(oracle_output)
        
        if verbose:
            print(f"[Scribe] Extracted principle: {oracle_data.get('principle_name', 'Unknown')}")
        
        # Generate comprehensive knowledge outputs
        outputs = _generate_knowledge_outputs(oracle_data, verbose)
        
        if verbose:
            print("[Scribe] Knowledge synthesis complete")
        
        return outputs
        
    except Exception as e:
        if verbose:
            print(f"[SCRIBE] Error: {e}")
        print(f"[Scribe] Error in knowledge synthesis: {e}")
        return _generate_fallback_knowledge(oracle_output)


def _extract_principle_data(oracle_output: str) -> Dict[str, Any]:
    """Extract principle data from Oracle output"""
    
    # Try direct JSON parsing first
    try:
        return json.loads(oracle_output)
    except json.JSONDecodeError:
        pass
    
    # Extract from formatted text
    extracted_data = {}
    
    # Extract principle name
    name_match = re.search(r'"principle_name":\s*"([^"]+)"', oracle_output)
    if name_match:
        extracted_data["principle_name"] = name_match.group(1)
    
    # Extract core principle
    core_match = re.search(r'"one_sentence_summary":\s*"([^"]+)"', oracle_output)
    if core_match:
        extracted_data["one_sentence_summary"] = core_match.group(1)
    
    # Extract domains
    domains_match = re.search(r'"domains":\s*\[(.*?)\]', oracle_output)
    if domains_match:
        domains_text = domains_match.group(1)
        domains = [d.strip().strip('"') for d in domains_text.split(',')]
        extracted_data["domains"] = domains
    
    # Extract predictions
    predictions_match = re.search(r'"predictions":\s*\[(.*?)\]', oracle_output)
    if predictions_match:
        predictions_text = predictions_match.group(1)
        predictions = [p.strip().strip('"') for p in predictions_text.split(',')]
        extracted_data["predictions"] = predictions
    
    # Extract study design
    study_match = re.search(r'"study_design":\s*\{([^}]+)\}', oracle_output)
    if study_match:
        study_text = study_match.group(1)
        # Extract key study parameters
        manipulation_match = re.search(r'"manipulation":\s*"([^"]+)"', study_text)
        if manipulation_match:
            extracted_data["manipulation"] = manipulation_match.group(1)
    
    return extracted_data


def _generate_knowledge_outputs(principle_data: Dict[str, Any], verbose: bool = False) -> str:
    """Generate comprehensive knowledge outputs"""
    
    principle_name = principle_data.get("principle_name", "Unknown Principle")
    core_principle = principle_data.get("one_sentence_summary", "")
    domains = principle_data.get("domains", [])
    predictions = principle_data.get("predictions", [])
    
    # Generate multiple knowledge outputs
    outputs = []
    
    # 1. Theoretical Framework
    theoretical_framework = _generate_theoretical_framework(principle_name, core_principle, domains)
    outputs.append(theoretical_framework)
    
    # 2. Research Paper Abstract
    research_abstract = _generate_research_abstract(principle_name, core_principle, domains)
    outputs.append(research_abstract)
    
    # 3. Knowledge Base Entry
    knowledge_entry = _generate_knowledge_entry(principle_name, core_principle, domains, predictions)
    outputs.append(knowledge_entry)
    
    # 4. Cross-Domain Analysis
    cross_domain = _generate_cross_domain_analysis(principle_name, core_principle, domains)
    outputs.append(cross_domain)
    
    # 5. Implementation Roadmap
    roadmap = _generate_implementation_roadmap(principle_name, core_principle, domains)
    outputs.append(roadmap)
    
    return "\n\n" + "="*80 + "\n\n".join(outputs)


def _generate_theoretical_framework(principle_name: str, core_principle: str, domains: List[str]) -> str:
    """Generate theoretical framework document"""
    
    return f"""# THEORETICAL FRAMEWORK: {principle_name.upper()}

## Abstract
{core_principle}

## 1. Introduction
This theoretical framework establishes the foundational principles for understanding {principle_name.lower()}. The framework integrates insights from multiple domains including {', '.join(domains)} to create a comprehensive understanding of the underlying mechanisms.

## 2. Core Principles
### 2.1 Fundamental Assumptions
- The principle operates across multiple domains of knowledge
- Cross-domain integration reveals deeper patterns
- Emergent properties arise from domain interactions

### 2.2 Theoretical Constructs
- **Domain Integration**: The synthesis of knowledge across {len(domains)} primary domains
- **Emergent Properties**: Novel characteristics that arise from domain interactions
- **Cross-Domain Validation**: Verification of principles across multiple knowledge areas

## 3. Domain Analysis
{chr(10).join([f"### 3.{i+1} {domain}" for i, domain in enumerate(domains)])}

## 4. Theoretical Implications
This framework suggests that {principle_name.lower()} represents a fundamental pattern that transcends individual domains and reveals deeper truths about the nature of knowledge and reality.

## 5. Future Research Directions
- Empirical validation across domains
- Development of measurement tools
- Application to novel domain combinations
"""


def _generate_research_abstract(principle_name: str, core_principle: str, domains: List[str]) -> str:
    """Generate research paper abstract"""
    
    return f"""# RESEARCH ABSTRACT

**Title**: {principle_name}: A Cross-Domain Theoretical Framework

**Abstract**:
{core_principle} This research presents a novel theoretical framework that integrates insights from {len(domains)} distinct domains: {', '.join(domains)}. Through systematic analysis and cross-domain synthesis, we identify emergent properties that transcend individual domain boundaries. Our findings suggest that {principle_name.lower()} represents a fundamental pattern in knowledge organization and reality perception. The framework provides a foundation for future research in cross-domain integration and emergent knowledge synthesis.

**Keywords**: {principle_name.lower()}, cross-domain integration, emergent properties, knowledge synthesis, theoretical framework

**Domain Coverage**: {', '.join(domains)}

**Research Significance**: This work establishes a new paradigm for understanding how knowledge domains interact and generate emergent insights that cannot be derived from single-domain analysis alone.
"""


def _generate_knowledge_entry(principle_name: str, core_principle: str, domains: List[str], predictions: List[str]) -> str:
    """Generate knowledge base entry"""
    
    return f"""# KNOWLEDGE BASE ENTRY

**Entry ID**: {principle_name.replace(' ', '_').upper()}_{datetime.now().strftime('%Y%m%d')}
**Last Updated**: {datetime.now().isoformat()}
**Status**: Active
**Confidence Level**: High

## Principle Definition
**Name**: {principle_name}
**Core Statement**: {core_principle}

## Domain Classification
**Primary Domains**: {', '.join(domains)}
**Cross-Domain Relevance**: High
**Integration Level**: Deep

## Key Insights
1. **Cross-Domain Synthesis**: The principle emerges from the integration of {len(domains)} distinct knowledge domains
2. **Emergent Properties**: Novel characteristics arise that are not present in individual domains
3. **Universal Applicability**: The principle applies across multiple knowledge areas

## Predictions
{chr(10).join([f"- {pred}" for pred in predictions]) if predictions else "- No specific predictions available"}

## Related Concepts
- Cross-domain integration
- Emergent knowledge
- Theoretical synthesis
- Knowledge architecture

## Usage Guidelines
- Apply across multiple domains for validation
- Use as foundation for cross-domain research
- Integrate with existing theoretical frameworks
"""


def _generate_cross_domain_analysis(principle_name: str, core_principle: str, domains: List[str]) -> str:
    """Generate cross-domain analysis document"""
    
    return f"""# CROSS-DOMAIN ANALYSIS

**Principle**: {principle_name}
**Analysis Date**: {datetime.now().strftime('%Y-%m-%d')}

## Executive Summary
This analysis examines how {principle_name.lower()} manifests across {len(domains)} distinct knowledge domains, revealing patterns of cross-domain consistency and emergent properties.

## Domain-by-Domain Analysis

{chr(10).join(['### ' + domain + chr(10) + '**Relevance**: High' + chr(10) + '**Manifestation**: The principle appears as [specific manifestation in this domain]' + chr(10) + '**Evidence**: [types of evidence available in this domain]' + chr(10) + '**Implications**: [how this domain contributes to the overall principle]' for domain in domains])}

## Cross-Domain Patterns

### 1. Consistency Patterns
- **High Consistency**: The principle maintains consistent characteristics across {len(domains)} domains
- **Adaptive Manifestation**: While consistent, the principle adapts to domain-specific contexts
- **Emergent Validation**: Cross-domain consistency validates the principle's fundamental nature

### 2. Integration Mechanisms
- **Knowledge Bridges**: Specific concepts that connect domains
- **Shared Frameworks**: Common theoretical structures across domains
- **Transfer Mechanisms**: Processes that enable knowledge transfer between domains

## Synthesis Insights
The cross-domain analysis reveals that {principle_name.lower()} represents a fundamental pattern that transcends individual knowledge areas. This suggests the principle operates at a meta-level of reality organization.

## Recommendations
1. **Further Research**: Conduct deeper analysis in each domain
2. **Integration Studies**: Explore how domains can be better integrated
3. **Application Development**: Develop tools for applying the principle across domains
"""


def _generate_implementation_roadmap(principle_name: str, core_principle: str, domains: List[str]) -> str:
    """Generate implementation roadmap"""
    
    return f"""# IMPLEMENTATION ROADMAP

**Principle**: {principle_name}
**Target**: Cross-Domain Knowledge Integration System
**Timeline**: 12-18 months

## Phase 1: Foundation (Months 1-3)
### 1.1 Theoretical Framework Development
- **Objective**: Establish comprehensive theoretical foundation
- **Deliverables**: 
  - Complete theoretical framework document
  - Domain integration models
  - Validation criteria
- **Success Metrics**: Framework approved by domain experts

### 1.2 Domain Analysis
- **Objective**: Deep analysis of each {len(domains)} domain
- **Deliverables**:
  - Domain-specific analysis reports
  - Cross-domain connection maps
  - Integration opportunities identification
- **Success Metrics**: Complete domain coverage

## Phase 2: Integration (Months 4-8)
### 2.1 Cross-Domain Synthesis
- **Objective**: Create integrated knowledge models
- **Deliverables**:
  - Cross-domain knowledge graphs
  - Integration algorithms
  - Validation frameworks
- **Success Metrics**: Successful cross-domain synthesis

### 2.2 Tool Development
- **Objective**: Build implementation tools
- **Deliverables**:
  - Knowledge integration platform
  - Analysis tools
  - Visualization systems
- **Success Metrics**: Functional tool suite

## Phase 3: Validation (Months 9-12)
### 3.1 Empirical Testing
- **Objective**: Validate across multiple domains
- **Deliverables**:
  - Test results
  - Performance metrics
  - Validation reports
- **Success Metrics**: Successful validation in {len(domains)} domains

### 3.2 Refinement
- **Objective**: Optimize based on test results
- **Deliverables**:
  - Refined implementation
  - Performance improvements
  - Documentation updates
- **Success Metrics**: Measurable performance improvements

## Phase 4: Deployment (Months 13-18)
### 4.1 System Deployment
- **Objective**: Deploy integrated system
- **Deliverables**:
  - Production system
  - User training materials
  - Support documentation
- **Success Metrics**: Successful deployment and adoption

### 4.2 Continuous Improvement
- **Objective**: Establish improvement processes
- **Deliverables**:
  - Monitoring systems
  - Feedback mechanisms
  - Update procedures
- **Success Metrics**: Continuous improvement cycle established

## Risk Mitigation
- **Technical Risks**: Prototype early, validate assumptions
- **Domain Risks**: Engage domain experts, validate understanding
- **Integration Risks**: Start with simple integrations, build complexity gradually
- **Adoption Risks**: Provide clear value proposition, demonstrate benefits

## Success Criteria
1. **Functional Integration**: Successfully integrate {len(domains)} domains
2. **Performance**: Achieve target performance metrics
3. **Adoption**: Gain user acceptance and adoption
4. **Impact**: Demonstrate measurable improvements in cross-domain knowledge synthesis
"""


def _generate_fallback_knowledge(oracle_output: str) -> str:
    """Generate fallback knowledge output when principle parsing fails"""
    
    return f"""# FALLBACK KNOWLEDGE SYNTHESIS

**Generated**: {datetime.now().isoformat()}
**Status**: Fallback Mode

## Raw Oracle Output
{oracle_output}

## Fallback Analysis
This knowledge synthesis was generated in fallback mode due to parsing difficulties with the Oracle output. The system attempted to extract meaningful information but encountered challenges in structured parsing.

## Recommendations
1. **Review Oracle Output**: Examine the raw output for formatting issues
2. **Manual Analysis**: Conduct manual analysis of the principle content
3. **System Improvement**: Enhance parsing capabilities for complex outputs

## Generated Content
- **Principle Name**: [Unable to extract]
- **Core Principle**: [Unable to extract]
- **Domains**: [Unable to extract]
- **Analysis**: [Unable to generate]

This fallback output serves as a placeholder until the system can properly parse and analyze the Oracle output.
"""


if __name__ == "__main__":
    # Test the Scribe agent
    test_output = """{
        "principle_name": "Test Principle",
        "one_sentence_summary": "This is a test principle for validation",
        "domains": ["Test Domain 1", "Test Domain 2"],
        "predictions": ["Prediction 1", "Prediction 2"]
    }"""
    
    result = run(None, test_output, verbose=True)
    print("Scribe Agent Test Result:")
    print(result)
