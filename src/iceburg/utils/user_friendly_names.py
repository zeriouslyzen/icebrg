"""
User-Friendly Name Mapping
Maps technical names to user-friendly display names for thinking/actions
"""

# User-friendly name mappings
USER_FRIENDLY_NAMES = {
    # Agents
    "surveyor": "Gathering Information",
    "dissident": "Exploring Alternatives",
    "synthesist": "Synthesizing Insights",
    "oracle": "Analyzing Evidence",
    "archaeologist": "Uncovering History",
    "supervisor": "Validating Results",
    "scribe": "Documenting Knowledge",
    "weaver": "Generating Code",
    "scrutineer": "Checking for Contradictions",
    "ide": "Preparing Development Environment",
    "swarm": "Coordinating Team",
    
    # Actions
    "prompt_interpreter": "Understanding Your Request",
    "system_integration": "Initializing Systems",
    "surveyor_analysis": "Searching Knowledge Base",
    "dissident_analysis": "Challenging Assumptions",
    "synthesist_analysis": "Combining Perspectives",
    "oracle_analysis": "Evaluating Evidence",
    "archaeologist_analysis": "Digging Deeper",
    "supervisor_validation": "Quality Check",
    "scrutineer_analysis": "Fact-Checking",
    "curiosity_engine": "Exploring New Ideas",
    "swarm_creation": "Assembling Team",
    "swarm_execution": "Working in Parallel",
    "insight_generation": "Finding Connections",
    "source_tracking": "Gathering Sources",
    
    # Thinking messages
    "analyzing_query": "Understanding your question",
    "processing": "Working on it",
    "searching": "Looking through knowledge",
    "synthesizing": "Putting pieces together",
    "validating": "Double-checking",
    "generating": "Creating response",
    
    # Modes
    "chat": "Quick Response",
    "fast": "Fast Mode",
    "research": "Deep Research",
    "device": "Device Design",
    "truth": "Truth Finding",
    "swarm": "Team Analysis",
    
    # Engines
    "MethodologyAnalyzer": "Research Methodology",
    "VectorStore": "Knowledge Search",
    "KnowledgeGraph": "Evidence Analysis",
    "EmergenceEngine": "Pattern Detection",
    "CuriosityEngine": "Exploration",
    "HybridReasoningEngine": "Advanced Reasoning",
    "SuppressionDetector": "Truth Detection",
    "ValidationEngine": "Quality Assurance",
    "HallucinationDetector": "Accuracy Check",
    "DeviceGenerator": "Design System",
    "MicroAgentSwarm": "Team Coordination",
    "SwarmingIntegration": "Parallel Processing",
    "InsightGenerator": "Insight Discovery",
    "SourceCitationTracker": "Source Management",
    
    # Algorithms
    "Enhanced Deliberation": "Deep Analysis",
    "Semantic Search": "Smart Search",
    "Parallel Execution": "Multi-Task Processing",
    "Surveyor Agent": "Information Gathering",
    "Dissident Agent": "Alternative Analysis",
    "Synthesist Agent": "Knowledge Synthesis",
    "Oracle Agent": "Evidence Evaluation",
    "Scrutineer Agent": "Contradiction Detection",
    "Archaeologist Agent": "Historical Research",
    "Supervisor Agent": "Quality Validation",
    "Scribe Agent": "Knowledge Documentation",
    "Weaver Agent": "Code Generation",
    "Curiosity-Driven Query Generation": "Exploration Queries",
    "Micro-Agent Swarm": "Team Processing",
    "Insight Generation": "Connection Discovery",
    "Citation Tracking": "Source Tracking",
}

# Context-aware messages based on mode/agent
CONTEXT_MESSAGES = {
    # Mode-specific
    "chat": {
        "default": "Quickly analyzing your question",
        "fast_path": "Finding the fastest answer",
        "deep_path": "Diving deeper into your question"
    },
    "research": {
        "default": "Conducting comprehensive research",
        "surveyor": "Gathering research materials",
        "dissident": "Exploring alternative perspectives",
        "synthesist": "Synthesizing research findings"
    },
    "truth": {
        "default": "Searching for truth",
        "suppression_detector": "Detecting suppressed information",
        "scrutineer": "Checking for contradictions"
    },
    "device": {
        "default": "Designing your device",
        "device_generator": "Creating specifications"
    },
    "swarm": {
        "default": "Coordinating team analysis",
        "swarm_creation": "Assembling specialists",
        "swarm_execution": "Working together"
    }
}

def get_user_friendly_name(technical_name: str, context: dict = None) -> str:
    """
    Get user-friendly name for technical name
    
    Args:
        technical_name: Technical name (e.g., "surveyor", "MethodologyAnalyzer")
        context: Optional context dict with mode, agent, etc.
    
    Returns:
        User-friendly display name
    """
    # Check direct mapping first
    if technical_name in USER_FRIENDLY_NAMES:
        return USER_FRIENDLY_NAMES[technical_name]
    
    # Try lowercase
    technical_lower = technical_name.lower()
    if technical_lower in USER_FRIENDLY_NAMES:
        return USER_FRIENDLY_NAMES[technical_lower]
    
    # Try to extract base name (e.g., "SurveyorAgent" -> "surveyor")
    if "agent" in technical_lower:
        base = technical_lower.replace("agent", "").replace("_", "").strip()
        if base in USER_FRIENDLY_NAMES:
            return USER_FRIENDLY_NAMES[base]
    
    # Try to extract engine name (e.g., "MethodologyAnalyzer" -> "MethodologyAnalyzer")
    if "engine" in technical_lower:
        base = technical_lower.replace("engine", "").replace("_", "").strip()
        if base in USER_FRIENDLY_NAMES:
            return USER_FRIENDLY_NAMES[base]
    
    # Fallback: Capitalize and format
    return technical_name.replace("_", " ").title()

def get_context_message(mode: str = None, agent: str = None, action: str = None) -> str:
    """
    Get context-aware message based on mode/agent/action
    
    Args:
        mode: Current mode (chat, research, truth, etc.)
        agent: Current agent (surveyor, dissident, etc.)
        action: Current action (surveyor_analysis, etc.)
    
    Returns:
        Context-aware message
    """
    if mode and mode in CONTEXT_MESSAGES:
        mode_messages = CONTEXT_MESSAGES[mode]
        
        # Try action-specific
        if action and action in mode_messages:
            return mode_messages[action]
        
        # Try agent-specific
        if agent and agent in mode_messages:
            return mode_messages[agent]
        
        # Use default
        if "default" in mode_messages:
            return mode_messages["default"]
    
    # Fallback
    return "Processing your request"

def format_thinking_message(agent: str = None, content: str = None, mode: str = None) -> str:
    """
    Format thinking message with user-friendly names
    
    Args:
        agent: Agent name (e.g., "surveyor")
        content: Original content
        mode: Current mode
    
    Returns:
        Formatted thinking message
    """
    if agent:
        friendly_name = get_user_friendly_name(agent)
        if content:
            # Remove agent name from content if it's redundant
            content_lower = content.lower()
            agent_lower = agent.lower()
            if agent_lower in content_lower:
                # Content already includes agent name, just use friendly name
                return f"{friendly_name}: {content}"
            else:
                return f"{friendly_name}: {content}"
        else:
            context_msg = get_context_message(mode=mode, agent=agent)
            return f"{friendly_name}: {context_msg}"
    
    if content:
        return content
    
    return "Thinking..."

def format_action_message(action: str = None, description: str = None, mode: str = None) -> str:
    """
    Format action message with user-friendly names
    
    Args:
        action: Action name (e.g., "surveyor_analysis")
        description: Original description
        mode: Current mode
    
    Returns:
        Formatted action message
    """
    if action:
        friendly_name = get_user_friendly_name(action)
        if description:
            # Check if description already includes friendly name
            if friendly_name.lower() in description.lower():
                return description
            return f"{friendly_name}: {description}"
        else:
            context_msg = get_context_message(mode=mode, action=action)
            return f"{friendly_name}: {context_msg}"
    
    if description:
        return description
    
    return "Processing..."

