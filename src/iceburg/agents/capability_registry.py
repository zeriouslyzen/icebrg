"""
ICEBURG Centralized Agent Capability Registry

Provides unified registry for all 45+ agents with comprehensive metadata:
- Input/output types
- Complexity and speed ratings
- Dependencies
- Capabilities
- Integration points
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Agent type classification"""
    CORE_ANALYSIS = "core_analysis"
    SYNTHESIS = "synthesis"
    VALIDATION = "validation"
    IMPLEMENTATION = "implementation"
    COORDINATION = "coordination"
    SPECIALIZED = "specialized"
    ARCHITECT = "architect"


class ComplexityLevel(Enum):
    """Complexity level for agent tasks"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4
    EXTREMELY_COMPLEX = 5


class SpeedRating(Enum):
    """Speed rating for agent execution"""
    ULTRA_FAST = 1  # < 1 second
    FAST = 2  # 1-5 seconds
    MODERATE = 3  # 5-15 seconds
    SLOW = 4  # 15-30 seconds
    VERY_SLOW = 5  # > 30 seconds


@dataclass
class AgentCapability:
    """Comprehensive agent capability definition"""
    agent_id: str
    agent_name: str
    agent_type: AgentType
    description: str
    input_types: List[str]
    output_types: List[str]
    capabilities: List[str]
    complexity_level: ComplexityLevel
    speed_rating: SpeedRating
    dependencies: List[str] = field(default_factory=list)
    required_engines: List[str] = field(default_factory=list)
    optional_engines: List[str] = field(default_factory=list)
    parallelizable: bool = True
    timeout_seconds: float = 30.0
    memory_mb: float = 512.0
    cpu_cores: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class AgentCapabilityRegistry:
    """
    Centralized registry for all ICEBURG agents with comprehensive metadata.
    
    Provides:
    - Dynamic capability discovery
    - Agent selection based on requirements
    - Dependency resolution
    - Performance optimization
    """
    
    def __init__(self):
        self.agents: Dict[str, AgentCapability] = {}
        self._initialize_registry()
        logger.info(f"Agent Capability Registry initialized with {len(self.agents)} agents")
    
    def _initialize_registry(self):
        """Initialize registry with all ICEBURG agents"""
        
        # Core Analysis Agents
        self._register_core_agents()
        
        # Synthesis Agents
        self._register_synthesis_agents()
        
        # Validation Agents
        self._register_validation_agents()
        
        # Implementation Agents
        self._register_implementation_agents()
        
        # Coordination Agents
        self._register_coordination_agents()
        
        # Specialized Agents
        self._register_specialized_agents()
        
        # Architect Agents
        self._register_architect_agents()
    
    def _register_core_agents(self):
        """Register core analysis agents"""
        
        # Surveyor Agent
        self.agents["surveyor"] = AgentCapability(
            agent_id="surveyor",
            agent_name="Surveyor",
            agent_type=AgentType.CORE_ANALYSIS,
            description="Information gathering and initial analysis. Research synthesis, domain exploration, evidence collection.",
            input_types=["query", "text", "multimodal"],
            output_types=["analysis", "synthesis", "evidence"],
            capabilities=["information_gathering", "research_synthesis", "domain_exploration", "evidence_collection", "semantic_search"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=["VectorStore"],
            optional_engines=["SourceCitationTracker"],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "surveyor_model", "priority": 1}
        )
        
        # Dissident Agent
        self.agents["dissident"] = AgentCapability(
            agent_id="dissident",
            agent_name="Dissident",
            agent_type=AgentType.CORE_ANALYSIS,
            description="Challenges assumptions and explores alternative perspectives. Generates counter-arguments and explores suppressed information.",
            input_types=["query", "surveyor_output", "text"],
            output_types=["alternative_analysis", "counter_arguments", "suppressed_info"],
            capabilities=["assumption_challenging", "alternative_perspectives", "counter_argumentation", "suppression_detection"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=["surveyor"],
            required_engines=["SuppressionDetector"],
            optional_engines=["VectorStore"],
            parallelizable=False,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "dissident_model", "priority": 2}
        )
        
        # Archaeologist Agent
        self.agents["archaeologist"] = AgentCapability(
            agent_id="archaeologist",
            agent_name="Archaeologist",
            agent_type=AgentType.CORE_ANALYSIS,
            description="Uncovers buried evidence, historical insights, and suppressed knowledge. Deep historical research and hidden information identification.",
            input_types=["query", "documents", "text"],
            output_types=["historical_insights", "buried_evidence", "suppressed_knowledge"],
            capabilities=["historical_research", "buried_evidence_detection", "suppressed_knowledge_recovery", "pattern_recognition"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["VectorStore", "KnowledgeGraph"],
            optional_engines=["EmergenceEngine"],
            parallelizable=True,
            timeout_seconds=60.0,
            memory_mb=2048.0,
            cpu_cores=2,
            metadata={"model": "archaeologist_model", "priority": 1}
        )
    
    def _register_synthesis_agents(self):
        """Register synthesis agents"""
        
        # Synthesist Agent
        self.agents["synthesist"] = AgentCapability(
            agent_id="synthesist",
            agent_name="Synthesist",
            agent_type=AgentType.SYNTHESIS,
            description="Synthesizes insights from various sources and agents. Integrates outputs from Surveyor, Dissident, and other agents.",
            input_types=["surveyor_output", "dissident_output", "enhanced_context", "multimodal_evidence"],
            output_types=["synthesis", "integrated_insights", "coherent_conclusions"],
            capabilities=["insight_synthesis", "cross_domain_integration", "pattern_identification", "multimodal_analysis"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=["surveyor", "dissident"],
            required_engines=["KnowledgeGraph"],
            optional_engines=["InsightGenerator", "EmergenceEngine"],
            parallelizable=False,
            timeout_seconds=45.0,
            memory_mb=2048.0,
            cpu_cores=2,
            metadata={"model": "synthesist_model", "priority": 3}
        )
        
        # Oracle Agent
        self.agents["oracle"] = AgentCapability(
            agent_id="oracle",
            agent_name="Oracle",
            agent_type=AgentType.SYNTHESIS,
            description="Synthesizes evidence into fundamental principles and generates new knowledge. Transforms synthesized information into core principles.",
            input_types=["synthesis_output", "evidence", "prior_principles"],
            output_types=["principles", "knowledge", "predictions", "study_designs"],
            capabilities=["principle_synthesis", "knowledge_generation", "evidence_weighting", "prediction_generation"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=["synthesist"],
            required_engines=["KnowledgeGraph", "VectorStore"],
            optional_engines=["EmergenceEngine", "CuriosityEngine"],
            parallelizable=False,
            timeout_seconds=60.0,
            memory_mb=3072.0,
            cpu_cores=2,
            metadata={"model": "oracle_model", "priority": 4}
        )
    
    def _register_validation_agents(self):
        """Register validation agents"""
        
        # Scrutineer Agent
        self.agents["scrutineer"] = AgentCapability(
            agent_id="scrutineer",
            agent_name="Scrutineer",
            agent_type=AgentType.VALIDATION,
            description="Scrutinizes synthesis for contradictions, gaps, and potential suppression. Acts as critical reviewer for internal outputs.",
            input_types=["synthesis_output", "evidence"],
            output_types=["validation", "contradiction_analysis", "suppression_indicators"],
            capabilities=["contradiction_detection", "gap_analysis", "suppression_detection", "validation"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=["synthesist"],
            required_engines=["SuppressionDetector", "ValidationEngine"],
            optional_engines=["HallucinationDetector"],
            parallelizable=False,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "scrutineer_model", "priority": 3}
        )
        
        # Supervisor Agent
        self.agents["supervisor"] = AgentCapability(
            agent_id="supervisor",
            agent_name="Supervisor",
            agent_type=AgentType.VALIDATION,
            description="Validates outputs, reasoning chains, and ensures overall quality and coherence. Reviews outputs of other agents.",
            input_types=["stage_outputs", "query", "agent_outputs"],
            output_types=["validation", "quality_assessment", "coherence_check"],
            capabilities=["output_validation", "reasoning_validation", "quality_control", "coherence_checking"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=["ValidationEngine"],
            optional_engines=["HallucinationDetector"],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "supervisor_model", "priority": 5}
        )
    
    def _register_implementation_agents(self):
        """Register implementation agents"""
        
        # Weaver Agent
        self.agents["weaver"] = AgentCapability(
            agent_id="weaver",
            agent_name="Weaver",
            agent_type=AgentType.IMPLEMENTATION,
            description="Generates code and executable logic from Oracle output. Translates principles into functional code.",
            input_types=["oracle_output", "principle_data"],
            output_types=["code", "executable_logic", "implementations"],
            capabilities=["code_generation", "principle_to_code", "domain_specific_code", "executable_generation"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=["oracle"],
            required_engines=[],
            optional_engines=["DeviceGenerator"],
            parallelizable=False,
            timeout_seconds=45.0,
            memory_mb=1536.0,
            cpu_cores=1,
            metadata={"model": "weaver_model", "priority": 5}
        )
        
        # Scribe Agent
        self.agents["scribe"] = AgentCapability(
            agent_id="scribe",
            agent_name="Scribe",
            agent_type=AgentType.IMPLEMENTATION,
            description="Generates structured knowledge outputs from Oracle principles. Formats principles into documentation and reports.",
            input_types=["oracle_output", "principle_data"],
            output_types=["documentation", "reports", "knowledge_artifacts"],
            capabilities=["documentation_generation", "knowledge_formatting", "report_generation", "academic_output"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=["oracle"],
            required_engines=[],
            optional_engines=["SourceCitationTracker"],
            parallelizable=False,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "scribe_model", "priority": 5}
        )
        
        # IDE Agent
        self.agents["ide"] = AgentCapability(
            agent_id="ide",
            agent_name="IDE Agent",
            agent_type=AgentType.IMPLEMENTATION,
            description="Provides safe command execution and code editing capabilities within a controlled environment.",
            input_types=["commands", "code", "file_paths"],
            output_types=["execution_results", "edited_code", "file_contents"],
            capabilities=["command_execution", "code_editing", "file_management", "safe_execution"],
            complexity_level=ComplexityLevel.MODERATE,
            speed_rating=SpeedRating.FAST,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=15.0,
            memory_mb=512.0,
            cpu_cores=1,
            metadata={"model": "ide_model", "priority": 6}
        )
    
    def _register_coordination_agents(self):
        """Register coordination agents"""
        
        # Prompt Interpreter
        self.agents["prompt_interpreter"] = AgentCapability(
            agent_id="prompt_interpreter",
            agent_name="Prompt Interpreter",
            agent_type=AgentType.COORDINATION,
            description="Initial analysis of user's query to determine intent, domain, and complexity. Linguistic breakdown and semantic analysis.",
            input_types=["query", "text"],
            output_types=["intent", "domain", "complexity", "word_breakdown"],
            capabilities=["intent_analysis", "domain_detection", "complexity_scoring", "linguistic_analysis", "etymology"],
            complexity_level=ComplexityLevel.MODERATE,
            speed_rating=SpeedRating.FAST,
            dependencies=[],
            required_engines=[],
            optional_engines=["WordBreakdownAnalyzer"],
            parallelizable=False,
            timeout_seconds=10.0,
            memory_mb=512.0,
            cpu_cores=1,
            metadata={"model": "prompt_interpreter_model", "priority": 0}
        )
        
        # Reflex Agent
        self.agents["reflex_agent"] = AgentCapability(
            agent_id="reflex_agent",
            agent_name="Reflex Agent",
            agent_type=AgentType.COORDINATION,
            description="Compresses verbose responses and extracts key bullets. Reduces verbosity while preserving linguistic depth.",
            input_types=["full_response", "text"],
            output_types=["compressed_response", "preview_bullets", "reflections"],
            capabilities=["response_compression", "bullet_extraction", "verbosity_removal", "reflection_extraction"],
            complexity_level=ComplexityLevel.SIMPLE,
            speed_rating=SpeedRating.ULTRA_FAST,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=2.0,
            memory_mb=256.0,
            cpu_cores=1,
            metadata={"model": None, "priority": 10}
        )
    
    def _register_specialized_agents(self):
        """Register specialized agents"""
        
        # Deliberation Agent
        self.agents["deliberation"] = AgentCapability(
            agent_id="deliberation",
            agent_name="Deliberation Agent",
            agent_type=AgentType.SPECIALIZED,
            description="Applies Enhanced Deliberation methodology. Performs meta-analysis and truth-seeking analysis.",
            input_types=["agent_output", "query"],
            output_types=["deliberation", "meta_analysis", "truth_analysis"],
            capabilities=["deliberation", "meta_analysis", "truth_seeking", "emergence_detection"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["MethodologyAnalyzer", "EmergenceEngine"],
            optional_engines=["CuriosityEngine"],
            parallelizable=True,
            timeout_seconds=45.0,
            memory_mb=2048.0,
            cpu_cores=2,
            metadata={"model": "deliberation_model", "priority": 3}
        )
        
        # Capability Gap Detector
        self.agents["capability_gap_detector"] = AgentCapability(
            agent_id="capability_gap_detector",
            agent_name="Capability Gap Detector",
            agent_type=AgentType.SPECIALIZED,
            description="Detects gaps in agent capabilities and identifies areas for improvement.",
            input_types=["agent_outputs", "requirements"],
            output_types=["gap_analysis", "improvement_recommendations"],
            capabilities=["gap_detection", "capability_analysis", "improvement_identification"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "capability_gap_model", "priority": 6}
        )
        
        # Hypothesis Testing Laboratory
        self.agents["hypothesis_testing_laboratory"] = AgentCapability(
            agent_id="hypothesis_testing_laboratory",
            agent_name="Hypothesis Testing Laboratory",
            agent_type=AgentType.SPECIALIZED,
            description="Tests hypotheses using statistical methods and simulation.",
            input_types=["hypothesis", "data", "test_parameters"],
            output_types=["test_results", "statistical_analysis", "validation"],
            capabilities=["hypothesis_testing", "statistical_analysis", "simulation", "validation"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=[],
            optional_engines=["VirtualPhysicsLab"],
            parallelizable=True,
            timeout_seconds=60.0,
            memory_mb=2048.0,
            cpu_cores=2,
            metadata={"model": "hypothesis_testing_model", "priority": 4}
        )
    
    def _register_architect_agents(self):
        """Register architect agents"""
        
        # Pyramid DAG Architect
        self.agents["pyramid_dag_architect"] = AgentCapability(
            agent_id="pyramid_dag_architect",
            agent_name="Pyramid DAG Architect",
            agent_type=AgentType.ARCHITECT,
            description="Implements pyramid DAG architecture for hierarchical task decomposition with Judge Agent verification.",
            input_types=["requirement", "specifications"],
            output_types=["dag_structure", "execution_plan", "validation"],
            capabilities=["dag_construction", "task_decomposition", "dependency_resolution", "execution_planning"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["MicroAgentSwarm"],
            optional_engines=["SwarmingIntegration"],
            parallelizable=False,
            timeout_seconds=120.0,
            memory_mb=4096.0,
            cpu_cores=4,
            metadata={"model": "pyramid_dag_model", "priority": 2}
        )
        
        # Enhanced Swarm Architect
        self.agents["enhanced_swarm_architect"] = AgentCapability(
            agent_id="enhanced_swarm_architect",
            agent_name="Enhanced Swarm Architect",
            agent_type=AgentType.ARCHITECT,
            description="Semantic routing and capability matching for optimal agent selection. Dual-audit mechanism and dynamic resource monitoring.",
            input_types=["requirement", "agent_swarm"],
            output_types=["agent_selection", "task_distribution", "coordination_plan"],
            capabilities=["semantic_routing", "capability_matching", "agent_selection", "task_distribution"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["MicroAgentSwarm", "SwarmingIntegration"],
            optional_engines=["BlackboardIntegration"],
            parallelizable=False,
            timeout_seconds=60.0,
            memory_mb=3072.0,
            cpu_cores=2,
            metadata={"model": "swarm_architect_model", "priority": 2}
        )
        
        # Swarm Architect
        self.agents["swarm_architect"] = AgentCapability(
            agent_id="swarm_architect",
            agent_name="Swarm Architect",
            agent_type=AgentType.ARCHITECT,
            description="Coordinates micro-agent swarm for parallel processing. Manages task distribution and agent coordination.",
            input_types=["tasks", "agent_swarm"],
            output_types=["swarm_coordination", "task_results", "aggregated_output"],
            capabilities=["swarm_coordination", "parallel_processing", "task_distribution", "result_aggregation"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["MicroAgentSwarm", "SwarmingIntegration"],
            optional_engines=["BlackboardIntegration"],
            parallelizable=False,
            timeout_seconds=60.0,
            memory_mb=3072.0,
            cpu_cores=2,
            metadata={"model": "swarm_architect_model", "priority": 2}
        )
        
        # Swarm Integrated Architect
        self.agents["swarm_integrated_architect"] = AgentCapability(
            agent_id="swarm_integrated_architect",
            agent_name="Swarm Integrated Architect",
            agent_type=AgentType.ARCHITECT,
            description="Integrates micro-agent swarm with ICEBURG's existing software lab. Replaces traditional Architect with swarm-powered version.",
            input_types=["requirement", "principle"],
            output_types=["software_result", "swarm_coordination", "integrated_output"],
            capabilities=["swarm_integration", "software_generation", "swarm_coordination", "lab_integration"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["MicroAgentSwarm", "SwarmingIntegration"],
            optional_engines=["BlackboardIntegration"],
            parallelizable=False,
            timeout_seconds=60.0,
            memory_mb=3072.0,
            cpu_cores=2,
            metadata={"model": "swarm_integrated_model", "priority": 2}
        )
        
        # Working Swarm Architect
        self.agents["working_swarm_architect"] = AgentCapability(
            agent_id="working_swarm_architect",
            agent_name="Working Swarm Architect",
            agent_type=AgentType.ARCHITECT,
            description="Swarm architect with correct agent capabilities. Uses actual agent capabilities for software generation.",
            input_types=["requirement", "complexity"],
            output_types=["software_result", "swarm_output", "architecture"],
            capabilities=["swarm_coordination", "capability_matching", "software_generation", "parallel_processing"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["MicroAgentSwarm"],
            optional_engines=["SwarmingIntegration"],
            parallelizable=False,
            timeout_seconds=60.0,
            memory_mb=3072.0,
            cpu_cores=2,
            metadata={"model": "working_swarm_model", "priority": 2}
        )
        
        # Emergent Architect
        self.agents["emergent_architect"] = AgentCapability(
            agent_id="emergent_architect",
            agent_name="Emergent Architect",
            agent_type=AgentType.ARCHITECT,
            description="Generates software architectures that emerge from principles. Creates emergent system designs.",
            input_types=["principle", "requirements"],
            output_types=["architecture", "emergent_design", "system_structure"],
            capabilities=["emergent_design", "architecture_generation", "principle_to_architecture", "system_design"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["EmergenceEngine"],
            optional_engines=["CuriosityEngine"],
            parallelizable=False,
            timeout_seconds=90.0,
            memory_mb=4096.0,
            cpu_cores=2,
            metadata={"model": "emergent_architect_model", "priority": 2}
        )
        
        # Visual Architect
        self.agents["visual_architect"] = AgentCapability(
            agent_id="visual_architect",
            agent_name="Visual Architect",
            agent_type=AgentType.ARCHITECT,
            description="Generates visual architectures and diagrams. Creates visual representations of system designs.",
            input_types=["architecture", "requirements"],
            output_types=["visual_diagrams", "architecture_visuals", "design_visualizations"],
            capabilities=["visual_generation", "diagram_creation", "architecture_visualization", "design_rendering"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=[],
            optional_engines=["VisualizationEngine"],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1536.0,
            cpu_cores=1,
            metadata={"model": "visual_architect_model", "priority": 6}
        )
        
        # Visual Red Team
        self.agents["visual_red_team"] = AgentCapability(
            agent_id="visual_red_team",
            agent_name="Visual Red Team",
            agent_type=AgentType.VALIDATION,
            description="Red team analysis for visual architectures. Tests visual designs for vulnerabilities and issues.",
            input_types=["visual_architecture", "design"],
            output_types=["red_team_analysis", "vulnerability_report", "security_assessment"],
            capabilities=["red_teaming", "vulnerability_detection", "security_analysis", "visual_testing"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=["visual_architect"],
            required_engines=[],
            optional_engines=["SecurityEngine"],
            parallelizable=False,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "visual_red_team_model", "priority": 6}
        )
        
        # Real Scientific Research
        self.agents["real_scientific_research"] = AgentCapability(
            agent_id="real_scientific_research",
            agent_name="Real Scientific Research",
            agent_type=AgentType.SPECIALIZED,
            description="Conducts real scientific research with rigorous methodology. Applies scientific method to research questions.",
            input_types=["research_question", "hypothesis", "data"],
            output_types=["research_results", "scientific_analysis", "research_paper"],
            capabilities=["scientific_research", "methodology_application", "data_analysis", "research_synthesis"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["MethodologyAnalyzer", "ResearchEngine"],
            optional_engines=["VirtualPhysicsLab"],
            parallelizable=False,
            timeout_seconds=120.0,
            memory_mb=4096.0,
            cpu_cores=4,
            metadata={"model": "research_model", "priority": 1}
        )
        
        # Virtual Scientific Ecosystem
        self.agents["virtual_scientific_ecosystem"] = AgentCapability(
            agent_id="virtual_scientific_ecosystem",
            agent_name="Virtual Scientific Ecosystem",
            agent_type=AgentType.SPECIALIZED,
            description="Creates virtual scientific ecosystems for experimentation. Simulates scientific environments.",
            input_types=["ecosystem_spec", "parameters"],
            output_types=["ecosystem_simulation", "experimental_results", "ecosystem_analysis"],
            capabilities=["ecosystem_simulation", "virtual_environment", "experimental_design", "scientific_simulation"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["VirtualPhysicsLab", "SimulationEngine"],
            optional_engines=["EmergenceEngine"],
            parallelizable=False,
            timeout_seconds=120.0,
            memory_mb=4096.0,
            cpu_cores=4,
            metadata={"model": "ecosystem_model", "priority": 1}
        )
        
        # Geospatial Financial Anthropological
        self.agents["geospatial_financial_anthropological"] = AgentCapability(
            agent_id="geospatial_financial_anthropological",
            agent_name="Geospatial Financial Anthropological",
            agent_type=AgentType.SPECIALIZED,
            description="Analyzes geospatial, financial, and anthropological patterns. Cross-domain analysis across geography, finance, and anthropology.",
            input_types=["geospatial_data", "financial_data", "anthropological_data"],
            output_types=["cross_domain_analysis", "pattern_identification", "integrated_insights"],
            capabilities=["geospatial_analysis", "financial_analysis", "anthropological_analysis", "cross_domain_synthesis"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=["KnowledgeGraph"],
            optional_engines=["EmergenceEngine"],
            parallelizable=True,
            timeout_seconds=60.0,
            memory_mb=2048.0,
            cpu_cores=2,
            metadata={"model": "geospatial_model", "priority": 3}
        )
        
        # Molecular Synthesis
        self.agents["molecular_synthesis"] = AgentCapability(
            agent_id="molecular_synthesis",
            agent_name="Molecular Synthesis",
            agent_type=AgentType.SPECIALIZED,
            description="Synthesizes molecular structures and chemical compounds. Generates molecular designs and predictions.",
            input_types=["molecular_spec", "chemical_constraints"],
            output_types=["molecular_design", "chemical_structure", "synthesis_pathway"],
            capabilities=["molecular_design", "chemical_synthesis", "structure_prediction", "pathway_generation"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["VirtualPhysicsLab"],
            optional_engines=["SimulationEngine"],
            parallelizable=False,
            timeout_seconds=90.0,
            memory_mb=3072.0,
            cpu_cores=2,
            metadata={"model": "molecular_model", "priority": 2}
        )
        
        # Comprehensive API Manager
        self.agents["comprehensive_api_manager"] = AgentCapability(
            agent_id="comprehensive_api_manager",
            agent_name="Comprehensive API Manager",
            agent_type=AgentType.SPECIALIZED,
            description="Manages comprehensive API integrations and external service connections. Coordinates API calls and data retrieval.",
            input_types=["api_spec", "service_requirements"],
            output_types=["api_integration", "service_coordination", "api_results"],
            capabilities=["api_management", "service_integration", "api_coordination", "external_service"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "api_manager_model", "priority": 5}
        )
        
        # Corporate Network Analyzer
        self.agents["corporate_network_analyzer"] = AgentCapability(
            agent_id="corporate_network_analyzer",
            agent_name="Corporate Network Analyzer",
            agent_type=AgentType.SPECIALIZED,
            description="Analyzes corporate networks, relationships, and organizational structures. Identifies patterns in corporate data.",
            input_types=["corporate_data", "network_structure"],
            output_types=["network_analysis", "relationship_mapping", "organizational_insights"],
            capabilities=["network_analysis", "corporate_analysis", "relationship_mapping", "organizational_analysis"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=["KnowledgeGraph"],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1536.0,
            cpu_cores=1,
            metadata={"model": "corporate_network_model", "priority": 5}
        )
        
        # Public Services Integration
        self.agents["public_services_integration"] = AgentCapability(
            agent_id="public_services_integration",
            agent_name="Public Services Integration",
            agent_type=AgentType.SPECIALIZED,
            description="Integrates with public services and government APIs. Manages public service data and interactions.",
            input_types=["service_request", "public_data"],
            output_types=["service_integration", "public_data_analysis", "service_results"],
            capabilities=["public_service_integration", "government_api", "public_data_analysis", "service_coordination"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "public_services_model", "priority": 5}
        )
        
        # Grounding Layer Agent
        self.agents["grounding_layer_agent"] = AgentCapability(
            agent_id="grounding_layer_agent",
            agent_name="Grounding Layer Agent",
            agent_type=AgentType.SPECIALIZED,
            description="Provides grounding layer for abstract concepts. Connects abstract reasoning to concrete implementations.",
            input_types=["abstract_concept", "concrete_context"],
            output_types=["grounded_concept", "concrete_implementation", "grounding_analysis"],
            capabilities=["concept_grounding", "abstraction_to_concrete", "implementation_grounding", "context_grounding"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=["KnowledgeGraph"],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "grounding_model", "priority": 4}
        )
        
        # Digital Twins Simulation
        self.agents["digital_twins_simulation"] = AgentCapability(
            agent_id="digital_twins_simulation",
            agent_name="Digital Twins Simulation",
            agent_type=AgentType.SPECIALIZED,
            description="Creates and simulates digital twins of physical systems. Models real-world systems in virtual environments.",
            input_types=["physical_system", "simulation_parameters"],
            output_types=["digital_twin", "simulation_results", "twin_analysis"],
            capabilities=["digital_twin_creation", "system_simulation", "twin_modeling", "virtual_modeling"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["SimulationEngine", "VirtualPhysicsLab"],
            optional_engines=["EmergenceEngine"],
            parallelizable=False,
            timeout_seconds=120.0,
            memory_mb=4096.0,
            cpu_cores=4,
            metadata={"model": "digital_twins_model", "priority": 1}
        )
        
        # Celestial Biological Framework
        self.agents["celestial_biological_framework"] = AgentCapability(
            agent_id="celestial_biological_framework",
            agent_name="Celestial Biological Framework",
            agent_type=AgentType.SPECIALIZED,
            description="Analyzes biological systems in celestial contexts. Cross-domain analysis of biology and astronomy.",
            input_types=["biological_data", "celestial_data"],
            output_types=["celestial_biological_analysis", "cross_domain_insights", "framework_application"],
            capabilities=["biological_analysis", "celestial_analysis", "cross_domain_synthesis", "framework_application"],
            complexity_level=ComplexityLevel.EXTREMELY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=["KnowledgeGraph", "EmergenceEngine"],
            optional_engines=["CuriosityEngine"],
            parallelizable=False,
            timeout_seconds=120.0,
            memory_mb=4096.0,
            cpu_cores=4,
            metadata={"model": "celestial_biological_model", "priority": 1}
        )
        
        # RAG Memory Integration
        self.agents["rag_memory_integration"] = AgentCapability(
            agent_id="rag_memory_integration",
            agent_name="RAG Memory Integration",
            agent_type=AgentType.SPECIALIZED,
            description="Integrates RAG (Retrieval-Augmented Generation) with memory systems. Enhances retrieval with persistent memory.",
            input_types=["query", "memory_context"],
            output_types=["rag_enhanced_output", "memory_integrated_result", "contextual_response"],
            capabilities=["rag_integration", "memory_retrieval", "context_enhancement", "persistent_memory"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=["VectorStore", "RAGEngine"],
            optional_engines=["MemoryEngine"],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1536.0,
            cpu_cores=1,
            metadata={"model": "rag_memory_model", "priority": 4}
        )
        
        # Teacher Student Tuning
        self.agents["teacher_student_tuning"] = AgentCapability(
            agent_id="teacher_student_tuning",
            agent_name="Teacher Student Tuning",
            agent_type=AgentType.SPECIALIZED,
            description="Implements teacher-student learning paradigm for model tuning. Distills knowledge from teacher to student models.",
            input_types=["teacher_model", "student_model", "training_data"],
            output_types=["tuned_model", "knowledge_distillation", "model_optimization"],
            capabilities=["knowledge_distillation", "model_tuning", "teacher_student_learning", "model_optimization"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.VERY_SLOW,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=False,
            timeout_seconds=180.0,
            memory_mb=4096.0,
            cpu_cores=4,
            metadata={"model": "teacher_student_model", "priority": 1}
        )
        
        # Runtime Agent Modifier
        self.agents["runtime_agent_modifier"] = AgentCapability(
            agent_id="runtime_agent_modifier",
            agent_name="Runtime Agent Modifier",
            agent_type=AgentType.SPECIALIZED,
            description="Modifies agents at runtime based on performance and requirements. Dynamically adjusts agent behavior.",
            input_types=["agent_spec", "modification_requirements"],
            output_types=["modified_agent", "runtime_changes", "optimization_results"],
            capabilities=["runtime_modification", "agent_optimization", "dynamic_adaptation", "behavior_modification"],
            complexity_level=ComplexityLevel.VERY_COMPLEX,
            speed_rating=SpeedRating.SLOW,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=False,
            timeout_seconds=45.0,
            memory_mb=2048.0,
            cpu_cores=2,
            metadata={"model": "runtime_modifier_model", "priority": 3}
        )
        
        # Dynamic Agent Factory
        self.agents["dynamic_agent_factory"] = AgentCapability(
            agent_id="dynamic_agent_factory",
            agent_name="Dynamic Agent Factory",
            agent_type=AgentType.COORDINATION,
            description="Creates agents dynamically at runtime based on requirements. Factory for generating specialized agents.",
            input_types=["agent_requirements", "capability_spec"],
            output_types=["agent_instance", "factory_result", "agent_configuration"],
            capabilities=["agent_creation", "dynamic_factory", "runtime_agent_generation", "capability_matching"],
            complexity_level=ComplexityLevel.COMPLEX,
            speed_rating=SpeedRating.MODERATE,
            dependencies=[],
            required_engines=[],
            optional_engines=[],
            parallelizable=True,
            timeout_seconds=30.0,
            memory_mb=1024.0,
            cpu_cores=1,
            metadata={"model": "dynamic_factory_model", "priority": 4}
        )
    
    def get_agent(self, agent_id: str) -> Optional[AgentCapability]:
        """Get agent capability by ID"""
        return self.agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, AgentCapability]:
        """Get all registered agents"""
        return self.agents.copy()
    
    def find_agents_by_capability(self, required_capability: str) -> List[AgentCapability]:
        """Find agents that have a specific capability"""
        matching_agents = []
        for agent in self.agents.values():
            if required_capability in agent.capabilities:
                matching_agents.append(agent)
        return matching_agents
    
    def find_agents_by_type(self, agent_type: AgentType) -> List[AgentCapability]:
        """Find agents by type"""
        return [agent for agent in self.agents.values() if agent.agent_type == agent_type]
    
    def find_best_agent(self, required_capabilities: List[str], 
                        input_types: Optional[List[str]] = None,
                        max_complexity: Optional[ComplexityLevel] = None,
                        min_speed: Optional[SpeedRating] = None) -> Optional[AgentCapability]:
        """
        Find the best agent for a task based on requirements.
        
        Args:
            required_capabilities: List of required capabilities
            input_types: Optional list of input types
            max_complexity: Optional maximum complexity level
            min_speed: Optional minimum speed rating
            
        Returns:
            Best matching agent or None
        """
        candidates = []
        
        for agent in self.agents.values():
            # Check capabilities
            capability_match = len(set(required_capabilities) & set(agent.capabilities))
            if capability_match == 0:
                continue
            
            # Check input types if specified
            if input_types:
                input_match = len(set(input_types) & set(agent.input_types))
                if input_match == 0:
                    continue
            
            # Check complexity if specified
            if max_complexity and agent.complexity_level.value > max_complexity.value:
                continue
            
            # Check speed if specified
            if min_speed and agent.speed_rating.value > min_speed.value:
                continue
            
            # Calculate match score
            score = (
                capability_match * 10 +  # Capability match weight
                (len(set(input_types or []) & set(agent.input_types)) * 5) +  # Input type match
                (5 - agent.complexity_level.value) +  # Lower complexity preferred
                (5 - agent.speed_rating.value)  # Faster preferred
            )
            
            candidates.append((score, agent))
        
        if not candidates:
            return None
        
        # Sort by score (descending) and return best match
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def resolve_dependencies(self, agent_ids: List[str]) -> List[str]:
        """
        Resolve agent dependencies and return execution order.
        
        Args:
            agent_ids: List of agent IDs to execute
            
        Returns:
            Ordered list of agent IDs respecting dependencies
        """
        # Build dependency graph
        graph = {}
        for agent_id in agent_ids:
            agent = self.agents.get(agent_id)
            if agent:
                graph[agent_id] = set(agent.dependencies)
        
        # Topological sort
        ordered = []
        visited = set()
        temp_visited = set()
        
        def visit(node: str):
            if node in temp_visited:
                # Circular dependency detected
                logger.warning(f"Circular dependency detected involving {node}")
                return
            if node in visited:
                return
            
            temp_visited.add(node)
            
            # Visit dependencies first
            for dep in graph.get(node, set()):
                if dep in agent_ids:  # Only visit if in our list
                    visit(dep)
            
            temp_visited.remove(node)
            visited.add(node)
            ordered.append(node)
        
        for agent_id in agent_ids:
            if agent_id not in visited:
                visit(agent_id)
        
        return ordered
    
    def get_parallelizable_groups(self, agent_ids: List[str]) -> List[List[str]]:
        """
        Group agents into parallelizable execution groups.
        
        Args:
            agent_ids: List of agent IDs to execute
            
        Returns:
            List of groups, where agents in each group can run in parallel
        """
        # Resolve dependencies first
        ordered = self.resolve_dependencies(agent_ids)
        
        # Build dependency map
        depends_on = {}
        for agent_id in ordered:
            agent = self.agents.get(agent_id)
            if agent:
                depends_on[agent_id] = set(agent.dependencies)
        
        # Group by dependency level
        groups = []
        completed = set()
        
        while len(completed) < len(ordered):
            # Find agents ready to execute (all dependencies completed)
            ready = []
            for agent_id in ordered:
                if agent_id in completed:
                    continue
                
                deps = depends_on.get(agent_id, set())
                if deps.issubset(completed):
                    agent = self.agents.get(agent_id)
                    if agent and agent.parallelizable:
                        ready.append(agent_id)
            
            if not ready:
                # No agents ready - might be circular dependency or missing agent
                remaining = [aid for aid in ordered if aid not in completed]
                if remaining:
                    logger.warning(f"Could not resolve dependencies for: {remaining}")
                    # Force add remaining agents
                    ready = remaining
            
            if ready:
                groups.append(ready)
                completed.update(ready)
            else:
                break
        
        return groups
    
    def get_agent_metadata(self, agent_id: str) -> Dict[str, Any]:
        """Get comprehensive metadata for an agent"""
        agent = self.agents.get(agent_id)
        if not agent:
            return {}
        
        return {
            "agent_id": agent.agent_id,
            "agent_name": agent.agent_name,
            "agent_type": agent.agent_type.value,
            "description": agent.description,
            "input_types": agent.input_types,
            "output_types": agent.output_types,
            "capabilities": agent.capabilities,
            "complexity_level": agent.complexity_level.value,
            "speed_rating": agent.speed_rating.value,
            "dependencies": agent.dependencies,
            "required_engines": agent.required_engines,
            "optional_engines": agent.optional_engines,
            "parallelizable": agent.parallelizable,
            "timeout_seconds": agent.timeout_seconds,
            "memory_mb": agent.memory_mb,
            "cpu_cores": agent.cpu_cores,
            "metadata": agent.metadata
        }


# Global registry instance
_registry: Optional[AgentCapabilityRegistry] = None


def get_registry() -> AgentCapabilityRegistry:
    """Get or create global agent capability registry"""
    global _registry
    if _registry is None:
        _registry = AgentCapabilityRegistry()
    return _registry


def register_agent(capability: AgentCapability):
    """Register a new agent capability"""
    registry = get_registry()
    registry.agents[capability.agent_id] = capability
    logger.info(f"Registered agent: {capability.agent_id}")


def get_agent_capability(agent_id: str) -> Optional[AgentCapability]:
    """Get agent capability by ID"""
    return get_registry().get_agent(agent_id)


def find_agents_by_capability(required_capability: str) -> List[AgentCapability]:
    """Find agents that have a specific capability"""
    return get_registry().find_agents_by_capability(required_capability)


def find_best_agent(required_capabilities: List[str], 
                    input_types: Optional[List[str]] = None,
                    max_complexity: Optional[ComplexityLevel] = None,
                    min_speed: Optional[SpeedRating] = None) -> Optional[AgentCapability]:
    """Find the best agent for a task"""
    return get_registry().find_best_agent(required_capabilities, input_types, max_complexity, min_speed)


