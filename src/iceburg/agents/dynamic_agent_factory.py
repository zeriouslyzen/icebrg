"""
Dynamic Agent Factory for ICEBURG
Creates new agent types during execution based on emergence patterns
"""

import os
import json
import uuid
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, Type, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class AgentTemplate:
    """Template for generating new agent types"""
    base_agent_type: str
    specialization: str
    capabilities: list
    prompt_modifications: Dict[str, str]
    reasoning_patterns: list
    domain_focus: str

class DynamicAgentFactory:
    """Factory for creating new agent types during execution"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.agents_dir = Path("src/iceburg/agents")
        self.generated_agents_dir = Path("data/generated_agents")
        self.generated_agents_dir.mkdir(parents=True, exist_ok=True)
        self.agent_registry = self._load_agent_registry()

    def _load_agent_registry(self) -> Dict[str, Dict[str, Any]]:
        """Load registry of generated agents"""
        registry_file = self.generated_agents_dir / "agent_registry.json"
        if registry_file.exists():
            try:
                with open(registry_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load agent registry: {e}")
        return {}

    def _save_agent_registry(self):
        """Save agent registry to disk"""
        registry_file = self.generated_agents_dir / "agent_registry.json"
        try:
            with open(registry_file, 'w') as f:
                json.dump(self.agent_registry, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save agent registry: {e}")

    def analyze_emergence_for_agent_creation(self, emergence_data: Dict[str, Any]) -> Optional[AgentTemplate]:
        """Analyze emergence data to determine if new agent type is needed"""

        # Check if emergence score warrants new agent
        emergence_score = emergence_data.get('emergence_score', 0)
        if emergence_score < 0.8:
            return None

        # Analyze patterns to determine specialization
        patterns = emergence_data.get('patterns', [])
        domains = emergence_data.get('domains', [])

        # Determine agent specialization based on patterns
        if 'cross_domain' in patterns:
            specialization = 'cross_domain_synthesizer'
        elif 'assumption_challenge' in patterns:
            specialization = 'assumption_challenger'
        elif 'novel_hypothesis' in patterns:
            specialization = 'hypothesis_generator'
        elif 'framework_departure' in patterns:
            specialization = 'paradigm_shifter'
        else:
            specialization = 'pattern_analyzer'

        # Create agent template
        template = AgentTemplate(
            base_agent_type='deliberation',
            specialization=specialization,
            capabilities=self._extract_capabilities(emergence_data),
            prompt_modifications=self._generate_prompt_modifications(emergence_data),
            reasoning_patterns=self._extract_reasoning_patterns(emergence_data),
            domain_focus=domains[0] if domains else 'general'
        )

        return template

    def _extract_capabilities(self, emergence_data: Dict[str, Any]) -> list:
        """Extract capabilities from emergence data"""
        capabilities = []

        # Analyze what the agent should be able to do
        if 'cross_domain' in emergence_data.get('patterns', []):
            capabilities.extend(['cross_domain_analysis', 'domain_bridging'])

        if 'novel_prediction' in emergence_data.get('patterns', []):
            capabilities.extend(['prediction_generation', 'hypothesis_testing'])

        if 'assumption_challenge' in emergence_data.get('patterns', []):
            capabilities.extend(['assumption_identification', 'alternative_framework_generation'])

        return capabilities

    def _generate_prompt_modifications(self, emergence_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate prompt modifications for the new agent"""
        modifications = {}

        # Modify system prompt based on specialization
        if 'cross_domain' in emergence_data.get('patterns', []):
            modifications['system_prompt'] = """
            You are a cross-domain synthesis specialist. Your expertise lies in identifying connections
            between seemingly unrelated fields and creating novel frameworks that bridge domain boundaries.
            Focus on finding analogous patterns and principles across different areas of knowledge.
            """

        if 'novel_prediction' in emergence_data.get('patterns', []):
            modifications['prediction_focus'] = """
            Prioritize generating testable, falsifiable predictions that can advance scientific understanding.
            Consider both near-term experimental validation and long-term theoretical implications.
            """

        return modifications

    def _extract_reasoning_patterns(self, emergence_data: Dict[str, Any]) -> list:
        """Extract reasoning patterns for the new agent"""
        patterns = []

        # Add reasoning patterns based on emergence characteristics
        if emergence_data.get('emergence_type') == 'cross_domain_synthesis':
            patterns.append('analogical_reasoning')
            patterns.append('pattern_transfer')

        if emergence_data.get('evidence_gaps'):
            patterns.append('gap_analysis')
            patterns.append('hypothesis_formulation')

        return patterns

    def create_agent_from_template(self, template: AgentTemplate, emergence_data: Dict[str, Any]) -> str:
        """Generate Python code for new agent based on template"""

        agent_name = f"{template.specialization}_{uuid.uuid4().hex[:8]}"
        agent_file = self.generated_agents_dir / f"{agent_name}.py"

        # Generate agent code
        agent_code = self._generate_agent_code(agent_name, template, emergence_data)

        # Write agent file
        with open(agent_file, 'w') as f:
            f.write(agent_code)

        # Update registry
        self.agent_registry[agent_name] = {
            'file_path': str(agent_file),
            'template': template.__dict__,
            'emergence_data': emergence_data,
            'created_at': time.time()
        }
        self._save_agent_registry()

        logger.info(f"Created new agent: {agent_name} with specialization: {template.specialization}")
        return agent_name

    def _generate_agent_code(self, agent_name: str, template: AgentTemplate, emergence_data: Dict[str, Any]) -> str:
        """Generate Python code for the new agent using LLM (like Architect agent)"""
        
        try:
            # Use ICEBURG's LLM directly to generate the actual code
            from ..llm import chat_complete
            
            # Create class name from agent name
            class_name = ''.join(word.capitalize() for word in agent_name.split('_'))
            
            # Create a code generation prompt
            code_prompt = f"""Generate a complete, production-ready Python agent class based on this specification:

Agent Name: {agent_name}
Class Name: {class_name}
Specialization: {template.specialization}
Capabilities: {', '.join(template.capabilities)}
Domain Focus: {template.domain_focus}
Reasoning Patterns: {', '.join(template.reasoning_patterns)}

Prompt Modifications:
{template.prompt_modifications.get('system_prompt', '')}

Emergence Data:
{str(emergence_data)[:500]}

Generate the COMPLETE Python code implementation including:
1. Imports (from __future__ import annotations, typing, IceburgConfig, logging)
2. Class definition with __init__ method
3. run() method that integrates with ICEBURG's LLM system using chat_complete()
4. get_capabilities() and get_specialization() methods
5. Proper error handling and logging
6. Integration with ICEBURG's config and LLM systems

The agent should use chat_complete() from ..llm import chat_complete to generate responses.
Use self.cfg for configuration and logger for logging.

Output ONLY valid Python code, no markdown or explanations. Do not include code blocks (```) markers."""
            
            # Use LLM directly to generate the code
            system_prompt = (
                "You are an expert Python developer specializing in AI agent development. "
                "Generate complete, production-ready agent classes that integrate with ICEBURG's LLM system."
            )
            
            generated_code = chat_complete(
                model=self.cfg.synthesist_model if hasattr(self.cfg, 'synthesist_model') else "llama3.1:8b",
                prompt=code_prompt,
                system=system_prompt,
                temperature=0.3,
                context_tag="dynamic_agent_code_gen"
            )
            
            if generated_code and "class" in generated_code:
                # Extract code if wrapped in markdown
                if "```python" in generated_code:
                    code_start = generated_code.find("```python") + 9
                    code_end = generated_code.find("```", code_start)
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                elif "```" in generated_code:
                    code_start = generated_code.find("```") + 3
                    code_end = generated_code.find("```", code_start)
                    if code_end > code_start:
                        generated_code = generated_code[code_start:code_end].strip()
                
                # Validate generated code
                from .code_validator import CodeValidator
                validator = CodeValidator()
                validation_result = validator.get_validation_details(generated_code)
                
                if validation_result.get("valid", False):
                    logger.info(f"Generated {len(generated_code)} characters of Python code for agent {agent_name}")
                    
                    # Log agent generation for fine-tuning
                    try:
                        from ..data_collection.fine_tuning_logger import FineTuningLogger
                        fine_tuning_logger = FineTuningLogger()
                        
                        metadata = {
                            "agent_name": agent_name,
                            "template": {
                                "specialization": template.specialization,
                                "capabilities": template.capabilities,
                                "domain_focus": template.domain_focus,
                                "reasoning_patterns": template.reasoning_patterns
                            },
                            "emergence_data": emergence_data,
                            "generation_method": "llm"
                        }
                        
                        fine_tuning_logger.log_agent_generation(
                            agent_name=agent_name,
                            generated_code=generated_code,
                            validation_result=validation_result,
                            metadata=metadata
                        )
                    except Exception as e:
                        logger.debug(f"Failed to log agent generation for fine-tuning: {e}")
                    
                    return generated_code
                else:
                    logger.warning(f"Generated code failed validation, using template fallback")
            else:
                logger.warning("LLM generation failed, using template")
        except Exception as e:
            logger.error(f"LLM code generation failed: {e}, using template fallback")
        
        # Fallback to template-based generation
        return self._generate_agent_code_template(agent_name, template, emergence_data)
    
    def _generate_agent_code_template(self, agent_name: str, template: AgentTemplate, emergence_data: Dict[str, Any]) -> str:
        """Generate Python code using template (fallback)"""
        
        # Create class name from agent name
        class_name = ''.join(word.capitalize() for word in agent_name.split('_'))

        # Generate imports
        imports = """
from __future__ import annotations
from typing import Dict, Any, List, Optional
from ..config import IceburgConfig
from ..llm import chat_complete
import logging

logger = logging.getLogger(__name__)
"""

        # Generate class definition
        class_def = f"""

class {class_name}:
    \"\"\"Dynamically generated agent with specialization: {template.specialization}\"\"\"

    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.specialization = "{template.specialization}"
        self.capabilities = {template.capabilities}
        self.domain_focus = "{template.domain_focus}"
        self.reasoning_patterns = {template.reasoning_patterns}
        self.emergence_data = {emergence_data}

    def run(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        \"\"\"Execute agent analysis using ICEBURG's LLM system\"\"\"

        # Enhanced prompt based on specialization
        system_prompt = f'''
        You are a specialized agent with expertise in {template.specialization}.
        Your capabilities include: {', '.join(template.capabilities)}
        You focus on domain: {template.domain_focus}
        Your reasoning patterns: {', '.join(template.reasoning_patterns)}

        {template.prompt_modifications.get('system_prompt', '')}

        Provide analysis for the following query with your specialized perspective.
        '''

        # Use ICEBURG's LLM system
        try:
            analysis = chat_complete(
                model=self.cfg.synthesist_model if hasattr(self.cfg, 'synthesist_model') else "llama3.1:8b",
                prompt=query,
                system=system_prompt,
                temperature=0.7,
                context_tag="{agent_name}"
            )
            
            return {{
                'agent_name': '{agent_name}',
                'specialization': '{template.specialization}',
                'analysis': analysis,
                'capabilities_used': self.capabilities,
                'confidence': 0.8
            }}
        except Exception as e:
            logger.error(f"Agent {agent_name} error: {{e}}")
            return {{
                'agent_name': '{agent_name}',
                'specialization': '{template.specialization}',
                'analysis': f'Error: {{str(e)}}',
                'capabilities_used': self.capabilities,
                'confidence': 0.0
            }}

    def get_capabilities(self) -> List[str]:
        \"\"\"Return agent capabilities\"\"\"
        return self.capabilities

    def get_specialization(self) -> str:
        \"\"\"Return agent specialization\"\"\"
        return self.specialization
"""

        return imports + class_def

    def load_generated_agent(self, agent_name: str) -> Optional[object]:
        """Load and instantiate a generated agent"""
        if agent_name not in self.agent_registry:
            return None

        agent_info = self.agent_registry[agent_name]
        agent_file = Path(agent_info['file_path'])

        if not agent_file.exists():
            logger.error(f"Agent file not found: {agent_file}")
            return None

        try:
            # Import the generated agent module
            import importlib.util
            spec = importlib.util.spec_from_file_location(agent_name, agent_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the agent class
            class_name = ''.join(word.capitalize() for word in agent_name.split('_'))
            agent_class = getattr(module, class_name)

            # Instantiate the agent
            return agent_class(self.cfg)

        except Exception as e:
            logger.error(f"Failed to load generated agent {agent_name}: {e}")
            return None

    def get_generated_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all generated agents"""
        return self.agent_registry.copy()

    def cleanup_unused_agents(self, keep_recent: int = 5):
        """Clean up old generated agents"""
        if not self.agent_registry:
            return

        # Sort by creation time
        sorted_agents = sorted(
            self.agent_registry.items(),
            key=lambda x: x[1]['created_at'],
            reverse=True
        )

        # Remove old agents
        to_remove = sorted_agents[keep_recent:]
        for agent_name, agent_info in to_remove:
            # Remove file
            agent_file = Path(agent_info['file_path'])
            if agent_file.exists():
                agent_file.unlink()

            # Remove from registry
            del self.agent_registry[agent_name]

        self._save_agent_registry()
        logger.info(f"Cleaned up generated agents, keeping {keep_recent} recent ones")
