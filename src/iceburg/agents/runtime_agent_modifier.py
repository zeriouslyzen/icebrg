"""
Runtime Agent Modifier for ICEBURG
Allows agents to modify their own behavior and structure during execution
"""

import json
import time
import copy
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class AdaptationRecord:
    """Record of agent adaptations"""
    timestamp: float
    adaptation_type: str  # 'prompt', 'behavior', 'capability', 'structure'
    old_state: Dict[str, Any]
    new_state: Dict[str, Any]
    reason: str
    performance_impact: Optional[float] = None

class RuntimeAgentModifier:
    """Enables runtime modification of agent behavior and capabilities"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.adaptation_history: Dict[str, List[AdaptationRecord]] = {}
        self.performance_tracker = {}

    def analyze_performance_for_adaptation(self, agent_name: str, recent_performance: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze agent performance to determine needed adaptations"""

        if not recent_performance:
            return None

        # Calculate performance metrics
        avg_confidence = sum(p.get('confidence', 0) for p in recent_performance) / len(recent_performance)
        avg_response_time = sum(p.get('response_time', 0) for p in recent_performance) / len(recent_performance)

        # Check for adaptation triggers
        adaptations = []

        # Low confidence adaptation
        if avg_confidence < 0.7:
            adaptations.append({
                'type': 'prompt_enhancement',
                'reason': f'Low confidence ({avg_confidence:.2f})',
                'priority': 'high'
            })

        # Slow response adaptation
        if avg_response_time > 30:  # seconds
            adaptations.append({
                'type': 'efficiency_optimization',
                'reason': f'Slow response time ({avg_response_time:.1f}s)',
                'priority': 'medium'
            })

        # Specialized domain adaptation
        domains = [p.get('domain', 'general') for p in recent_performance]
        if len(set(domains)) > 3:
            adaptations.append({
                'type': 'domain_specialization',
                'reason': f'Multiple domains detected: {set(domains)}',
                'priority': 'low'
            })

        return adaptations if adaptations else None

    def modify_agent_prompt(self, agent_name: str, current_prompt: str, adaptations: List[Dict[str, Any]]) -> str:
        """Modify agent prompt based on adaptations"""

        modified_prompt = current_prompt

        for adaptation in adaptations:
            if adaptation['type'] == 'prompt_enhancement':
                # Add confidence boosting instructions
                enhancement = """
                ENHANCED CONFIDENCE PROTOCOL: When providing analysis, explicitly state your confidence level
                and the reasoning behind it. If confidence is below 0.8, provide alternative interpretations
                and suggest additional research directions.
                """
                modified_prompt += enhancement

            elif adaptation['type'] == 'domain_specialization':
                # Add domain-specific instructions
                domains = adaptation.get('domains', [])
                if domains:
                    specialization = f"""
                    DOMAIN SPECIALIZATION: You have demonstrated expertise in {', '.join(domains)}.
                    When analyzing queries in these domains, leverage your specialized knowledge and
                    provide more detailed, domain-specific insights.
                    """
                    modified_prompt += specialization

            elif adaptation['type'] == 'efficiency_optimization':
                # Add efficiency instructions
                optimization = """
                EFFICIENCY PROTOCOL: Focus on providing concise, high-impact analysis.
                Prioritize the most important insights and conclusions over exhaustive detail.
                """
                modified_prompt += optimization

        return modified_prompt

    def modify_agent_capabilities(self, agent_name: str, current_capabilities: List[str],
                                 adaptations: List[Dict[str, Any]]) -> List[str]:
        """Modify agent capabilities based on adaptations"""

        modified_capabilities = current_capabilities.copy()

        for adaptation in adaptations:
            if adaptation['type'] == 'capability_expansion':
                new_capabilities = adaptation.get('new_capabilities', [])
                modified_capabilities.extend(new_capabilities)

            elif adaptation['type'] == 'capability_specialization':
                # Focus on specific capabilities
                specializations = adaptation.get('specializations', [])
                if specializations:
                    modified_capabilities = [cap for cap in modified_capabilities
                                           if any(spec in cap.lower() for spec in specializations)]

        return modified_capabilities

    def record_adaptation(self, agent_name: str, adaptation_type: str,
                         old_state: Dict[str, Any], new_state: Dict[str, Any], reason: str):
        """Record an adaptation for tracking and rollback"""

        record = AdaptationRecord(
            timestamp=time.time(),
            adaptation_type=adaptation_type,
            old_state=old_state,
            new_state=new_state,
            reason=reason
        )

        if agent_name not in self.adaptation_history:
            self.adaptation_history[agent_name] = []

        self.adaptation_history[agent_name].append(record)

        # Keep only last 50 adaptations per agent
        if len(self.adaptation_history[agent_name]) > 50:
            self.adaptation_history[agent_name] = self.adaptation_history[agent_name][-50:]

    def rollback_adaptation(self, agent_name: str, steps: int = 1) -> bool:
        """Rollback recent adaptations"""

        if agent_name not in self.adaptation_history or not self.adaptation_history[agent_name]:
            return False

        # Get adaptations to rollback
        adaptations = self.adaptation_history[agent_name][-steps:]
        rollback_successful = True

        for adaptation in adaptations:
            try:
                # This would need to be implemented based on specific agent architecture
                logger.info(f"Rolling back {adaptation.adaptation_type} adaptation for {agent_name}")
            except Exception as e:
                logger.error(f"Failed to rollback adaptation: {e}")
                rollback_successful = False

        # Remove rolled back adaptations from history
        self.adaptation_history[agent_name] = self.adaptation_history[agent_name][:-steps]

        return rollback_successful

    def get_adaptation_history(self, agent_name: str) -> List[AdaptationRecord]:
        """Get adaptation history for an agent"""
        return self.adaptation_history.get(agent_name, [])

    def analyze_adaptation_effectiveness(self, agent_name: str) -> Dict[str, Any]:
        """Analyze how adaptations have affected agent performance"""

        if agent_name not in self.adaptation_history:
            return {'adaptations': 0, 'effectiveness': 'unknown'}

        adaptations = self.adaptation_history[agent_name]

        # Simple analysis - in practice would be more sophisticated
        adaptation_types = {}
        for adaptation in adaptations:
            adaptation_type = adaptation.adaptation_type
            adaptation_types[adaptation_type] = adaptation_types.get(adaptation_type, 0) + 1

        return {
            'total_adaptations': len(adaptations),
            'adaptation_types': adaptation_types,
            'last_adaptation': adaptations[-1].timestamp if adaptations else None
        }

    def create_adaptive_agent_wrapper(self, base_agent_class):
        """Create a wrapper that adds adaptive capabilities to any agent"""

        class AdaptiveAgentWrapper:
            def __init__(self, cfg, base_agent):
                self.cfg = cfg
                self.base_agent = base_agent
                self.modifier = RuntimeAgentModifier(cfg)
                self.current_prompt = getattr(base_agent, 'system_prompt', '')
                self.current_capabilities = getattr(base_agent, 'capabilities', [])
                self.adaptation_enabled = True

            def run(self, query: str, context: Dict[str, Any] = None):
                """Run agent with adaptive capabilities"""

                # Check if adaptation is needed
                if self.adaptation_enabled:
                    performance_data = self._get_recent_performance()
                    adaptations = self.modifier.analyze_performance_for_adaptation(
                        self.base_agent.__class__.__name__, performance_data
                    )

                    if adaptations:
                        self._apply_adaptations(adaptations)

                # Run base agent
                return self.base_agent.run(query, context)

            def _get_recent_performance(self) -> List[Dict[str, Any]]:
                """Get recent performance data (simplified)"""
                # In practice, this would pull from actual performance tracking
                return []

            def _apply_adaptations(self, adaptations: List[Dict[str, Any]]):
                """Apply adaptations to agent"""
                old_prompt = self.current_prompt
                old_capabilities = self.current_capabilities.copy()

                # Apply prompt modifications
                if any(a['type'] == 'prompt_enhancement' for a in adaptations):
                    self.current_prompt = self.modifier.modify_agent_prompt(
                        self.base_agent.__class__.__name__, self.current_prompt, adaptations
                    )

                # Apply capability modifications
                if any(a['type'] in ['capability_expansion', 'capability_specialization'] for a in adaptations):
                    self.current_capabilities = self.modifier.modify_agent_capabilities(
                        self.base_agent.__class__.__name__, self.current_capabilities, adaptations
                    )

                # Record the adaptation
                self.modifier.record_adaptation(
                    self.base_agent.__class__.__name__,
                    'adaptive_modification',
                    {'prompt': old_prompt, 'capabilities': old_capabilities},
                    {'prompt': self.current_prompt, 'capabilities': self.current_capabilities},
                    f'Applied {len(adaptations)} adaptations'
                )

            def get_adaptation_status(self) -> Dict[str, Any]:
                """Get current adaptation status"""
                return {
                    'adaptations_enabled': self.adaptation_enabled,
                    'adaptation_history': self.modifier.get_adaptation_history(self.base_agent.__class__.__name__),
                    'current_capabilities': self.current_capabilities,
                    'adaptation_effectiveness': self.modifier.analyze_adaptation_effectiveness(self.base_agent.__class__.__name__)
                }

        return AdaptiveAgentWrapper

    def enable_adaptation_for_agent(self, agent_instance):
        """Enable runtime adaptation for a specific agent"""
        if hasattr(agent_instance, 'adaptation_enabled'):
            agent_instance.adaptation_enabled = True
            logger.info(f"Enabled runtime adaptation for {agent_instance.__class__.__name__}")
        else:
            # Wrap agent with adaptive capabilities
            wrapped_agent = self.create_adaptive_agent_wrapper(self.cfg, agent_instance)
            logger.info(f"Wrapped {agent_instance.__class__.__name__} with adaptive capabilities")

    def disable_adaptation_for_agent(self, agent_instance):
        """Disable runtime adaptation for a specific agent"""
        if hasattr(agent_instance, 'adaptation_enabled'):
            agent_instance.adaptation_enabled = False
            logger.info(f"Disabled runtime adaptation for {agent_instance.__class__.__name__}")
