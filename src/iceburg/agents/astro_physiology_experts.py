"""
Astro-Physiology Expert Agents
Specialized agents for post-analysis consultation
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from ..config import IceburgConfig
from ..llm import chat_complete

logger = logging.getLogger(__name__)


# V2: New specialized expert agents

async def sleep_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Sleep Expert Agent
    Analyzes circadian rhythms, melatonin production, and sleep quality indicators
    """
    expert_name = "sleep"
    try:
        logger.info(f"🌌 Sleep expert: Starting consultation")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        current_conditions = algorithmic_data.get("current_conditions", {})
        
        prompt = f"""You are a sleep expert analyzing celestial-biological data for sleep optimization.

Molecular Imprint:
- Voltage gates: {molecular_imprint.get('voltage_gates', {})}
- Biophysical parameters: {behavioral_predictions}

Current Celestial Conditions:
- Schumann resonance: {current_conditions.get('electromagnetic_environment', {}).get('schumann_resonance_fundamental', 7.83)} Hz

Provide:
1. Optimal sleep schedule recommendations (bedtime, wake time)
2. Circadian rhythm insights
3. Melatonin production timing
4. Sleep hygiene recommendations
5. Environmental factors affecting sleep

Keep it scientific and actionable."""
        
        from ..llm import chat_complete
        model = getattr(cfg, 'primary_model', None) or getattr(cfg, 'surveyor_model', None) or 'llama3.1:8b'
        
        response = await asyncio.to_thread(
            chat_complete,
            model,
            prompt,
            system="You are a sleep expert specializing in circadian biology and celestial influences on sleep.",
            temperature=temperature,
            options={"num_ctx": 2048, "num_predict": 500},
            context_tag="SleepExpert"
        )
        
        return {
            "expert": expert_name,
            "recommendations": response.strip() if response else "Sleep analysis unavailable.",
            "optimal_sleep_times": {},
            "sleep_hygiene": []
        }
        
    except Exception as e:
        logger.error(f"Error in {expert_name} expert: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "recommendations": f"Sleep expert consultation unavailable: {str(e)}"
        }


async def stress_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Stress Expert Agent
    Analyzes stress response patterns, cortisol rhythms, and nervous system state
    """
    expert_name = "stress"
    try:
        logger.info(f"🌌 Stress expert: Starting consultation")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        
        prompt = f"""You are a stress management expert analyzing celestial-biological data.

Biophysical Parameters:
- Neural excitability: {behavioral_predictions.get('neural_excitability', 0):.2f}
- Inhibitory control: {behavioral_predictions.get('inhibitory_control', 0):.2f}
- Cellular stability: {behavioral_predictions.get('cellular_stability', 0):.2f}

TCM Health Indicators:
- Organ correlations: {tcm_predictions}

Provide:
1. Stress response patterns
2. Optimal relaxation times
3. Stress management techniques
4. Coping strategies
5. Nervous system support recommendations

Keep it scientific and actionable."""
        
        from ..llm import chat_complete
        model = getattr(cfg, 'primary_model', None) or getattr(cfg, 'surveyor_model', None) or 'llama3.1:8b'
        
        response = await asyncio.to_thread(
            chat_complete,
            model,
            prompt,
            system="You are a stress management expert specializing in nervous system regulation and stress response optimization.",
            temperature=temperature,
            options={"num_ctx": 2048, "num_predict": 500},
            context_tag="StressExpert"
        )
        
        return {
            "expert": expert_name,
            "recommendations": response.strip() if response else "Stress analysis unavailable.",
            "optimal_relaxation_times": {},
            "stress_management_techniques": []
        }
        
    except Exception as e:
        logger.error(f"Error in {expert_name} expert: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "recommendations": f"Stress expert consultation unavailable: {str(e)}"
        }


async def hormone_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Hormone Expert Agent
    Analyzes hormonal patterns, endocrine system state, and hormone production cycles
    """
    expert_name = "hormone"
    try:
        logger.info(f"🌌 Hormone expert: Starting consultation")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        tcm_time_analysis = algorithmic_data.get("tcm_time_analysis", {})
        
        prompt = f"""You are a hormone expert analyzing celestial-biological data for endocrine optimization.

Biophysical Parameters:
- Neurotransmitter release: {behavioral_predictions.get('neurotransmitter_release', 0):.2f}
- Cellular stability: {behavioral_predictions.get('cellular_stability', 0):.2f}

TCM Organ Correlations:
- Organ systems: {tcm_predictions}
- Time-of-day analysis: {tcm_time_analysis.get('optimal_times', {})}

Provide:
1. Hormonal pattern analysis
2. Endocrine system state
3. Hormone production cycles
4. Optimization strategies
5. Timing recommendations
6. Support suggestions

Keep it scientific and actionable."""
        
        from ..llm import chat_complete
        model = getattr(cfg, 'primary_model', None) or getattr(cfg, 'surveyor_model', None) or 'llama3.1:8b'
        
        response = await asyncio.to_thread(
            chat_complete,
            model,
            prompt,
            system="You are a hormone expert specializing in endocrine system function and hormonal optimization.",
            temperature=temperature,
            options={"num_ctx": 2048, "num_predict": 500},
            context_tag="HormoneExpert"
        )
        
        return {
            "expert": expert_name,
            "recommendations": response.strip() if response else "Hormone analysis unavailable.",
            "hormonal_patterns": {},
            "optimization_strategies": []
        }
        
    except Exception as e:
        logger.error(f"Error in {expert_name} expert: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "recommendations": f"Hormone expert consultation unavailable: {str(e)}"
        }


async def digestive_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Digestive Expert Agent
    Analyzes digestive system state, gut health indicators, and optimal digestion times
    """
    expert_name = "digestive"
    try:
        logger.info(f"🌌 Digestive expert: Starting consultation")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        tcm_time_analysis = algorithmic_data.get("tcm_time_analysis", {})
        
        prompt = f"""You are a digestive health expert analyzing celestial-biological data.

TCM Organ Correlations:
- Organ systems: {tcm_predictions}
- Time-of-day analysis: {tcm_time_analysis.get('optimal_times', {})}

Molecular Imprint:
- Cellular dependencies: {molecular_imprint.get('cellular_dependencies', {})}

Provide:
1. Digestive system state analysis
2. Gut health indicators
3. Optimal digestion times
4. Dietary timing recommendations
5. Digestive support strategies
6. Gut health recommendations

Keep it scientific and actionable."""
        
        from ..llm import chat_complete
        model = getattr(cfg, 'primary_model', None) or getattr(cfg, 'surveyor_model', None) or 'llama3.1:8b'
        
        response = await asyncio.to_thread(
            chat_complete,
            model,
            prompt,
            system="You are a digestive health expert specializing in gut health and optimal digestion timing.",
            temperature=temperature,
            options={"num_ctx": 2048, "num_predict": 500},
            context_tag="DigestiveExpert"
        )
        
        return {
            "expert": expert_name,
            "recommendations": response.strip() if response else "Digestive analysis unavailable.",
            "optimal_digestion_times": {},
            "dietary_timing": []
        }
        
    except Exception as e:
        logger.error(f"Error in {expert_name} expert: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "recommendations": f"Digestive expert consultation unavailable: {str(e)}"
        }


async def health_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Health Expert Agent
    Provides health insights based on molecular imprint and TCM predictions
    """
    expert_name = "health"
    try:
        logger.info(f"🌌 Health expert: Starting consultation with {len(algorithmic_data)} data keys")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        tcm_time_analysis = algorithmic_data.get("tcm_time_analysis", {})
        behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
        
        logger.info(f"🌌 Health expert: molecular_imprint keys={list(molecular_imprint.keys()) if molecular_imprint else []}, tcm_predictions={bool(tcm_predictions)}, behavioral_predictions={bool(behavioral_predictions)}")
        
        prompt = f"""You are a health expert analyzing celestial-biological data.

Molecular Imprint:
- Voltage gates: {molecular_imprint.get('voltage_gates', {})}
- Biophysical parameters: {behavioral_predictions}

TCM Health Indicators:
- Organ correlations: {tcm_predictions}
- Time-of-day analysis: {tcm_time_analysis.get('optimal_times', {})}

Provide:
1. Key health insights (2-3 sentences)
2. Organ systems to monitor
3. Time-of-day recommendations for optimal health
4. Any concerns or strengths

Keep it scientific and actionable."""

        model = cfg.surveyor_model or cfg.primary_model or 'llama3.1:8b'
        # Add timeout to prevent hanging (15 seconds per expert)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat_complete,
                    model,
                    prompt,
                    temperature=temperature,
                    options={"num_predict": 300}
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"🌌 {expert_name} expert consultation timed out")
            return {
                "expert": expert_name,
                "error": "Timeout - LLM response took too long",
                "insights": f"{expert_name.title()} consultation unavailable due to timeout."
            }
        
        result = {
            "expert": expert_name,
            "insights": response,
            "organ_systems": list(tcm_predictions.keys()) if tcm_predictions else [],
            "optimal_times": tcm_time_analysis.get("optimal_times", {})
        }
        
        logger.info(f"🌌 Health expert: Completed, insights length={len(response)}, organ_systems={len(result['organ_systems'])}")
        return result
        
    except Exception as e:
        logger.warning(f"Error in health expert consultation: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "insights": "Health consultation unavailable."
        }


async def nutrition_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Nutrition Expert Agent
    Provides dietary recommendations based on molecular imprint and TCM element correlations
    """
    expert_name = "nutrition"
    try:
        logger.info(f"🌌 Nutrition expert: Starting consultation")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
        
        # Extract TCM elements
        elements = {}
        for planet, data in tcm_predictions.items():
            if isinstance(data, dict):
                element = data.get("element", "")
                if element:
                    elements[element] = elements.get(element, []) + [planet]
        
        prompt = f"""You are a nutrition expert analyzing celestial-biological data.

Molecular Profile:
- Voltage gate sensitivities: {molecular_imprint.get('voltage_gates', {})}
- Biophysical parameters: {behavioral_predictions}

TCM Element Correlations:
- Elements: {elements}

Provide:
1. Dietary recommendations (2-3 sentences)
2. Foods to emphasize based on TCM elements
3. Nutrients that may support voltage gate function
4. Timing recommendations for meals

Keep it scientific and practical."""

        model = cfg.surveyor_model or cfg.primary_model or 'llama3.1:8b'
        # Add timeout to prevent hanging (15 seconds per expert)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat_complete,
                    model,
                    prompt,
                    temperature=temperature,
                    options={"num_predict": 300}
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"🌌 {expert_name} expert consultation timed out")
            return {
                "expert": expert_name,
                "error": "Timeout - LLM response took too long",
                "insights": f"{expert_name.title()} consultation unavailable due to timeout."
            }
        
        result = {
            "expert": expert_name,
            "insights": response,
            "recommended_elements": list(elements.keys()),
            "dietary_focus": "Based on TCM element correlations and molecular profile"
        }
        
        logger.info(f"🌌 Nutrition expert: Completed, insights length={len(response)}, recommended_elements={len(result['recommended_elements'])}")
        return result
        
    except Exception as e:
        logger.warning(f"Error in nutrition expert consultation: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "insights": "Nutrition consultation unavailable."
        }


async def movement_expert_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Movement Expert Agent
    Provides movement and exercise recommendations based on biophysical parameters
    """
    expert_name = "movement"
    try:
        logger.info(f"🌌 Movement expert: Starting consultation")
        behavioral_predictions = algorithmic_data.get("behavioral_predictions", {})
        tcm_predictions = algorithmic_data.get("tcm_predictions", {})
        tcm_time_analysis = algorithmic_data.get("tcm_time_analysis", {})
        
        prompt = f"""You are a movement and exercise expert analyzing celestial-biological data.

Biophysical Parameters:
- Neural excitability: {behavioral_predictions.get('neural_excitability', 0):.2%}
- Inhibitory control: {behavioral_predictions.get('inhibitory_control', 0):.2%}
- Neurotransmitter release: {behavioral_predictions.get('neurotransmitter_release', 0):.2%}
- Cellular stability: {behavioral_predictions.get('cellular_stability', 0):.2%}

TCM Organ Correlations:
- Primary organs: {list(tcm_predictions.keys()) if tcm_predictions else []}
- Optimal activity times: {tcm_time_analysis.get('optimal_times', {})}

Provide:
1. Exercise recommendations (2-3 sentences)
2. Movement types that align with biophysical profile
3. Timing recommendations for physical activity
4. Intensity and duration suggestions

Keep it scientific and practical."""

        model = cfg.surveyor_model or cfg.primary_model or 'llama3.1:8b'
        # Add timeout to prevent hanging (15 seconds per expert)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat_complete,
                    model,
                    prompt,
                    temperature=temperature,
                    options={"num_predict": 300}
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"🌌 {expert_name} expert consultation timed out")
            return {
                "expert": expert_name,
                "error": "Timeout - LLM response took too long",
                "insights": f"{expert_name.title()} consultation unavailable due to timeout."
            }
        
        result = {
            "expert": expert_name,
            "insights": response,
            "recommended_intensity": "Based on neural excitability and cellular stability",
            "optimal_times": tcm_time_analysis.get("optimal_times", {})
        }
        
        logger.info(f"🌌 Movement expert: Completed, insights length={len(response)}, optimal_times={len(result['optimal_times'])}")
        return result
        
    except Exception as e:
        logger.warning(f"Error in movement expert consultation: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "insights": "Movement consultation unavailable."
        }


async def chart_reader_consultation(
    algorithmic_data: Dict[str, Any],
    cfg: IceburgConfig,
    temperature: float = 0.3
) -> Dict[str, Any]:
    """
    Chart Reader Agent
    Provides astrological chart interpretation based on molecular imprint
    """
    expert_name = "chart_reader"
    try:
        logger.info(f"🌌 Chart reader: Starting consultation")
        molecular_imprint = algorithmic_data.get("molecular_imprint", {})
        celestial_positions = molecular_imprint.get("celestial_positions", {})
        synthesized_analysis = algorithmic_data.get("synthesized_analysis", {})
        meridian_connections = synthesized_analysis.get("meridian_connections", {})
        
        logger.info(f"🌌 Chart reader: celestial_positions={bool(celestial_positions)}, synthesized_analysis={bool(synthesized_analysis)}, meridian_connections={len(meridian_connections) if meridian_connections else 0}")
        
        prompt = f"""You are a celestial chart reader analyzing molecular imprint data.

Celestial Positions:
- Available bodies: {list(celestial_positions.keys()) if celestial_positions else []}

Meridian Connections:
- Celestial → Organ mappings: {meridian_connections}

Provide:
1. Key celestial patterns (2-3 sentences)
2. Meridian connection insights
3. Dominant celestial influences
4. How these patterns manifest physiologically

Keep it scientific, focusing on electromagnetic and gravitational influences."""

        model = cfg.surveyor_model or cfg.primary_model or 'llama3.1:8b'
        # Add timeout to prevent hanging (15 seconds per expert)
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    chat_complete,
                    model,
                    prompt,
                    temperature=temperature,
                    options={"num_predict": 300}
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"🌌 {expert_name} expert consultation timed out")
            return {
                "expert": expert_name,
                "error": "Timeout - LLM response took too long",
                "insights": f"{expert_name.title()} consultation unavailable due to timeout."
            }
        
        result = {
            "expert": expert_name,
            "insights": response,
            "meridian_connections": meridian_connections,
            "dominant_influences": list(meridian_connections.keys())[:3] if meridian_connections else []
        }
        
        logger.info(f"🌌 Chart reader: Completed, insights length={len(response)}, meridian_connections={len(meridian_connections)}")
        return result
        
    except Exception as e:
        logger.warning(f"Error in chart reader consultation: {e}", exc_info=True)
        return {
            "expert": expert_name,
            "error": str(e),
            "insights": "Chart reading consultation unavailable."
        }

