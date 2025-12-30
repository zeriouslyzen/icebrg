"""
Visual Reasoner Agent for ICEBURG

Integrates V-JEPA visual understanding with ICEBURG's reasoning pipeline.
Handles:
- Visual spatial reasoning for ARC-AGI grids
- Grid pattern transformation detection
- Visual similarity analysis

Part of the ARC-AGI Enhancement Project - V-JEPA Integration
"""

import re
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..config import IceburgConfig
from ..llm import chat_complete

logger = logging.getLogger(__name__)

# Try to import V-JEPA (optional dependency)
try:
    from ..vjepa import VJEPAEncoder, VJEPAConfig, get_encoder, ARC_COLORS
    VJEPA_AVAILABLE = True
except ImportError:
    VJEPA_AVAILABLE = False
    logger.warning("[VISUAL_REASONER] V-JEPA not available - install torch and transformers")


@dataclass
class VisualAnalysis:
    """Result of visual reasoning."""
    query_type: str
    answer: str
    visual_features: Optional[Dict[str, Any]]
    confidence: float
    explanation: str
    used_vjepa: bool


VISUAL_REASONER_SYSTEM = (
    "ROLE: Visual Reasoning Specialist - Expert in spatial pattern analysis.\n"
    "TASK: Analyze visual grids, detect patterns, and reason about spatial transformations.\n"
    "\n"
    "CAPABILITIES:\n"
    "- Analyze ARC-AGI style grids (2D arrays of colors)\n"
    "- Detect spatial patterns and transformations\n"
    "- Compare visual similarity between grids\n"
    "- Predict missing or transformed elements\n"
    "\n"
    "OUTPUT FORMAT:\n"
    "1. PATTERN TYPE: [rotation/reflection/translation/color_change/scaling/complex]\n"
    "2. DESCRIPTION: [describe the visual pattern or transformation]\n"
    "3. PREDICTION: [if applicable, predict output]\n"
    "4. CONFIDENCE: [0.0-1.0]\n"
)


def parse_grid_from_text(text: str) -> Optional[List[List[int]]]:
    """
    Parse a grid from text representation.
    
    Handles formats:
    - [[0,1,2],[3,4,5]]
    - 0 1 2\\n3 4 5
    - Grid with colors as numbers
    """
    # Try JSON-like format
    import json
    try:
        grid = json.loads(text)
        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
            return grid
    except (json.JSONDecodeError, TypeError):
        pass
    
    # Try space/newline separated format
    lines = text.strip().split('\n')
    grid = []
    for line in lines:
        line = line.strip()
        if line:
            # Split by spaces, commas, or just digits
            nums = re.findall(r'\d+', line)
            if nums:
                grid.append([int(n) for n in nums])
    
    if grid and all(len(row) == len(grid[0]) for row in grid):
        return grid
    
    return None


def analyze_grid_properties(grid: List[List[int]]) -> Dict[str, Any]:
    """Analyze basic properties of a grid."""
    if not grid or not grid[0]:
        return {"valid": False}
    
    height = len(grid)
    width = len(grid[0])
    
    # Count colors/values
    values = set()
    value_counts = {}
    for row in grid:
        for val in row:
            values.add(val)
            value_counts[val] = value_counts.get(val, 0) + 1
    
    # Check for symmetry
    h_symmetric = all(
        grid[i] == grid[height - 1 - i] 
        for i in range(height // 2)
    )
    v_symmetric = all(
        all(grid[i][j] == grid[i][width - 1 - j] for j in range(width // 2))
        for i in range(height)
    )
    
    return {
        "valid": True,
        "height": height,
        "width": width,
        "unique_values": len(values),
        "values": sorted(values),
        "value_counts": value_counts,
        "is_square": height == width,
        "horizontal_symmetric": h_symmetric,
        "vertical_symmetric": v_symmetric,
        "total_cells": height * width,
    }


def detect_transformation(
    input_grid: List[List[int]],
    output_grid: List[List[int]],
    use_vjepa: bool = True
) -> Dict[str, Any]:
    """
    Detect the transformation between input and output grids.
    
    Uses V-JEPA for visual embedding comparison if available.
    """
    result = {
        "transformation_type": "unknown",
        "confidence": 0.0,
        "details": {},
    }
    
    input_props = analyze_grid_properties(input_grid)
    output_props = analyze_grid_properties(output_grid)
    
    if not input_props["valid"] or not output_props["valid"]:
        return result
    
    # Check for identity (no change)
    if input_grid == output_grid:
        return {
            "transformation_type": "identity",
            "confidence": 1.0,
            "details": {"unchanged": True}
        }
    
    # Check for rotation
    rotations = _check_rotations(input_grid, output_grid)
    if rotations["match"]:
        return {
            "transformation_type": "rotation",
            "confidence": 0.95,
            "details": rotations
        }
    
    # Check for reflection
    reflections = _check_reflections(input_grid, output_grid)
    if reflections["match"]:
        return {
            "transformation_type": "reflection",
            "confidence": 0.95,
            "details": reflections
        }
    
    # Check for color substitution
    color_sub = _check_color_substitution(input_grid, output_grid)
    if color_sub["match"]:
        return {
            "transformation_type": "color_substitution",
            "confidence": 0.9,
            "details": color_sub
        }
    
    # Use V-JEPA for complex patterns
    if use_vjepa and VJEPA_AVAILABLE:
        try:
            encoder = get_encoder()
            if encoder.is_available():
                similarity = encoder.similarity(input_grid, output_grid)
                result["vjepa_similarity"] = similarity
                result["confidence"] = max(0.5, 1.0 - abs(0.5 - similarity))
        except Exception as e:
            logger.debug(f"[VISUAL_REASONER] V-JEPA analysis failed: {e}")
    
    # Fallback to basic analysis
    if input_props["height"] != output_props["height"] or input_props["width"] != output_props["width"]:
        result["transformation_type"] = "resize"
        result["details"] = {
            "input_size": (input_props["height"], input_props["width"]),
            "output_size": (output_props["height"], output_props["width"]),
        }
        result["confidence"] = 0.8
    
    return result


def _check_rotations(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
    """Check if output is a rotation of input."""
    def rotate_90(grid):
        return [list(row) for row in zip(*grid[::-1])]
    
    rotated = input_grid
    for degrees in [90, 180, 270]:
        rotated = rotate_90(rotated)
        if rotated == output_grid:
            return {"match": True, "degrees": degrees}
    
    return {"match": False}


def _check_reflections(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
    """Check if output is a reflection of input."""
    # Horizontal flip
    h_flip = [row[::-1] for row in input_grid]
    if h_flip == output_grid:
        return {"match": True, "axis": "horizontal"}
    
    # Vertical flip
    v_flip = input_grid[::-1]
    if v_flip == output_grid:
        return {"match": True, "axis": "vertical"}
    
    return {"match": False}


def _check_color_substitution(input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
    """Check if output is a color substitution of input."""
    if len(input_grid) != len(output_grid):
        return {"match": False}
    if any(len(input_grid[i]) != len(output_grid[i]) for i in range(len(input_grid))):
        return {"match": False}
    
    mapping = {}
    for i in range(len(input_grid)):
        for j in range(len(input_grid[i])):
            in_val = input_grid[i][j]
            out_val = output_grid[i][j]
            
            if in_val in mapping:
                if mapping[in_val] != out_val:
                    return {"match": False}
            else:
                mapping[in_val] = out_val
    
    # Check if it's not identity
    if all(k == v for k, v in mapping.items()):
        return {"match": False}
    
    return {"match": True, "mapping": mapping}


def run(
    cfg: IceburgConfig,
    query: str,
    input_grid: Optional[List[List[int]]] = None,
    output_grid: Optional[List[List[int]]] = None,
    use_vjepa: bool = True,
    verbose: bool = False
) -> str:
    """
    Main entry point for Visual Reasoner.
    
    Args:
        cfg: ICEBURG configuration
        query: The visual reasoning question
        input_grid: Optional input grid (2D list of ints)
        output_grid: Optional output grid for transformation detection
        use_vjepa: Whether to use V-JEPA encoder
        verbose: Print debug info
        
    Returns:
        Analysis result with visual reasoning
    """
    result_parts = []
    used_vjepa = False
    
    # Try to parse grids from query if not provided
    if input_grid is None:
        input_grid = parse_grid_from_text(query)
    
    # Analyze input grid
    if input_grid is not None:
        props = analyze_grid_properties(input_grid)
        
        if props["valid"]:
            result_parts.append(f"**Grid Size:** {props['height']}×{props['width']}")
            result_parts.append(f"**Unique Values:** {props['unique_values']} ({props['values']})")
            
            symmetry = []
            if props["horizontal_symmetric"]:
                symmetry.append("horizontal")
            if props["vertical_symmetric"]:
                symmetry.append("vertical")
            if symmetry:
                result_parts.append(f"**Symmetry:** {', '.join(symmetry)}")
            
            # Detect transformation if output provided
            if output_grid is not None:
                transform = detect_transformation(input_grid, output_grid, use_vjepa=use_vjepa)
                
                result_parts.append(f"\n**Transformation:** {transform['transformation_type']}")
                result_parts.append(f"**Confidence:** {transform['confidence']:.0%}")
                
                if transform.get("details"):
                    for key, val in transform["details"].items():
                        if key != "match":
                            result_parts.append(f"**{key.replace('_', ' ').title()}:** {val}")
                
                if "vjepa_similarity" in transform:
                    result_parts.append(f"**Visual Similarity (V-JEPA):** {transform['vjepa_similarity']:.2%}")
                    used_vjepa = True
                
                if transform["confidence"] >= 0.8:
                    return "\n".join(result_parts)
    
    # Use V-JEPA for deep visual analysis if available
    if use_vjepa and VJEPA_AVAILABLE and input_grid is not None:
        try:
            encoder = get_encoder()
            if encoder.initialize():
                embedding = encoder.encode_grid(input_grid)
                if embedding is not None:
                    result_parts.append(f"\n**V-JEPA Embedding Shape:** {embedding.shape}")
                    used_vjepa = True
        except Exception as e:
            if verbose:
                print(f"[VISUAL_REASONER] V-JEPA embedding failed: {e}")
    
    # Fall back to LLM for complex patterns
    if verbose:
        print("[VISUAL_REASONER] Using LLM for complex visual analysis")
    
    prompt = f"VISUAL REASONING TASK:\n{query}\n\n"
    
    if input_grid is not None:
        prompt += f"INPUT GRID:\n{input_grid}\n\n"
    if output_grid is not None:
        prompt += f"OUTPUT GRID:\n{output_grid}\n\n"
    
    prompt += "Analyze the visual pattern or transformation and explain your reasoning."
    
    try:
        llm_result = chat_complete(
            cfg.synthesist_model,
            prompt,
            system=VISUAL_REASONER_SYSTEM,
            temperature=0.3,
            options={"num_ctx": 2048, "num_predict": 500},
            context_tag="VisualReasoner"
        )
        
        if result_parts:
            return "\n".join(result_parts) + f"\n\n**LLM Analysis:**\n{llm_result}"
        return llm_result
        
    except Exception as e:
        if verbose:
            print(f"[VISUAL_REASONER] LLM Error: {e}")
        if result_parts:
            return "\n".join(result_parts)
        raise


def grid_to_ascii(grid: List[List[int]], symbols: str = ".#@*+%&!?~") -> str:
    """
    Convert a grid to ASCII art for text-based visualization.
    
    Args:
        grid: 2D grid of integers
        symbols: Characters to use for each value (0=first char, etc.)
        
    Returns:
        ASCII art string representation
    """
    lines = []
    for row in grid:
        line = ""
        for val in row:
            if 0 <= val < len(symbols):
                line += symbols[val]
            else:
                line += "?"
        lines.append(line)
    return "\n".join(lines)
