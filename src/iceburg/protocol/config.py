from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Optional, Union

import yaml


_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "config", "protocol.yaml"
)


@dataclass
class ProtocolConfig:
    """Runtime configuration for the modular protocol."""

    # Core mode settings
    fast_mode_enabled: bool = True
    hybrid_mode_enabled: bool = True
    smart_mode_enabled: bool = True
    legacy_mode_enabled: bool = False
    
    # CIM Stack flags
    force_molecular: bool = False
    force_bioelectric: bool = False
    force_hypothesis_testing: bool = False
    
    # AGI Capabilities flags
    force_agi: bool = False
    
    # Blockchain and verification flags
    enable_blockchain_verification: bool = True  # Enabled by default
    
    # Multimodal processing flags
    enable_multimodal_processing: bool = True  # Enabled by default
    enable_visual_generation: bool = True  # Enabled by default
    
    # Other protocol flags
    verbose: bool = False
    evidence_strict: bool = False
    fast: bool = False
    hybrid: bool = False

    feature_flags: Dict[str, bool] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, mapping: Dict[str, Any]) -> "ProtocolConfig":
        flags = mapping.get("feature_flags", {})
        params = mapping.get("parameters", {})
        return cls(
            fast_mode_enabled=mapping.get("fast_mode_enabled", True),
            hybrid_mode_enabled=mapping.get("hybrid_mode_enabled", True),
            smart_mode_enabled=mapping.get("smart_mode_enabled", True),
            legacy_mode_enabled=mapping.get("legacy_mode_enabled", False),
            feature_flags=flags,
            parameters=params,
        )


@lru_cache(1)
def load_config(path: Optional[str] = None) -> ProtocolConfig:
    """Load protocol configuration from YAML file and environment overrides."""

    cfg_path = path or _DEFAULT_CONFIG_PATH
    mapping: Dict[str, Any] = {}

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as fh:
            mapping = yaml.safe_load(fh) or {}

    # Environment overrides
    env_fast = os.getenv("ICEBURG_PROTOCOL_FAST", None)
    env_hybrid = os.getenv("ICEBURG_PROTOCOL_HYBRID", None)
    env_smart = os.getenv("ICEBURG_PROTOCOL_SMART", None)
    env_legacy = os.getenv("ICEBURG_PROTOCOL_LEGACY", None)

    if env_fast is not None:
        mapping["fast_mode_enabled"] = env_fast == "1"
    if env_hybrid is not None:
        mapping["hybrid_mode_enabled"] = env_hybrid == "1"
    if env_smart is not None:
        mapping["smart_mode_enabled"] = env_smart == "1"
    if env_legacy is not None:
        mapping["legacy_mode_enabled"] = env_legacy == "1"

    return ProtocolConfig.from_mapping(mapping)
