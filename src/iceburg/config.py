import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from .model_select import resolve_models, resolve_models_small, resolve_models_hybrid


load_dotenv()


@dataclass(frozen=True)
class IceburgConfig:
    data_dir: Path
    surveyor_model: str
    dissident_model: str
    synthesist_model: str
    oracle_model: str
    embed_model: str
    # Provider abstraction
    llm_provider: str = "ollama"  # ollama | google | anthropic | openai
    provider_url: str = ""        # base URL for HTTP providers
    timeout_s: int = 60
    enable_code_generation: bool = True
    disable_memory: bool = False
    enable_software_lab: bool = True  # Enabled by default
    max_context_length: int = 8192  # Increased context window
    fast: bool = False


def load_config() -> IceburgConfig:
    base_dir = Path(os.getenv("ICEBURG_DATA_DIR", "./data")).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    preferred_surveyor = os.getenv("ICEBURG_SURVEYOR_MODEL", "qwen2.5:7b")
    preferred_dissident = os.getenv("ICEBURG_DISSIDENT_MODEL", "mistral:7b")
    preferred_synthesist = os.getenv("ICEBURG_SYNTHESIST_MODEL", "deepseek-v2:16b")
    preferred_oracle = os.getenv("ICEBURG_ORACLE_MODEL", "deepseek-v2:16b")
    preferred_embed = os.getenv("ICEBURG_EMBED_MODEL", "nomic-embed-text")
    enable_code_gen = os.getenv("ICEBURG_ENABLE_CODE_GENERATION", "0") == "1"

    surveyor_m, dissident_m, synthesist_m, oracle_m, embed_m = resolve_models(
        preferred_surveyor,
        preferred_dissident,
        preferred_synthesist,
        preferred_oracle,
        preferred_embed,
    )

    return IceburgConfig(
        data_dir=base_dir,
        surveyor_model=surveyor_m,
        dissident_model=dissident_m,
        synthesist_model=synthesist_m,
        oracle_model=oracle_m,
        embed_model=embed_m,
        llm_provider=os.getenv("ICEBURG_LLM_PROVIDER", "ollama"),  # Default to Ollama (local)
        provider_url=os.getenv("ICEBURG_PROVIDER_URL", ""),
        timeout_s=int(os.getenv("ICEBURG_PROVIDER_TIMEOUT_S", "60")),
        enable_code_generation=enable_code_gen,
        fast=False,
    )


def load_config_fast() -> IceburgConfig:
    base = load_config()
    s, d, y, o, e = resolve_models_small()
    return IceburgConfig(
        data_dir=base.data_dir,
        surveyor_model=s,
        dissident_model=d,
        synthesist_model=y,
        oracle_model=o,
        embed_model=e,
        enable_code_generation=base.enable_code_generation,
        fast=True,
    )


def load_config_hybrid() -> IceburgConfig:
    base = load_config()
    s, d, y, o, e = resolve_models_hybrid()
    return IceburgConfig(
        data_dir=base.data_dir,
        surveyor_model=s,
        dissident_model=d,
        synthesist_model=y,
        oracle_model=o,
        embed_model=e,
        enable_code_generation=base.enable_code_generation,
        fast=False,
    )


def load_config_with_model(
    primary_model: Optional[str] = None,
    use_small_models: bool = False,
    fast: bool = False
) -> IceburgConfig:
    """Load config with custom model selection"""
    base = load_config()
    
    if use_small_models:
        # CRITICAL: resolve_models_small() can return large models if they're first in available list
        # Force Surveyor to use a small model explicitly
        from .model_select import _available_model_names
        available = _available_model_names()
        small_candidates = ["deepseek-r1:7b", "llama3.1:8b", "llama3:8b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b", "phi3:mini", "deepseek-coder:6.7b"]
        s = None
        for candidate in small_candidates:
            if candidate in available:
                s = candidate
                break
        if not s:
            # Fallback: find any model with 8b or smaller in name
            for model in available:
                if any(marker in model.lower() for marker in [":8b", ":7b", ":3b", ":1b", "mini"]):
                    if not any(large in model.lower() for large in ["70b", "65b", "32b", "40b"]):
                        s = model
                        break
        if not s:
            s = "qwen2.5:7b"  # Final fallback
        # Other agents can use resolve_models_small
        _, d, y, o, e = resolve_models_small()
        # But ensure they're not huge models either
        if any(large in str(d).lower() for large in ["70b", "65b", "32b", "40b"]):
            d = s
        if any(large in str(y).lower() for large in ["70b", "65b", "32b", "40b"]):
            y = s
        if any(large in str(o).lower() for large in ["70b", "65b", "32b", "40b"]):
            o = s
    elif primary_model:
        # Use the primary model for all agents (faster, smaller)
        # BUT: For Surveyor, always prefer smaller models to avoid OOM errors
        from .model_select import _available_model_names, _first_present
        available = _available_model_names()
        
        # Check if primary_model is a large model (>8B) - these can cause OOM
        large_model_markers = ["70b", "65b", "32b", "40b"]
        is_large_model = any(marker in primary_model.lower() for marker in large_model_markers)
        
        if is_large_model:
            # Force Surveyor to use smaller model to prevent OOM
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"⚠️ Primary model {primary_model} is large - using smaller model for Surveyor to prevent OOM")
            # Explicitly find a small model for Surveyor (8b or smaller) - must be EXACT match
            small_candidates = ["deepseek-r1:7b", "llama3.1:8b", "llama3:8b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b", "phi3:mini", "deepseek-coder:6.7b"]
            s = None
            for candidate in small_candidates:
                if candidate in available:
                    s = candidate
                    break
            if not s:
                # Fallback: find any model with 8b or smaller in name
                for model in available:
                    if any(marker in model.lower() for marker in [":8b", ":7b", ":3b", ":1b", "mini"]):
                        if not any(large in model.lower() for large in ["70b", "65b", "32b", "40b"]):
                            s = model
                            break
            if not s:
                s = "qwen2.5:7b"  # Final fallback
            logger.info(f"✅ Surveyor will use {s} instead of {primary_model} to prevent OOM")
            d = y = o = primary_model if (primary_model in available or not available) else s
        elif primary_model in available or not available:
            s = d = y = o = primary_model
        else:
            # Fallback to small models if primary not available
            s, d, y, o, e = resolve_models_small()
            s = d = y = o = s  # Use the first available small model
        e = base.embed_model
    else:
        s, d, y, o, e = resolve_models(
            base.surveyor_model,
            base.dissident_model,
            base.synthesist_model,
            base.oracle_model,
            base.embed_model
        )
    
    return IceburgConfig(
        data_dir=base.data_dir,
        surveyor_model=s,
        dissident_model=d,
        synthesist_model=y,
        oracle_model=o,
        embed_model=e,
        llm_provider=base.llm_provider,
        provider_url=base.provider_url,
        timeout_s=base.timeout_s,
        enable_code_generation=base.enable_code_generation,
        fast=fast or use_small_models,
    )
