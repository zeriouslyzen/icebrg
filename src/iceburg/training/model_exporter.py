"""
ICEBURG Model Exporter
======================

Exports fine-tuned models in multiple formats:
- Ollama Modelfile format (for local inference)
- HuggingFace format (for sharing and further fine-tuning)
- GGUF/GGML quantized formats (for efficient deployment)
"""

from __future__ import annotations
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class ExportFormat(Enum):
    """Supported export formats."""
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"
    GGUF = "gguf"
    ONNX = "onnx"


class QuantizationType(Enum):
    """Quantization types for GGUF export."""
    Q4_0 = "q4_0"
    Q4_K_M = "q4_k_m"
    Q5_0 = "q5_0"
    Q5_K_M = "q5_k_m"
    Q8_0 = "q8_0"
    F16 = "f16"


@dataclass
class ExportResult:
    """Result of model export."""
    success: bool
    format: ExportFormat
    output_path: Path
    file_size_mb: float
    export_time_seconds: float
    metadata: Dict[str, Any]
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "format": self.format.value,
            "output_path": str(self.output_path),
            "file_size_mb": self.file_size_mb,
            "export_time_seconds": self.export_time_seconds,
            "metadata": self.metadata,
            "error_message": self.error_message
        }


class ModelExporter:
    """
    Exports ICEBURG fine-tuned models in multiple formats.
    
    Supports:
    - HuggingFace format (transformers compatible)
    - Ollama Modelfile (local deployment)
    - GGUF quantized (via llama.cpp)
    """
    
    # Default system prompts for each model type
    SYSTEM_PROMPTS = {
        "iceburg-base": "You are ICEBURG, a truth-seeking AI assistant that uses multi-agent verification to ensure accuracy and minimize hallucination.",
        "iceburg-surveyor": "You are ICEBURG Surveyor, a research agent specialized in gathering comprehensive information, exploring domains, and synthesizing evidence from multiple sources.",
        "iceburg-dissident": "You are ICEBURG Dissident, an adversarial agent specialized in challenging assumptions, detecting contradictions, and presenting alternative perspectives to ensure truth through conflict.",
        "iceburg-synthesist": "You are ICEBURG Synthesist, a connection agent specialized in cross-domain synthesis, integrating insights from multiple agents, and finding unexpected connections.",
        "iceburg-oracle": "You are ICEBURG Oracle, a truth-validation agent specialized in extracting fundamental principles, validating conclusions, and making final truth determinations."
    }
    
    def __init__(self, default_output_dir: Optional[Path] = None):
        """
        Initialize model exporter.
        
        Args:
            default_output_dir: Default directory for exports
        """
        self.default_output_dir = default_output_dir or Path("models/iceburg/exports")
        self.default_output_dir.mkdir(parents=True, exist_ok=True)
        
    def export(
        self,
        model_path: Path,
        model_name: str,
        formats: List[ExportFormat] = None,
        quantization: QuantizationType = QuantizationType.Q4_K_M,
        output_dir: Optional[Path] = None
    ) -> Dict[ExportFormat, ExportResult]:
        """
        Export model in specified formats.
        
        Args:
            model_path: Path to fine-tuned model
            model_name: Name for the exported model
            formats: List of formats to export (default: all)
            quantization: Quantization type for GGUF export
            output_dir: Output directory (uses default if None)
            
        Returns:
            Dictionary of format -> ExportResult
        """
        if formats is None:
            formats = [ExportFormat.HUGGINGFACE, ExportFormat.OLLAMA]
            
        output_dir = output_dir or self.default_output_dir
        results = {}
        
        for fmt in formats:
            logger.info(f"Exporting {model_name} to {fmt.value} format...")
            
            if fmt == ExportFormat.HUGGINGFACE:
                results[fmt] = self._export_huggingface(model_path, model_name, output_dir)
            elif fmt == ExportFormat.OLLAMA:
                results[fmt] = self._export_ollama(model_path, model_name, output_dir)
            elif fmt == ExportFormat.GGUF:
                results[fmt] = self._export_gguf(model_path, model_name, output_dir, quantization)
            elif fmt == ExportFormat.ONNX:
                results[fmt] = self._export_onnx(model_path, model_name, output_dir)
                
        return results
        
    def _export_huggingface(
        self,
        model_path: Path,
        model_name: str,
        output_dir: Path
    ) -> ExportResult:
        """Export in HuggingFace format."""
        import time
        start_time = time.time()
        
        try:
            hf_dir = output_dir / model_name / "huggingface"
            hf_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            if model_path.exists():
                for item in model_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, hf_dir)
                    elif item.is_dir() and item.name not in ["Modelfile"]:
                        shutil.copytree(item, hf_dir / item.name, dirs_exist_ok=True)
                        
            # Create model card
            self._create_model_card(hf_dir, model_name)
            
            # Calculate size
            total_size = sum(f.stat().st_size for f in hf_dir.rglob("*") if f.is_file())
            size_mb = total_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                format=ExportFormat.HUGGINGFACE,
                output_path=hf_dir,
                file_size_mb=size_mb,
                export_time_seconds=time.time() - start_time,
                metadata={"files": [f.name for f in hf_dir.iterdir()]}
            )
            
        except Exception as e:
            logger.error(f"HuggingFace export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.HUGGINGFACE,
                output_path=Path(""),
                file_size_mb=0,
                export_time_seconds=time.time() - start_time,
                metadata={},
                error_message=str(e)
            )
            
    def _export_ollama(
        self,
        model_path: Path,
        model_name: str,
        output_dir: Path
    ) -> ExportResult:
        """Export as Ollama Modelfile."""
        import time
        start_time = time.time()
        
        try:
            ollama_dir = output_dir / model_name / "ollama"
            ollama_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine model type from name
            model_type = self._detect_model_type(model_name)
            system_prompt = self.SYSTEM_PROMPTS.get(model_type, self.SYSTEM_PROMPTS["iceburg-base"])
            
            # Find base model path
            hf_path = model_path / "huggingface" if (model_path / "huggingface").exists() else model_path
            
            # Create Modelfile
            modelfile_content = self._create_modelfile(
                model_name=model_name,
                model_path=hf_path,
                system_prompt=system_prompt,
                model_type=model_type
            )
            
            modelfile_path = ollama_dir / "Modelfile"
            with open(modelfile_path, "w") as f:
                f.write(modelfile_content)
                
            # Create README for Ollama usage
            readme_path = ollama_dir / "README.md"
            with open(readme_path, "w") as f:
                f.write(f"""# {model_name} - Ollama Model

## Installation

1. Make sure Ollama is installed: https://ollama.ai/

2. Create the model:
```bash
cd {ollama_dir}
ollama create {model_name} -f Modelfile
```

3. Run the model:
```bash
ollama run {model_name}
```

## Model Information

- **Type**: {model_type}
- **Created**: {datetime.now().isoformat()}
- **Framework**: ICEBURG Internal Fine-Tuning

## Usage Examples

```bash
# Interactive chat
ollama run {model_name}

# API call
curl http://localhost:11434/api/generate -d '{{
  "model": "{model_name}",
  "prompt": "What is your approach to truth-seeking?"
}}'
```
""")
            
            # Calculate size
            size_mb = modelfile_path.stat().st_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                format=ExportFormat.OLLAMA,
                output_path=ollama_dir,
                file_size_mb=size_mb,
                export_time_seconds=time.time() - start_time,
                metadata={
                    "modelfile": str(modelfile_path),
                    "model_type": model_type,
                    "system_prompt": system_prompt[:100] + "..."
                }
            )
            
        except Exception as e:
            logger.error(f"Ollama export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.OLLAMA,
                output_path=Path(""),
                file_size_mb=0,
                export_time_seconds=time.time() - start_time,
                metadata={},
                error_message=str(e)
            )
            
    def _export_gguf(
        self,
        model_path: Path,
        model_name: str,
        output_dir: Path,
        quantization: QuantizationType
    ) -> ExportResult:
        """Export as GGUF quantized model."""
        import time
        start_time = time.time()
        
        try:
            gguf_dir = output_dir / model_name / "gguf"
            gguf_dir.mkdir(parents=True, exist_ok=True)
            
            gguf_file = gguf_dir / f"{model_name}-{quantization.value}.gguf"
            
            # Check if llama.cpp convert script is available
            convert_script = shutil.which("convert-hf-to-gguf.py")
            
            if not convert_script:
                # Try common locations
                common_paths = [
                    Path.home() / "llama.cpp" / "convert-hf-to-gguf.py",
                    Path("/opt/llama.cpp/convert-hf-to-gguf.py"),
                ]
                
                for path in common_paths:
                    if path.exists():
                        convert_script = str(path)
                        break
                        
            if convert_script:
                # Run conversion
                hf_path = model_path / "huggingface" if (model_path / "huggingface").exists() else model_path
                
                cmd = [
                    "python", convert_script,
                    str(hf_path),
                    "--outfile", str(gguf_file),
                    "--outtype", quantization.value
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
                    
                size_mb = gguf_file.stat().st_size / (1024 * 1024)
                
                return ExportResult(
                    success=True,
                    format=ExportFormat.GGUF,
                    output_path=gguf_file,
                    file_size_mb=size_mb,
                    export_time_seconds=time.time() - start_time,
                    metadata={"quantization": quantization.value}
                )
            else:
                # Create instructions for manual conversion
                instructions_path = gguf_dir / "CONVERSION_INSTRUCTIONS.md"
                with open(instructions_path, "w") as f:
                    f.write(f"""# GGUF Conversion Instructions

llama.cpp is required for GGUF conversion but was not found.

## Installation

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

## Conversion

```bash
python convert-hf-to-gguf.py {model_path}/huggingface --outfile {gguf_file} --outtype {quantization.value}
```

## Quantization

For additional quantization:
```bash
./quantize {gguf_file} {model_name}-{quantization.value}.gguf {quantization.value}
```
""")
                
                return ExportResult(
                    success=False,
                    format=ExportFormat.GGUF,
                    output_path=gguf_dir,
                    file_size_mb=0,
                    export_time_seconds=time.time() - start_time,
                    metadata={"instructions": str(instructions_path)},
                    error_message="llama.cpp not found. See CONVERSION_INSTRUCTIONS.md"
                )
                
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.GGUF,
                output_path=Path(""),
                file_size_mb=0,
                export_time_seconds=time.time() - start_time,
                metadata={},
                error_message=str(e)
            )
            
    def _export_onnx(
        self,
        model_path: Path,
        model_name: str,
        output_dir: Path
    ) -> ExportResult:
        """Export as ONNX model."""
        import time
        start_time = time.time()
        
        try:
            onnx_dir = output_dir / model_name / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            
            # ONNX export requires optimum
            try:
                from optimum.exporters.onnx import main_export
            except ImportError:
                return ExportResult(
                    success=False,
                    format=ExportFormat.ONNX,
                    output_path=onnx_dir,
                    file_size_mb=0,
                    export_time_seconds=time.time() - start_time,
                    metadata={},
                    error_message="optimum library not installed. Run: pip install optimum[onnxruntime]"
                )
                
            hf_path = model_path / "huggingface" if (model_path / "huggingface").exists() else model_path
            
            main_export(
                str(hf_path),
                str(onnx_dir),
                task="text-generation"
            )
            
            # Calculate size
            total_size = sum(f.stat().st_size for f in onnx_dir.rglob("*.onnx"))
            size_mb = total_size / (1024 * 1024)
            
            return ExportResult(
                success=True,
                format=ExportFormat.ONNX,
                output_path=onnx_dir,
                file_size_mb=size_mb,
                export_time_seconds=time.time() - start_time,
                metadata={"files": [f.name for f in onnx_dir.glob("*.onnx")]}
            )
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return ExportResult(
                success=False,
                format=ExportFormat.ONNX,
                output_path=Path(""),
                file_size_mb=0,
                export_time_seconds=time.time() - start_time,
                metadata={},
                error_message=str(e)
            )
            
    def _detect_model_type(self, model_name: str) -> str:
        """Detect model type from name."""
        model_name_lower = model_name.lower()
        
        if "surveyor" in model_name_lower:
            return "iceburg-surveyor"
        elif "dissident" in model_name_lower:
            return "iceburg-dissident"
        elif "synthesist" in model_name_lower:
            return "iceburg-synthesist"
        elif "oracle" in model_name_lower:
            return "iceburg-oracle"
        else:
            return "iceburg-base"
            
    def _create_modelfile(
        self,
        model_name: str,
        model_path: Path,
        system_prompt: str,
        model_type: str
    ) -> str:
        """Create Ollama Modelfile content."""
        return f'''# ICEBURG Fine-Tuned Model: {model_name}
# Type: {model_type}
# Created: {datetime.now().isoformat()}
# Framework: ICEBURG Internal Fine-Tuning

# Base model - point to HuggingFace format
FROM {model_path}

# Template for ChatML format
TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{- end }}}}
{{{{- range .Messages }}}}
<|im_start|>{{{{ .Role }}}}
{{{{ .Content }}}}<|im_end|>
{{{{- end }}}}
<|im_start|>assistant
"""

# Model parameters optimized for truth-seeking
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 4096

# System prompt
SYSTEM """{system_prompt}"""

# License
LICENSE """
ICEBURG Internal Fine-Tuned Model
For use within ICEBURG multi-agent system.
Base model license applies.
"""
'''
        
    def _create_model_card(self, output_dir: Path, model_name: str) -> None:
        """Create HuggingFace model card."""
        model_type = self._detect_model_type(model_name)
        
        readme_content = f'''---
tags:
  - iceburg
  - multi-agent
  - truth-seeking
  - fine-tuned
license: apache-2.0
language:
  - en
---

# {model_name}

ICEBURG fine-tuned model for {model_type.replace("iceburg-", "")} tasks.

## Model Description

This model was fine-tuned using ICEBURG's internal fine-tuning framework, which includes:

- **Truth Filtering**: Training data filtered using ICEBURG's Instant Truth System
- **Emergence Processing**: Curriculum weighted by emergence patterns
- **Agent Specialization**: Optimized for specific agent role

## Model Type

**{model_type}**

{self.SYSTEM_PROMPTS.get(model_type, "")}

## Training

- **Framework**: ICEBURG Internal Fine-Tuning
- **Method**: LoRA/QLoRA
- **Hardware**: Optimized for M4 Mac (MPS)

## Usage

### With Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{model_name}")
tokenizer = AutoTokenizer.from_pretrained("{model_name}")

# Generate
inputs = tokenizer("What is truth?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=500)
print(tokenizer.decode(outputs[0]))
```

### With Ollama

```bash
ollama create {model_name} -f Modelfile
ollama run {model_name}
```

## Part of ICEBURG

This model is designed to work within the ICEBURG multi-agent research system,
providing specialized capabilities for truth-seeking and research synthesis.
'''
        
        readme_path = output_dir / "README.md"
        with open(readme_path, "w") as f:
            f.write(readme_content)
            
            
# Convenience function
def export_model(
    model_path: str,
    model_name: str,
    formats: List[str] = None
) -> Dict[str, ExportResult]:
    """
    Export a fine-tuned model.
    
    Args:
        model_path: Path to model
        model_name: Name for export
        formats: List of format names ("huggingface", "ollama", "gguf")
        
    Returns:
        Dictionary of format -> ExportResult
    """
    exporter = ModelExporter()
    
    if formats is None:
        format_enums = [ExportFormat.HUGGINGFACE, ExportFormat.OLLAMA]
    else:
        format_mapping = {
            "huggingface": ExportFormat.HUGGINGFACE,
            "ollama": ExportFormat.OLLAMA,
            "gguf": ExportFormat.GGUF,
            "onnx": ExportFormat.ONNX
        }
        format_enums = [format_mapping[f.lower()] for f in formats if f.lower() in format_mapping]
        
    results = exporter.export(Path(model_path), model_name, format_enums)
    
    return {fmt.value: result for fmt, result in results.items()}

