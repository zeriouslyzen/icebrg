#!/usr/bin/env python3
"""
ICEBURG Model Inference CLI
============================

Run inference with trained ICEBURG models using transformers/peft.

Usage:
    python scripts/run_iceburg_model.py --model surveyor --prompt "Research quantum computing"
    python scripts/run_iceburg_model.py --model dissident --prompt "Challenge the AI hype"
    python scripts/run_iceburg_model.py --interactive --model oracle
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# System prompts for each model type
SYSTEM_PROMPTS = {
    "surveyor": "You are ICEBURG Surveyor, a research agent specialized in gathering comprehensive information, exploring domains, and synthesizing evidence from multiple authoritative sources.",
    "dissident": "You are ICEBURG Dissident, an adversarial agent specialized in challenging assumptions, detecting contradictions, and presenting alternative perspectives.",
    "synthesist": "You are ICEBURG Synthesist, a connection agent specialized in cross-domain synthesis, integrating insights from multiple fields, and discovering unexpected connections.",
    "oracle": "You are ICEBURG Oracle, a truth-validation agent specialized in extracting fundamental principles, validating conclusions, and making final truth determinations."
}


class ICEBURGModelRunner:
    """Runner for ICEBURG fine-tuned models."""
    
    def __init__(self, model_type: str):
        """
        Initialize model runner.
        
        Args:
            model_type: Type of model (surveyor, dissident, synthesist, oracle)
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.device = None
        self._loaded = False
        
    def load(self) -> bool:
        """Load the model and tokenizer."""
        if self._loaded:
            return True
            
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
            
            # Get device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
                
            logger.info(f"Using device: {self.device}")
            
            # Find the best model
            from iceburg.training.model_registry import get_model_registry
            registry = get_model_registry()
            model_info = registry.get_best_model(self.model_type)
            
            if not model_info:
                logger.error(f"No {self.model_type} model found in registry")
                return False
                
            logger.info(f"Loading model: {model_info.name}")
            logger.info(f"  Path: {model_info.path}")
            
            # Load base model - we need to know what base model was used
            # For now, assume Qwen2.5-0.5B since that's what we trained with
            base_model_name = "Qwen/Qwen2.5-0.5B"
            
            logger.info(f"Loading base model: {base_model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float32 if self.device.type == "mps" else torch.float16,
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            adapter_path = model_info.path / "huggingface"
            if not adapter_path.exists():
                adapter_path = model_info.path  # Try root path
                
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate a response to a prompt.
        
        Args:
            prompt: User prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling
            
        Returns:
            Generated response
        """
        if not self._loaded:
            if not self.load():
                return "Error: Failed to load model"
                
        try:
            import torch
            
            # Build ChatML formatted input
            system_prompt = SYSTEM_PROMPTS.get(self.model_type, SYSTEM_PROMPTS["surveyor"])
            
            chat_input = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""
            
            # Tokenize
            inputs = self.tokenizer(
                chat_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # Extract just the assistant response
            if "<|im_start|>assistant" in response:
                response = response.split("<|im_start|>assistant")[-1]
            if "<|im_end|>" in response:
                response = response.split("<|im_end|>")[0]
                
            return response.strip()
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {e}"
            
    def interactive(self):
        """Run interactive chat session."""
        print("\n" + "="*60)
        print(f"ICEBURG {self.model_type.upper()} - Interactive Mode")
        print("="*60)
        print("\nType 'quit' or 'exit' to end the session.")
        print("Type 'clear' to clear the screen.\n")
        
        if not self.load():
            print("Failed to load model. Exiting.")
            return
            
        while True:
            try:
                prompt = input("\nYou: ").strip()
                
                if not prompt:
                    continue
                    
                if prompt.lower() in ["quit", "exit"]:
                    print("\nGoodbye!")
                    break
                    
                if prompt.lower() == "clear":
                    print("\033c", end="")
                    continue
                    
                print(f"\n{self.model_type.capitalize()}: ", end="", flush=True)
                response = self.generate(prompt)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break


def main():
    parser = argparse.ArgumentParser(description="ICEBURG Model Inference")
    
    parser.add_argument(
        "--model",
        choices=["surveyor", "dissident", "synthesist", "oracle"],
        default="surveyor",
        help="Model type to use (default: surveyor)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Prompt to generate response for"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    runner = ICEBURGModelRunner(args.model)
    
    if args.interactive:
        runner.interactive()
    elif args.prompt:
        print(f"\nLoading {args.model} model...")
        if runner.load():
            print(f"\nPrompt: {args.prompt}")
            print(f"\nResponse:")
            response = runner.generate(
                args.prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature
            )
            print(response)
        else:
            print("Failed to load model")
            return 1
    else:
        print("Specify --prompt or --interactive")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

