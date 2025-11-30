from __future__ import annotations
from typing import Dict, Any, List, Optional
import json
import uuid
from datetime import datetime
from pathlib import Path


class TrainingDataGenerator:
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("./data")
        self.training_dir = self.data_dir / "training_data"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Training data storage
        self.supervised_file = self.training_dir / "supervised_learning.jsonl"
        self.reinforcement_file = self.training_dir / "reinforcement_learning.jsonl"
        self.few_shot_file = self.training_dir / "few_shot_learning.jsonl"
        self.meta_learning_file = self.training_dir / "meta_learning.jsonl"
    
    def generate_training_data(self, emergence_intel: Dict[str, Any]) -> Dict[str, Any]:
        """Convert emergence intelligence into training data for new LLMs"""
        
        # Extract high-value emergence patterns
        high_value_patterns = self.extract_high_value_patterns(emergence_intel)
        
        # Generate training examples
        training_examples = []
        for pattern in high_value_patterns:
            example = self.create_training_example(pattern)
            training_examples.append(example)
        
        # Format for different training approaches
        training_formats = {
            "supervised_learning": self.format_for_supervised(training_examples),
            "reinforcement_learning": self.format_for_rl(training_examples),
            "few_shot_learning": self.format_for_few_shot(training_examples),
            "meta_learning": self.format_for_meta_learning(training_examples)
        }
        
        # Store training data
        self.store_training_data(training_formats)
        
        return training_formats
    
    def extract_high_value_patterns(self, emergence_intel: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract high-value patterns from emergence intelligence"""
        
        patterns = []
        
        # High emergence score patterns
        emergence_score = emergence_intel.get("emergence_score", 0.0)
        if emergence_score > 0.7:
            patterns.append({
                "type": "high_emergence",
                "score": emergence_score,
                "context": emergence_intel.get("core_principle", ""),
                "emergence_insight": f"High emergence score ({emergence_score}) indicates significant novel insight",
                "confidence": emergence_intel.get("confidence_score", 0.0)
            })
        
        # Cross-domain synthesis patterns
        domains = emergence_intel.get("domains", [])
        if len(domains) > 1:
            patterns.append({
                "type": "cross_domain_synthesis",
                "score": 0.8,
                "context": f"Domains: {', '.join(domains)}",
                "emergence_insight": f"Cross-domain synthesis across {len(domains)} domains",
                "confidence": emergence_intel.get("confidence_score", 0.0)
            })
        
        # Novel prediction patterns
        predictions = emergence_intel.get("key_predictions", [])
        if predictions:
            patterns.append({
                "type": "novel_prediction",
                "score": 0.7,
                "context": f"Predictions: {'; '.join(predictions)}",
                "emergence_insight": f"Generated {len(predictions)} novel, testable predictions",
                "confidence": emergence_intel.get("confidence_score", 0.0)
            })
        
        # Assumption challenge patterns
        if "assumption_challenge" in emergence_intel.get("patterns", []):
            patterns.append({
                "type": "assumption_challenge",
                "score": 0.8,
                "context": emergence_intel.get("core_principle", ""),
                "emergence_insight": "Successfully challenged underlying assumptions",
                "confidence": emergence_intel.get("confidence_score", 0.0)
            })
        
        return patterns
    
    def create_training_example(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Create training example from emergence pattern"""
        
        return {
            "example_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "pattern_type": pattern["type"],
            "input": f"Analyze: {pattern['context']}",
            "output": pattern["emergence_insight"],
            "metadata": {
                "emergence_score": pattern["score"],
                "pattern_type": pattern["type"],
                "confidence": pattern["confidence"],
                "training_value": self.calculate_training_value(pattern)
            }
        }
    
    def calculate_training_value(self, pattern: Dict[str, Any]) -> float:
        """Calculate the training value of a pattern"""
        
        base_value = pattern.get("score", 0.0)
        confidence = pattern.get("confidence", 0.0)
        
        # Higher emergence scores are more valuable for training
        # Higher confidence increases training value
        training_value = (base_value * 0.7) + (confidence * 0.3)
        
        return min(training_value, 1.0)
    
    def format_for_supervised(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format examples for supervised learning"""
        
        supervised_examples = []
        for example in examples:
            supervised_examples.append({
                "input": example["input"],
                "output": example["output"],
                "metadata": example["metadata"]
            })
        
        return supervised_examples
    
    def format_for_rl(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format examples for reinforcement learning"""
        
        rl_examples = []
        for example in examples:
            # RL format: state, action, reward
            rl_examples.append({
                "state": example["input"],
                "action": example["output"],
                "reward": example["metadata"]["training_value"],
                "metadata": example["metadata"]
            })
        
        return rl_examples
    
    def format_for_few_shot(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format examples for few-shot learning"""
        
        few_shot_examples = []
        for example in examples:
            # Few-shot format: task description + examples
            few_shot_examples.append({
                "task": "Detect emergence patterns in analysis",
                "examples": [example],
                "query": example["input"],
                "expected": example["output"],
                "metadata": example["metadata"]
            })
        
        return few_shot_examples
    
    def format_for_meta_learning(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format examples for meta-learning"""
        
        meta_examples = []
        for example in examples:
            # Meta-learning format: learning to learn
            meta_examples.append({
                "task_type": "emergence_detection",
                "task_description": f"Learn to detect {example['pattern_type']} patterns",
                "example": example,
                "learning_objective": "Improve emergence detection across domains",
                "metadata": example["metadata"]
            })
        
        return meta_examples
    
    def store_training_data(self, training_formats: Dict[str, Any]) -> None:
        """Store training data to files"""
        
        try:
            # Store supervised learning data
            with open(self.supervised_file, "a", encoding="utf-8") as f:
                for example in training_formats["supervised_learning"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            # Store reinforcement learning data
            with open(self.reinforcement_file, "a", encoding="utf-8") as f:
                for example in training_formats["reinforcement_learning"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            # Store few-shot learning data
            with open(self.few_shot_file, "a", encoding="utf-8") as f:
                for example in training_formats["few_shot_learning"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            # Store meta-learning data
            with open(self.meta_learning_file, "a", encoding="utf-8") as f:
                for example in training_formats["meta_learning"]:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
                    
        except Exception as e:
    
    def get_training_data_summary(self) -> Dict[str, Any]:
        """Get a summary of generated training data"""
        
        summary = {}
        
        try:
            # Count examples in each file
            for file_path, file_type in [
                (self.supervised_file, "supervised_learning"),
                (self.reinforcement_file, "reinforcement_learning"),
                (self.few_shot_file, "few_shot_learning"),
                (self.meta_learning_file, "meta_learning")
            ]:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        summary[file_type] = len(lines)
                else:
                    summary[file_type] = 0
                    
        except Exception as e:
            summary = {}
        
        return summary
    
    def clear_training_data(self) -> None:
        """Clear all training data (for testing/reset)"""
        
        try:
            for file_path in [self.supervised_file, self.reinforcement_file, 
                self.few_shot_file, self.meta_learning_file]:
                if file_path.exists():
                    file_path.unlink()
        except Exception as e:
