"""
Fine-Tuning Data Logger
Collects full conversation pairs, reasoning chains, and quality metrics
for future LLM fine-tuning
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class FineTuningLogger:
    """Logger for fine-tuning data collection."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize fine-tuning logger.
        
        Args:
            data_dir: Data directory path (defaults to data/fine_tuning)
        """
        if data_dir is None:
            data_dir = Path("data/fine_tuning")
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.conversations_file = self.data_dir / "conversations.jsonl"
        self.reasoning_chains_file = self.data_dir / "reasoning_chains.jsonl"
        self.quality_metrics_file = self.data_dir / "quality_metrics.jsonl"
        self.agent_generations_file = self.data_dir / "agent_generations.jsonl"
        
        # Enable/disable flag (opt-in only)
        self.enabled = os.getenv("ICEBURG_ENABLE_FINE_TUNING_DATA", "0") == "1"
        
        if self.enabled:
            logger.info(f"Fine-tuning data collection enabled. Data directory: {self.data_dir}")
        else:
            logger.debug("Fine-tuning data collection disabled (set ICEBURG_ENABLE_FINE_TUNING_DATA=1 to enable)")
    
    def log_conversation(
        self,
        messages: List[Dict[str, str]],
        metadata: Dict[str, Any],
        quality_score: Optional[float] = None
    ) -> None:
        """
        Log full conversation for fine-tuning.
        
        Args:
            messages: List of messages in ChatML format [{"role": "system/user/assistant", "content": "..."}]
            metadata: Conversation metadata (model, mode, agent, etc.)
            quality_score: Quality score (0.0-1.0) - only log if >= 0.8
        """
        if not self.enabled:
            return
        
        # Filter by quality (only log high-quality conversations)
        if quality_score is not None and quality_score < 0.8:
            logger.debug(f"Skipping low-quality conversation (score: {quality_score:.2f})")
            return
        
        # Ensure messages are in correct format
        if not isinstance(messages, list) or len(messages) == 0:
            logger.warning("Invalid messages format, skipping")
            return
        
        # Validate message format
        for msg in messages:
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                logger.warning("Invalid message format, skipping")
                return
        
        entry = {
            "messages": messages,
            "metadata": {
                **metadata,
                "quality_score": quality_score,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            with open(self.conversations_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(f"Logged conversation with {len(messages)} messages")
        except Exception as e:
            logger.error(f"Error logging conversation: {e}")
    
    def log_reasoning_chain(
        self,
        reasoning_chain: List[Dict[str, Any]],
        final_response: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Log reasoning chain for fine-tuning.
        
        Args:
            reasoning_chain: List of reasoning steps [{"step": 1, "agent": "...", "thinking": "...", "action": "...", "output": {...}}]
            final_response: Final response text
            metadata: Reasoning chain metadata (query, agents_used, etc.)
        """
        if not self.enabled:
            return
        
        if not isinstance(reasoning_chain, list) or len(reasoning_chain) == 0:
            logger.warning("Invalid reasoning chain format, skipping")
            return
        
        entry = {
            "reasoning_chain": reasoning_chain,
            "final_response": final_response,
            "metadata": {
                **metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            with open(self.reasoning_chains_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(f"Logged reasoning chain with {len(reasoning_chain)} steps")
        except Exception as e:
            logger.error(f"Error logging reasoning chain: {e}")
    
    def log_quality_metrics(
        self,
        conversation_id: str,
        quality_metrics: Dict[str, Any]
    ) -> None:
        """
        Log quality metrics for fine-tuning.
        
        Args:
            conversation_id: Conversation identifier
            quality_metrics: Quality metrics (user_rating, quality_score, accuracy, etc.)
        """
        if not self.enabled:
            return
        
        entry = {
            "conversation_id": conversation_id,
            "quality_metrics": quality_metrics,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            with open(self.quality_metrics_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(f"Logged quality metrics for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error logging quality metrics: {e}")
    
    def log_agent_generation(
        self,
        agent_name: str,
        generated_code: str,
        validation_result: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> None:
        """
        Log agent generation for fine-tuning.
        
        Args:
            agent_name: Name of generated agent
            generated_code: Generated Python code
            validation_result: Code validation results (valid, syntax_valid, etc.)
            metadata: Generation metadata (template, emergence_data, etc.)
        """
        if not self.enabled:
            return
        
        # Only log validated, successful generations
        if not validation_result.get("valid", False):
            logger.debug(f"Skipping invalid agent generation: {agent_name}")
            return
        
        entry = {
            "agent_name": agent_name,
            "generated_code": generated_code,
            "validation_result": validation_result,
            "metadata": {
                **metadata,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        try:
            with open(self.agent_generations_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.debug(f"Logged agent generation: {agent_name}")
        except Exception as e:
            logger.error(f"Error logging agent generation: {e}")
    
    def export_for_fine_tuning(
        self,
        output_file: Path,
        format: str = "chatml",  # "chatml", "alpaca", "sharegpt"
        min_quality: float = 0.8,
        min_conversations: int = 0
    ) -> int:
        """
        Export data in format suitable for fine-tuning.
        
        Args:
            output_file: Output file path
            format: Export format ("chatml", "alpaca", "sharegpt")
            min_quality: Minimum quality score (0.0-1.0)
            min_conversations: Minimum number of messages in conversation
            
        Returns:
            Number of exported conversations
        """
        if not self.conversations_file.exists():
            logger.warning(f"Conversations file not found: {self.conversations_file}")
            return 0
        
        # Read all conversations
        conversations = []
        try:
            with open(self.conversations_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        quality_score = entry.get("metadata", {}).get("quality_score", 0.0)
                        messages = entry.get("messages", [])
                        
                        # Filter by quality and minimum messages
                        if quality_score >= min_quality and len(messages) >= min_conversations:
                            conversations.append(entry)
        except Exception as e:
            logger.error(f"Error reading conversations: {e}")
            return 0
        
        # Convert to target format
        if format == "chatml":
            formatted = self._convert_to_chatml(conversations)
        elif format == "alpaca":
            formatted = self._convert_to_alpaca(conversations)
        elif format == "sharegpt":
            formatted = self._convert_to_sharegpt(conversations)
        else:
            logger.error(f"Unknown format: {format}")
            return 0
        
        # Write to output file
        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                for entry in formatted:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info(f"Exported {len(formatted)} conversations to {output_file} (format: {format})")
            return len(formatted)
        except Exception as e:
            logger.error(f"Error writing export file: {e}")
            return 0
    
    def _convert_to_chatml(self, conversations: List[Dict]) -> List[Dict]:
        """Convert conversations to ChatML format."""
        formatted = []
        for conv in conversations:
            messages = conv.get("messages", [])
            if messages:
                formatted.append({
                    "messages": messages
                })
        return formatted
    
    def _convert_to_alpaca(self, conversations: List[Dict]) -> List[Dict]:
        """Convert conversations to Alpaca format."""
        formatted = []
        for conv in conversations:
            messages = conv.get("messages", [])
            
            # Extract last user message and assistant response
            user_msg = None
            assistant_msg = None
            system_msg = None
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    system_msg = content
                elif role == "user":
                    user_msg = content
                elif role == "assistant":
                    assistant_msg = content
            
            if user_msg and assistant_msg:
                # Combine system message with instruction if present
                instruction = user_msg
                if system_msg:
                    instruction = f"{system_msg}\n\n{instruction}"
                
                formatted.append({
                    "instruction": instruction,
                    "input": "",
                    "output": assistant_msg
                })
        return formatted
    
    def _convert_to_sharegpt(self, conversations: List[Dict]) -> List[Dict]:
        """Convert conversations to ShareGPT format."""
        formatted = []
        for conv in conversations:
            messages = conv.get("messages", [])
            if messages:
                # ShareGPT format uses "conversations" key
                formatted.append({
                    "conversations": messages
                })
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        stats = {
            "enabled": self.enabled,
            "data_dir": str(self.data_dir),
            "conversations": 0,
            "reasoning_chains": 0,
            "quality_metrics": 0,
            "agent_generations": 0
        }
        
        if not self.enabled:
            return stats
        
        # Count entries in each file
        for file_path, key in [
            (self.conversations_file, "conversations"),
            (self.reasoning_chains_file, "reasoning_chains"),
            (self.quality_metrics_file, "quality_metrics"),
            (self.agent_generations_file, "agent_generations")
        ]:
            if file_path.exists():
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        count = sum(1 for line in f if line.strip())
                    stats[key] = count
                except Exception as e:
                    logger.error(f"Error counting {key}: {e}")
        
        return stats

