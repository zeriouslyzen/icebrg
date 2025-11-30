"""
Keystroke Prediction System
Implements Keystroke-Level Model (KLM) for user behavior prediction
"""

from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
import time
import json
from collections import deque
from dataclasses import dataclass, field


@dataclass
class KeystrokeEvent:
    """Individual keystroke event"""
    key: str
    timestamp: float
    key_code: Optional[int] = None
    modifiers: List[str] = field(default_factory=list)


@dataclass
class TypingPattern:
    """Typing pattern analysis"""
    words: List[str]
    word_times: List[float]
    average_wpm: float
    pause_patterns: List[float]
    common_sequences: List[Tuple[str, str]]


class KeystrokePredictor:
    """Predicts user prompts based on keystroke patterns"""
    
    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self.keystroke_history: deque = deque(maxlen=max_history)
        self.pattern_history: deque = deque(maxlen=50)
        self.prediction_cache: Dict[str, float] = {}
        self.user_consent: bool = False
        
    def set_consent(self, consented: bool):
        """Set user consent for keystroke tracking"""
        self.user_consent = consented
        
    def track_keystroke(self, key: str, key_code: Optional[int] = None, modifiers: Optional[List[str]] = None):
        """Track a keystroke event"""
        if not self.user_consent:
            return
            
        event = KeystrokeEvent(
            key=key,
            timestamp=time.time(),
            key_code=key_code,
            modifiers=modifiers or []
        )
        self.keystroke_history.append(event)
        
    def analyze_pattern(self) -> Optional[TypingPattern]:
        """Analyze typing patterns from keystroke history"""
        if len(self.keystroke_history) < 5:
            return None
            
        # Extract words and timing
        words = []
        word_times = []
        current_word = ""
        word_start_time = None
        
        for event in self.keystroke_history:
            if event.key == " " or event.key == "Enter":
                if current_word:
                    words.append(current_word)
                    if word_start_time:
                        word_times.append(event.timestamp - word_start_time)
                    word_start_time = None
                    current_word = ""
            elif event.key.isalnum() or event.key in ".,!?;:":
                if not word_start_time:
                    word_start_time = event.timestamp
                current_word += event.key
                
        # Calculate average WPM
        if word_times:
            total_time = sum(word_times)
            total_words = len(words)
            if total_time > 0:
                average_wpm = (total_words / total_time) * 60
            else:
                average_wpm = 0.0
        else:
            average_wpm = 0.0
            
        # Analyze pause patterns
        pause_patterns = []
        for i in range(1, len(self.keystroke_history)):
            pause = self.keystroke_history[i].timestamp - self.keystroke_history[i-1].timestamp
            pause_patterns.append(pause)
            
        # Find common sequences
        common_sequences = []
        if len(words) >= 2:
            for i in range(len(words) - 1):
                seq = (words[i], words[i+1])
                common_sequences.append(seq)
                
        return TypingPattern(
            words=words,
            word_times=word_times,
            average_wpm=average_wpm,
            pause_patterns=pause_patterns,
            common_sequences=common_sequences
        )
        
    def predict_next_prompt(self, current_text: str) -> Optional[Dict[str, Any]]:
        """Predict likely next prompt based on current typing"""
        if not self.user_consent or len(current_text) < 3:
            return None
            
        pattern = self.analyze_pattern()
        if not pattern:
            return None
            
        # Simple prediction based on common sequences and current text
        predictions = []
        
        # Check for common query patterns
        text_lower = current_text.lower()
        if text_lower.startswith("what"):
            predictions.append({
                "type": "question",
                "confidence": 0.7,
                "suggested_completion": "is",
                "context_preload": ["definition", "explanation"]
            })
        elif text_lower.startswith("how"):
            predictions.append({
                "type": "question",
                "confidence": 0.7,
                "suggested_completion": "does",
                "context_preload": ["process", "method"]
            })
        elif text_lower.startswith("why"):
            predictions.append({
                "type": "question",
                "confidence": 0.7,
                "suggested_completion": "is",
                "context_preload": ["reason", "explanation"]
            })
            
        # Check for common sequences
        if pattern.common_sequences:
            last_word = pattern.words[-1] if pattern.words else ""
            for seq in pattern.common_sequences[-5:]:  # Check last 5 sequences
                if seq[0] == last_word:
                    predictions.append({
                        "type": "sequence",
                        "confidence": 0.5,
                        "suggested_completion": seq[1],
                        "context_preload": []
                    })
                    
        if predictions:
            # Return highest confidence prediction
            best = max(predictions, key=lambda x: x["confidence"])
            return {
                "prediction": best,
                "pattern_analysis": {
                    "wpm": pattern.average_wpm,
                    "word_count": len(pattern.words),
                    "pause_avg": sum(pattern.pause_patterns) / len(pattern.pause_patterns) if pattern.pause_patterns else 0
                }
            }
            
        return None
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get keystroke statistics"""
        if not self.user_consent:
            return {"consent": False}
            
        pattern = self.analyze_pattern()
        if not pattern:
            return {
                "consent": True,
                "events_tracked": len(self.keystroke_history),
                "pattern_available": False
            }
            
        return {
            "consent": True,
            "events_tracked": len(self.keystroke_history),
            "pattern_available": True,
            "average_wpm": pattern.average_wpm,
            "words_typed": len(pattern.words),
            "common_sequences": pattern.common_sequences[-10:]  # Last 10 sequences
        }

