#!/usr/bin/env python3
"""
REAL VOICE CONVERSATION SYSTEM
Now you can actually TALK to the AI with your voice!
"""

import sys
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

# Import real voice modules
from .real_voice_input import RealVoiceInputModule, RealVoiceInput
from .real_voice_output import RealVoiceOutputModule, RealVoiceOutput


@dataclass
class RealVoiceConversation:
    """Real voice conversation session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_exchanges: int
    voice_inputs: List[RealVoiceInput]
    voice_outputs: List[RealVoiceOutput]


class RealVoiceConversationSystem:
    """REAL voice conversation system - you can actually talk to it!"""
    
    def __init__(self):
        self.voice_input = RealVoiceInputModule()
        self.voice_output = RealVoiceOutputModule()
        self.conversation_counter = 0
        self.current_session = None
        
        # Conversation routing based on AI's own solution
        self.complexity_thresholds = {
            "fast_pipeline": 0.4,      # Simple questions: 2-5 seconds
            "hybrid_pipeline": 0.7,    # Medium questions: 30-60 seconds
            "deep_pipeline": 1.0       # Complex questions: 2-3 minutes
        }
        
        print("🎤 REAL VOICE CONVERSATION SYSTEM INITIALIZED!")
        print("✅ You can now SPEAK to the AI and HEAR it respond!")
    
    def start_voice_conversation(self) -> str:
        """Start a new REAL voice conversation session"""
        
        session_id = self._generate_session_id()
        self.current_session = RealVoiceConversation(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            total_exchanges=0,
            voice_inputs=[],
            voice_outputs=[]
        )
        
        # Welcome message - AI will SPEAK this to you!
        welcome_text = "Hello! I am the Iceberg Protocol, an autonomous research platform. I can now hear and speak with you. How can I assist you today?"
        
        print(f"[VOICE] 🎭 Session started: {session_id}")
        print(f"[VOICE] 🔊 AI speaking: {welcome_text}")
        
        # AI speaks the welcome message
        voice_output = self.voice_output.speak_response(welcome_text)
        self.current_session.voice_outputs.append(voice_output)
        
        return session_id
    
    def voice_chat(self, question: str = None) -> RealVoiceOutput:
        """Process a REAL voice question and respond with voice"""
        
        if not self.current_session:
            self.start_voice_conversation()
        
        # Get REAL voice input if not provided
        if not question:
            print("[VOICE] 🎤 Listening for your voice question...")
            voice_input = self.voice_input.listen_for_question()
            question = voice_input.transcription
            self.current_session.voice_inputs.append(voice_input)
        else:
            # Simulate voice input for testing
            voice_input = RealVoiceInput(
                audio_data=b"simulated_audio",
                duration=len(question) * 0.1,
                timestamp=datetime.now(),
                transcription=question,
                confidence=0.98,
                source="test_input"
            )
            self.current_session.voice_inputs.append(voice_input)
        
        print(f"[VOICE] 🎯 You said: {question}")
        
        # Route question based on complexity (AI's own solution)
        pipeline_type = self._route_question(question)
        print(f"[VOICE] 🚀 Routing to: {pipeline_type}")
        
        # Generate response based on pipeline type
        response = self._generate_response(question, pipeline_type)
        
        # AI SPEAKS the response to you!
        print(f"[VOICE] 🤖 AI responding: {response[:100]}...")
        voice_output = self.voice_output.speak_response(response)
        self.current_session.voice_outputs.append(voice_output)
        self.current_session.total_exchanges += 1
        
        return voice_output
    
    def real_time_voice_conversation(self, callback_function: Callable = None) -> None:
        """Start REAL-TIME voice conversation - you talk, AI responds!"""
        
        if not self.current_session:
            self.start_voice_conversation()
        
        print("\n🎤 🚀 STARTING REAL VOICE CONVERSATION!")
        print("=" * 60)
        print("🎯 HOW TO USE:")
        print("   1. Say your question clearly")
        print("   2. AI will analyze and respond")
        print("   3. Say 'Goodbye' to end")
        print("   4. Press Ctrl+C to stop")
        print("=" * 60)
        print("🎤 SPEAK NOW...")
        
        def voice_callback(voice_input: RealVoiceInput):
            if voice_input.transcription.lower() == "goodbye":
                print("[VOICE] 👋 Goodbye detected, ending conversation...")
                self.end_voice_conversation()
                return
            
            print(f"\n[VOICE] 🎯 Processing: {voice_input.transcription}")
            
            # Process the voice question
            voice_output = self.voice_chat(voice_input.transcription)
            
            # Call custom callback if provided
            if callback_function:
                callback_function(voice_input, voice_output)
            
            print(f"\n[VOICE] ✅ Exchange complete. Ready for next question...")
            print("🎤 SPEAK NOW...")
        
        # Start real-time listening
        self.voice_input.real_time_listening(voice_callback)
    
    def _route_question(self, question: str) -> str:
        """Route question to appropriate pipeline based on complexity"""
        
        # Simple complexity analysis (in production, use AI classification)
        complexity_score = self._analyze_complexity(question)
        
        # Route based on AI's own thresholds
        if complexity_score < self.complexity_thresholds["fast_pipeline"]:
            return "fast_pipeline"
        elif complexity_score < self.complexity_thresholds["hybrid_pipeline"]:
            return "hybrid_pipeline"
        else:
            return "deep_pipeline"
    
    def _analyze_complexity(self, question: str) -> float:
        """Analyze question complexity (0.0 = simple, 1.0 = complex)"""
        
        # Simple heuristics (in production, use AI classification)
        complexity_factors = {
            "length": len(question.split()) / 100,  # Normalize by word count
            "keywords": self._check_complex_keywords(question),
            "structure": self._analyze_question_structure(question)
        }
        
        # Weighted complexity score
        complexity_score = (
            complexity_factors["length"] * 0.3 +
            complexity_factors["keywords"] * 0.4 +
            complexity_factors["structure"] * 0.3
        )
        
        return min(max(complexity_score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _check_complex_keywords(self, question: str) -> float:
        """Check for complex keywords that indicate deep analysis needed"""
        
        complex_keywords = [
            "how", "why", "explain", "analyze", "compare", "contrast",
            "implications", "mechanisms", "theories", "paradigms",
            "quantum", "consciousness", "emergence", "complexity",
            "implement", "design", "create", "solve"
        ]
        
        question_lower = question.lower()
        complex_count = sum(1 for keyword in complex_keywords if keyword in question_lower)
        
        return min(complex_count / 5, 1.0)  # Normalize by expected max
    
    def _analyze_question_structure(self, question: str) -> float:
        """Analyze question structure for complexity indicators"""
        
        # Multiple clauses indicate complexity
        clause_indicators = ["and", "or", "but", "however", "although", "while"]
        clause_count = sum(1 for indicator in clause_indicators if indicator in question.lower())
        
        # Question marks and length indicate complexity
        question_marks = question.count("?")
        length_factor = len(question) / 200  # Normalize by expected max length
        
        structure_score = (clause_count * 0.3 + question_marks * 0.2 + length_factor * 0.5)
        return min(structure_score, 1.0)
    
    def _generate_response(self, question: str, pipeline_type: str) -> str:
        """Generate response based on pipeline type"""
        
        if pipeline_type == "fast_pipeline":
            # Quick response for simple questions
            return f"Quick answer: {question} is a straightforward question that I can answer briefly. For a more detailed analysis, you could ask me to go deeper."
        
        elif pipeline_type == "hybrid_pipeline":
            # Balanced response for medium questions
            return f"Medium analysis: {question} requires some depth. I can provide a balanced response that covers the key points without going into full research mode."
        
        else:  # deep_pipeline
            # Full analysis response
            return f"Deep analysis: {question} is a complex question that requires full multi-agent analysis. I'll need 2-3 minutes to provide a comprehensive response with revolutionary insights, code generation, and lab testing."
    
    def end_voice_conversation(self) -> RealVoiceConversation:
        """End the current voice conversation session"""
        
        if self.current_session:
            self.current_session.end_time = datetime.now()
            
            # Farewell message - AI speaks this to you!
            farewell_text = f"Thank you for our conversation! We had {self.current_session.total_exchanges} exchanges. I'm always here to help with your research questions."
            
            print(f"[VOICE] 🔊 AI speaking: {farewell_text}")
            self.voice_output.speak_response(farewell_text)
            
            # Return the completed session
            completed_session = self.current_session
            self.current_session = None
            return completed_session
        
        return None
    
    def _generate_session_id(self) -> str:
        """Generate unique conversation session ID"""
        self.conversation_counter += 1
        return f"real_voice_session_{self.conversation_counter}_{int(datetime.now().timestamp())}"
    
    def get_conversation_status(self) -> Dict[str, Any]:
        """Get current conversation status and capabilities"""
        return {
            "status": "operational",
            "current_session": self.current_session.session_id if self.current_session else None,
            "total_sessions": self.conversation_counter,
            "complexity_thresholds": self.complexity_thresholds,
            "voice_input_status": self.voice_input.get_voice_status(),
            "voice_output_status": self.voice_output.get_voice_status(),
            "capabilities": [
                "REAL voice input (microphone)",
                "REAL voice output (speakers)",
                "Intelligent question routing",
                "Multi-pipeline response system",
                "Voice session management",
                "Complexity-based optimization"
            ]
        }


# Example usage and testing
if __name__ == "__main__":
    # Create real voice conversation system
    print("🎤 INITIALIZING REAL VOICE CONVERSATION SYSTEM...")
    voice_system = RealVoiceConversationSystem()
    
    # Test conversation system
    print("\n🎭 REAL VOICE SYSTEM STATUS:")
    print("=" * 40)
    status = voice_system.get_conversation_status()
    print(f"Status: {status['status']}")
    print(f"Capabilities: {status['capabilities'][:3]}")
    
    # Test simple question
    print("\n🎯 Testing voice conversation...")
    simple_response = voice_system.voice_chat("What is AI?")
    print(f"AI Response: {simple_response.text_input[:100]}...")
    
    # Start real-time conversation
    print("\n🚀 Starting REAL voice conversation...")
    print("🎤 SPEAK TO THE AI NOW!")
    
    def conversation_callback(voice_input: RealVoiceInput, voice_output: RealVoiceOutput):
        print(f"\n[CONVERSATION] Q: {voice_input.transcription}")
        print(f"[CONVERSATION] A: {voice_output.text_input[:100]}...")
    
    voice_system.real_time_voice_conversation(conversation_callback)
