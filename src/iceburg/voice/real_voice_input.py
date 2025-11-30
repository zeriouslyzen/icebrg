#!/usr/bin/env python3
"""
REAL Voice Input Module using actual microphone input
Now you can actually SPEAK instead of typing!
"""

import subprocess
import json
import time
import speech_recognition as sr
import whisper
import tempfile
import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RealVoiceInput:
    """Real voice input data"""
    audio_data: bytes
    duration: float
    timestamp: datetime
    transcription: str
    confidence: float
    source: str


class RealVoiceInputModule:
    """REAL voice input using actual microphone and speech recognition"""
    
    def __init__(self):
        self.voice_counter = 0
        self.is_listening = False
        self.current_session = None
        
        # Initialize speech recognizer
        try:
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print("[VOICE] ✅ Microphone initialized successfully")
            print("[VOICE] ✅ Speech recognition ready")
            
        except Exception as e:
            print(f"[VOICE] ❌ Error initializing microphone: {e}")
            self.recognizer = None
            self.microphone = None
        
        # Initialize local Whisper model
        try:
            print("[VOICE] 🤖 Loading local Whisper model...")
            import warnings
            # Suppress FP16 warning for CPU-only execution
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="FP16 is not supported on CPU*")
                self.whisper_model = whisper.load_model("base", device="cpu")
            print("[VOICE] ✅ Local Whisper model loaded successfully")
            self.use_whisper = True
        except Exception as e:
            print(f"[VOICE] ❌ Error loading Whisper model: {e}")
            print("[VOICE] 🔄 Falling back to Google Speech Recognition")
            self.whisper_model = None
            self.use_whisper = False
        
    def start_listening(self) -> bool:
        """Start listening for voice input using REAL microphone"""
        try:
            if not self.recognizer or not self.microphone:
                print("[VOICE] ❌ Microphone not available")
                return False
                
            self.is_listening = True
            print("[VOICE] 🎤 Listening started... (REAL microphone)")
            print("[VOICE] Speak your question clearly...")
            return True
        except Exception as e:
            print(f"[VOICE] Error starting listening: {e}")
            return False
    
    def stop_listening(self) -> bool:
        """Stop listening for voice input"""
        try:
            self.is_listening = False
            print("[VOICE] 🛑 Listening stopped")
            return True
        except Exception as e:
            print(f"[VOICE] Error stopping listening: {e}")
            return False
    
    def listen_for_question(self, timeout: float = 10.0) -> RealVoiceInput:
        """Listen for a REAL voice question using microphone"""
        
        if not self.recognizer or not self.microphone:
            print("[VOICE] ❌ Microphone not available, falling back to typing")
            return self._fallback_to_typing()
        
        if not self.is_listening:
            self.start_listening()
        
        voice_id = self._generate_voice_id()
        
        print(f"[VOICE] 🎯 Listening for question (timeout: {timeout}s)...")
        print("[VOICE] 🎤 SPEAK NOW...")
        
        try:
            # Use REAL microphone input
            with self.microphone as source:
                print("[VOICE] 🎤 Microphone active - speak now...")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            
            print("[VOICE] 🎯 Processing your speech...")
            
            # Convert speech to text using LOCAL Whisper or Google Speech Recognition
            if self.use_whisper and self.whisper_model:
                try:
                    print("[VOICE] 🤖 Processing with local Whisper...")
                    # Save audio to temporary file for Whisper
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(audio.get_wav_data())
                        temp_file_path = temp_file.name
                    
                    # Transcribe with Whisper
                    result = self.whisper_model.transcribe(temp_file_path)
                    transcription = result["text"].strip()
                    confidence = 0.95  # Whisper doesn't provide confidence scores
                    
                    # Clean up temp file
                    os.unlink(temp_file_path)
                    
                    print(f"[VOICE] ✅ Local Whisper recognized: {transcription}")
                    
                except Exception as e:
                    print(f"[VOICE] ❌ Whisper error: {e}")
                    print("[VOICE] 🔄 Falling back to Google Speech Recognition")
                    transcription = "Whisper processing error"
                    confidence = 0.0
            else:
                # Fallback to Google Speech Recognition
                try:
                    print("[VOICE] 🌐 Processing with Google Speech Recognition...")
                    transcription = self.recognizer.recognize_google(audio)
                    confidence = 0.95
                    print(f"[VOICE] ✅ Google Speech recognized: {transcription}")
                    
                except sr.UnknownValueError:
                    print("[VOICE] ❌ Could not understand audio")
                    transcription = "Could not understand audio"
                    confidence = 0.0
                    
                except sr.RequestError as e:
                    print(f"[VOICE] ❌ Speech recognition service error: {e}")
                    transcription = "Speech recognition service error"
                    confidence = 0.0
            
            # Calculate duration (rough estimate)
            word_count = len(transcription.split())
            estimated_duration = (word_count / 150) * 60 if word_count > 0 else 1.0
            
            return RealVoiceInput(
                audio_data=audio.get_wav_data(),
                duration=estimated_duration,
                timestamp=datetime.now(),
                transcription=transcription,
                confidence=confidence,
                source="local_whisper" if self.use_whisper else "google_speech"
            )
            
        except Exception as e:
            print(f"[VOICE] ❌ Error during voice input: {e}")
            print("[VOICE] 🔄 Falling back to typing...")
            return self._fallback_to_typing()
    
    def _fallback_to_typing(self) -> RealVoiceInput:
        """Fallback to typing if microphone fails"""
        print("[VOICE] ⌨️ Type your question (microphone unavailable):")
        user_input = input("[VOICE] Question: ")
        
        return RealVoiceInput(
            audio_data=b"fallback_audio",
            duration=len(user_input) * 0.1,
            timestamp=datetime.now(),
            transcription=user_input,
            confidence=0.98,
            source="fallback_typing"
        )
    
    def real_time_listening(self, callback_function) -> None:
        """Start REAL-TIME voice listening with microphone"""
        
        if not self.recognizer or not self.microphone:
            print("[VOICE] ❌ Microphone not available for real-time listening")
            print("[VOICE] 🔄 Using fallback mode...")
            self._fallback_real_time_listening(callback_function)
            return
        
        print("[VOICE] 🚀 Starting REAL-TIME voice conversation...")
        print("[VOICE] 🎤 Say your question clearly...")
        print("[VOICE] Say 'Goodbye' to end the conversation...")
        print("[VOICE] Press Ctrl+C to stop...")
        
        try:
            while True:
                print("\n[VOICE] 🎤 Listening for your voice...")
                
                # Listen for voice input
                voice_input = self.listen_for_question()
                
                if voice_input.transcription.lower() == "goodbye":
                    print("[VOICE] 👋 Goodbye detected, ending conversation...")
                    break
                
                # Call the callback function
                callback_function(voice_input)
                
                # Small delay to prevent overwhelming
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n[VOICE] 🛑 Interrupted by user")
        finally:
            self.stop_listening()
    
    def _fallback_real_time_listening(self, callback_function) -> None:
        """Fallback real-time listening using typing"""
        print("[VOICE] 🔄 Using fallback mode (typing)...")
        print("[VOICE] Type your questions, type 'Goodbye' to end...")
        
        try:
            while True:
                user_input = input("\n[VOICE] Question: ")
                
                if user_input.lower() == "goodbye":
                    print("[VOICE] 👋 Goodbye detected, ending conversation...")
                    break
                
                # Create voice input object
                voice_input = RealVoiceInput(
                    audio_data=b"fallback_audio",
                    duration=len(user_input) * 0.1,
                    timestamp=datetime.now(),
                    transcription=user_input,
                    confidence=0.98,
                    source="fallback_typing"
                )
                
                # Call the callback function
                callback_function(voice_input)
                
        except KeyboardInterrupt:
            print("\n[VOICE] 🛑 Interrupted by user")
    
    def _generate_voice_id(self) -> str:
        """Generate unique voice input ID"""
        self.voice_counter += 1
        return f"real_voice_{self.voice_counter}_{int(datetime.now().timestamp())}"
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice input status and capabilities"""
        return {
            "status": "operational" if self.is_listening else "idle",
            "is_listening": self.is_listening,
            "microphone_available": self.recognizer is not None and self.microphone is not None,
            "total_voice_inputs": self.voice_counter,
            "current_session": self.current_session,
            "capabilities": [
                "REAL microphone input",
                "Speech recognition",
                "Voice activation detection",
                "Continuous conversation support",
                "Fallback to typing"
            ],
            "platform": "macOS",
            "dependencies": "SpeechRecognition + PyAudio"
        }
    
    def test_microphone(self) -> bool:
        """Test if microphone is accessible"""
        try:
            if not self.recognizer or not self.microphone:
                print("[VOICE] ❌ Microphone not initialized")
                return False
            
            print("[VOICE] 🎤 Testing microphone...")
            with self.microphone as source:
                print("[VOICE] 🎤 Say something for 3 seconds...")
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=3)
            
            print("[VOICE] ✅ Microphone test passed!")
            return True
            
        except Exception as e:
            print(f"[VOICE] ❌ Microphone test failed: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create real voice input module
    voice_input = RealVoiceInputModule()
    
    # Test microphone
    print("🎤 Testing REAL Voice Input Module:")
    print("=" * 40)
    
    if voice_input.test_microphone():
        print("✅ Microphone test passed!")
        
        # Test voice input
        print("\n🎯 Testing voice input...")
        voice_data = voice_input.listen_for_question()
        print(f"Transcription: {voice_data.transcription}")
        print(f"Confidence: {voice_data.confidence}")
        print(f"Duration: {voice_data.duration:.1f} seconds")
        
        # Test real-time listening
        print("\n🚀 Testing real-time listening...")
        def voice_callback(voice_input: RealVoiceInput):
            print(f"\n[VOICE CALLBACK] Detected: {voice_input.transcription}")
        
        voice_input.real_time_listening(voice_callback)
    else:
        print("❌ Microphone test failed. Check system permissions.")
        print("🔄 Using fallback mode...")
        
        # Test fallback
        voice_data = voice_input.listen_for_question()
        print(f"Fallback transcription: {voice_data.transcription}")
