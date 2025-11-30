#!/usr/bin/env python3
"""
Real Voice Output Module using macOS built-in text-to-speech
No external dependencies required!
"""

import subprocess
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RealVoiceOutput:
    """Real voice output data"""
    text_input: str
    audio_data: bytes
    duration: float
    timestamp: datetime
    voice_model: str
    quality: str
    source: str


class RealVoiceOutputModule:
    """Real voice output using macOS built-in text-to-speech"""
    
    def __init__(self):
        self.voice_counter = 0
        self.is_speaking = False
        self.current_voice = "Alex"  # Default macOS voice
        
        # Available macOS voices
        self.available_voices = [
            "Alex", "Victoria", "Daniel", "Samantha", "Tom", "Karen",
            "Fred", "Ralph", "Nick", "Bells", "Whisper", "Trinoids"
        ]
    
    def speak_response(self, text_response: str, voice_model: str = None) -> RealVoiceOutput:
        """Convert text response to speech using macOS"""
        
        voice_id = self._generate_voice_id()
        
        # Use specified voice or default
        if voice_model and voice_model in self.available_voices:
            self.current_voice = voice_model
        
        # Calculate estimated duration (150 words per minute)
        word_count = len(text_response.split())
        estimated_duration = (word_count / 150) * 60 if word_count > 0 else 1.0
        
        print(f"[VOICE] 🔊 Speaking with voice: {self.current_voice}")
        print(f"[VOICE] 📝 Text: {text_response[:100]}{'...' if len(text_response) > 100 else ''}")
        
        # Use macOS 'say' command for text-to-speech
        try:
            self.is_speaking = True
            
            # Run the say command
            cmd = ['say', '-v', self.current_voice, text_response]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                print(f"[VOICE] ✅ Speech completed successfully")
            else:
                print(f"[VOICE] ⚠️ Speech completed with warnings: {stderr.decode()}")
            
        except Exception as e:
            print(f"[VOICE] ❌ Speech error: {e}")
        finally:
            self.is_speaking = False
        
        # Generate simulated audio data
        audio_data = self._simulate_audio_from_text(text_response)
        
        return RealVoiceOutput(
            text_input=text_response,
            audio_data=audio_data,
            duration=estimated_duration,
            timestamp=datetime.utcnow(),
            voice_model=self.current_voice,
            quality="high",
            source="macOS_text_to_speech"
        )
    
    def stream_response(self, text_response: str, chunk_size: int = 50) -> List[RealVoiceOutput]:
        """Stream long responses in chunks for real-time speaking"""
        
        # Split response into chunks
        words = text_response.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        print(f"[VOICE] 📡 Streaming response in {len(chunks)} chunks...")
        
        # Convert each chunk to speech
        voice_outputs = []
        for i, chunk in enumerate(chunks):
            print(f"[VOICE] 🎯 Speaking chunk {i+1}/{len(chunks)}...")
            voice_output = self.speak_response(chunk)
            voice_outputs.append(voice_output)
            
            # Small pause between chunks
            time.sleep(0.5)
        
        return voice_outputs
    
    def save_audio_file(self, text_response: str, file_path: str) -> RealVoiceOutput:
        """Save text-to-speech as audio file using macOS"""
        
        voice_id = self._generate_voice_output_id()
        
        print(f"[VOICE] 💾 Saving speech to: {file_path}")
        
        # Use macOS 'say' command to save audio file
        try:
            cmd = ['say', '-v', self.current_voice, '-o', file_path, text_response]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                print(f"[VOICE] ✅ Audio file saved successfully")
            else:
                print(f"[VOICE] ❌ Audio file save failed: {process.stderr}")
                
        except Exception as e:
            print(f"[VOICE] ❌ Audio file save error: {e}")
        
        # Calculate duration
        word_count = len(text_response.split())
        estimated_duration = (word_count / 150) * 60 if word_count > 0 else 1.0
        
        # Generate simulated audio data
        audio_data = self._simulate_audio_from_text(text_response)
        
        return RealVoiceOutput(
            text_input=text_response,
            audio_data=audio_data,
            duration=estimated_duration,
            timestamp=datetime.utcnow(),
            voice_model=self.current_voice,
            quality="high",
            source="macOS_audio_file"
        )
    
    def list_available_voices(self) -> List[str]:
        """List all available macOS voices"""
        try:
            cmd = ['say', '-v', '?']
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                voices = []
                for line in result.stdout.split('\n'):
                    if line.strip():
                        # Parse voice information
                        parts = line.split()
                        if len(parts) >= 2:
                            voice_name = parts[0]
                            voices.append(voice_name)
                
                return voices[:20]  # Return first 20 voices
            else:
                return self.available_voices  # Fallback to known voices
                
        except Exception as e:
            print(f"[VOICE] ❌ Error listing voices: {e}")
            return self.available_voices
    
    def test_voice(self, voice_name: str) -> bool:
        """Test if a specific voice is available"""
        try:
            cmd = ['say', '-v', voice_name, 'Hello, this is a test.']
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            return process.returncode == 0
        except Exception as e:
            print(f"[VOICE] ❌ Voice test error for {voice_name}: {e}")
            return False
    
    def _simulate_audio_from_text(self, text: str) -> bytes:
        """Simulate audio data from text"""
        # In production, this would be actual audio data
        audio_length = len(text) * 100  # Simulate audio length based on text
        return f"macos_audio_{audio_length}_bytes".encode()
    
    def _generate_voice_id(self) -> str:
        """Generate unique voice output ID"""
        self.voice_counter += 1
        return f"real_voice_out_{self.voice_counter}_{int(datetime.utcnow().timestamp())}"
    
    def _generate_voice_output_id(self) -> str:
        """Generate unique voice output ID for file operations"""
        self.voice_counter += 1
        return f"real_voice_file_{self.voice_counter}_{int(datetime.utcnow().timestamp())}"
    
    def get_voice_status(self) -> Dict[str, Any]:
        """Get current voice output status and capabilities"""
        return {
            "status": "operational" if not self.is_speaking else "speaking",
            "is_speaking": self.is_speaking,
            "current_voice": self.current_voice,
            "available_voices": self.available_voices,
            "total_voice_outputs": self.voice_counter,
            "capabilities": [
                "High-quality text-to-speech",
                "Real-time response streaming",
                "Multiple voice models",
                "Audio file export",
                "macOS native integration"
            ],
            "platform": "macOS",
            "dependencies": "None (built-in)"
        }
    
    def set_voice(self, voice_name: str) -> bool:
        """Set the voice model"""
        if voice_name in self.available_voices:
            if self.test_voice(voice_name):
                self.current_voice = voice_name
                print(f"[VOICE] ✅ Voice changed to: {voice_name}")
                return True
            else:
                print(f"[VOICE] ❌ Voice {voice_name} is not available")
                return False
        else:
            print(f"[VOICE] ❌ Voice {voice_name} not found in available voices")
            return False


# Example usage and testing
if __name__ == "__main__":
    # Create real voice output module
    voice_output = RealVoiceOutputModule()
    
    # Test voice output
    print("🔊 Testing Real Voice Output Module:")
    print("=" * 40)
    
    # List available voices
    print("🎭 Available voices:")
    available_voices = voice_output.list_available_voices()
    for i, voice in enumerate(available_voices[:5]):  # Show first 5
        print(f"  {i+1}. {voice}")
    
    # Test speaking
    print("\n🎯 Testing text-to-speech...")
    sample_text = "Hello! I am the Iceberg Protocol, an autonomous research platform. How can I assist you today?"
    voice_result = voice_output.speak_response(sample_text)
    print(f"Spoken text: {voice_result.text_input[:50]}...")
    print(f"Voice model: {voice_result.voice_model}")
    print(f"Duration: {voice_result.duration:.1f} seconds")
    
    # Test streaming
    print("\n📡 Testing response streaming...")
    long_text = "This is a longer response that demonstrates the streaming capability. The system can break down complex responses into manageable chunks and speak them in real-time, making it easier to follow along with complex explanations."
    stream_results = voice_output.stream_response(long_text, chunk_size=20)
    print(f"Streamed in {len(stream_results)} chunks")
    
    # Test file saving
    print("\n💾 Testing audio file export...")
    file_result = voice_output.save_audio_file(sample_text, "test_response.aiff")
    print(f"Saved audio file with duration: {file_result.duration:.1f} seconds")
    
    # Test voice changing
    print("\n🔄 Testing voice changing...")
    if "Victoria" in available_voices:
        voice_output.set_voice("Victoria")
        voice_output.speak_response("This is Victoria speaking. Voice change successful!")
    else:
        print("Victoria voice not available for testing")
