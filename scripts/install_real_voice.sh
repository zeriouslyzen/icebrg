#!/bin/bash

echo "🎤 Installing True Voice Recognition for Iceberg Protocol..."
echo "=================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew not found. Installing Homebrew first..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "✅ Homebrew already installed"
fi

echo ""
echo "🔧 Installing PortAudio (required for PyAudio)..."
brew install portaudio

echo ""
echo "🐍 Installing Python voice packages..."
pip install PyAudio SpeechRecognition

echo ""
echo "🎯 Installing Whisper for local speech recognition..."
pip install openai-whisper

echo ""
echo "🔊 Installing ElevenLabs for high-quality TTS..."
pip install elevenlabs

echo ""
echo "✅ Voice recognition installation complete!"
echo ""
echo "🎤 Now you can:"
echo "   - SPEAK to the AI (real voice input)"
echo "   - HEAR the AI speak back (high-quality output)"
echo "   - Have natural voice conversations"
echo ""
echo "🚀 Run: python src/iceburg/voice/real_voice_conversation.py"
