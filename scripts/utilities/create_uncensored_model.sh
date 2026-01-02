#!/bin/bash
# Create uncensored Ollama model for ICEBURG

MODEL_NAME="llama3.1-uncensored"
BASE_MODEL="llama3.1:8b"

echo "Creating uncensored modelfile: $MODEL_NAME"

cat > /tmp/Modelfile.$MODEL_NAME << MODELFILE
FROM $BASE_MODEL

PARAMETER num_ctx 16384

# NOTE: The SYSTEM prompt here is a baseline. 
# ICEBURG agents typically override this at runtime with specific role prompts.
MODELFILE

echo "Creating model: $MODEL_NAME"
ollama create $MODEL_NAME -f /tmp/Modelfile.$MODEL_NAME

echo ""
echo "✅ Model created: $MODEL_NAME"
echo ""
echo "To use in ICEBURG, set environment variables:"
echo "export ICEBURG_SURVEYOR_MODEL=\"$MODEL_NAME\""
echo "export ICEBURG_DISSIDENT_MODEL=\"$MODEL_NAME\""
echo "export ICEBURG_SYNTHESIST_MODEL=\"$MODEL_NAME\""
echo "export ICEBURG_ORACLE_MODEL=\"$MODEL_NAME\""
