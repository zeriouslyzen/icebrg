#!/bin/bash
# Start ICEBURG in Autonomous Mode

echo "Starting ICEBURG in Autonomous Mode..."
echo "======================================"

# Set environment variables
export ICEBURG_AUTONOMOUS_LEARNING=true
export ICEBURG_ENHANCED_SWARM=true
export ICEBURG_SCIENTIFIC_RESEARCH=true
export ICEBURG_HYPOTHESIS_TESTING=true
export ICEBURG_EXPERIMENT_DESIGN=true
export ICEBURG_DATA_ANALYSIS=true
export ICEBURG_SCIENTIFIC_VALIDATION=true
export ICEBURG_VISUAL_GENERATION=true
export ICEBURG_CODE_GENERATION=true
export ICEBURG_APP_GENERATION=true

# Start ICEBURG
python3 -m iceburg.autonomous.research_orchestrator

echo "ICEBURG autonomous mode started!"

