#!/bin/bash
# Test Astro-Physiology Engine in "REAL MODE"
# Injects dummy keys to trigger the "Real Data" code paths in ingestion.py

echo "🧪 TESTING ASTRO-PHYSIOLOGY ENGINE (REAL MODE)..."
echo "------------------------------------------------"

# Export Dummy Keys to trigger "If api_key:" checks
export ICEBURG_NOAA_API_KEY="test_noaa_key_123"
export ICEBURG_NCDC_TOKEN="qkJWRWRijPXrbBDxSaWkkdHkwsKSHbcP"
export ICEBURG_OURA_TOKEN="test_oura_token_456"

# Run the verification script
python3 verify_astro_engine.py
