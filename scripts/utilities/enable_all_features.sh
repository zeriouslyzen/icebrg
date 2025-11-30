#!/bin/bash
# Enable all ICEBURG features

echo "🚀 Enabling all ICEBURG features..."
echo ""

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    touch .env
fi

# Enable features in .env
echo "Setting environment variables..."
grep -q "ICEBURG_ENABLE_SOFTWARE_LAB" .env && \
    sed -i '' 's/ICEBURG_ENABLE_SOFTWARE_LAB=.*/ICEBURG_ENABLE_SOFTWARE_LAB=1/' .env || \
    echo "ICEBURG_ENABLE_SOFTWARE_LAB=1" >> .env

grep -q "ICEBURG_ENABLE_CODE_GENERATION" .env && \
    sed -i '' 's/ICEBURG_ENABLE_CODE_GENERATION=.*/ICEBURG_ENABLE_CODE_GENERATION=1/' .env || \
    echo "ICEBURG_ENABLE_CODE_GENERATION=1" >> .env

grep -q "ICEBURG_ENABLE_WEB" .env && \
    sed -i '' 's/ICEBURG_ENABLE_WEB=.*/ICEBURG_ENABLE_WEB=1/' .env || \
    echo "ICEBURG_ENABLE_WEB=1" >> .env

# Enable protocol features via environment
export ICEBURG_PROTOCOL_BLOCKCHAIN=1
export ICEBURG_PROTOCOL_MULTIMODAL=1
export ICEBURG_PROTOCOL_VISUAL=1

echo "✅ Features enabled:"
echo "   • Software Lab: ENABLED"
echo "   • Code Generation: ENABLED"
echo "   • Web Search: ENABLED"
echo "   • Blockchain Verification: ENABLED (via protocol config)"
echo "   • Multimodal Processing: ENABLED (via protocol config)"
echo "   • Visual Generation: ENABLED (via protocol config)"
echo ""
echo "📝 Note: Protocol features (blockchain, multimodal, visual) are enabled in code."
echo "   Environment variables are set for this session."
echo ""
echo "✅ All features enabled!"

