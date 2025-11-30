#!/bin/bash
# Quick deletion of unused ICEBURG models

echo "🗑️  ICEBURG Model Cleanup"
echo "=========================="
echo ""
echo "This will delete unused models to save disk space."
echo ""
echo "Models to delete:"
echo "  • glm-4.6:cloud (Cloud model, not local)"
echo "  • bakllava:7b (4.7 GB - Large vision model)"
echo "  • codellama:7b-instruct (3.8 GB - Duplicate)"
echo "  • gemma:7b (5.0 GB - Alternative model)"
echo "  • yi:6b (3.5 GB - Alternative model)"
echo ""
echo "Estimated space savings: ~17 GB"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Deleting models..."

# Delete cloud model
echo "Deleting glm-4.6:cloud..."
ollama rm glm-4.6:cloud 2>&1 | grep -v "NotOpenSSLWarning" || echo "  (Already deleted or not found)"

# Delete unused models
echo "Deleting bakllava:7b..."
ollama rm bakllava:7b 2>&1 | grep -v "NotOpenSSLWarning" || echo "  (Already deleted or not found)"

echo "Deleting codellama:7b-instruct..."
ollama rm codellama:7b-instruct 2>&1 | grep -v "NotOpenSSLWarning" || echo "  (Already deleted or not found)"

echo "Deleting gemma:7b..."
ollama rm gemma:7b 2>&1 | grep -v "NotOpenSSLWarning" || echo "  (Already deleted or not found)"

echo "Deleting yi:6b..."
ollama rm yi:6b 2>&1 | grep -v "NotOpenSSLWarning" || echo "  (Already deleted or not found)"

echo ""
echo "✅ Deletion complete!"
echo ""
echo "Remaining models:"
ollama list 2>&1 | grep -v "NotOpenSSLWarning" | head -20

