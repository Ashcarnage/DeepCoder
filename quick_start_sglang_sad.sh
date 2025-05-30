#!/bin/bash

# ==============================================================================
# Quick Start SGLang SAD Training Launcher
# ==============================================================================

echo "🚀 Quick Start: SGLang + SAD Training for Qwen 30B MOE"
echo "====================================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Ensure we're in the right directory
cd /workspace/persistent

echo -e "${BLUE}[INFO]${NC} Current directory: $(pwd)"
echo -e "${BLUE}[INFO]${NC} Checking requirements..."

# Check if main files exist
if [[ ! -f "fast_sglang_sad_trainer.py" ]]; then
    echo -e "${YELLOW}[WARNING]${NC} Main trainer not found, copying from DeepCoder..."
    cp DeepCoder/fast_sglang_sad_trainer.py .
fi

if [[ ! -f "setup_sglang_sad_training.sh" ]]; then
    echo -e "${YELLOW}[WARNING]${NC} Setup script not found, copying from DeepCoder..."
    cp DeepCoder/setup_sglang_sad_training.sh .
    chmod +x setup_sglang_sad_training.sh
fi

# Quick GPU check
echo -e "${BLUE}[INFO]${NC} GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo -e "${GREEN}[READY]${NC} Starting automatic SGLang + SAD training..."
echo -e "${BLUE}[INFO]${NC} This will:"
echo "   1. Setup the environment"
echo "   2. Start SGLang server with Qwen 30B MOE"
echo "   3. Run SAD training with real-time plots"
echo ""

# Ask for confirmation
read -p "Continue with automatic training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo -e "${GREEN}[STARTING]${NC} Automatic SGLang + SAD Training..."

# Run the full setup and training
./setup_sglang_sad_training.sh all

echo ""
echo -e "${GREEN}[COMPLETE]${NC} Training finished!"
echo ""
echo "📊 Check results in:"
echo "   - training_plots/ (visualization)"
echo "   - trained_models/ (metrics & adapters)"
echo ""
echo "🌐 SGLang server: http://localhost:30000"
echo "🎯 To run training again: ./quick_start_sglang_sad.sh" 