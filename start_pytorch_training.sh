#!/bin/bash

echo "🚀 Starting PyTorch Agentic Training Setup"
echo "=========================================="

# Install PyTorch dependencies
echo "📦 Installing PyTorch dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate datasets tokenizers
pip install peft bitsandbytes
pip install plotly matplotlib seaborn pandas
pip install groq requests python-dotenv
pip install jsonlines rich fire

echo "✅ Dependencies installed!"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p checkpoints/pytorch_training
mkdir -p logs/pytorch_training
mkdir -p plots/training

echo "✅ Directories created!"
echo ""

# Check GPU availability
echo "🔍 Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Start training
echo "🚀 Starting PyTorch Agentic Training..."
echo "======================================"
python3 scripts/pytorch_agentic_trainer.py 