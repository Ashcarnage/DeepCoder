#!/bin/bash
# Environment Setup Script for DeepCoder Project
# Run this script after mounting the persistent volume on a new RunPod

set -e  # Exit on any error

echo "🚀 Setting up DeepCoder environment..."
echo "========================================"

# 1. Check if persistent volume is mounted
if [ ! -d "/workspace/persistent" ]; then
    echo "❌ /workspace/persistent not found!"
    echo "Make sure you've mounted the persistent volume correctly."
    exit 1
fi

# 2. Check if project exists in persistent storage
if [ ! -d "/workspace/persistent/DeepCoder" ]; then
    echo "❌ /workspace/persistent/DeepCoder not found!"
    echo "Project files are missing from persistent storage."
    echo "You may need to restore from GitHub or backup."
    exit 1
fi

# 3. Create symlink to project (if not already exists)
if [ ! -L "/DeepCoder" ] && [ ! -d "/DeepCoder" ]; then
    echo "🔗 Creating symlink to persistent project..."
    ln -sf /workspace/persistent/DeepCoder /DeepCoder
elif [ -d "/DeepCoder" ] && [ ! -L "/DeepCoder" ]; then
    echo "⚠️  Regular /DeepCoder directory found. Backing up and creating symlink..."
    mv /DeepCoder /DeepCoder.backup.$(date +%s)
    ln -sf /workspace/persistent/DeepCoder /DeepCoder
fi

cd /DeepCoder

# 4. Update system packages
echo "📦 Updating system packages..."
apt-get update && apt-get install -y git curl wget

# 5. Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 6. Set up HuggingFace cache (optional - can use existing cache)
if [ ! -d "/workspace/persistent/cache" ]; then
    echo "💾 Creating persistent cache directory..."
    mkdir -p /workspace/persistent/cache/huggingface
    
    # Optionally link to persistent cache
    # ln -sf /workspace/persistent/cache/huggingface /root/.cache/huggingface
fi

# 7. Verify model files
echo "🤖 Verifying model files..."
if [ -d "/workspace/persistent/models/qwen3-30b-a3b" ]; then
    echo "✅ Qwen model found in persistent storage"
    MODEL_SIZE=$(du -sh /workspace/persistent/models/qwen3-30b-a3b | cut -f1)
    echo "   Model size: $MODEL_SIZE"
else
    echo "❌ Qwen model not found! You may need to download it."
    echo "   Run: huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /workspace/persistent/models/qwen3-30b-a3b"
fi

# 8. Environment variables setup
echo "🔧 Environment variables setup..."
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << EOF
# API Keys (set these with your actual keys)
# GROQ_API_KEY=your_groq_api_key_here
# TOGETHER_API_KEY=your_together_ai_key_here  
# HUGGINGFACE_API_KEY=your_huggingface_token_here
# WANDB_API_KEY=your_wandb_key_here
# HF_TOKEN=your_huggingface_token_here

# Model paths
QWEN_MODEL_PATH=/workspace/persistent/models/qwen3-30b-a3b
EOF
    echo "✅ Created .env template - ADD YOUR API KEYS!"
else
    echo "✅ .env file already exists"
fi

# 9. Verify installation
echo "🔍 Verifying installation..."
python scripts/check_workspace_persistence.py

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Edit .env file and add your API keys"
echo "2. Source environment: source .env"
echo "3. Test data collection: python scripts/collect_data.py --validate-setup"
echo "4. Generate synthetic data: python scripts/collect_data.py --sources synthetic --max-items 10"
echo ""
echo "💡 TIP: Get free API keys from:"
echo "   • Groq: https://console.groq.com/"
echo "   • Together AI: https://api.together.xyz/"
echo "   • HuggingFace: https://huggingface.co/settings/tokens"
echo ""
echo "🔗 Project is now symlinked from persistent storage at:"
echo "   /DeepCoder -> /workspace/persistent/DeepCoder" 