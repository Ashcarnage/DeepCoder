#!/bin/bash

# ==============================================================================
# SGLang + SAD Training Environment Setup Script
# Recreates the complete environment for Qwen 30B MOE with SGLang and SAD training
# ==============================================================================

set -e  # Exit on any error

echo "🚀 Starting SGLang + SAD Training Environment Setup..."
echo "========================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ==============================================================================
# 1. SYSTEM REQUIREMENTS CHECK
# ==============================================================================

log_info "Checking system requirements..."

# Check if we're in the right directory
if [[ ! -d "/workspace/persistent" ]]; then
    log_error "Must be run from /workspace/persistent directory"
    exit 1
fi

cd /workspace/persistent

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    log_error "NVIDIA GPU not detected! SGLang requires GPU."
    exit 1
fi

log_success "System requirements met"

# ==============================================================================
# 2. PYTHON ENVIRONMENT SETUP
# ==============================================================================

log_info "Setting up Python environment..."

# Install required packages
pip install --upgrade pip

# Core ML packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# SGLang installation
pip install "sglang[all]"

# Training dependencies
pip install transformers datasets accelerate bitsandbytes
pip install peft lora

# API clients
pip install groq openai

# Visualization
pip install matplotlib numpy pandas tqdm

# Data processing
pip install jsonlines requests

log_success "Python environment configured"

# ==============================================================================
# 3. VERIFY MODEL FILES
# ==============================================================================

log_info "Verifying Qwen 30B MOE model files..."

QWEN_MODEL_PATH="/workspace/persistent/models/qwen3-30b-a3b"

if [[ ! -d "$QWEN_MODEL_PATH" ]]; then
    log_error "Qwen 30B model not found at $QWEN_MODEL_PATH"
    log_info "Please ensure the model is downloaded to the persistent workspace"
    exit 1
fi

# Check key model files
MODEL_FILES=("config.json" "tokenizer.json" "model-00001-of-00016.safetensors")
for file in "${MODEL_FILES[@]}"; do
    if [[ ! -f "$QWEN_MODEL_PATH/$file" ]]; then
        log_error "Missing model file: $file"
        exit 1
    fi
done

log_success "Qwen 30B MOE model files verified"

# ==============================================================================
# 4. ENVIRONMENT VARIABLES
# ==============================================================================

log_info "Setting up environment variables..."

# Export CUDA settings for optimal performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# SGLang optimization settings
export SGLANG_DISABLE_DISK_CACHE=1
export SGLANG_ENABLE_TORCH_COMPILE=1

log_success "Environment variables configured"

# ==============================================================================
# 5. CREATE DIRECTORY STRUCTURE
# ==============================================================================

log_info "Creating directory structure..."

mkdir -p /workspace/persistent/training_plots
mkdir -p /workspace/persistent/trained_models
mkdir -p /workspace/persistent/logs
mkdir -p /workspace/persistent/scripts
mkdir -p /workspace/persistent/configs

log_success "Directory structure created"

# ==============================================================================
# 6. COPY TRAINING SCRIPTS
# ==============================================================================

log_info "Verifying training scripts..."

REQUIRED_SCRIPTS=(
    "fast_sglang_sad_trainer.py"
    "optimized_sglang_sad_trainer.py"
    "production_sad_trainer.py"
)

for script in "${REQUIRED_SCRIPTS[@]}"; do
    if [[ ! -f "/workspace/persistent/$script" ]]; then
        log_error "Missing training script: $script"
        exit 1
    fi
done

log_success "Training scripts verified"

# ==============================================================================
# 7. START SGLANG SERVER FUNCTION
# ==============================================================================

start_sglang_server() {
    log_info "Starting optimized SGLang server for Qwen 30B MOE..."
    
    # Kill any existing SGLang processes
    pkill -f sglang || true
    sleep 5
    
    cd "$QWEN_MODEL_PATH"
    
    # Start SGLang with optimized settings
    python -m sglang.launch_server \
        --model-path . \
        --host 0.0.0.0 \
        --port 30000 \
        --mem-fraction-static 0.85 \
        --max-total-tokens 65536 \
        --max-prefill-tokens 32768 \
        --chunked-prefill-size 8192 \
        --max-running-requests 2048 \
        --context-length 32768 \
        --reasoning-parser qwen3 \
        --trust-remote-code \
        --served-model-name qwen3-30b-a3b \
        --disable-custom-all-reduce &
    
    SGLANG_PID=$!
    
    log_info "SGLang server starting with PID: $SGLANG_PID"
    log_info "Waiting for server to be ready..."
    
    # Wait for server to be ready
    for i in {1..120}; do
        if curl -s http://localhost:30000/health > /dev/null 2>&1; then
            log_success "SGLang server is ready!"
            return 0
        fi
        echo -n "."
        sleep 5
    done
    
    log_error "SGLang server failed to start within 10 minutes"
    return 1
}

# ==============================================================================
# 8. RUN SAD TRAINING FUNCTION
# ==============================================================================

run_sad_training() {
    log_info "Starting SAD training with real-time visualization..."
    
    # Set Groq API key (you'll need to update this)
    export GROQ_API_KEY="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"
    
    cd /workspace/persistent
    
    # Run the fast SGLang SAD trainer
    python fast_sglang_sad_trainer.py
    
    log_success "SAD training completed!"
}

# ==============================================================================
# 9. MAIN EXECUTION
# ==============================================================================

main() {
    case "${1:-all}" in
        "setup")
            log_info "Running setup only..."
            ;;
        "sglang")
            log_info "Starting SGLang server only..."
            start_sglang_server
            ;;
        "train")
            log_info "Running training only (assumes SGLang is running)..."
            run_sad_training
            ;;
        "all")
            log_info "Running complete setup and training..."
            start_sglang_server
            sleep 30  # Give SGLang time to fully initialize
            run_sad_training
            ;;
        "stop")
            log_info "Stopping SGLang server..."
            pkill -f sglang || true
            log_success "SGLang server stopped"
            ;;
        *)
            echo "Usage: $0 {setup|sglang|train|all|stop}"
            echo "  setup  - Setup environment only"
            echo "  sglang - Start SGLang server only"
            echo "  train  - Run SAD training only"
            echo "  all    - Complete setup and training (default)"
            echo "  stop   - Stop SGLang server"
            exit 1
            ;;
    esac
}

# ==============================================================================
# 10. FINAL INSTRUCTIONS
# ==============================================================================

show_instructions() {
    echo ""
    log_success "Setup completed! Here's how to use the system:"
    echo ""
    echo "🚀 Quick Start:"
    echo "   ./setup_sglang_sad_training.sh all    # Complete setup and training"
    echo ""
    echo "📊 Individual Components:"
    echo "   ./setup_sglang_sad_training.sh setup  # Setup environment only"
    echo "   ./setup_sglang_sad_training.sh sglang # Start SGLang server"
    echo "   ./setup_sglang_sad_training.sh train  # Run SAD training"
    echo "   ./setup_sglang_sad_training.sh stop   # Stop SGLang server"
    echo ""
    echo "📁 Important Files:"
    echo "   - Model: /workspace/persistent/models/qwen3-30b-a3b/"
    echo "   - Trainer: /workspace/persistent/fast_sglang_sad_trainer.py"
    echo "   - Results: /workspace/persistent/training_plots/"
    echo "   - Logs: /workspace/persistent/trained_models/"
    echo ""
    echo "🌐 SGLang Server: http://localhost:30000"
    echo "📈 Training will generate real-time plots every 5 steps"
    echo ""
    log_warning "Remember to update GROQ_API_KEY in the script!"
}

# Run main function
main "$@"

# Show instructions if running complete setup
if [[ "${1:-all}" == "all" ]] || [[ "${1:-all}" == "setup" ]]; then
    show_instructions
fi

log_success "🎉 SGLang + SAD Training Environment Ready!" 