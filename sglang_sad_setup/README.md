# SGLang + SAD Training Complete Setup Package

## 🚀 Quick Start
```bash
# One-click launch (if already set up)
./quick_start_sglang_sad.sh

# Full setup from scratch
./setup_sglang_sad_training.sh all
```

## 📁 Package Contents

### Core Files
- **`setup_sglang_sad_training.sh`** - Complete automated setup script (9.4KB)
- **`quick_start_sglang_sad.sh`** - One-click launcher for existing setups (2.1KB)
- **`fast_sglang_sad_trainer.py`** - Main training script with real-time visualization (25.8KB)

### Documentation
- **`SETUP_INSTRUCTIONS.md`** - Comprehensive setup and deployment guide (6.7KB)
- **`README_SGLang_SAD_Training.md`** - Technical documentation and troubleshooting (6.7KB)
- **`README.md`** - This overview file

## 🎯 What This Package Does

### Structured Agent Distillation (SAD) Training
- **Teacher Model**: DeepSeek R1 Distill Llama 70B (via Groq API)
- **Student Model**: Local Qwen 30B MOE 
- **Method**: LoRA fine-tuning with specialized loss weighting
- **Output**: Agentic reasoning capabilities transferred to local model

### Key Features
- ✅ **Automated Setup**: One-command installation of all dependencies
- ✅ **Optimized Configuration**: Pre-tuned SGLang server settings for 30B MOE
- ✅ **Real-time Visualization**: Live training plots and metrics
- ✅ **Rate Limit Handling**: Automatic Groq API throttling
- ✅ **Backup & Recovery**: Comprehensive checkpoint system
- ✅ **Production Ready**: Tested on RTX 4090 with 64GB RAM

## 📊 Proven Results

### Last Successful Training Run
- **30 training steps** completed across 3 epochs
- **4.14s average response time** 
- **180+ tokens/s throughput**
- **Agentic reasoning** successfully transferred

### Generated Output Example
```json
{
  "reasoning_content": "Okay, I need to write a Python function to find the maximum element in a list. Let me think about how to approach this..."
}
```

## 🔧 System Requirements

### Minimum Hardware
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) or equivalent
- **RAM**: 64GB system memory
- **Storage**: 100GB free space
- **OS**: Linux with CUDA 11.8+

### Required Environment Variables
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export HF_TOKEN="your_huggingface_token_here"
```

## 📈 Performance Metrics

### Training Configuration
- **Context Length**: 32,768 tokens
- **LoRA Rank**: 16 (alpha=32, dropout=0.1)
- **SAD Loss Weights**: reasoning=2.0x, action=1.5x, base=1.0x
- **Memory Usage**: ~85% GPU utilization
- **Training Time**: ~45 minutes for 30 steps

### SGLang Server Optimization
- **Memory Fraction**: 0.85 (static allocation)
- **Max Total Tokens**: 65,536
- **Chunked Prefill**: 8,192 tokens
- **CUDA Graphs**: Enabled for 120+ tokens/s

## 🛠️ Usage Instructions

### First Time Setup
1. **Set Environment Variables**:
   ```bash
   export GROQ_API_KEY="your_key"
   export HF_TOKEN="your_token"
   ```

2. **Run Full Setup**:
   ```bash
   chmod +x setup_sglang_sad_training.sh
   ./setup_sglang_sad_training.sh all
   ```

3. **Start Training**:
   ```bash
   ./quick_start_sglang_sad.sh
   ```

### Subsequent Runs
```bash
# Just run the quick start script
./quick_start_sglang_sad.sh
```

## 📋 Output Files

### Training Results
- `./results/sad_training_*.pth` - Model checkpoints
- `./results/lora_adapter_*.bin` - LoRA adapters
- `./plots/sad_training_results_*.png` - Training visualizations
- `./results/sad_detailed_results_*.json` - Performance metrics

### Logs
- `./logs/sglang_server_*.log` - SGLang server logs
- `./logs/sad_training_*.log` - Training process logs

## 🔍 Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce `mem-fraction-static` from 0.85 to 0.7
2. **Server Not Responding**: Check `curl http://localhost:30000/v1/models`
3. **Training Data Missing**: Verify `/workspace/persistent/models/qwen3-30b-a3b`
4. **Rate Limits**: Training handles Groq API limits automatically

### Performance Tuning
- **Increase Throughput**: Adjust `chunked-prefill-size`
- **Reduce Memory**: Lower `max-total-tokens`
- **Faster Training**: Enable gradient checkpointing

## 🎯 Success Validation

### Setup Verification
- [ ] SGLang server responds to health check
- [ ] Model loads without CUDA OOM errors
- [ ] Training data file exists and is readable
- [ ] Environment variables are set

### Training Verification
- [ ] All 30 training steps complete
- [ ] Loss decreases over time
- [ ] Real-time plots generate successfully
- [ ] Model outputs show agentic reasoning patterns

## 📚 Additional Resources

- **Detailed Setup**: See `SETUP_INSTRUCTIONS.md`
- **Technical Docs**: See `README_SGLang_SAD_Training.md`
- **Training Script**: See `fast_sglang_sad_trainer.py`

---

**Package Version**: 1.0  
**Last Updated**: May 30, 2025  
**Tested Environment**: NVIDIA RTX 4090, 64GB RAM, Ubuntu 22.04  
**Location**: `/DeepCoder/sglang_sad_setup/` 