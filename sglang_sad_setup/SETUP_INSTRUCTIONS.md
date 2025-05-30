# SGLang + SAD Training Setup Instructions

## Overview
This directory contains all essential files for setting up and running Structured Agent Distillation (SAD) training using SGLang with the Qwen 30B MOE model.

## Quick Start (Recommended)
```bash
# One-click setup and launch
./quick_start_sglang_sad.sh
```

## Manual Setup
```bash
# Full automated setup with all components
./setup_sglang_sad_training.sh all

# Or specific components:
./setup_sglang_sad_training.sh dependencies    # Install dependencies only
./setup_sglang_sad_training.sh sglang         # Setup SGLang only
./setup_sglang_sad_training.sh training       # Setup training environment only
```

## Files in This Directory

### 1. `quick_start_sglang_sad.sh` 
- **Purpose**: One-click launcher for existing setups
- **Usage**: Run when SGLang and dependencies are already installed
- **Features**:
  - Checks system readiness
  - Starts SGLang server with optimized configuration
  - Launches training with real-time visualization
  - Handles graceful shutdown

### 2. `setup_sglang_sad_training.sh`
- **Purpose**: Complete environment setup from scratch
- **Usage**: Run on fresh systems or for full reinstallation
- **Features**:
  - Installs all dependencies (PyTorch, SGLang, LoRA adapters)
  - Downloads and validates model files
  - Sets up training data paths
  - Configures optimal memory settings
  - Creates backup and recovery points

### 3. `README_SGLang_SAD_Training.md`
- **Purpose**: Technical documentation and troubleshooting
- **Contents**: Detailed configuration parameters, known issues, performance tuning

## Prerequisites

### System Requirements
- **GPU**: NVIDIA GPU with 48GB+ VRAM (RTX 4090, A100, H100)
- **RAM**: 64GB+ system memory recommended
- **Storage**: 100GB+ free space for models and training data
- **OS**: Linux with CUDA 11.8+ support

### Environment Variables (Required)
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export HF_TOKEN="your_huggingface_token_here"  # For model downloads
```

## Training Configuration

### Model Setup
- **Teacher Model**: DeepSeek R1 Distill Llama 70B (via Groq API)
- **Student Model**: Local Qwen 30B MOE (`/workspace/persistent/models/qwen3-30b-a3b`)
- **LoRA Config**: rank=16, alpha=32, dropout=0.1
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Parameters
- **SAD Loss Weights**: reasoning=2.0x, action=1.5x, base=1.0x
- **Batch Size**: 1 (due to memory constraints)
- **Learning Rate**: 5e-5 with cosine scheduling
- **Context Length**: 32,768 tokens
- **Training Steps**: 30 (configurable)

### SGLang Server Configuration
```bash
--model-path /workspace/persistent/models/qwen3-30b-a3b
--mem-fraction-static 0.85
--max-total-tokens 65536
--max-prefill-tokens 32768
--chunked-prefill-size 8192
--reasoning-parser qwen3
--context-length 32768
--port 30000
```

## Expected Performance

### Training Metrics (Last Successful Run)
- **Training Steps**: 30 completed across 3 epochs
- **Average Response Time**: 4.14 seconds
- **SGLang Throughput**: 180+ tokens/second
- **Memory Usage**: ~85% GPU memory utilization
- **Total Training Time**: ~45 minutes

### Output Files Generated
- **Model Checkpoints**: `./results/sad_training_*.pth`
- **LoRA Adapters**: `./results/lora_adapter_*.bin`
- **Training Plots**: `./plots/sad_training_results_*.png`
- **Detailed Logs**: `./logs/sad_training_*.log`
- **Performance Metrics**: `./results/sad_detailed_results_*.json`

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   ```bash
   # Reduce memory fraction
   # Edit setup script: mem-fraction-static from 0.85 to 0.7
   ```

2. **"SGLang server not responding"**
   ```bash
   # Check server status
   curl http://localhost:30000/v1/models
   
   # Restart server
   pkill -f "python.*sglang"
   ./quick_start_sglang_sad.sh
   ```

3. **"Training data not found"**
   ```bash
   # Verify data path
   ls -la /DeepCoder/data/training_data/agentic_train.jsonl
   ```

4. **"Groq API rate limit exceeded"**
   ```bash
   # Check rate limits: 30 req/min, 6K tokens/min
   # Training automatically handles rate limiting
   ```

### Performance Optimization

1. **Increase Throughput**:
   - Use `--chunked-prefill-size 8192` for better batching
   - Enable `--cuda-graph` for faster inference
   - Set optimal `--mem-fraction-static` (0.8-0.9)

2. **Reduce Memory Usage**:
   - Lower `--max-total-tokens` if needed
   - Use gradient checkpointing in training
   - Enable mixed precision training

3. **Faster Training**:
   - Increase batch size if memory allows
   - Use multiple GPU setup with tensor parallelism
   - Pre-cache teacher responses to reduce API calls

## Recovery and Backup

### Automatic Backup Points
The setup script creates backups at:
- Model checkpoints: Every 5 training steps
- Configuration files: Before each run
- Training logs: Timestamped for each session

### Manual Recovery
```bash
# Restore from backup
cp ./results/backup_* ./results/
./quick_start_sglang_sad.sh
```

## Advanced Configuration

### Custom Training Data
Replace `/DeepCoder/data/training_data/agentic_train.jsonl` with your own data in the format:
```json
{"content": {"turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}}
```

### Multi-GPU Setup
```bash
# Edit setup script for tensor parallelism
--tp-size 2  # For 2 GPUs
--port 30000,30001
```

### Custom LoRA Configuration
Modify the training script to adjust:
- `lora_rank`: 8, 16, 32, 64
- `lora_alpha`: 16, 32, 64
- `target_modules`: Add/remove model layers

## Support and Monitoring

### Real-time Monitoring
Training provides live visualization of:
- Loss curves (total, reasoning, action)
- Response time distribution  
- Token throughput metrics
- Memory utilization

### Logs and Debugging
- **SGLang Server Logs**: `./logs/sglang_server_*.log`
- **Training Logs**: `./logs/sad_training_*.log`
- **Error Logs**: `./logs/error_*.log`

## Success Validation

### Verify Successful Setup
1. SGLang server responding: `curl http://localhost:30000/v1/models`
2. Model loaded correctly: Check server logs for "Model loaded successfully"
3. Training data accessible: Verify JSONL file exists and is readable
4. GPU memory available: `nvidia-smi` shows sufficient free memory

### Verify Successful Training
1. Training completes all steps without errors
2. Loss decreases over training steps
3. Generated responses show agentic reasoning patterns
4. Model checkpoints and plots are created

---

**Last Updated**: May 30, 2025  
**Version**: 1.0  
**Location**: `/DeepCoder/sglang_sad_setup/`
**Tested Configuration**: NVIDIA RTX 4090, 64GB RAM, Ubuntu 22.04 