# 🚀 Qwen SAD System Setup Scripts

This folder contains all the scripts needed to quickly set up and manage your SAD-enhanced Qwen 30B model system after restarting your environment.

## 📋 Quick Start

After restarting your runpod/environment:

1. **Install dependencies:**
   ```bash
   cd /root/DeepCoder
   ./setup/install_requirements.sh
   ```

2. **Start the complete system:**
   ```bash
   ./setup/start_qwen_sad_system.sh
   ```

That's it! The system will:
- Start SGLang server with your trained model
- Wait for it to be ready
- Launch the interactive CLI

## 📁 Script Overview

### 🎯 Main Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `start_qwen_sad_system.sh` | **Main startup script** - Starts SGLang server and CLI | `./setup/start_qwen_sad_system.sh` |
| `stop_sglang_server.sh` | Cleanly stops the SGLang server | `./setup/stop_sglang_server.sh` |
| `check_system_status.sh` | Comprehensive system health check | `./setup/check_system_status.sh` |
| `install_requirements.sh` | Installs all Python dependencies | `./setup/install_requirements.sh` |

### 🔧 What Each Script Does

#### `start_qwen_sad_system.sh`
- ✅ Checks for existing SGLang processes and stops them
- ✅ Verifies model and CLI files exist
- ✅ Starts SGLang server with exact parameters for your setup
- ✅ Waits for server to be ready (health checks)
- ✅ Offers to launch interactive CLI
- ✅ Provides process management (PID tracking)

**SGLang Command Used:**
```bash
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
    --disable-custom-all-reduce
```

#### `stop_sglang_server.sh`
- ✅ Gracefully stops SGLang server using PID file
- ✅ Force kills if graceful shutdown fails
- ✅ Cleans up processes and frees port 30000
- ✅ Removes stale PID files

#### `check_system_status.sh`
- ✅ Checks model directory and size
- ✅ Verifies CLI files exist
- ✅ Shows SGLang server status and memory usage
- ✅ Tests server health endpoints
- ✅ Displays GPU status and utilization
- ✅ Shows recent log entries
- ✅ Provides management command suggestions

#### `install_requirements.sh`
- ✅ Updates pip to latest version
- ✅ Installs all requirements from `requirements.txt`
- ✅ Verifies critical packages (SGLang, Rich, OpenAI)
- ✅ Checks CUDA availability

## 🎮 Interactive Usage Examples

### Start Everything (Typical Workflow)
```bash
cd /root/DeepCoder
./setup/start_qwen_sad_system.sh
# Follow prompts to launch CLI
```

### Check If System Is Running
```bash
./setup/check_system_status.sh
```

### Stop Server (Keep Files)
```bash
./setup/stop_sglang_server.sh
```

### Start Just the CLI (If Server Already Running)
```bash
python interactive_qwen_cli.py
```

## 📂 File Locations

- **Model**: `/workspace/persistent/models/qwen3-30b-a3b/`
- **CLI**: `/root/DeepCoder/interactive_qwen_cli.py`
- **Setup Scripts**: `/root/DeepCoder/setup/`
- **Server Logs**: `/workspace/persistent/models/qwen3-30b-a3b/sglang_server.log`
- **PID File**: `/workspace/persistent/models/qwen3-30b-a3b/sglang_server.pid`

## 🔍 Troubleshooting

### Server Won't Start
```bash
# Check what's using port 30000
sudo lsof -i:30000

# Check for zombie processes
ps aux | grep sglang

# Force clean and restart
./setup/stop_sglang_server.sh
./setup/start_qwen_sad_system.sh
```

### Model Not Found
```bash
# Verify model location
ls -la /workspace/persistent/models/qwen3-30b-a3b/

# Check if model files are complete
du -sh /workspace/persistent/models/qwen3-30b-a3b/
```

### Dependencies Missing
```bash
# Reinstall everything
./setup/install_requirements.sh

# Check specific packages
python -c "import sglang, rich, openai; print('All imports successful')"
```

### Memory Issues
```bash
# Check GPU memory
nvidia-smi

# Check system memory
free -h

# View SGLang memory usage
./setup/check_system_status.sh
```

## 🎯 Features

- **Automated Setup**: One command to start everything
- **Health Monitoring**: Comprehensive status checks
- **Process Management**: Clean startup/shutdown with PID tracking
- **Error Handling**: Robust error detection and recovery
- **Logging**: Detailed logs for debugging
- **Interactive**: User-friendly prompts and colored output

## 🚀 Advanced Usage

### Background Server Mode
```bash
# Start server but don't launch CLI
./setup/start_qwen_sad_system.sh
# Choose 'N' when prompted for CLI

# Later, start CLI separately
python interactive_qwen_cli.py
```

### Monitor Server Logs
```bash
tail -f /workspace/persistent/models/qwen3-30b-a3b/sglang_server.log
```

### Custom Model Path
Edit the `MODEL_PATH` variable in the scripts if your model is in a different location.

## 💾 Backup Recommendation

Before shutting down your runpod, ensure these files are backed up:
- `setup/` folder (this directory)
- `interactive_qwen_cli.py`
- `requirements.txt`
- Any training results or LoRA weights

## 🎉 Success Indicators

When everything is working correctly, you should see:
- ✅ SGLang server running on port 30000
- ✅ Health checks responding
- ✅ Interactive CLI launches successfully
- ✅ Model generates responses with SAD capabilities

---

*This setup ensures your SAD-enhanced Qwen 30B system is ready to use immediately after environment restart!* 