#!/bin/bash
# 🚀 Complete Qwen SAD System Startup Script
# ==========================================
# 
# This script will:
# 1. Start SGLang server with your trained Qwen 30B model
# 2. Wait for server to be ready
# 3. Launch the interactive CLI
#
# Usage: ./start_qwen_sad_system.sh

set -e  # Exit on any error

# Configuration
MODEL_PATH="/workspace/persistent/models/qwen3-30b-a3b"
CLI_PATH="/root/DeepCoder"
SGLANG_PORT=30000
SGLANG_HOST="localhost"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}🚀 Starting Qwen SAD System${NC}"
echo -e "${CYAN}================================${NC}"
echo ""

# Function to check if SGLang server is running
check_server() {
    curl -s http://${SGLANG_HOST}:${SGLANG_PORT}/health > /dev/null 2>&1
}

# Function to wait for server
wait_for_server() {
    echo -e "${YELLOW}⏳ Waiting for SGLang server to be ready...${NC}"
    local timeout=300  # 5 minutes timeout
    local count=0
    
    while ! check_server; do
        if [ $count -ge $timeout ]; then
            echo -e "${RED}❌ Timeout waiting for server to start${NC}"
            exit 1
        fi
        
        if [ $((count % 10)) -eq 0 ]; then
            echo -e "${CYAN}   Still waiting... (${count}s elapsed)${NC}"
        fi
        
        sleep 1
        ((count++))
    done
    
    echo -e "${GREEN}✅ SGLang server is ready!${NC}"
}

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model directory not found: $MODEL_PATH${NC}"
    echo -e "${YELLOW}💡 Please ensure the Qwen model is downloaded to the correct location${NC}"
    exit 1
fi

# Check if interactive CLI exists
if [ ! -f "$CLI_PATH/interactive_qwen_cli.py" ]; then
    echo -e "${RED}❌ Interactive CLI not found: $CLI_PATH/interactive_qwen_cli.py${NC}"
    echo -e "${YELLOW}💡 Please ensure you're running this from the DeepCoder directory${NC}"
    exit 1
fi

# Kill any existing SGLang processes
echo -e "${YELLOW}🔍 Checking for existing SGLang processes...${NC}"
if pgrep -f "sglang.launch_server" > /dev/null; then
    echo -e "${YELLOW}⚠️  Found existing SGLang processes. Stopping them...${NC}"
    pkill -f "sglang.launch_server" || true
    sleep 3
fi

# Check if port is available
if lsof -i:${SGLANG_PORT} > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port ${SGLANG_PORT} is in use. Attempting to free it...${NC}"
    lsof -ti:${SGLANG_PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo -e "${BLUE}📂 Model Path: ${MODEL_PATH}${NC}"
echo -e "${BLUE}🌐 Server: ${SGLANG_HOST}:${SGLANG_PORT}${NC}"
echo -e "${BLUE}💻 CLI Path: ${CLI_PATH}${NC}"
echo ""

# Change to model directory
echo -e "${CYAN}📁 Changing to model directory...${NC}"
cd "$MODEL_PATH"

# Start SGLang server in background
echo -e "${CYAN}🔥 Starting SGLang server with SAD-trained Qwen 30B...${NC}"
echo -e "${YELLOW}Command: python -m sglang.launch_server --model-path . --host 0.0.0.0 --port 30000 --mem-fraction-static 0.85 --max-total-tokens 65536 --max-prefill-tokens 32768 --chunked-prefill-size 8192 --max-running-requests 2048 --context-length 32768 --reasoning-parser qwen3 --trust-remote-code --served-model-name qwen3-30b-a3b --disable-custom-all-reduce${NC}"
echo ""

# Start the server in background
nohup python -m sglang.launch_server \
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
    --disable-custom-all-reduce \
    > sglang_server.log 2>&1 &

# Store the PID
SGLANG_PID=$!
echo $SGLANG_PID > sglang_server.pid
echo -e "${GREEN}✅ SGLang server started with PID: $SGLANG_PID${NC}"
echo -e "${CYAN}📋 Logs: $MODEL_PATH/sglang_server.log${NC}"

# Wait for server to be ready
wait_for_server

# Change back to CLI directory
cd "$CLI_PATH"

# Show system status
echo ""
echo -e "${BOLD}${GREEN}🎉 System Ready!${NC}"
echo -e "${CYAN}================================${NC}"
echo -e "${GREEN}✅ SGLang Server: Running on ${SGLANG_HOST}:${SGLANG_PORT}${NC}"
echo -e "${GREEN}✅ SAD-Enhanced Qwen 30B: Loaded${NC}"
echo -e "${GREEN}✅ Interactive CLI: Ready to launch${NC}"
echo ""

# Give option to start CLI
echo -e "${YELLOW}🎯 Ready to launch interactive CLI?${NC}"
echo -e "${CYAN}   [Y] Start CLI now${NC}"
echo -e "${CYAN}   [N] Exit (server will keep running)${NC}"
echo -e "${CYAN}   [S] Show server status${NC}"

read -p "Choice [Y/n/s]: " choice
case ${choice,,} in  # Convert to lowercase
    s)
        echo ""
        echo -e "${BLUE}📊 Server Status:${NC}"
        echo -e "${CYAN}PID: $(cat $MODEL_PATH/sglang_server.pid 2>/dev/null || echo 'Unknown')${NC}"
        echo -e "${CYAN}Health: $(curl -s http://${SGLANG_HOST}:${SGLANG_PORT}/health 2>/dev/null && echo 'OK' || echo 'Not responding')${NC}"
        echo -e "${CYAN}Logs: tail -f $MODEL_PATH/sglang_server.log${NC}"
        ;;
    n)
        echo -e "${YELLOW}🚪 Exiting. SGLang server will continue running.${NC}"
        echo -e "${CYAN}💡 To start CLI later: cd $CLI_PATH && python interactive_qwen_cli.py${NC}"
        echo -e "${CYAN}💡 To stop server: kill $(cat $MODEL_PATH/sglang_server.pid 2>/dev/null || echo 'PID_NOT_FOUND')${NC}"
        ;;
    *)
        echo ""
        echo -e "${BOLD}${BLUE}🎮 Launching Interactive Qwen CLI...${NC}"
        echo -e "${CYAN}================================${NC}"
        sleep 1
        python interactive_qwen_cli.py
        ;;
esac

echo ""
echo -e "${BOLD}${GREEN}🎊 Session Complete!${NC}" 