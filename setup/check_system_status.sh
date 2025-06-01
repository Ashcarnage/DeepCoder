#!/bin/bash
# 📊 System Status Check Script
# =============================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

MODEL_PATH="/workspace/persistent/models/qwen3-30b-a3b"
CLI_PATH="/root/DeepCoder"
SGLANG_PORT=30000
SGLANG_HOST="localhost"
PID_FILE="$MODEL_PATH/sglang_server.pid"

echo -e "${BOLD}${BLUE}📊 Qwen SAD System Status${NC}"
echo -e "${CYAN}==========================${NC}"
echo ""

# Check model directory
echo -e "${BOLD}📂 Model Directory:${NC}"
if [ -d "$MODEL_PATH" ]; then
    echo -e "   ${GREEN}✅ Model directory exists: $MODEL_PATH${NC}"
    MODEL_SIZE=$(du -sh "$MODEL_PATH" 2>/dev/null | cut -f1)
    echo -e "   ${CYAN}📏 Size: $MODEL_SIZE${NC}"
else
    echo -e "   ${RED}❌ Model directory not found: $MODEL_PATH${NC}"
fi
echo ""

# Check CLI files
echo -e "${BOLD}💻 CLI Files:${NC}"
if [ -f "$CLI_PATH/interactive_qwen_cli.py" ]; then
    echo -e "   ${GREEN}✅ Interactive CLI found${NC}"
else
    echo -e "   ${RED}❌ Interactive CLI not found${NC}"
fi

if [ -f "$CLI_PATH/setup/start_qwen_sad_system.sh" ]; then
    echo -e "   ${GREEN}✅ Startup script found${NC}"
else
    echo -e "   ${YELLOW}⚠️  Startup script not found${NC}"
fi
echo ""

# Check SGLang server status
echo -e "${BOLD}🔥 SGLang Server:${NC}"

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo -e "   ${CYAN}📋 PID file: $PID${NC}"
    
    # Check if process is running
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "   ${GREEN}✅ Process running (PID: $PID)${NC}"
        
        # Check memory usage
        MEMORY=$(ps -o pid,vsz,rss,comm -p "$PID" 2>/dev/null | tail -n 1)
        if [ -n "$MEMORY" ]; then
            echo -e "   ${CYAN}💾 Memory: $MEMORY${NC}"
        fi
    else
        echo -e "   ${RED}❌ Process not running (stale PID file)${NC}"
    fi
else
    echo -e "   ${YELLOW}📋 No PID file found${NC}"
fi

# Check if SGLang processes are running
SGLANG_PROCS=$(pgrep -f "sglang.launch_server" | wc -l)
if [ "$SGLANG_PROCS" -gt 0 ]; then
    echo -e "   ${GREEN}✅ SGLang processes: $SGLANG_PROCS running${NC}"
else
    echo -e "   ${RED}❌ No SGLang processes found${NC}"
fi

# Check port
if lsof -i:${SGLANG_PORT} > /dev/null 2>&1; then
    PORT_PROC=$(lsof -i:${SGLANG_PORT} | tail -n 1 | awk '{print $2}')
    echo -e "   ${GREEN}✅ Port ${SGLANG_PORT}: In use (PID: $PORT_PROC)${NC}"
else
    echo -e "   ${RED}❌ Port ${SGLANG_PORT}: Not in use${NC}"
fi

# Check server health
echo -e "   ${CYAN}🏥 Health check:${NC}"
if curl -s http://${SGLANG_HOST}:${SGLANG_PORT}/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}✅ Server responding to health checks${NC}"
    
    # Try to get model info
    MODEL_INFO=$(curl -s http://${SGLANG_HOST}:${SGLANG_PORT}/get_model_info 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}✅ Model info accessible${NC}"
    fi
else
    echo -e "   ${RED}❌ Server not responding${NC}"
fi
echo ""

# Check GPU status
echo -e "${BOLD}🎮 GPU Status:${NC}"
if command -v nvidia-smi > /dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "   ${GREEN}✅ GPU accessible${NC}"
        echo -e "   ${CYAN}📊 $GPU_INFO${NC}"
    else
        echo -e "   ${YELLOW}⚠️  Could not get GPU info${NC}"
    fi
else
    echo -e "   ${RED}❌ nvidia-smi not found${NC}"
fi
echo ""

# Check logs
echo -e "${BOLD}📋 Recent Logs:${NC}"
LOG_FILE="$MODEL_PATH/sglang_server.log"
if [ -f "$LOG_FILE" ]; then
    echo -e "   ${GREEN}✅ Log file exists: $LOG_FILE${NC}"
    LOG_SIZE=$(ls -lh "$LOG_FILE" | awk '{print $5}')
    echo -e "   ${CYAN}📏 Size: $LOG_SIZE${NC}"
    
    # Show last few lines
    echo -e "   ${CYAN}📝 Last 3 lines:${NC}"
    tail -n 3 "$LOG_FILE" 2>/dev/null | sed 's/^/      /'
else
    echo -e "   ${YELLOW}⚠️  No log file found${NC}"
fi
echo ""

# Summary
echo -e "${BOLD}📋 Quick Summary:${NC}"
if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null && curl -s http://${SGLANG_HOST}:${SGLANG_PORT}/health > /dev/null 2>&1; then
    echo -e "   ${GREEN}🎉 System is running and healthy!${NC}"
    echo -e "   ${CYAN}💡 Ready to use: python $CLI_PATH/interactive_qwen_cli.py${NC}"
else
    echo -e "   ${RED}⚠️  System is not fully operational${NC}"
    echo -e "   ${CYAN}💡 To start: $CLI_PATH/setup/start_qwen_sad_system.sh${NC}"
fi

echo ""
echo -e "${BOLD}🔧 Management Commands:${NC}"
echo -e "   ${CYAN}Start system: $CLI_PATH/setup/start_qwen_sad_system.sh${NC}"
echo -e "   ${CYAN}Stop server:  $CLI_PATH/setup/stop_sglang_server.sh${NC}"
echo -e "   ${CYAN}View logs:    tail -f $LOG_FILE${NC}"
echo -e "   ${CYAN}Start CLI:    cd $CLI_PATH && python interactive_qwen_cli.py${NC}" 