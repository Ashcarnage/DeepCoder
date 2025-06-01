#!/bin/bash
# 🛑 Stop SGLang Server Script
# ============================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

MODEL_PATH="/workspace/persistent/models/qwen3-30b-a3b"
PID_FILE="$MODEL_PATH/sglang_server.pid"

echo -e "${BOLD}${RED}🛑 Stopping SGLang Server${NC}"
echo -e "${BLUE}==========================${NC}"

# Check if PID file exists
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}📋 Found PID file: $PID${NC}"
    
    # Check if process is running
    if kill -0 "$PID" 2>/dev/null; then
        echo -e "${YELLOW}⏳ Stopping process $PID...${NC}"
        kill "$PID"
        
        # Wait for graceful shutdown
        for i in {1..10}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo -e "${GREEN}✅ Process stopped gracefully${NC}"
                rm -f "$PID_FILE"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo -e "${YELLOW}⚠️  Forcing kill...${NC}"
            kill -9 "$PID" 2>/dev/null || true
            rm -f "$PID_FILE"
            echo -e "${GREEN}✅ Process force killed${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  Process not running, removing stale PID file${NC}"
        rm -f "$PID_FILE"
    fi
else
    echo -e "${YELLOW}📋 No PID file found${NC}"
fi

# Kill any remaining SGLang processes
echo -e "${YELLOW}🔍 Checking for any remaining SGLang processes...${NC}"
if pgrep -f "sglang.launch_server" > /dev/null; then
    echo -e "${YELLOW}⚠️  Found remaining processes, killing them...${NC}"
    pkill -f "sglang.launch_server" || true
    sleep 2
fi

# Check port 30000
if lsof -i:30000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port 30000 still in use, freeing it...${NC}"
    lsof -ti:30000 | xargs kill -9 2>/dev/null || true
fi

echo -e "${GREEN}✅ SGLang server stopped successfully${NC}"
echo -e "${BLUE}💡 To restart: cd /root/DeepCoder && ./setup/start_qwen_sad_system.sh${NC}" 