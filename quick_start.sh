#!/bin/bash
# 🚀 Quick Start - Qwen SAD System
# =================================
# 
# Run this immediately after restarting your runpod to get everything working!
# Usage: ./quick_start.sh

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}🚀 Qwen SAD System - Quick Start${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""
echo -e "${GREEN}This will:${NC}"
echo -e "${CYAN}  1. Install all dependencies${NC}"
echo -e "${CYAN}  2. Start SGLang server with your trained model${NC}"
echo -e "${CYAN}  3. Launch the interactive CLI${NC}"
echo ""

read -p "Ready to start? [Y/n]: " choice
case ${choice,,} in
    n)
        echo -e "${CYAN}💡 Manual setup available in ./setup/ folder${NC}"
        exit 0
        ;;
    *)
        echo -e "${GREEN}🎯 Starting automatic setup...${NC}"
        ;;
esac

echo ""
echo -e "${BOLD}Step 1: Installing dependencies...${NC}"
./setup/install_requirements.sh

echo ""
echo -e "${BOLD}Step 2: Starting system...${NC}"
./setup/start_qwen_sad_system.sh 