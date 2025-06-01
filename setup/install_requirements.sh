#!/bin/bash
# 📦 Install Requirements Script
# ==============================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}📦 Installing Qwen SAD System Requirements${NC}"
echo -e "${CYAN}==========================================${NC}"
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found. Please run this from the DeepCoder directory.${NC}"
    exit 1
fi

# Update pip
echo -e "${YELLOW}⬆️  Updating pip...${NC}"
python -m pip install --upgrade pip

# Install requirements
echo -e "${YELLOW}📦 Installing Python requirements...${NC}"
pip install -r requirements.txt

# Verify SGLang installation
echo -e "${YELLOW}🔍 Verifying SGLang installation...${NC}"
if python -c "import sglang" 2>/dev/null; then
    echo -e "${GREEN}✅ SGLang is installed${NC}"
else
    echo -e "${RED}❌ SGLang not found, installing...${NC}"
    pip install "sglang[all]"
fi

# Verify Rich installation
echo -e "${YELLOW}🔍 Verifying Rich installation...${NC}"
if python -c "import rich" 2>/dev/null; then
    echo -e "${GREEN}✅ Rich is installed${NC}"
else
    echo -e "${RED}❌ Rich not found, installing...${NC}"
    pip install rich
fi

# Verify OpenAI client installation
echo -e "${YELLOW}🔍 Verifying OpenAI client installation...${NC}"
if python -c "import openai" 2>/dev/null; then
    echo -e "${GREEN}✅ OpenAI client is installed${NC}"
else
    echo -e "${RED}❌ OpenAI client not found, installing...${NC}"
    pip install openai
fi

# Check for CUDA
echo -e "${YELLOW}🎮 Checking CUDA availability...${NC}"
if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo -e "${GREEN}✅ PyTorch with CUDA is available${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not detected or PyTorch not installed properly${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}✅ Installation complete!${NC}"
echo -e "${CYAN}💡 Next steps:${NC}"
echo -e "${CYAN}   1. Ensure your Qwen model is in: /workspace/persistent/models/qwen3-30b-a3b/${NC}"
echo -e "${CYAN}   2. Run: ./setup/start_qwen_sad_system.sh${NC}" 