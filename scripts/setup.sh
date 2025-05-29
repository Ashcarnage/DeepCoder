#!/bin/bash

# DeepCoder Setup Script
# Automates the installation and configuration of the DeepCoder project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}"
    echo "=================================="
    echo "       DeepCoder Setup"
    echo "=================================="
    echo -e "${NC}"
}

# Check if Python 3.8+ is available
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        REQUIRED_VERSION="3.8"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_status "Python $PYTHON_VERSION found ✓"
        else
            print_error "Python $PYTHON_VERSION found, but Python 3.8+ is required"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.8+"
        exit 1
    fi
}

# Check if GPU is available
check_gpu() {
    print_status "Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
        print_status "GPU detected: $GPU_INFO"
    else
        print_warning "nvidia-smi not found. GPU acceleration may not be available."
    fi
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip first
    python3 -m pip install --upgrade pip
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        python3 -m pip install -r requirements.txt
    else
        print_error "requirements.txt not found!"
        exit 1
    fi
    
    print_status "Dependencies installed successfully ✓"
}

# Create environment file
create_env_file() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f ".env" ]; then
        if [ -f "env.example" ]; then
            cp env.example .env
            print_status "Created .env file from env.example"
            print_warning "Please edit .env file and add your API keys"
        else
            print_error "env.example not found!"
            exit 1
        fi
    else
        print_status ".env file already exists ✓"
    fi
}

# Create data directories
create_directories() {
    print_status "Creating project directories..."
    
    DIRS=("data" "models" "logs" "checkpoints" "cache")
    
    for dir in "${DIRS[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            print_status "Created directory: $dir"
        else
            print_status "Directory exists: $dir ✓"
        fi
    done
}

# Download sample data (optional)
setup_sample_data() {
    print_status "Setting up sample data..."
    
    # Create sample problems file
    cat > data/sample_problems.json << 'EOF'
[
    {
        "id": "sample_1",
        "prompt": "Write a Python function that takes a list of integers and returns a new list containing only the even numbers, sorted in ascending order.",
        "difficulty": "easy",
        "category": "list_processing"
    },
    {
        "id": "sample_2", 
        "prompt": "What is the capital of France, and what is its approximate population?",
        "difficulty": "easy",
        "category": "knowledge_retrieval"
    },
    {
        "id": "sample_3",
        "prompt": "Implement a Python function to calculate the factorial of a non-negative integer. Then use it to calculate 5!.",
        "difficulty": "medium",
        "category": "math_coding"
    }
]
EOF
    
    print_status "Sample problems created ✓"
}

# Make scripts executable
make_scripts_executable() {
    print_status "Making scripts executable..."
    
    find scripts -name "*.py" -exec chmod +x {} \;
    find scripts -name "*.sh" -exec chmod +x {} \;
    
    print_status "Scripts made executable ✓"
}

# Run verification
run_verification() {
    print_status "Running environment verification..."
    
    if python3 scripts/verify_environment.py; then
        print_status "Environment verification passed ✓"
    else
        print_warning "Environment verification failed. Please check the output above."
        return 1
    fi
}

# Main setup function
main() {
    print_header
    
    print_status "Starting DeepCoder setup..."
    
    # Check prerequisites
    check_python
    check_gpu
    
    # Install and configure
    install_dependencies
    create_env_file
    create_directories
    setup_sample_data
    make_scripts_executable
    
    # Verify setup
    if run_verification; then
        echo
        print_status "Setup completed successfully! 🎉"
        echo
        echo -e "${BLUE}Next steps:${NC}"
        echo "1. Edit .env file and add your GROQ_API_KEY"
        echo "2. Run: python3 scripts/verify_environment.py"
        echo "3. Start with: python3 src/data_generation/generate_trajectories.py"
        echo
    else
        print_error "Setup completed with warnings. Please resolve issues above."
        exit 1
    fi
}

# Run main function
main "$@" 