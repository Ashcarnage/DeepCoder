# Getting Started with DeepCoder

## 🎯 Project Overview
DeepCoder is a comprehensive pipeline for distilling the reasoning and coding capabilities of DeepSeek R1 into a more efficient Qwen 3B MoE model using Structured Agent Distillation (SAD).

## 📋 What We've Built So Far

### ✅ Complete Foundation (Phase 1 Setup)
1. **Project Structure** - Organized codebase with clear separation of concerns
2. **Requirements & Dependencies** - Complete dependency specification with pinned versions
3. **Configuration System** - Comprehensive YAML-based configuration
4. **Environment Setup** - Automated setup and verification scripts
5. **Documentation** - Detailed PRDs, task breakdowns, and implementation status

### 🏗️ Architecture Overview
```
Teacher Model (DeepSeek R1) → Data Generation → Preprocessing → 
Student Training (Qwen 3B) → Evaluation → API Deployment
```

## 🚀 Quick Start

### 1. Initial Setup
```bash
# Clone and enter the project
cd DeepCoder

# Run automated setup
./scripts/setup.sh

# This will:
# - Install all dependencies
# - Create necessary directories
# - Set up configuration files
# - Run environment verification
```

### 2. Configure Environment
```bash
# Edit the environment file
cp env.example .env
nano .env  # Add your GROQ_API_KEY and other settings
```

### 3. Verify Setup
```bash
# Run comprehensive verification
python scripts/verify_environment.py

# This checks:
# - Python version and dependencies
# - GPU availability
# - API connectivity
# - Project structure
# - Configuration validity
```

## 📊 Current Status

### ✅ Completed (Ready to Use)
- **Environment Setup**: Complete dependency management and verification
- **Project Structure**: Organized codebase with clear architecture
- **Configuration System**: Comprehensive settings management
- **Documentation**: Detailed requirements, tasks, and implementation guide

### 🔄 Next Steps (Implementation Required)
1. **Teacher Model Integration** (Week 1)
   - Groq API wrapper with DeepSeek R1
   - Langchain agent setup
   - Tool implementations (Python execution, knowledge retrieval)

2. **Data Generation Pipeline** (Week 1)
   - Problem dataset curation
   - Trajectory generation loop
   - Quality validation and filtering

3. **Preprocessing & SAD** (Week 2)
   - Trajectory parsing and segmentation
   - Token-level alignment for SAD
   - Training dataset preparation

4. **Student Model Training** (Week 3)
   - Qwen 3B setup with QLoRA
   - Custom SAD loss implementation
   - Training pipeline execution

5. **Evaluation & API** (Week 4)
   - Benchmark integration
   - Performance comparison
   - FastAPI deployment

## 🛠️ Development Workflow

### Phase-Based Development
Each phase has verification checkpoints:

```bash
# Phase 1: Environment Setup ✅ DONE
python scripts/verify_environment.py

# Phase 2: Data Generation (Next)
python scripts/verify_phase1.py  # When ready

# Phase 3: Preprocessing 
python scripts/verify_phase2.py  # After phase 2

# Continue for phases 3-5...
```

### Key Commands
```bash
# Environment verification
python scripts/verify_environment.py

# Data generation (when implemented)
python src/data_generation/generate_trajectories.py

# Training (when implemented)
python src/training/train_student.py

# API server (when implemented)
uvicorn src.api.main:app --reload
```

## 📁 Project Structure
```
DeepCoder/
├── docs/                     # 📋 Documentation and PRDs
│   ├── PROJECT_REQUIREMENTS.md
│   ├── TASK_BREAKDOWN.md
│   └── IMPLEMENTATION_STATUS.md
├── src/                      # 🔧 Source code (implementation pending)
│   ├── data_generation/      # Teacher model and trajectory generation
│   ├── preprocessing/        # SAD data preprocessing
│   ├── training/            # Student model training
│   ├── evaluation/          # Benchmarking and evaluation
│   └── api/                 # FastAPI deployment
├── configs/                 # ⚙️ Configuration files
│   └── config.yaml          # Main project configuration
├── scripts/                 # 🛠️ Utility scripts
│   ├── setup.sh            # Automated setup
│   └── verify_environment.py # Environment verification
├── tests/                   # 🧪 Test files (pending)
├── data/                    # 📊 Generated datasets
├── models/                  # 🤖 Trained models
└── logs/                    # 📝 Training and API logs
```

## 🎯 Success Metrics

### Phase Completion Criteria
- **Phase 1**: Environment setup passes all verifications ✅
- **Phase 2**: 1000+ valid trajectories generated with 90%+ success rate
- **Phase 3**: SAD preprocessing with 95%+ span identification accuracy
- **Phase 4**: Student model training with stable convergence
- **Phase 5**: Production API with <2s response time

### Overall Project Goals
- **Performance**: 85%+ of teacher model capability
- **Efficiency**: 10x faster inference with 5x lower cost
- **Quality**: High-quality reasoning and tool usage
- **Deployment**: Production-ready API endpoint

## 🔗 Key Resources

### Documentation
- [Project Requirements](docs/PROJECT_REQUIREMENTS.md) - Complete technical specifications
- [Task Breakdown](docs/TASK_BREAKDOWN.md) - Detailed implementation plan
- [Implementation Status](docs/IMPLEMENTATION_STATUS.md) - Current progress tracking

### Configuration
- [config.yaml](configs/config.yaml) - Main project configuration
- [env.example](env.example) - Environment variables template

### Scripts
- [setup.sh](scripts/setup.sh) - Automated project setup
- [verify_environment.py](scripts/verify_environment.py) - Environment verification

## 🚨 Important Notes

### Prerequisites
- **Python 3.8+** with GPU support recommended
- **GROQ API Key** for DeepSeek R1 access
- **24GB+ GPU** for optimal training performance
- **100GB+ storage** for datasets and models

### Security Considerations
- Python code execution requires sandboxing in production
- API rate limiting and authentication needed
- Secure handling of API keys and user inputs

### Performance Optimization
- QLoRA training reduces memory requirements
- Gradient checkpointing for large models
- Mixed precision training (BF16) for speed
- Incremental data processing for large datasets

## 🤝 Contributing

### Development Process
1. **Follow the phase-based approach** - Complete each phase before moving to the next
2. **Run verification scripts** - Ensure each phase passes verification
3. **Update implementation status** - Track progress in the status document
4. **Test thoroughly** - Validate functionality at each step

### Code Quality
- Follow Python best practices and type hints
- Add comprehensive error handling
- Write unit tests for critical components
- Document complex functions and classes

## 🆘 Troubleshooting

### Common Issues
1. **Unsloth installation fails**: Check CUDA version compatibility
2. **Groq API errors**: Verify API key and rate limits
3. **GPU memory issues**: Reduce batch size or use gradient checkpointing
4. **Package conflicts**: Use virtual environment with pinned versions

### Getting Help
- Check [Implementation Status](docs/IMPLEMENTATION_STATUS.md) for known issues
- Review verification script output for specific problems
- Ensure all prerequisites are met before proceeding

---

**Ready to start?** Run `./scripts/setup.sh` and then `python scripts/verify_environment.py` to verify your setup is complete! 