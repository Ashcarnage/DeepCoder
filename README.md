# DeepCoder: Agent Reasoning Distillation Pipeline

## Overview
DeepCoder is a comprehensive pipeline for distilling the reasoning and coding capabilities of DeepSeek R1 (teacher model) into a more efficient Qwen 3B MoE model (student model) using Structured Agent Distillation (SAD).

## Key Features
- **Teacher Model**: DeepSeek R1 via Groq API + Langchain
- **Student Model**: Qwen 3B with 3B active parameters (MoE)
- **Distillation Method**: Structured Agent Distillation (SAD)
- **Agent Framework**: Thought-Action-Observation cycles
- **Tools**: Python execution, Knowledge retrieval
- **Output**: Production-ready API endpoint

## Project Structure
```
DeepCoder/
├── docs/                 # Project documentation and PRDs
├── src/                  # Source code
│   ├── data_generation/  # Teacher model data generation
│   ├── preprocessing/    # SAD data preprocessing
│   ├── training/         # Student model fine-tuning
│   ├── evaluation/       # Model evaluation and testing
│   └── api/              # FastAPI endpoint
├── configs/              # Configuration files
├── notebooks/            # Jupyter notebooks for experimentation
├── tests/                # Unit tests
└── scripts/              # Utility scripts
```

## Quick Start
1. Set up environment variables (GROQ_API_KEY)
2. Install dependencies: `pip install -r requirements.txt`
3. Run data generation: `python src/data_generation/generate_trajectories.py`
4. Fine-tune student model: `python src/training/train_student.py`
5. Start API server: `uvicorn src.api.main:app --reload`

## Verification Points
Each phase includes verification scripts to ensure proper functionality before proceeding to the next phase.