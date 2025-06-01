#!/usr/bin/env python3
"""
🚀 ROBUST REAL Structured Agent Distillation (SAD) Training
===========================================================

This performs ACTUAL LoRA training with weight updates on Qwen 30B:

✅ REAL LoRA weight updates (not simulation)
✅ Waits for SGLang server to be ready  
✅ Uses existing high-quality training data
✅ Memory management to avoid conflicts
✅ Before/after evaluation with actual model changes
✅ Sophisticated tool usage training
✅ Quantifiable improvements with actual weight updates

STRATEGY:
1. Wait for SGLang server to be ready for baseline testing
2. Stop SGLang server during training (free VRAM)
3. Load Qwen 30B for LoRA training (use freed VRAM)
4. Perform actual gradient updates and save LoRA weights
5. Restart SGLang and demonstrate improvements
6. Show real before/after comparison with weight changes
"""

import os
import sys
import json
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# ML Libraries
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# API Clients
from openai import OpenAI
import requests

import logging
import warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class RobustSADConfig:
    """Configuration for robust real SAD training."""
    
    # Model configuration
    student_model_path: str = "/workspace/persistent/models/qwen3-30b-a3b"
    
    # Existing training data
    training_data_file: str = "/workspace/persistent/teacher_tool_training/essential_training_data.json"
    
    # LoRA configuration (optimized for real training)
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # Training configuration
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 1
    max_seq_length: int = 1024
    gradient_accumulation_steps: int = 4
    
    # Output configuration
    output_dir: str = "/workspace/persistent/robust_real_sad_training"
    lora_model_dir: str = "/workspace/persistent/robust_real_sad_training/lora_weights"
    
    # SGLang configuration
    sglang_port: int = 30000
    sglang_host: str = "localhost"
    sglang_wait_timeout: int = 300  # 5 minutes max wait

class ToolAwareDataset(Dataset):
    """Dataset for tool-aware training using existing data."""
    
    def __init__(self, training_data: List[Dict], tokenizer, max_length: int = 1024):
        self.training_data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, idx):
        item = self.training_data[idx]
        prompt = item["prompt"]
        response = item["response"]
        
        # Create conversation format with system message for tool usage
        conversation = f"""System: You are an expert assistant with access to tools. Use them systematically.

User: {prompt}

"""
        return conversation 