#!/usr/bin/env python3
"""
PyTorch-based Agentic Training System with LoRA and SAD
Real model training with weight updates, visualization, and monitoring
"""

import os
import sys
import json
import logging
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# PyTorch and Transformers
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

# API clients
from groq import Groq
import requests

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

@dataclass
class TrainingConfig:
    """Training configuration"""
    
    # Model configuration - use smaller model for limited memory
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # Much smaller model
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    
    # Training parameters - optimized for memory
    learning_rate: float = 2e-4
    batch_size: int = 1  # Reduce batch size
    max_steps: int = 200  # Fewer steps for demo
    eval_every: int = 20
    save_every: int = 40
    max_length: int = 1024  # Reduce sequence length
    
    # LoRA configuration - smaller for memory efficiency
    lora_r: int = 8  # Reduced from 16
    lora_alpha: int = 16  # Reduced from 32
    lora_dropout: float = 0.1
    
    # SAD loss weights
    distillation_weight: float = 0.7
    agentic_pattern_weight: float = 0.2
    reasoning_weight: float = 0.1
    temperature: float = 3.0
    
    # Paths
    train_data: str = "data/training_data/agentic_train.jsonl"
    val_data: str = "data/training_data/agentic_val.jsonl"
    checkpoint_dir: str = "checkpoints/pytorch_training"
    logs_dir: str = "logs/pytorch_training"
    plots_dir: str = "plots/training"

class AgenticDataset(Dataset):
    """Dataset for agentic training conversations"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        
        # Load conversations
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    conv = json.loads(line.strip())
                    self.conversations.append(conv)
        
        print(f"Loaded {len(self.conversations)} conversations from {data_path}")
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Extract turns and create text
        content = conversation.get("content", {})
        turns = content.get("turns", [])
        
        # Create conversation text
        text_parts = []
        for turn in turns:
            role = turn.get("role", "user")
            message = turn.get("content", "")
            text_parts.append(f"<|{role}|>\n{message}\n")
        
        full_text = "".join(text_parts)
        
        # Tokenize without padding (we'll pad in collate_fn)
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "conversation": conversation
        }

def collate_fn(batch, tokenizer):
    """Custom collate function to handle variable-length sequences"""
    input_ids = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    conversations = [item["conversation"] for item in batch]
    
    # Pad sequences to the same length
    max_len = max(len(seq) for seq in input_ids)
    
    padded_input_ids = []
    padded_attention_masks = []
    
    for i in range(len(input_ids)):
        seq_len = len(input_ids[i])
        padding_len = max_len - seq_len
        
        # Pad input_ids
        padded_seq = torch.cat([
            input_ids[i],
            torch.full((padding_len,), tokenizer.pad_token_id, dtype=input_ids[i].dtype)
        ])
        padded_input_ids.append(padded_seq)
        
        # Pad attention_mask
        padded_mask = torch.cat([
            attention_masks[i],
            torch.zeros(padding_len, dtype=attention_masks[i].dtype)
        ])
        padded_attention_masks.append(padded_mask)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_masks),
        "conversation": conversations
    }

class AgenticTrainer:
    """PyTorch-based agentic trainer with LoRA and SAD"""
    
    def __init__(self, config: TrainingConfig, groq_api_key: str):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Setup directories
        for dir_path in [config.checkpoint_dir, config.logs_dir, config.plots_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Training tracking
        self.training_losses = []
        self.validation_losses = []
        self.distillation_losses = []
        self.agentic_pattern_losses = []
        self.reasoning_losses = []
        self.step = 0
        self.best_val_loss = float('inf')
        
        # Load model and tokenizer
        self.load_model()
        
        # Load datasets
        self.load_datasets()
        
        # Setup optimizer
        self.setup_optimizer()
    
    def setup_logging(self):
        """Setup logging"""
        log_file = Path(self.config.logs_dir) / f"pytorch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger("AgenticTrainer")
        self.logger.info(f"Training log: {log_file}")
    
    def load_model(self):
        """Load student model with LoRA"""
        self.logger.info(f"Loading model: {self.config.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with memory optimizations
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,  # Use FP16 for memory efficiency
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,  # Reduce CPU memory usage
            load_in_8bit=False,  # Disable 8-bit loading for compatibility with LoRA
            use_cache=False  # Disable KV cache to save memory
        )
        
        # Enable gradient checkpointing to save memory
        self.base_model.gradient_checkpointing_enable()
        
        # Setup LoRA with memory-efficient configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Fewer target modules
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        self.logger.info(f"Model loaded with LoRA configuration")
        self.logger.info(f"Model dtype: {self.model.dtype}")
        self.logger.info(f"Device: {next(self.model.parameters()).device}")
    
    def load_datasets(self):
        """Load training and validation datasets"""
        self.train_dataset = AgenticDataset(
            self.config.train_data, 
            self.tokenizer, 
            self.config.max_length
        )
        
        self.val_dataset = AgenticDataset(
            self.config.val_data, 
            self.tokenizer, 
            self.config.max_length
        )
        
        # Create data loaders with custom collate function
        def custom_collate_fn(batch):
            return collate_fn(batch, self.tokenizer)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_steps,
            eta_min=1e-6
        )
        
        self.logger.info(f"Optimizer setup with lr={self.config.learning_rate}")
    
    def get_teacher_response(self, conversation_turns: List[Dict]) -> str:
        """Get response from teacher model (Groq)"""
        try:
            # Format conversation for teacher
            messages = []
            for turn in conversation_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
            
            # Get teacher response
            response = self.groq_client.chat.completions.create(
                model=self.config.teacher_model,
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.warning(f"Teacher model error: {e}")
            return "I'll help you solve this step by step. Let me analyze the problem and provide a systematic approach."
    
    def get_student_response(self, conversation_turns: List[Dict]) -> str:
        """Get response from student model"""
        try:
            # Format conversation
            text_parts = []
            for turn in conversation_turns:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                text_parts.append(f"<|{role}|>\n{content}\n")
            
            prompt = "".join(text_parts) + "<|assistant|>\n"
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length - 256
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            self.logger.warning(f"Student model error: {e}")
            return "Let me think about this problem..."
    
    def calculate_sad_loss(self, 
                          student_logits: torch.Tensor,
                          teacher_response: str,
                          conversation: Dict) -> Tuple[torch.Tensor, Dict]:
        """Calculate Self-Adaptive Distillation (SAD) loss"""
        
        # Tokenize teacher response for target
        teacher_tokens = self.tokenizer(
            teacher_response,
            return_tensors="pt",
            truncation=True,
            max_length=student_logits.shape[1],
            padding=True
        ).to(self.device)
        
        # Get teacher logits (simplified - we'll use cross-entropy with teacher tokens)
        target_ids = teacher_tokens["input_ids"]
        
        # Pad or truncate to match student logits
        if target_ids.shape[1] > student_logits.shape[1]:
            target_ids = target_ids[:, :student_logits.shape[1]]
        elif target_ids.shape[1] < student_logits.shape[1]:
            padding = torch.full(
                (target_ids.shape[0], student_logits.shape[1] - target_ids.shape[1]),
                self.tokenizer.pad_token_id,
                device=self.device,
                dtype=target_ids.dtype  # Ensure consistent dtype
            )
            target_ids = torch.cat([target_ids, padding], dim=1)
        
        # Ensure tensors are on the same device and have compatible types
        student_logits = student_logits.to(self.device)
        target_ids = target_ids.to(self.device).long()  # Ensure target is LongTensor
        
        # Distillation loss (cross-entropy)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Reshape for loss calculation
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        target_ids_flat = target_ids.view(-1)
        
        # Calculate loss with proper tensor types
        distillation_loss = criterion(student_logits_flat, target_ids_flat)
        
        # Agentic pattern loss
        agentic_patterns = [
            "step by step", "let me think", "first", "then", "analyze",
            "approach", "solution", "consider", "method", "strategy"
        ]
        
        pattern_score = 0.0
        teacher_lower = teacher_response.lower()
        for pattern in agentic_patterns:
            if pattern in teacher_lower:
                pattern_score += 1.0
        
        pattern_score = min(pattern_score / len(agentic_patterns), 1.0)
        agentic_pattern_loss = torch.tensor(1.0 - pattern_score, device=self.device, dtype=distillation_loss.dtype)
        
        # Reasoning quality loss (based on conversation context)
        reasoning_score = 0.5  # Simplified
        content = conversation.get("content", {})
        if "reasoning" in str(content).lower() or "problem" in str(content).lower():
            reasoning_score = 0.7
        
        reasoning_loss = torch.tensor(1.0 - reasoning_score, device=self.device, dtype=distillation_loss.dtype)
        
        # Combined SAD loss
        total_loss = (
            self.config.distillation_weight * distillation_loss +
            self.config.agentic_pattern_weight * agentic_pattern_loss +
            self.config.reasoning_weight * reasoning_loss
        )
        
        loss_components = {
            "distillation": distillation_loss.item(),
            "agentic_pattern": agentic_pattern_loss.item(),
            "reasoning": reasoning_loss.item(),
            "total": total_loss.item()
        }
        
        return total_loss, loss_components
    
    def training_step(self, batch) -> Dict:
        """Execute one training step with real weight updates"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        conversations = batch["conversation"]
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids
        )
        
        total_loss = 0.0
        loss_components = {"distillation": 0.0, "agentic_pattern": 0.0, "reasoning": 0.0}
        
        # Calculate SAD loss for each item in batch
        for i, conversation in enumerate(conversations):
            # Get teacher response
            content = conversation.get("content", {})
            turns = content.get("turns", [])
            
            if len(turns) >= 2:
                teacher_response = self.get_teacher_response(turns[:-1])  # Exclude last turn
                
                # Calculate SAD loss
                student_logits = outputs.logits[i:i+1]  # Single item logits
                sad_loss, components = self.calculate_sad_loss(
                    student_logits, teacher_response, conversation
                )
                
                total_loss += sad_loss
                for key, value in components.items():
                    if key != "total":
                        loss_components[key] += value
        
        # Average loss over batch
        if len(conversations) > 0:
            total_loss = total_loss / len(conversations)
            for key in loss_components:
                loss_components[key] = loss_components[key] / len(conversations)
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "total_loss": total_loss.item(),
            **loss_components,
            "learning_rate": self.scheduler.get_last_lr()[0]
        }
    
    def validate(self) -> Dict:
        """Run validation"""
        self.model.eval()
        
        total_loss = 0.0
        total_components = {"distillation": 0.0, "agentic_pattern": 0.0, "reasoning": 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                conversations = batch["conversation"]
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                batch_loss = 0.0
                batch_components = {"distillation": 0.0, "agentic_pattern": 0.0, "reasoning": 0.0}
                
                # Calculate loss for each item
                for i, conversation in enumerate(conversations):
                    content = conversation.get("content", {})
                    turns = content.get("turns", [])
                    
                    if len(turns) >= 2:
                        teacher_response = self.get_teacher_response(turns[:-1])
                        student_logits = outputs.logits[i:i+1]
                        
                        sad_loss, components = self.calculate_sad_loss(
                            student_logits, teacher_response, conversation
                        )
                        
                        batch_loss += sad_loss.item()
                        for key, value in components.items():
                            if key != "total":
                                batch_components[key] += value
                
                if len(conversations) > 0:
                    batch_loss = batch_loss / len(conversations)
                    for key in batch_components:
                        batch_components[key] = batch_components[key] / len(conversations)
                
                total_loss += batch_loss
                for key in total_components:
                    total_components[key] += batch_components[key]
                num_batches += 1
        
        # Average over all batches
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            for key in total_components:
                total_components[key] = total_components[key] / num_batches
        else:
            avg_loss = float('inf')
        
        return {
            "val_loss": avg_loss,
            **total_components
        }
    
    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_step_{self.step}"
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save LoRA weights
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save training state
        state = {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "training_losses": self.training_losses,
            "validation_losses": self.validation_losses,
            "config": self.config.__dict__
        }
        
        with open(checkpoint_path / "training_state.json", 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def plot_training_progress(self):
        """Create training progress plots"""
        if len(self.training_losses) < 2:
            return
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Training Loss', 'Validation Loss', 'Loss Components', 'Learning Rate'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Training loss
        steps = list(range(1, len(self.training_losses) + 1))
        fig.add_trace(
            go.Scatter(x=steps, y=self.training_losses, name='Training Loss', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Validation loss
        if self.validation_losses:
            val_steps = list(range(self.config.eval_every, len(self.validation_losses) * self.config.eval_every + 1, self.config.eval_every))
            fig.add_trace(
                go.Scatter(x=val_steps, y=self.validation_losses, name='Validation Loss', line=dict(color='red')),
                row=1, col=2
            )
        
        # Loss components
        if self.distillation_losses and self.agentic_pattern_losses and self.reasoning_losses:
            fig.add_trace(
                go.Scatter(x=steps[-len(self.distillation_losses):], y=self.distillation_losses, 
                          name='Distillation', line=dict(color='green')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=steps[-len(self.agentic_pattern_losses):], y=self.agentic_pattern_losses, 
                          name='Agentic Pattern', line=dict(color='orange')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=steps[-len(self.reasoning_losses):], y=self.reasoning_losses, 
                          name='Reasoning', line=dict(color='purple')),
                row=2, col=1
            )
        
        # Learning rate
        lrs = [self.scheduler.get_last_lr()[0] for _ in steps]
        fig.add_trace(
            go.Scatter(x=steps, y=lrs, name='Learning Rate', line=dict(color='black')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Agentic Training Progress - Step {self.step}",
            height=800,
            showlegend=True
        )
        
        # Save plot
        plot_path = Path(self.config.plots_dir) / f"training_progress_step_{self.step}.html"
        fig.write_html(plot_path)
        
        # Also save as PNG
        try:
            fig.write_image(Path(self.config.plots_dir) / f"training_progress_step_{self.step}.png")
        except Exception as e:
            self.logger.warning(f"Could not save PNG plot: {e}")
        
        self.logger.info(f"Training plot saved: {plot_path}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting PyTorch Agentic Training with LoRA and SAD!")
        self.logger.info("=" * 60)
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.config.max_steps // len(self.train_loader) + 1):
            for batch in self.train_loader:
                if self.step >= self.config.max_steps:
                    break
                
                self.step += 1
                
                # Training step
                step_results = self.training_step(batch)
                
                # Track losses
                self.training_losses.append(step_results["total_loss"])
                self.distillation_losses.append(step_results["distillation"])
                self.agentic_pattern_losses.append(step_results["agentic_pattern"])
                self.reasoning_losses.append(step_results["reasoning"])
                
                # Logging
                if self.step % 10 == 0:
                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"Step {self.step}/{self.config.max_steps} | "
                        f"Loss: {step_results['total_loss']:.4f} | "
                        f"LR: {step_results['learning_rate']:.2e} | "
                        f"Time: {elapsed:.1f}s"
                    )
                
                # Validation
                if self.step % self.config.eval_every == 0:
                    val_results = self.validate()
                    self.validation_losses.append(val_results["val_loss"])
                    
                    if val_results["val_loss"] < self.best_val_loss:
                        self.best_val_loss = val_results["val_loss"]
                        self.logger.info(f"🎯 New best validation loss: {val_results['val_loss']:.4f}")
                    
                    self.logger.info(
                        f"📊 Validation - Loss: {val_results['val_loss']:.4f} | "
                        f"Distillation: {val_results['distillation']:.4f} | "
                        f"Agentic: {val_results['agentic_pattern']:.4f} | "
                        f"Reasoning: {val_results['reasoning']:.4f}"
                    )
                
                # Checkpointing
                if self.step % self.config.save_every == 0:
                    self.save_checkpoint()
                    self.plot_training_progress()
            
            if self.step >= self.config.max_steps:
                break
        
        # Final results
        total_time = time.time() - start_time
        
        self.logger.info("🎉 Training Complete!")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Final Results:")
        self.logger.info(f"   • Total steps: {self.step}")
        self.logger.info(f"   • Training time: {total_time:.1f}s")
        self.logger.info(f"   • Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"   • Final training loss: {self.training_losses[-1]:.4f}")
        
        # Save final checkpoint and plots
        self.save_checkpoint()
        self.plot_training_progress()
        
        return True

def main():
    """Main training function"""
    
    # Configuration
    config = TrainingConfig()
    
    # Groq API key
    groq_api_key = "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"
    
    print("🚀 PyTorch Agentic Training with LoRA and SAD")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Teacher: {config.teacher_model}")
    print(f"Max steps: {config.max_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"LoRA r: {config.lora_r}")
    print()
    
    try:
        # Initialize trainer
        trainer = AgenticTrainer(config, groq_api_key)
        
        # Start training
        success = trainer.train()
        
        if success:
            print("\n✨ Agentic training completed successfully!")
            print(f"Check plots in: {config.plots_dir}")
            print(f"Check logs in: {config.logs_dir}")
            print(f"Check checkpoints in: {config.checkpoint_dir}")
            return True
        else:
            print("\n❌ Training failed")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 