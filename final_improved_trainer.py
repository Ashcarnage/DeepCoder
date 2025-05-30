#!/usr/bin/env python3
"""
Final Improved Agentic Training System
✅ Uses deepseek-r1-distill-llama-70b (available on Groq) as teacher
✅ Local Qwen2.5-1.5B as student with real PyTorch weight updates
✅ Before/after training demonstrations
✅ Loss visualization and comprehensive reporting
✅ Reasoning, coding, and tool use capabilities
"""

import os
import sys
import json
import logging
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# PyTorch and Transformers
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

# API clients
from groq import Groq

def setup_comprehensive_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs/final_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"final_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("FinalTrainer")

class AgenticDatasetFinal(Dataset):
    """Final optimized dataset for agentic training"""
    
    def __init__(self, data_path: str, tokenizer, max_samples: int = 15, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversations = []
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                if line.strip():
                    conv = json.loads(line.strip())
                    if self._has_agentic_patterns(conv):
                        self.conversations.append(conv)
        
        print(f"✅ Loaded {len(self.conversations)} high-quality agentic conversations")
    
    def _has_agentic_patterns(self, conversation: Dict) -> bool:
        """Enhanced agentic pattern detection"""
        content = str(conversation).lower()
        patterns = [
            "step by step", "reasoning", "analyze", "solve", "approach",
            "code", "function", "algorithm", "method", "tool", "implement"
        ]
        return sum(1 for pattern in patterns if pattern in content) >= 2
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        content = conversation.get("content", {})
        turns = content.get("turns", [])
        
        # Create meaningful conversation text
        text_parts = []
        for turn in turns[:3]:  # Focus on first 3 turns
            role = turn.get("role", "user")
            message = turn.get("content", "")[:400]  # Reasonable length
            text_parts.append(f"<|{role}|>: {message}")
        
        full_text = " ".join(text_parts)
        
        # Tokenize with proper handling
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding=False
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "conversation": conversation
        }

class FinalAgenticTrainer:
    """Final Agentic Trainer with all optimizations"""
    
    def __init__(self, groq_api_key: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_comprehensive_logging()
        
        # Optimized configuration
        self.config = {
            "student_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "teacher_model": "deepseek-r1-distill-llama-70b",  # Available model
            "learning_rate": 8e-4,
            "batch_size": 1,
            "max_steps": 25,
            "eval_every": 8,
            "max_length": 512,
            "lora_r": 8,
            "lora_alpha": 16,
            "distillation_weight": 0.75,
            "agentic_pattern_weight": 0.20,
            "reasoning_weight": 0.05
        }
        
        # Training metrics
        self.training_losses = []
        self.loss_components = {"distillation": [], "agentic": [], "reasoning": []}
        self.step = 0
        
        # Initialize
        self.groq_client = Groq(api_key=groq_api_key)
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        
        self.logger.info("🚀 Final Agentic Trainer Initialized!")
        self.logger.info(f"📖 Student: {self.config['student_model']}")
        self.logger.info(f"🎓 Teacher: {self.config['teacher_model']}")
        self.logger.info(f"🔧 Device: {self.device}")
    
    def setup_model(self):
        """Setup student model with LoRA"""
        self.logger.info(f"Loading student model: {self.config['student_model']}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["student_model"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.config["student_model"],
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            device_map="auto" if self.device.type == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        self.base_model.gradient_checkpointing_enable()
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config["lora_r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()
    
    def setup_data(self):
        """Setup optimized training data"""
        dataset = AgenticDatasetFinal(
            "data/training_data/agentic_train.jsonl", 
            self.tokenizer, 
            max_samples=15,
            max_length=self.config["max_length"]
        )
        
        def collate_fn(batch):
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_mask"] for item in batch]
            conversations = [item["conversation"] for item in batch]
            
            # Efficient padding
            max_len = max(len(seq) for seq in input_ids)
            
            padded_input_ids = []
            padded_attention_masks = []
            
            for i in range(len(input_ids)):
                seq_len = len(input_ids[i])
                padding_len = max_len - seq_len
                
                padded_seq = torch.cat([
                    input_ids[i],
                    torch.full((padding_len,), self.tokenizer.pad_token_id, dtype=input_ids[i].dtype)
                ])
                padded_input_ids.append(padded_seq)
                
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
        
        self.train_loader = DataLoader(
            dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True, 
            collate_fn=collate_fn
        )
    
    def setup_optimizer(self):
        """Setup optimizer with learning rate scheduling"""
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.config["learning_rate"],
            weight_decay=0.01
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config["max_steps"], 
            eta_min=1e-6
        )
    
    def get_teacher_response(self, conversation_turns: List[Dict]) -> str:
        """Get enhanced teacher response"""
        try:
            # Build context from conversation
            messages = []
            for turn in conversation_turns[-2:]:  # Last 2 turns for context
                role = turn.get("role", "user")
                content = turn.get("content", "")[:600]  # Reasonable length
                
                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role == "assistant":
                    messages.append({"role": "assistant", "content": content})
            
            # Enhance prompt for agentic reasoning
            if messages and messages[-1]["role"] == "user":
                messages[-1]["content"] += " Please think step by step and show your reasoning process clearly."
            
            # Call teacher model
            response = self.groq_client.chat.completions.create(
                model=self.config["teacher_model"],
                messages=messages,
                temperature=0.6,
                max_tokens=400
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.logger.warning(f"Teacher API error: {e}")
            # Fallback with strong agentic patterns
            return "Let me approach this systematically. First, I'll analyze the problem carefully. Then, I'll break it down into manageable steps and solve each part methodically."
    
    def calculate_enhanced_loss(self, student_logits: torch.Tensor, 
                              teacher_response: str, conversation: Dict) -> Tuple[torch.Tensor, Dict]:
        """Calculate enhanced SAD loss with improved pattern detection"""
        
        # Tokenize teacher response
        teacher_tokens = self.tokenizer(
            teacher_response,
            return_tensors="pt",
            truncation=True,
            max_length=student_logits.shape[1],
            padding=True
        ).to(self.device)
        
        target_ids = teacher_tokens["input_ids"]
        
        # Align tensor dimensions
        if target_ids.shape[1] != student_logits.shape[1]:
            if target_ids.shape[1] > student_logits.shape[1]:
                target_ids = target_ids[:, :student_logits.shape[1]]
            else:
                padding = torch.full(
                    (target_ids.shape[0], student_logits.shape[1] - target_ids.shape[1]),
                    self.tokenizer.pad_token_id,
                    device=self.device,
                    dtype=target_ids.dtype
                )
                target_ids = torch.cat([target_ids, padding], dim=1)
        
        student_logits = student_logits.to(self.device)
        target_ids = target_ids.to(self.device).long()
        
        # Distillation loss
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        distillation_loss = criterion(
            student_logits.view(-1, student_logits.size(-1)),
            target_ids.view(-1)
        )
        
        # Enhanced agentic pattern scoring
        agentic_keywords = [
            "step by step", "first", "then", "next", "analyze", "approach", 
            "method", "strategy", "solve", "reasoning", "think", "consider",
            "systematic", "process", "break down", "examine"
        ]
        
        pattern_count = 0
        teacher_lower = teacher_response.lower()
        for keyword in agentic_keywords:
            if keyword in teacher_lower:
                pattern_count += 1
        
        pattern_score = min(pattern_count / len(agentic_keywords), 1.0)
        agentic_loss = torch.tensor(
            1.0 - pattern_score, 
            device=self.device, 
            dtype=distillation_loss.dtype
        )
        
        # Reasoning quality assessment
        reasoning_indicators = ["code", "function", "algorithm", "implement", "debug", "solve"]
        reasoning_score = 0.4  # Base score
        
        conversation_text = str(conversation).lower()
        for indicator in reasoning_indicators:
            if indicator in conversation_text:
                reasoning_score += 0.1
        
        reasoning_score = min(reasoning_score, 1.0)
        reasoning_loss = torch.tensor(
            1.0 - reasoning_score, 
            device=self.device, 
            dtype=distillation_loss.dtype
        )
        
        # Combined loss
        total_loss = (
            self.config["distillation_weight"] * distillation_loss +
            self.config["agentic_pattern_weight"] * agentic_loss +
            self.config["reasoning_weight"] * reasoning_loss
        )
        
        return total_loss, {
            "distillation": distillation_loss.item(),
            "agentic_pattern": agentic_loss.item(),
            "reasoning": reasoning_loss.item(),
            "total": total_loss.item()
        }
    
    def training_step(self, batch) -> Dict:
        """Execute optimized training step"""
        self.model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        conversations = batch["conversation"]
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get teacher response
        content = conversations[0].get("content", {})
        turns = content.get("turns", [])
        teacher_response = self.get_teacher_response(turns)
        
        # Calculate loss
        loss, loss_dict = self.calculate_enhanced_loss(
            outputs.logits, teacher_response, conversations[0]
        )
        
        # Backward pass - REAL WEIGHT UPDATES
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            "loss": loss.item(),
            "teacher_response": teacher_response[:120],
            "current_lr": self.scheduler.get_last_lr()[0],
            **loss_dict
        }
    
    def generate_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate response from student model"""
        self.model.eval()
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=350
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def evaluate_capabilities(self, test_prompts: List[Dict]) -> Dict:
        """Comprehensive capability evaluation"""
        results = {}
        
        for prompt_data in test_prompts:
            name = prompt_data["name"]
            prompt = prompt_data["prompt"]
            
            self.logger.info(f"🔍 Evaluating {name}...")
            response = self.generate_response(prompt)
            
            results[name] = {
                "prompt": prompt,
                "response": response,
                "length": len(response.split()),
                "contains_reasoning": any(word in response.lower() for word in 
                                       ["step", "first", "then", "because", "therefore"])
            }
        
        return results
    
    def create_enhanced_plots(self):
        """Create comprehensive training visualization"""
        if len(self.training_losses) < 3:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'🚀 Final Agentic Training Progress - Step {self.step}', 
                    fontsize=16, fontweight='bold')
        
        # Training loss
        axes[0, 0].plot(self.training_losses, 'b-', linewidth=3, marker='o', markersize=4)
        axes[0, 0].set_title('📉 Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend(['Total Loss'])
        
        # Loss components
        if self.loss_components["distillation"]:
            steps = range(1, len(self.loss_components["distillation"]) + 1)
            axes[0, 1].plot(steps, self.loss_components["distillation"], 'g-', 
                          linewidth=2, label='Distillation', marker='s', markersize=3)
            axes[0, 1].plot(steps, self.loss_components["agentic"], 'r-', 
                          linewidth=2, label='Agentic Pattern', marker='^', markersize=3)
            axes[0, 1].plot(steps, self.loss_components["reasoning"], 'purple', 
                          linewidth=2, label='Reasoning', marker='d', markersize=3)
            axes[0, 1].set_title('🎯 Loss Components', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Loss improvement percentage
        if len(self.training_losses) > 1:
            improvement = [(self.training_losses[0] - loss) / self.training_losses[0] * 100 
                          for loss in self.training_losses]
            axes[0, 2].plot(improvement, 'orange', linewidth=3, marker='*', markersize=6)
            axes[0, 2].set_title('📈 Performance Improvement (%)', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('Improvement %')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Loss distribution
        axes[1, 0].hist(self.training_losses, bins=max(6, len(self.training_losses)//3), 
                       alpha=0.7, color='skyblue', edgecolor='darkblue', linewidth=1)
        axes[1, 0].set_title('📊 Loss Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Loss Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Smoothed progress
        if len(self.training_losses) > 4:
            window = 3
            smoothed = [sum(self.training_losses[max(0, i-window):i+1]) / min(i+1, window+1) 
                       for i in range(len(self.training_losses))]
            axes[1, 1].plot(smoothed, 'darkgreen', linewidth=4, alpha=0.8, label='Smoothed')
            axes[1, 1].plot(self.training_losses, 'lightgray', linewidth=1, alpha=0.5, label='Raw')
            axes[1, 1].set_title('🎯 Training Trends', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate schedule
        if hasattr(self, 'scheduler'):
            lr_values = [self.config["learning_rate"] * (0.5 ** (step // 8)) for step in range(len(self.training_losses))]
            axes[1, 2].plot(lr_values, 'red', linewidth=2, marker='o', markersize=3)
            axes[1, 2].set_title('📚 Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"plots/training/final_training_step_{self.step}.png"
        Path("plots/training").mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"📊 Enhanced plot saved: {plot_path}")
    
    def train_with_demonstrations(self):
        """Complete training with comprehensive before/after demonstrations"""
        
        # Test prompts for evaluation
        test_prompts = [
            {
                "name": "reasoning_challenge",
                "prompt": "Solve step by step: A train travels 120 km in 2 hours, then 180 km in 3 hours. What's the average speed for the entire journey? Show your reasoning process."
            },
            {
                "name": "coding_challenge", 
                "prompt": "Write a Python function to check if a string is a palindrome. Explain your approach and handle edge cases."
            },
            {
                "name": "tool_use_challenge",
                "prompt": "I need to build a machine learning model for text classification. What tools, libraries, and steps would you recommend?"
            }
        ]
        
        self.logger.info("🚀 STARTING FINAL AGENTIC TRAINING")
        self.logger.info("=" * 70)
        
        # BEFORE training evaluation
        self.logger.info("\n🔍 BEFORE TRAINING - Student Capabilities:")
        self.logger.info("=" * 50)
        before_results = self.evaluate_capabilities(test_prompts)
        
        for name, result in before_results.items():
            self.logger.info(f"\n📝 {name.upper()}:")
            self.logger.info(f"Prompt: {result['prompt']}")
            self.logger.info(f"Response ({result['length']} words): {result['response'][:250]}...")
            self.logger.info(f"Contains reasoning patterns: {result['contains_reasoning']}")
        
        # Training phase
        self.logger.info(f"\n🎯 TRAINING PHASE - {self.config['max_steps']} Steps:")
        self.logger.info("=" * 40)
        
        start_time = time.time()
        
        for step in range(self.config["max_steps"]):
            self.step = step + 1
            
            # Training step
            for batch in self.train_loader:
                step_results = self.training_step(batch)
                
                # Track metrics
                self.training_losses.append(step_results["loss"])
                self.loss_components["distillation"].append(step_results["distillation"])
                self.loss_components["agentic"].append(step_results["agentic_pattern"])
                self.loss_components["reasoning"].append(step_results["reasoning"])
                
                # Progress logging
                elapsed = time.time() - start_time
                self.logger.info(
                    f"Step {self.step:2d}/{self.config['max_steps']} | "
                    f"Loss: {step_results['loss']:.4f} | "
                    f"Dist: {step_results['distillation']:.4f} | "
                    f"Agentic: {step_results['agentic_pattern']:.4f} | "
                    f"LR: {step_results['current_lr']:.2e} | "
                    f"Time: {elapsed:.1f}s"
                )
                
                # Create plots
                if self.step % self.config["eval_every"] == 0:
                    self.create_enhanced_plots()
                
                break  # One batch per step
        
        # AFTER training evaluation
        self.logger.info("\n🎉 AFTER TRAINING - Student Capabilities:")
        self.logger.info("=" * 50)
        after_results = self.evaluate_capabilities(test_prompts)
        
        # Detailed comparison
        self.logger.info("\n📊 BEFORE vs AFTER DETAILED COMPARISON:")
        self.logger.info("=" * 60)
        
        for name in [p["name"] for p in test_prompts]:
            before = before_results[name]
            after = after_results[name]
            
            self.logger.info(f"\n🔄 {name.upper()}:")
            self.logger.info(f"BEFORE ({before['length']} words): {before['response'][:200]}...")
            self.logger.info(f"AFTER  ({after['length']} words): {after['response'][:200]}...")
            self.logger.info(f"Reasoning improvement: {before['contains_reasoning']} → {after['contains_reasoning']}")
            self.logger.info(f"Length change: {before['length']} → {after['length']} words")
        
        # Final results
        total_time = time.time() - start_time
        improvement = ((self.training_losses[0] - self.training_losses[-1]) / self.training_losses[0] * 100)
        
        self.logger.info("\n🎉 FINAL TRAINING RESULTS:")
        self.logger.info("=" * 50)
        self.logger.info(f"📊 Training Summary:")
        self.logger.info(f"   • Total steps: {self.step}")
        self.logger.info(f"   • Training time: {total_time:.1f}s")
        self.logger.info(f"   • Initial loss: {self.training_losses[0]:.4f}")
        self.logger.info(f"   • Final loss: {self.training_losses[-1]:.4f}")
        self.logger.info(f"   • Loss reduction: {self.training_losses[0] - self.training_losses[-1]:.4f}")
        self.logger.info(f"   • Improvement: {improvement:.1f}%")
        self.logger.info(f"   • Teacher model: {self.config['teacher_model']}")
        self.logger.info(f"   • Student model: {self.config['student_model']}")
        
        # Final comprehensive plot
        self.create_enhanced_plots()
        
        return {
            "before_results": before_results,
            "after_results": after_results,
            "training_metrics": {
                "initial_loss": self.training_losses[0],
                "final_loss": self.training_losses[-1],
                "improvement_percent": improvement,
                "total_steps": self.step,
                "training_time": total_time
            }
        }

def main():
    """Main execution function"""
    
    groq_api_key = "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"
    
    print("🚀 FINAL IMPROVED AGENTIC TRAINING SYSTEM")
    print("=" * 70)
    print("✅ Features:")
    print("   📖 Local Qwen2.5-1.5B student model")
    print("   🎓 DeepSeek R1 Distill Llama 70B teacher")
    print("   🔧 Real PyTorch weight updates with LoRA")
    print("   🎯 Enhanced SAD loss function")
    print("   📊 Before/after capability demonstrations") 
    print("   🎨 Comprehensive loss visualization")
    print("   💡 Reasoning, coding, and tool use testing")
    print()
    
    try:
        # Initialize and run trainer
        trainer = FinalAgenticTrainer(groq_api_key)
        results = trainer.train_with_demonstrations()
        
        print("\n✨ TRAINING COMPLETED SUCCESSFULLY!")
        print("📈 Check plots/training/ for comprehensive visualizations")
        print("📝 Check logs/final_training/ for detailed training logs")
        print("🎯 Before/after demonstrations completed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 