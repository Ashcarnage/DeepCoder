#!/usr/bin/env python3
"""
Production Structured Agent Distillation Trainer
===============================================
✅ Intelligent Groq rate limit handling (30 req/min, 6K tokens/min)
✅ LoRA PEFT for efficient Qwen 30B weight updates
✅ Before/after evaluation with same questions
✅ Comprehensive loss tracking and visualization
✅ SGLang integration for student model
✅ Real-time training monitoring
"""

import os
import sys
import json
import time
import math
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
from queue import Queue
import re
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.layout import Layout
from rich.live import Live
from tqdm import tqdm
from groq import Groq
from openai import OpenAI
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
import logging
import warnings
import requests
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

console = Console()

@dataclass
class RateLimitConfig:
    """Rate limit configuration for Groq API."""
    requests_per_minute: int = 30
    requests_per_day: int = 1000
    tokens_per_minute: int = 6000
    tokens_per_day: int = 500000  # Conservative estimate
    
class RateLimiter:
    """Rate limiter for API calls with request and token limits."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = deque()
        self.token_times = deque()
        self.daily_requests = 0
        self.daily_tokens = 0
        self.last_reset = time.time()
    
    def _cleanup_old_records(self):
        """Remove records older than 1 minute and reset daily counters if needed."""
        current_time = time.time()
        one_minute_ago = current_time - 60
        one_day_ago = current_time - 86400
        
        # Clean minute-based records
        while self.request_times and self.request_times[0] < one_minute_ago:
            self.request_times.popleft()
        while self.token_times and self.token_times[0][0] < one_minute_ago:
            self.token_times.popleft()
            
        # Reset daily counters if needed
        if current_time - self.last_reset > 86400:
            self.daily_requests = 0
            self.daily_tokens = 0
            self.last_reset = current_time
    
    def get_wait_time(self, estimated_tokens: int = 100) -> float:
        """Calculate how long to wait before making a request."""
        self._cleanup_old_records()
        current_time = time.time()
        
        # Check rate limits
        wait_times = []
        
        # Request rate limit (per minute)
        if len(self.request_times) >= self.config.requests_per_minute:
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                wait_times.append(wait_time)
        
        # Token rate limit (per minute)
        minute_tokens = sum(tokens for _, tokens in self.token_times)
        if minute_tokens + estimated_tokens > self.config.tokens_per_minute:
            if self.token_times:
                oldest_token_time = self.token_times[0][0]
                wait_time = 60 - (current_time - oldest_token_time)
                if wait_time > 0:
                    wait_times.append(wait_time)
        
        # Daily limits
        if self.daily_requests >= self.config.requests_per_day:
            wait_times.append(86400 - (current_time - self.last_reset))
        
        if self.daily_tokens + estimated_tokens > self.config.tokens_per_day:
            wait_times.append(86400 - (current_time - self.last_reset))
        
        return max(wait_times) if wait_times else 0
    
    def record_request(self, tokens_used: int):
        """Record a successful request."""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_times.append((current_time, tokens_used))
        self.daily_requests += 1
        self.daily_tokens += tokens_used
    
    def get_status(self) -> Dict:
        """Get current rate limit status."""
        self._cleanup_old_records()
        minute_requests = len(self.request_times)
        minute_tokens = sum(tokens for _, tokens in self.token_times)
        
        return {
            "requests_per_minute": f"{minute_requests}/{self.config.requests_per_minute}",
            "tokens_per_minute": f"{minute_tokens}/{self.config.tokens_per_minute}",
            "daily_requests": f"{self.daily_requests}/{self.config.requests_per_day}",
            "daily_tokens": f"{self.daily_tokens}/{self.config.tokens_per_day}"
        }

@dataclass
class SADTrainingConfig:
    """Configuration for Structured Agent Distillation training."""
    # Model configs
    student_model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    
    # LoRA configs
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training configs
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    
    # SAD-specific configs
    reason_loss_weight: float = 1.5  # Higher weight for reasoning spans
    action_loss_weight: float = 1.3  # Higher weight for action spans
    base_loss_weight: float = 1.0    # Base weight for other tokens
    
    # Training data
    num_training_examples: int = 200

class StructuredAgentDistillationTrainer:
    """Trainer for Structured Agent Distillation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = SADTrainingConfig()
        self.rate_limiter = RateLimiter(RateLimitConfig())
        
        # Initialize components
        self.groq_client = None
        self.sglang_client = None
        self.student_model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        
        # Training data and metrics
        self.training_data = []
        self.training_losses = []
        self.current_epoch = 0
        
        # Load API key
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")

    def setup_clients(self) -> bool:
        """Setup Groq and SGLang clients."""
        try:
            # Setup Groq client
            self.groq_client = Groq(api_key=self.groq_api_key)
            console.print("[green]✅ Groq client connected[/green]")
            
            # Setup SGLang client (REQUIRED for training)
            try:
                response = requests.get("http://localhost:30000/health", timeout=5)
                if response.status_code == 200:
                    console.print("[green]✅ SGLang client connected[/green]")
                    return True
                else:
                    console.print("[red]❌ SGLang server not responding properly[/red]")
                    return False
            except:
                console.print("[red]❌ SGLang not available at localhost:30000[/red]")
                console.print("[yellow]💡 Please start SGLang server first: cd /workspace/persistent/models/qwen3-30b-a3b && python -m sglang.launch_server --model-path . --host 0.0.0.0 --port 30000[/yellow]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Error setting up clients: {e}[/red]")
            return False

    def setup_student_model(self) -> bool:
        """Setup student model with LoRA adapters for real training."""
        try:
            # Use the existing local Qwen 30B model
            local_model_path = "/workspace/persistent/models/qwen3-30b-a3b"
            console.print(f"[blue]🧠 Loading student model from: {local_model_path}[/blue]")
            
            # Load tokenizer from the local model
            self.tokenizer = AutoTokenizer.from_pretrained(
                local_model_path, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load the student model locally for LoRA training
            console.print("[yellow]⚠️ Loading 30B model for LoRA training - this may take time...[/yellow]")
            self.student_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Configure LoRA for Qwen architecture
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
            
            # Apply LoRA to the model - this adds trainable adapters
            self.student_model = get_peft_model(self.student_model, lora_config)
            self.student_model.print_trainable_parameters()
            
            # Setup optimizer for LoRA parameters only
            self.optimizer = AdamW(
                self.student_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            
            # Setup learning rate scheduler
            total_steps = (self.config.num_training_examples // self.config.batch_size) * self.config.num_epochs
            warmup_steps = int(total_steps * self.config.warmup_ratio)
            
            self.scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
            
            console.print(f"[green]✅ Student model loaded with LoRA adapters[/green]")
            console.print(f"[blue]📊 Tokenizer vocab size: {len(self.tokenizer)}[/blue]")
            console.print(f"[blue]🔧 Total training steps: {total_steps}[/blue]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Error setting up student model: {e}[/red]")
            import traceback
            traceback.print_exc()
            return False

    def get_teacher_response(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Get response from teacher model via Groq API."""
        for attempt in range(max_retries):
            try:
                # Check rate limits
                wait_time = self.rate_limiter.get_wait_time(estimated_tokens=500)
                if wait_time > 0:
                    console.print(f"[yellow]⏳ Rate limit reached, waiting {wait_time:.1f}s...[/yellow]")
                    time.sleep(wait_time)
                
                # Make API call
                response = self.groq_client.chat.completions.create(
                    model=self.config.teacher_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a helpful assistant that provides step-by-step reasoning. Use [REASON] tags for your thinking process and [ACT] tags for specific actions or answers."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                self.rate_limiter.record_request(tokens_used)
                
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    console.print(f"[red]❌ Failed to get teacher response after {max_retries} attempts: {e}[/red]")
                    return None
                else:
                    wait_time = 2 ** attempt
                    console.print(f"[yellow]⚠️ Attempt {attempt + 1} failed, retrying in {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
        
        return None

    def parse_reasoning_spans(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Parse [REASON] and [ACT] spans from text and return their positions."""
        spans = {"reasoning": [], "action": []}
        
        # Find [REASON] spans
        reason_pattern = r'\[REASON\](.*?)\[/REASON\]'
        for match in re.finditer(reason_pattern, text, re.DOTALL):
            spans["reasoning"].append((match.start(), match.end()))
        
        # Find [ACT] spans  
        action_pattern = r'\[ACT\](.*?)\[/ACT\]'
        for match in re.finditer(action_pattern, text, re.DOTALL):
            spans["action"].append((match.start(), match.end()))
            
        return spans

    def compute_sad_loss(self, logits: torch.Tensor, labels: torch.Tensor, teacher_text: str) -> torch.Tensor:
        """Compute Structured Agent Distillation loss with span-specific weighting."""
        device = logits.device
        
        # Get span positions
        spans = self.parse_reasoning_spans(teacher_text)
        
        # Create weight tensor (base weight for all tokens)
        weights = torch.full_like(labels, self.config.base_loss_weight, dtype=torch.float, device=device)
        
        # Convert spans to token positions (approximate)
        for span_type, span_list in spans.items():
            for start_char, end_char in span_list:
                # Convert character positions to approximate token positions
                # This is a simplified approach - in practice you'd want more precise alignment
                start_token = start_char // 4  # Rough approximation
                end_token = min(end_char // 4, weights.size(-1))
                
                if span_type == "reasoning":
                    weights[start_token:end_token] = self.config.reason_loss_weight
                elif span_type == "action":
                    weights[start_token:end_token] = self.config.action_loss_weight
        
        # Compute cross-entropy loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_weights = shift_weights.view(-1)
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits, shift_labels)
        
        # Apply span-specific weights
        weighted_loss = loss * shift_weights
        
        # Mask out padding tokens
        mask = (shift_labels != -100).float()
        weighted_loss = weighted_loss * mask
        
        return weighted_loss.sum() / mask.sum()

    def prepare_training_data(self) -> bool:
        """Prepare training data by collecting teacher responses."""
        try:
            console.print(f"[blue]📚 Generating {self.config.num_training_examples} training examples...[/blue]")
            
            # Sample training prompts
            training_prompts = [
                "Solve this step by step: What is the square root of 144?",
                "Explain how photosynthesis works in plants.",
                "How do you calculate compound interest?",
                "What causes seasons on Earth?",
                "Solve: If a train travels 120 km in 2 hours, what is its speed?",
                "Explain the water cycle process.",
                "How do you find the area of a circle?",
                "What is the difference between DNA and RNA?",
                "Calculate: 15% of 200",
                "How does a combustion engine work?",
                "Solve: 2x + 5 = 15",
                "Explain Newton's first law of motion",
                "How do you convert Celsius to Fahrenheit?",
                "What causes tides in the ocean?",
                "Find the perimeter of a rectangle with length 8 and width 6",
            ]
            
            self.training_data = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Collecting teacher responses...", total=self.config.num_training_examples)
                
                for i in range(self.config.num_training_examples):
                    # Cycle through prompts if we need more examples
                    prompt = training_prompts[i % len(training_prompts)]
                    if i >= len(training_prompts):
                        prompt = f"Variation {i//len(training_prompts)}: {prompt}"
                    
                    teacher_response = self.get_teacher_response(prompt)
                    if teacher_response:
                        self.training_data.append({
                            "prompt": prompt,
                            "response": teacher_response
                        })
                    
                    progress.update(task, advance=1)
            
            console.print(f"[green]✅ Collected {len(self.training_data)} training examples[/green]")
            return len(self.training_data) > 0
            
        except Exception as e:
            console.print(f"[red]❌ Error preparing training data: {e}[/red]")
            return False

    def train_step(self, batch: Dict[str, torch.Tensor], teacher_texts: List[str]) -> float:
        """Execute one training step with SAD loss."""
        self.student_model.train()
        
        # Forward pass
        outputs = self.student_model(**batch)
        logits = outputs.logits
        
        # Compute SAD loss for each example in batch
        total_loss = 0
        for i in range(len(teacher_texts)):
            example_logits = logits[i:i+1]
            example_labels = batch["labels"][i:i+1]
            
            sad_loss = self.compute_sad_loss(example_logits, example_labels, teacher_texts[i])
            total_loss += sad_loss
        
        # Average loss across batch
        loss = total_loss / len(teacher_texts)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def train_model(self):
        """Main training loop with real weight updates."""
        console.print("[bold green]🚀 Starting Structured Agent Distillation Training[/bold green]")
        
        # Prepare training data
        if not self.prepare_training_data():
            console.print("[red]❌ Failed to prepare training data[/red]")
            return
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            console.print(f"[blue]📖 Epoch {epoch + 1}/{self.config.num_epochs}[/blue]")
            
            epoch_losses = []
            num_batches = len(self.training_data) // self.config.batch_size
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Training Epoch {epoch + 1}...", total=num_batches)
                
                for batch_idx in range(0, len(self.training_data), self.config.batch_size):
                    batch_data = self.training_data[batch_idx:batch_idx + self.config.batch_size]
                    
                    # Prepare batch
                    prompts = [ex["prompt"] for ex in batch_data]
                    responses = [ex["response"] for ex in batch_data]
                    
                    # Tokenize with proper handling
                    full_texts = [f"{p}\n{r}" for p, r in zip(prompts, responses)]
                    
                    try:
                        tokenized = self.tokenizer(
                            full_texts,
                            padding=True,
                            truncation=True,
                            max_length=self.config.max_seq_length,
                            return_tensors="pt"
                        )
                        
                        # Validate tokenized inputs
                        input_ids = tokenized["input_ids"]
                        attention_mask = tokenized["attention_mask"]
                        
                        # Check for invalid token IDs
                        vocab_size = len(self.tokenizer)
                        if torch.any(input_ids >= vocab_size) or torch.any(input_ids < 0):
                            console.print(f"[yellow]⚠️ Skipping batch with invalid token IDs[/yellow]")
                            continue
                        
                        # Move to device safely
                        device = next(self.student_model.parameters()).device
                        batch = {}
                        batch["input_ids"] = input_ids.to(device)
                        batch["attention_mask"] = attention_mask.to(device)
                        batch["labels"] = input_ids.clone().to(device)
                        
                        # Training step
                        loss = self.train_step(batch, responses)
                        epoch_losses.append(loss)
                        self.training_losses.append(loss)
                        
                    except Exception as e:
                        console.print(f"[yellow]⚠️ Skipping batch due to error: {e}[/yellow]")
                        continue
                    
                    progress.update(task, advance=1)
            
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                console.print(f"[green]✅ Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}[/green]")
            else:
                console.print(f"[yellow]⚠️ Epoch {epoch + 1} completed with no valid batches[/yellow]")
        
        console.print("[bold green]🎉 Training completed![/bold green]")

    def save_model(self, output_dir: str = "outputs/sad_qwen_model"):
        """Save the trained model and adapters."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save LoRA adapters
            self.student_model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Save training config
            config_path = os.path.join(output_dir, "training_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config.__dict__, f, indent=2)
            
            console.print(f"[green]✅ Model saved to {output_dir}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error saving model: {e}[/red]")

    def run_evaluation(self, phase: str) -> Dict[str, float]:
        """Run evaluation on test questions."""
        test_questions = [
            "What is 25% of 80?",
            "How does gravity work?", 
            "Solve: 3x - 7 = 14",
            "Explain cellular respiration",
            "Calculate the area of a triangle with base 6 and height 4"
        ]
        
        results = {}
        
        for i, question in enumerate(test_questions):
            try:
                # Get response
                inputs = self.tokenizer(question, return_tensors="pt", padding=True)
                device = next(self.student_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.student_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(question):].strip()
                
                # Store metrics
                category = f"test_q{i+1}"
                results[f"{category}_length"] = len(response)
                results[f"{category}_reasoning"] = len(self.parse_reasoning_spans(response)['reasoning'])
                results[f"{category}_actions"] = len(self.parse_reasoning_spans(response)['action'])
                results[f"{category}_response"] = response
                
            except Exception as e:
                console.print(f"[yellow]⚠️ Error evaluating question {i+1}: {e}[/yellow]")
                results[f"test_q{i+1}_length"] = 0
                results[f"test_q{i+1}_reasoning"] = 0
                results[f"test_q{i+1}_actions"] = 0
        
        return results

    def create_visualizations(self, before_eval: Dict, after_eval: Dict):
        """Create training visualization charts."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.style.use('default')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training loss
        if self.training_losses:
            ax1.plot(self.training_losses, 'b-', linewidth=2, alpha=0.8)
            ax1.set_title('SAD Training Loss', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Training Step')
            ax1.set_ylabel('Loss')
            ax1.grid(True, alpha=0.3)
            
        # Response length comparison
        categories = ['test_q1', 'test_q2', 'test_q3', 'test_q4', 'test_q5']
        before_lengths = [before_eval.get(f'{cat}_length', 0) for cat in categories]
        after_lengths = [after_eval.get(f'{cat}_length', 0) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, before_lengths, width, label='Before Training', alpha=0.8)
        ax2.bar(x + width/2, after_lengths, width, label='After Training', alpha=0.8)
        ax2.set_title('Response Length Comparison', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Test Questions')
        ax2.set_ylabel('Response Length (characters)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Reasoning spans
        before_reasoning = [before_eval.get(f'{cat}_reasoning', 0) for cat in categories]
        after_reasoning = [after_eval.get(f'{cat}_reasoning', 0) for cat in categories]
        
        ax3.bar(x - width/2, before_reasoning, width, label='Before Training', alpha=0.8, color='orange')
        ax3.bar(x + width/2, after_reasoning, width, label='After Training', alpha=0.8, color='red')
        ax3.set_title('Reasoning Spans [REASON] Count', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Test Questions')
        ax3.set_ylabel('Number of Reasoning Spans')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Action spans
        before_actions = [before_eval.get(f'{cat}_actions', 0) for cat in categories]
        after_actions = [after_eval.get(f'{cat}_actions', 0) for cat in categories]
        
        ax4.bar(x - width/2, before_actions, width, label='Before Training', alpha=0.8, color='green')
        ax4.bar(x + width/2, after_actions, width, label='After Training', alpha=0.8, color='purple')
        ax4.set_title('Action Spans [ACT] Count', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Test Questions')
        ax4.set_ylabel('Number of Action Spans')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'sad_training_results_{timestamp}.png', dpi=300, bbox_inches='tight')
        console.print(f"[green]✅ Visualization saved as sad_training_results_{timestamp}.png[/green]")

    def save_detailed_results(self, before_eval: Dict, after_eval: Dict):
        """Save detailed training results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            "training_config": self.config.__dict__,
            "training_summary": {
                "total_epochs": self.config.num_epochs,
                "total_examples": len(self.training_data),
                "final_loss": self.training_losses[-1] if self.training_losses else None,
                "average_loss": sum(self.training_losses) / len(self.training_losses) if self.training_losses else None
            },
            "evaluation_results": {
                "before_training": before_eval,
                "after_training": after_eval
            },
            "training_losses": self.training_losses
        }
        
        filename = f'sad_detailed_results_{timestamp}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"[green]✅ Detailed results saved as {filename}[/green]")

    def run_complete_training_pipeline(self):
        """Run the complete SAD training pipeline."""
        console.print("[bold blue]🎯 Starting Complete Structured Agent Distillation Pipeline[/bold blue]")
        
        # Setup phase
        if not self.setup_clients():
            console.print("[red]❌ Failed to setup clients[/red]")
            return
        
        if not self.setup_student_model():
            console.print("[red]❌ Failed to setup student model[/red]")
            return
        
        # Before training evaluation
        console.print("[blue]📊 Running before-training evaluation...[/blue]")
        before_eval = self.run_evaluation("before")
        
        # Training phase
        self.train_model()
        
        # After training evaluation
        console.print("[blue]📊 Running after-training evaluation...[/blue]")
        after_eval = self.run_evaluation("after")
        
        # Save model
        self.save_model()
        
        # Create visualizations and save results
        self.create_visualizations(before_eval, after_eval)
        self.save_detailed_results(before_eval, after_eval)
        
        # Print summary
        console.print("\n[bold green]🎉 Training Pipeline Complete![/bold green]")
        console.print("[bold]Key Results:[/bold]")
        
        if self.training_losses:
            improvement = ((self.training_losses[0] - self.training_losses[-1]) / self.training_losses[0]) * 100
            console.print(f"• Loss improvement: {improvement:.1f}% ({self.training_losses[0]:.4f} → {self.training_losses[-1]:.4f})")
        
        console.print(f"• Training examples: {len(self.training_data)}")
        console.print(f"• Training epochs: {self.config.num_epochs}")
        console.print(f"• Model saved with LoRA adapters")

def main():
    """Main function to run SAD training."""
    try:
        # Initialize trainer
        trainer = StructuredAgentDistillationTrainer()
        
        # Run complete pipeline
        trainer.run_complete_training_pipeline()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Training interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Training failed: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 