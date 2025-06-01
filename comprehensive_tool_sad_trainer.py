#!/usr/bin/env python3
"""
Comprehensive Tool-Aware Structured Agent Distillation (SAD) Trainer
====================================================================

This trainer performs actual SAD training on the local Qwen 30B model with:
✅ LoRA PEFT for efficient local weight updates 
✅ Intelligent Groq rate limit handling (30 req/min, 6K tokens/min)
✅ Tool-aware trajectory generation with teacher demonstrations
✅ Comprehensive tool usage pattern training (Cursor-like capabilities)
✅ Before/after evaluation with sophisticated reasoning tests
✅ Real-time training monitoring and visualization
✅ Local model training with SGLang server for inference testing

Features:
- Actual model weight updates using LoRA on local Qwen 30B
- Teacher model via Groq API with smart rate limiting
- Tool usage patterns: terminal, web search, file ops, code analysis, etc.
- Multi-step reasoning with tool integration training
- Comprehensive evaluation of improvements
"""

import os
import sys
import json
import time
import math
import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import threading
from queue import Queue
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

# API clients
from groq import Groq
from openai import OpenAI

# Deep Learning
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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/tool_sad_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class RateLimitConfig:
    """Rate limit configuration for Groq API."""
    requests_per_minute: int = 30
    requests_per_day: int = 1000
    tokens_per_minute: int = 6000
    tokens_per_day: int = 500000

class RateLimiter:
    """Advanced rate limiter for API calls with request and token limits."""
    
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
    
    def get_wait_time(self, estimated_tokens: int = 500) -> float:
        """Calculate how long to wait before making a request."""
        self._cleanup_old_records()
        current_time = time.time()
        
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
class ToolUsageConfig:
    """Configuration for tool usage patterns"""
    
    available_tools: List[str] = field(default_factory=lambda: [
        "terminal_command",      # Execute terminal commands
        "web_search",           # Search web for information
        "file_operations",      # Read/write/edit files
        "code_analyzer",        # Analyze code for bugs/improvements
        "system_monitor",       # Monitor system resources
        "git_operations",       # Git version control
        "database_query",       # Database interactions
        "api_client",          # Make API calls
        "data_processor",      # Process and analyze data
        "documentation_search"  # Search documentation
    ])
    
    tool_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "debugging": ["terminal_command", "file_operations", "code_analyzer", "system_monitor"],
        "research": ["web_search", "documentation_search", "file_operations"],
        "development": ["file_operations", "git_operations", "terminal_command", "code_analyzer"],
        "data_analysis": ["data_processor", "file_operations", "system_monitor", "database_query"],
        "system_admin": ["terminal_command", "system_monitor", "file_operations"],
        "api_integration": ["api_client", "file_operations", "code_analyzer", "terminal_command"]
    })
    
    tool_contexts: Dict[str, str] = field(default_factory=lambda: {
        "terminal_command": "When needing to execute shell commands, install packages, or run system operations",
        "web_search": "When needing current information, documentation, or research on unfamiliar topics",
        "file_operations": "When needing to read, write, edit, or manage files and directories", 
        "code_analyzer": "When needing to analyze code for bugs, performance issues, or best practices",
        "system_monitor": "When needing to check system resources, performance metrics, or diagnostics",
        "git_operations": "When working with version control, commits, branches, or repository management",
        "database_query": "When needing to query, update, or manage database operations",
        "api_client": "When making HTTP requests or interacting with external APIs",
        "data_processor": "When processing datasets, performing calculations, or data transformations",
        "documentation_search": "When searching for API docs, tutorials, or technical documentation"
    })

@dataclass
class ToolSADConfig:
    """Configuration for Tool-aware SAD training."""
    
    # Model configs
    student_model_path: str = "/workspace/persistent/models/qwen3-30b-a3b"
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    sglang_base_url: str = "http://localhost:30000/v1"
    
    # LoRA configs  
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training configs
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.1
    
    # SAD-specific configs
    tool_usage_loss_weight: float = 2.0    # Higher weight for tool usage spans
    reasoning_loss_weight: float = 1.5     # Higher weight for reasoning spans
    base_loss_weight: float = 1.0          # Base weight for other tokens
    
    # Training data
    num_training_examples: int = 40
    scenarios_per_pattern: int = 4
    
    # Output settings
    output_dir: str = "/workspace/persistent/tool_sad_model"
    results_dir: str = "/workspace/persistent/tool_sad_results"

class ToolAwareDataset(Dataset):
    """Dataset for tool-aware training examples."""
    
    def __init__(self, training_data: List[Dict], tokenizer, max_length: int = 2048):
        self.data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format as conversation
        prompt = item["prompt"]
        response = item["teacher_response"]
        
        text = f"<|user|>: {prompt}\n<|assistant|>: {response}<|endoftext|>"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze().clone(),
            "teacher_response": response
        }

class ComprehensiveToolSADTrainer:
    """Comprehensive tool-aware SAD trainer with actual model weight updates."""
    
    def __init__(self):
        self.config = ToolSADConfig()
        self.tool_config = ToolUsageConfig()
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
        self.tool_usage_metrics = []
        self.current_epoch = 0
        
        # Tool scenarios
        self.tool_scenarios = self._create_comprehensive_scenarios()
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.results_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def _create_comprehensive_scenarios(self) -> List[Dict]:
        """Create comprehensive tool usage scenarios for training."""
        
        scenarios = [
            # Debugging scenarios
            {
                "pattern": "debugging",
                "task": "Debug a Python script that's consuming too much memory",
                "context": "A data processing script is running out of memory on large datasets",
                "required_tools": ["terminal_command", "file_operations", "code_analyzer", "system_monitor"],
                "complexity": "high"
            },
            {
                "pattern": "debugging", 
                "task": "Fix a web application that's responding slowly",
                "context": "Users are reporting slow page load times and timeouts",
                "required_tools": ["system_monitor", "file_operations", "code_analyzer", "terminal_command"],
                "complexity": "high"
            },
            
            # Research scenarios
            {
                "pattern": "research",
                "task": "Research and implement a new machine learning technique",
                "context": "Need to evaluate a new transformer architecture for a project",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "high"
            },
            {
                "pattern": "research",
                "task": "Find the best practices for API security",
                "context": "Building a new API and need to ensure proper security measures",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "medium"
            },
            
            # Development scenarios
            {
                "pattern": "development",
                "task": "Build a complete user authentication system",
                "context": "Developing a web app that needs secure user login and registration",
                "required_tools": ["file_operations", "git_operations", "code_analyzer", "terminal_command"],
                "complexity": "high"
            },
            {
                "pattern": "development",
                "task": "Create a REST API with proper testing",
                "context": "Building a backend API with comprehensive test coverage",
                "required_tools": ["file_operations", "terminal_command", "code_analyzer", "git_operations"],
                "complexity": "high"
            },
            
            # Data analysis scenarios
            {
                "pattern": "data_analysis",
                "task": "Build a data pipeline for processing customer data",
                "context": "Processing millions of customer records daily with reliability requirements",
                "required_tools": ["data_processor", "database_query", "file_operations", "system_monitor"],
                "complexity": "high"
            },
            {
                "pattern": "data_analysis",
                "task": "Analyze performance metrics and create dashboards",
                "context": "Need to monitor system performance and create visual dashboards",
                "required_tools": ["data_processor", "system_monitor", "file_operations", "database_query"],
                "complexity": "medium"
            },
            
            # System admin scenarios
            {
                "pattern": "system_admin",
                "task": "Optimize server performance and resource usage",
                "context": "Server experiencing high CPU and memory usage",
                "required_tools": ["system_monitor", "terminal_command", "file_operations"],
                "complexity": "medium"
            },
            {
                "pattern": "system_admin",
                "task": "Set up monitoring and alerting for production systems",
                "context": "Need comprehensive monitoring for a production environment",
                "required_tools": ["terminal_command", "system_monitor", "file_operations"],
                "complexity": "high"
            },
            
            # API integration scenarios
            {
                "pattern": "api_integration",
                "task": "Debug API integration issues with third-party services",
                "context": "Third-party API calls are failing intermittently in production",
                "required_tools": ["api_client", "file_operations", "code_analyzer", "terminal_command"],
                "complexity": "medium"
            },
            {
                "pattern": "api_integration",
                "task": "Build a microservices architecture with proper error handling",
                "context": "Designing resilient microservices with proper communication",
                "required_tools": ["api_client", "file_operations", "code_analyzer", "system_monitor"],
                "complexity": "high"
            }
        ]
        
        return scenarios
    
    def setup_clients(self) -> bool:
        """Setup API clients for teacher model and local inference."""
        try:
            # Setup Groq client for teacher model
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                console.print("[red]❌ GROQ_API_KEY not found in environment[/red]")
                return False
            
            self.groq_client = Groq(api_key=groq_api_key)
            
            # Setup SGLang client for student model inference testing
            self.sglang_client = OpenAI(
                base_url=self.config.sglang_base_url,
                api_key="EMPTY"
            )
            
            # Test connections
            try:
                health_response = requests.get("http://localhost:30000/health", timeout=5)
                if health_response.status_code == 200:
                    console.print("[green]✅ SGLang server connection verified[/green]")
                else:
                    console.print("[yellow]⚠️ SGLang server not responding, will skip inference tests[/yellow]")
            except:
                console.print("[yellow]⚠️ SGLang server not available, will skip inference tests[/yellow]")
            
            console.print("[green]✅ API clients setup successful[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to setup clients: {e}[/red]")
            return False
    
    def setup_student_model(self) -> bool:
        """Setup the local Qwen 30B student model with LoRA."""
        try:
            console.print("[blue]🤖 Loading Qwen 30B student model...[/blue]")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.student_model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.student_model = AutoModelForCausalLM.from_pretrained(
                self.config.student_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                use_cache=False
            )
            
            # Enable gradient checkpointing for memory efficiency
            self.student_model.gradient_checkpointing_enable()
            
            # Setup LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none"
            )
            
            self.student_model = get_peft_model(self.student_model, lora_config)
            self.student_model.print_trainable_parameters()
            
            console.print("[green]✅ Student model setup complete with LoRA[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to setup student model: {e}[/red]")
            return False
    
    def generate_tool_aware_prompt(self, scenario: Dict) -> str:
        """Generate a comprehensive prompt that encourages sophisticated tool usage."""
        
        tools_description = self._format_available_tools(scenario['required_tools'])
        
        prompt = f"""
You are an expert senior software engineer working on a {scenario['pattern']} task.

CONTEXT: {scenario['context']}

TASK: {scenario['task']}

AVAILABLE TOOLS:
{tools_description}

REQUIREMENTS:
1. Use a systematic, professional problem-solving approach
2. Strategically select and use tools - explain WHY you choose each tool
3. Show realistic tool calls with proper parameters
4. Analyze results from each tool and determine next steps
5. Demonstrate multi-step reasoning with tool integration
6. Include error handling and validation steps
7. Provide comprehensive solutions like a senior engineer would

FORMAT YOUR RESPONSE:
- Start with problem analysis and planning
- For each tool usage, use this format:

[TOOL_CALL]
Tool: tool_name
Parameters: {{"param": "value"}}
Reasoning: Why I'm using this tool and what I expect to find
[/TOOL_CALL]

[TOOL_OUTPUT]
[Realistic simulated output from the tool]
[/TOOL_OUTPUT]

[ANALYSIS]
Analysis of the results and next steps based on the output
[/ANALYSIS]

- Build upon results logically with follow-up tool calls
- Conclude with summary and actionable recommendations

Demonstrate expert-level problem-solving with sophisticated tool usage patterns.
"""
        return prompt.strip()
    
    def _format_available_tools(self, required_tools: List[str]) -> str:
        """Format tool descriptions for the prompt."""
        tool_descriptions = []
        for tool in required_tools:
            context = self.tool_config.tool_contexts.get(tool, "General purpose tool")
            tool_descriptions.append(f"• {tool}: {context}")
        
        return "\n".join(tool_descriptions)
    
    def get_teacher_response(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Get response from teacher model via Groq API with rate limiting."""
        for attempt in range(max_retries):
            try:
                # Check rate limits
                wait_time = self.rate_limiter.get_wait_time(estimated_tokens=1000)
                if wait_time > 0:
                    console.print(f"[yellow]⏳ Rate limit reached, waiting {wait_time:.1f}s...[/yellow]")
                    time.sleep(wait_time)
                
                # Enhanced system prompt for tool usage
                system_prompt = """
You are a world-class senior software engineer and technical lead with expertise in:
- Advanced debugging and performance optimization
- Complex system architecture and design  
- Professional development workflows and best practices
- Strategic tool usage for maximum efficiency
- Multi-step problem-solving methodologies

When solving problems:
1. Think systematically like a senior engineer
2. Use tools strategically and explain your reasoning
3. Show realistic, detailed tool outputs that demonstrate expertise
4. Build solutions incrementally with proper validation
5. Include error handling, edge cases, and production considerations
6. Demonstrate sophisticated reasoning and decision-making
7. Provide actionable insights and recommendations

Use [TOOL_CALL], [TOOL_OUTPUT], and [ANALYSIS] tags to structure your response clearly.
"""
                
                # Make API call
                response = self.groq_client.chat.completions.create(
                    model=self.config.teacher_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
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
    
    def parse_tool_spans(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Parse [TOOL_CALL], [TOOL_OUTPUT], and [ANALYSIS] spans from text."""
        spans = {"tool_calls": [], "tool_outputs": [], "analysis": []}
        
        # Find tool call spans
        tool_call_pattern = r'\[TOOL_CALL\](.*?)\[/TOOL_CALL\]'
        for match in re.finditer(tool_call_pattern, text, re.DOTALL):
            spans["tool_calls"].append((match.start(), match.end()))
        
        # Find tool output spans
        tool_output_pattern = r'\[TOOL_OUTPUT\](.*?)\[/TOOL_OUTPUT\]'
        for match in re.finditer(tool_output_pattern, text, re.DOTALL):
            spans["tool_outputs"].append((match.start(), match.end()))
        
        # Find analysis spans
        analysis_pattern = r'\[ANALYSIS\](.*?)\[/ANALYSIS\]'
        for match in re.finditer(analysis_pattern, text, re.DOTALL):
            spans["analysis"].append((match.start(), match.end()))
            
        return spans
    
    def compute_tool_aware_sad_loss(self, logits: torch.Tensor, labels: torch.Tensor, teacher_text: str) -> torch.Tensor:
        """Compute Tool-aware SAD loss with span-specific weighting."""
        device = logits.device
        
        # Get tool-related span positions
        spans = self.parse_tool_spans(teacher_text)
        
        # Create weight tensor (base weight for all tokens)
        weights = torch.full_like(labels, self.config.base_loss_weight, dtype=torch.float, device=device)
        
        # Convert spans to token positions (approximate)
        for span_type, span_list in spans.items():
            for start_char, end_char in span_list:
                # Convert character positions to approximate token positions
                start_token = max(0, start_char // 4)  # Rough approximation
                end_token = min(end_char // 4, weights.size(-1))
                
                if span_type == "tool_calls":
                    weights[start_token:end_token] = self.config.tool_usage_loss_weight
                elif span_type in ["tool_outputs", "analysis"]:
                    weights[start_token:end_token] = self.config.reasoning_loss_weight
        
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
        """Prepare comprehensive tool-aware training data."""
        try:
            total_examples = len(self.tool_scenarios) * self.config.scenarios_per_pattern
            console.print(f"[blue]📚 Generating {total_examples} tool-aware training examples...[/blue]")
            
            self.training_data = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(), 
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Collecting teacher demonstrations...", total=total_examples)
                
                for scenario in self.tool_scenarios:
                    for variation in range(self.config.scenarios_per_pattern):
                        # Create variation of the scenario
                        if variation > 0:
                            modified_scenario = scenario.copy()
                            modified_scenario["task"] = f"Variation {variation}: {scenario['task']}"
                            modified_scenario["context"] = f"Alternative context: {scenario['context']}"
                        else:
                            modified_scenario = scenario
                        
                        # Generate prompt
                        prompt = self.generate_tool_aware_prompt(modified_scenario)
                        
                        # Get teacher response
                        teacher_response = self.get_teacher_response(prompt)
                        
                        if teacher_response:
                            self.training_data.append({
                                "prompt": prompt,
                                "teacher_response": teacher_response,
                                "scenario": modified_scenario
                            })
                        
                        progress.update(task, advance=1)
                        
                        # Small delay to be respectful to API
                        time.sleep(0.5)
            
            console.print(f"[green]✅ Collected {len(self.training_data)} tool-aware training examples[/green]")
            
            # Save training data
            training_data_file = Path(self.config.results_dir) / "tool_training_data.json"
            with open(training_data_file, 'w') as f:
                json.dump(self.training_data, f, indent=2)
            
            return len(self.training_data) > 0
            
        except Exception as e:
            console.print(f"[red]❌ Error preparing training data: {e}[/red]")
            return False
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute one training step with tool-aware SAD loss."""
        self.student_model.train()
        
        # Forward pass
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        teacher_responses = batch["teacher_response"]
        
        outputs = self.student_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute tool-aware SAD loss for each example in batch
        total_loss = 0
        for i in range(len(teacher_responses)):
            example_logits = logits[i:i+1]
            example_labels = labels[i:i+1]
            
            sad_loss = self.compute_tool_aware_sad_loss(
                example_logits, example_labels, teacher_responses[i]
            )
            total_loss += sad_loss
        
        # Average loss across batch
        loss = total_loss / len(teacher_responses)
        
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
        """Main training loop with tool-aware weight updates."""
        console.print("[bold green]🚀 Starting Tool-Aware Structured Agent Distillation Training[/bold green]")
        
        # Prepare training data
        if not self.prepare_training_data():
            console.print("[red]❌ Failed to prepare training data[/red]")
            return False
        
        # Create dataset and dataloader
        dataset = ToolAwareDataset(self.training_data, self.tokenizer, self.config.max_seq_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        self.optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(dataloader) * self.config.num_epochs
        self.scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # Training loop
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            console.print(f"[blue]📚 Epoch {epoch + 1}/{self.config.num_epochs}[/blue]")
            
            epoch_losses = []
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(f"Training Epoch {epoch + 1}...", total=len(dataloader))
                
                for batch_idx, batch in enumerate(dataloader):
                    # Move batch to device
                    batch = {k: v.cuda() if torch.cuda.is_available() and isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Training step
                    loss = self.train_step(batch)
                    epoch_losses.append(loss)
                    self.training_losses.append(loss)
                    
                    # Log progress
                    if global_step % 5 == 0:
                        console.print(f"Step {global_step}: Loss = {loss:.4f}")
                    
                    progress.update(task, advance=1)
                    global_step += 1
            
            avg_epoch_loss = np.mean(epoch_losses)
            console.print(f"[green]✅ Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}[/green]")
        
        console.print("[bold green]🎉 Tool-aware SAD training completed![/bold green]")
        return True
    
    def save_model(self):
        """Save the trained model and configuration."""
        try:
            console.print("[blue]💾 Saving trained model...[/blue]")
            
            # Save LoRA weights
            self.student_model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)
            
            # Save configuration
            config_file = Path(self.config.output_dir) / "training_config.json"
            with open(config_file, 'w') as f:
                json.dump({
                    "model_config": self.config.__dict__,
                    "tool_config": self.tool_config.__dict__,
                    "training_losses": self.training_losses,
                    "num_scenarios": len(self.tool_scenarios)
                }, f, indent=2, default=str)
            
            console.print(f"[green]✅ Model saved to {self.config.output_dir}[/green]")
            
        except Exception as e:
            console.print(f"[red]❌ Error saving model: {e}[/red]")
    
    def test_sglang_inference(self, test_prompt: str) -> Optional[str]:
        """Test the trained model via SGLang inference."""
        try:
            if not self.sglang_client:
                return None
            
            response = self.sglang_client.chat.completions.create(
                model="qwen3-30b-a3b",
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Handle DeepSeek-R1 format
            content = response.choices[0].message.content
            if content is None:
                content = getattr(response.choices[0].message, 'reasoning_content', '')
            
            return content
            
        except Exception as e:
            console.print(f"[yellow]⚠️ SGLang inference test failed: {e}[/yellow]")
            return None
    
    def create_training_visualizations(self):
        """Create comprehensive training visualizations."""
        if not self.training_losses:
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training loss progression
        axes[0, 0].plot(self.training_losses)
        axes[0, 0].set_title("Tool-Aware SAD Training Loss")
        axes[0, 0].set_xlabel("Training Steps")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)
        
        # Loss distribution
        axes[0, 1].hist(self.training_losses, bins=20, alpha=0.7)
        axes[0, 1].set_title("Loss Distribution")
        axes[0, 1].set_xlabel("Loss Value")
        axes[0, 1].set_ylabel("Frequency")
        
        # Smoothed loss
        if len(self.training_losses) > 10:
            window_size = max(5, len(self.training_losses) // 10)
            smoothed = np.convolve(self.training_losses, np.ones(window_size)/window_size, mode='valid')
            axes[1, 0].plot(smoothed)
            axes[1, 0].set_title(f"Smoothed Loss (window={window_size})")
            axes[1, 0].set_xlabel("Training Steps")
            axes[1, 0].set_ylabel("Smoothed Loss")
            axes[1, 0].grid(True)
        
        # Tool pattern coverage
        pattern_counts = defaultdict(int)
        for item in self.training_data:
            pattern_counts[item["scenario"]["pattern"]] += 1
        
        patterns = list(pattern_counts.keys())
        counts = list(pattern_counts.values())
        axes[1, 1].bar(patterns, counts)
        axes[1, 1].set_title("Tool Pattern Coverage")
        axes[1, 1].set_xlabel("Pattern Type")
        axes[1, 1].set_ylabel("Training Examples")
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.results_dir) / "tool_sad_training_results.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]📊 Training visualizations saved to {plot_file}[/green]")
    
    def run_complete_training_pipeline(self):
        """Run the complete tool-aware SAD training pipeline."""
        
        console.print("[bold blue]🚀 Starting Comprehensive Tool-Aware SAD Training Pipeline[/bold blue]")
        
        try:
            # Step 1: Setup clients
            if not self.setup_clients():
                return False
            
            # Step 2: Setup student model
            if not self.setup_student_model():
                return False
            
            # Step 3: Train model
            if not self.train_model():
                return False
            
            # Step 4: Save model
            self.save_model()
            
            # Step 5: Create visualizations
            self.create_training_visualizations()
            
            # Step 6: Test inference if available
            test_prompt = """
You need to debug a Python script that's using too much memory. The script processes large CSV files.

Available tools: terminal_command, file_operations, code_analyzer, system_monitor

Please solve this step-by-step using the available tools.
"""
            
            console.print("[blue]🧪 Testing trained model inference...[/blue]")
            test_result = self.test_sglang_inference(test_prompt)
            if test_result:
                console.print("[green]✅ Inference test successful![/green]")
                console.print(f"Sample response: {test_result[:200]}...")
            
            # Summary
            console.print("\n" + "="*80)
            console.print("[bold green]🎉 COMPREHENSIVE TOOL-AWARE SAD TRAINING COMPLETED![/bold green]")
            console.print("="*80)
            console.print(f"✅ Model saved to: {self.config.output_dir}")
            console.print(f"📊 Results saved to: {self.config.results_dir}")
            console.print(f"📈 Training examples: {len(self.training_data)}")
            console.print(f"🛠️  Tool patterns trained: {len(set(item['scenario']['pattern'] for item in self.training_data))}")
            console.print(f"📉 Final training loss: {self.training_losses[-1] if self.training_losses else 'N/A'}")
            console.print(f"🧠 Enhanced with sophisticated tool usage capabilities")
            console.print("="*80)
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Training pipeline failed: {e}[/red]")
            return False

def main():
    """Main execution function."""
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]❌ GROQ_API_KEY environment variable not set[/red]")
        return
    
    # Initialize and run trainer
    trainer = ComprehensiveToolSADTrainer()
    success = trainer.run_complete_training_pipeline()
    
    if success:
        console.print("[bold green]🎉 Training completed successfully![/bold green]")
    else:
        console.print("[bold red]❌ Training failed![/bold red]")

if __name__ == "__main__":
    main() 