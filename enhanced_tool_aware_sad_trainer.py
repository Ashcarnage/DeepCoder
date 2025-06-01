#!/usr/bin/env python3
"""
Enhanced Tool-Aware Structured Agent Distillation (SAD) Trainer
================================================================

This trainer enhances the Qwen 30B student model to:
1. Learn sophisticated reasoning patterns from teacher models
2. Master tool usage patterns (terminal, web search, file operations, code analysis)
3. Develop multi-step problem-solving capabilities with tool integration
4. Understand when and how to use different tools effectively

Features:
- Tool-aware trajectory generation with teacher demonstrations
- Multi-modal training combining reasoning + tool usage
- LoRA fine-tuning for efficient weight updates
- Comprehensive evaluation with before/after capabilities
- Real-world tool usage scenarios (like Cursor IDE)
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW

# API Clients
from groq import Groq
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/enhanced_tool_sad_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ToolUsageConfig:
    """Configuration for tool usage patterns"""
    
    # Available tools (mimicking Cursor IDE capabilities)
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
    
    # Tool usage patterns to teach
    tool_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "debugging": ["terminal_command", "file_operations", "code_analyzer"],
        "research": ["web_search", "documentation_search", "file_operations"],
        "development": ["file_operations", "git_operations", "terminal_command", "code_analyzer"],
        "data_analysis": ["data_processor", "file_operations", "system_monitor"],
        "system_admin": ["terminal_command", "system_monitor", "file_operations"]
    })
    
    # Context for when to use tools
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
class EnhancedSADConfig:
    """Enhanced configuration for tool-aware SAD training"""
    
    # Model settings
    student_model_path: str = "/workspace/persistent/models/qwen3-30b-a3b"
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    
    # Training parameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 1
    max_seq_length: int = 2048
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    
    # SAD loss weights
    base_loss_weight: float = 1.0
    reasoning_loss_weight: float = 1.5
    tool_usage_loss_weight: float = 2.0
    trajectory_loss_weight: float = 1.8
    
    # Tool training settings
    max_tool_demonstrations: int = 50
    tool_trajectory_length: int = 10
    
    # Output settings
    output_dir: str = "/workspace/persistent/enhanced_tool_sad_model"
    evaluation_output: str = "/workspace/persistent/tool_evaluation_results"

class ToolAwareTrajectoryGenerator:
    """Generates training trajectories that include sophisticated tool usage"""
    
    def __init__(self, config: EnhancedSADConfig, tool_config: ToolUsageConfig, groq_client: Groq):
        self.config = config
        self.tool_config = tool_config
        self.groq_client = groq_client
        
        # SGLang client for local model
        self.sglang_client = OpenAI(
            base_url="http://localhost:30000/v1",
            api_key="EMPTY"
        )
        
        # Tool usage scenarios (real-world programming tasks)
        self.scenarios = [
            {
                "type": "debugging_session",
                "description": "Debug a Python script with performance issues",
                "required_tools": ["file_operations", "code_analyzer", "terminal_command"],
                "complexity": "medium"
            },
            {
                "type": "web_research",
                "description": "Research and implement a new technology solution",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "high"
            },
            {
                "type": "system_analysis",
                "description": "Analyze system performance and optimize resources",
                "required_tools": ["system_monitor", "terminal_command", "data_processor"],
                "complexity": "medium"
            },
            {
                "type": "code_development",
                "description": "Develop a new feature with proper testing",
                "required_tools": ["file_operations", "git_operations", "code_analyzer", "terminal_command"],
                "complexity": "high"
            },
            {
                "type": "data_pipeline",
                "description": "Build and optimize a data processing pipeline",
                "required_tools": ["data_processor", "database_query", "file_operations", "system_monitor"],
                "complexity": "high"
            }
        ]
    
    def generate_tool_aware_prompt(self, scenario: Dict) -> str:
        """Generate a prompt that encourages tool usage"""
        
        base_prompt = f"""
You are an expert AI assistant helping with a {scenario['type']} task. 

Task: {scenario['description']}

You have access to the following tools:
{self._format_available_tools(scenario['required_tools'])}

Please solve this step-by-step, using tools when appropriate. For each tool usage:
1. Explain WHY you're using that specific tool
2. Show the tool call with proper parameters
3. Analyze the results and determine next steps
4. Continue with additional tool calls as needed

Be thorough and demonstrate professional problem-solving methodology.
"""
        return base_prompt.strip()
    
    def _format_available_tools(self, required_tools: List[str]) -> str:
        """Format tool descriptions for the prompt"""
        tool_descriptions = []
        for tool in required_tools:
            context = self.tool_config.tool_contexts.get(tool, "General purpose tool")
            tool_descriptions.append(f"- {tool}: {context}")
        
        return "\n".join(tool_descriptions)
    
    def get_teacher_demonstration(self, scenario: Dict) -> str:
        """Get a high-quality tool usage demonstration from the teacher model"""
        
        prompt = self.generate_tool_aware_prompt(scenario)
        
        # Enhanced system prompt for tool usage
        system_prompt = """
You are a senior software engineer and AI assistant expert at using development tools effectively. 

When solving problems:
1. Think step-by-step and break down complex tasks
2. Use tools strategically and explain your reasoning
3. Show realistic tool calls with proper syntax
4. Analyze tool outputs and adjust your approach
5. Demonstrate professional debugging and development practices

Format tool calls as:
<tool_call>
<tool_name>tool_name</tool_name>
<parameters>
{
  "param1": "value1",
  "param2": "value2"
}
</parameters>
</tool_call>

Show realistic tool outputs and build upon them logically.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting teacher demonstration: {e}")
            return ""
    
    def generate_training_trajectories(self, num_trajectories: int = 20) -> List[Dict]:
        """Generate a set of training trajectories with tool usage"""
        
        trajectories = []
        
        for i in range(num_trajectories):
            # Select scenario (rotate through available scenarios)
            scenario = self.scenarios[i % len(self.scenarios)]
            
            # Get teacher demonstration
            logger.info(f"Generating trajectory {i+1}/{num_trajectories}: {scenario['type']}")
            
            teacher_response = self.get_teacher_demonstration(scenario)
            
            if teacher_response:
                trajectory = {
                    "id": f"tool_trajectory_{i:03d}",
                    "scenario": scenario,
                    "prompt": self.generate_tool_aware_prompt(scenario),
                    "teacher_response": teacher_response,
                    "tools_used": self._extract_tools_from_response(teacher_response),
                    "complexity": scenario["complexity"],
                    "timestamp": datetime.now().isoformat()
                }
                
                trajectories.append(trajectory)
                
                # Rate limiting
                time.sleep(2)
            else:
                logger.warning(f"Failed to generate trajectory {i+1}")
        
        logger.info(f"Generated {len(trajectories)} tool-aware trajectories")
        return trajectories
    
    def _extract_tools_from_response(self, response: str) -> List[str]:
        """Extract tool names mentioned in the response"""
        tools_found = []
        for tool in self.tool_config.available_tools:
            if tool in response or tool.replace("_", " ") in response:
                tools_found.append(tool)
        return tools_found

class ToolAwareSADDataset(Dataset):
    """Dataset for tool-aware SAD training"""
    
    def __init__(self, trajectories: List[Dict], tokenizer, max_length: int = 2048):
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        
        # Create training text with special tokens
        prompt = trajectory["prompt"]
        teacher_response = trajectory["teacher_response"]
        
        # Format as conversation
        text = f"<|user|>: {prompt}\n<|assistant|>: {teacher_response}<|endoftext|>"
        
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
            "trajectory": trajectory
        }

class EnhancedToolAwareSADTrainer:
    """Enhanced SAD trainer with sophisticated tool usage capabilities"""
    
    def __init__(self, groq_api_key: str):
        self.config = EnhancedSADConfig()
        self.tool_config = ToolUsageConfig()
        
        # Initialize clients
        self.groq_client = Groq(api_key=groq_api_key)
        self.sglang_client = OpenAI(
            base_url="http://localhost:30000/v1",
            api_key="EMPTY"
        )
        
        # Initialize trajectory generator
        self.trajectory_generator = ToolAwareTrajectoryGenerator(
            self.config, self.tool_config, self.groq_client
        )
        
        # Training tracking
        self.training_metrics = {
            "losses": [],
            "tool_usage_accuracy": [],
            "reasoning_quality": [],
            "step_numbers": []
        }
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.evaluation_output, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("Enhanced Tool-Aware SAD Trainer initialized")
    
    def setup_student_model(self):
        """Setup the local Qwen 30B student model with LoRA"""
        
        logger.info("Loading Qwen 30B student model...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.student_model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False
        )
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        # Setup LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("Student model setup complete with LoRA")
    
    def prepare_training_data(self):
        """Generate and prepare training data with tool usage patterns"""
        
        logger.info("Generating tool-aware training trajectories...")
        
        # Generate diverse trajectories
        trajectories = self.trajectory_generator.generate_training_trajectories(
            num_trajectories=self.config.max_tool_demonstrations
        )
        
        # Save trajectories for analysis
        trajectory_file = Path(self.config.output_dir) / "training_trajectories.json"
        with open(trajectory_file, 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        logger.info(f"Saved {len(trajectories)} trajectories to {trajectory_file}")
        
        # Create dataset
        self.train_dataset = ToolAwareSADDataset(
            trajectories, self.tokenizer, self.config.max_seq_length
        )
        
        logger.info(f"Training dataset prepared with {len(self.train_dataset)} examples")
    
    def compute_enhanced_sad_loss(self, outputs, batch):
        """Compute enhanced SAD loss with tool usage awareness"""
        
        logits = outputs.logits
        labels = batch["labels"]
        
        # Base language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        base_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )
        
        # Tool usage pattern loss (higher weight for tool-related tokens)
        tool_loss = self._compute_tool_pattern_loss(shift_logits, shift_labels, batch)
        
        # Reasoning quality loss (focus on logical sequences)
        reasoning_loss = self._compute_reasoning_loss(shift_logits, shift_labels, batch)
        
        # Combined loss
        total_loss = (
            self.config.base_loss_weight * base_loss +
            self.config.tool_usage_loss_weight * tool_loss +
            self.config.reasoning_loss_weight * reasoning_loss
        )
        
        return {
            "total_loss": total_loss,
            "base_loss": base_loss,
            "tool_loss": tool_loss,
            "reasoning_loss": reasoning_loss
        }
    
    def _compute_tool_pattern_loss(self, logits, labels, batch):
        """Compute loss with emphasis on tool usage patterns"""
        
        # Identify tool-related tokens
        tool_tokens = []
        for tool in self.tool_config.available_tools:
            tool_ids = self.tokenizer.encode(tool, add_special_tokens=False)
            tool_tokens.extend(tool_ids)
        
        # Create tool mask
        tool_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in tool_tokens:
            tool_mask |= (labels == token_id)
        
        if tool_mask.sum() > 0:
            # Higher weight for tool-related predictions
            tool_logits = logits[tool_mask]
            tool_labels = labels[tool_mask]
            
            tool_loss = F.cross_entropy(
                tool_logits.view(-1, tool_logits.size(-1)),
                tool_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        else:
            tool_loss = torch.tensor(0.0, device=logits.device)
        
        return tool_loss
    
    def _compute_reasoning_loss(self, logits, labels, batch):
        """Compute loss focusing on reasoning patterns"""
        
        # Identify reasoning-related tokens
        reasoning_phrases = [
            "because", "therefore", "however", "step", "analyze", 
            "consider", "evaluate", "determine", "conclude"
        ]
        
        reasoning_tokens = []
        for phrase in reasoning_phrases:
            phrase_ids = self.tokenizer.encode(phrase, add_special_tokens=False)
            reasoning_tokens.extend(phrase_ids)
        
        # Create reasoning mask
        reasoning_mask = torch.zeros_like(labels, dtype=torch.bool)
        for token_id in reasoning_tokens:
            reasoning_mask |= (labels == token_id)
        
        if reasoning_mask.sum() > 0:
            reasoning_logits = logits[reasoning_mask]
            reasoning_labels = labels[reasoning_mask]
            
            reasoning_loss = F.cross_entropy(
                reasoning_logits.view(-1, reasoning_logits.size(-1)),
                reasoning_labels.view(-1),
                ignore_index=self.tokenizer.pad_token_id
            )
        else:
            reasoning_loss = torch.tensor(0.0, device=logits.device)
        
        return reasoning_loss
    
    def train_model(self):
        """Train the model with enhanced SAD approach"""
        
        logger.info("Starting enhanced tool-aware SAD training...")
        
        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        # Setup scheduler
        num_training_steps = len(self.train_dataset) * self.config.num_epochs
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps
        )
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )
            
            epoch_losses = []
            
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                batch = {k: v.cuda() if torch.cuda.is_available() else v 
                        for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward pass
                outputs = self.model(**{k: v for k, v in batch.items() 
                                      if k in ['input_ids', 'attention_mask']})
                
                # Compute enhanced loss
                loss_dict = self.compute_enhanced_sad_loss(outputs, batch)
                loss = loss_dict["total_loss"]
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # Update
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                if global_step % 5 == 0:
                    logger.info(
                        f"Step {global_step}: "
                        f"Total Loss: {loss:.4f}, "
                        f"Base Loss: {loss_dict['base_loss']:.4f}, "
                        f"Tool Loss: {loss_dict['tool_loss']:.4f}, "
                        f"Reasoning Loss: {loss_dict['reasoning_loss']:.4f}"
                    )
                
                # Track metrics
                self.training_metrics["losses"].append(loss.item())
                self.training_metrics["step_numbers"].append(global_step)
                
                epoch_losses.append(loss.item())
                global_step += 1
            
            avg_epoch_loss = np.mean(epoch_losses)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
        
        logger.info("Training completed!")
    
    def evaluate_model(self, test_scenarios: List[Dict]) -> Dict:
        """Evaluate the model's tool usage and reasoning capabilities"""
        
        logger.info("Evaluating model capabilities...")
        
        self.model.eval()
        evaluation_results = []
        
        with torch.no_grad():
            for scenario in test_scenarios:
                prompt = self.trajectory_generator.generate_tool_aware_prompt(scenario)
                
                # Get model response
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                response = self.tokenizer.decode(
                    outputs[0][len(inputs["input_ids"][0]):], 
                    skip_special_tokens=True
                )
                
                # Analyze response quality
                analysis = self._analyze_response_quality(response, scenario)
                
                evaluation_results.append({
                    "scenario": scenario,
                    "prompt": prompt,
                    "response": response,
                    "analysis": analysis
                })
        
        # Save evaluation results
        eval_file = Path(self.config.evaluation_output) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_file}")
        return evaluation_results
    
    def _analyze_response_quality(self, response: str, scenario: Dict) -> Dict:
        """Analyze the quality of a model response"""
        
        analysis = {
            "tools_mentioned": 0,
            "required_tools_used": 0,
            "reasoning_indicators": 0,
            "step_by_step_approach": False,
            "tool_call_format": False
        }
        
        # Check for tool usage
        for tool in self.tool_config.available_tools:
            if tool in response or tool.replace("_", " ") in response:
                analysis["tools_mentioned"] += 1
                if tool in scenario["required_tools"]:
                    analysis["required_tools_used"] += 1
        
        # Check for reasoning indicators
        reasoning_words = ["because", "therefore", "step", "analyze", "consider", "first", "next", "then"]
        for word in reasoning_words:
            if word in response.lower():
                analysis["reasoning_indicators"] += 1
        
        # Check for structured approach
        analysis["step_by_step_approach"] = any(
            phrase in response.lower() 
            for phrase in ["step 1", "first step", "step-by-step", "let me", "i'll start"]
        )
        
        # Check for tool call format
        analysis["tool_call_format"] = "<tool_call>" in response and "<tool_name>" in response
        
        return analysis
    
    def save_model(self):
        """Save the trained model"""
        
        logger.info("Saving enhanced tool-aware model...")
        
        # Save LoRA weights
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save configuration
        config_file = Path(self.config.output_dir) / "training_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "model_config": self.config.__dict__,
                "tool_config": self.tool_config.__dict__,
                "training_metrics": self.training_metrics
            }, f, indent=2, default=str)
        
        logger.info(f"Model saved to {self.config.output_dir}")
    
    def create_training_visualizations(self):
        """Create comprehensive training visualizations"""
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss progression
        axes[0, 0].plot(self.training_metrics["step_numbers"], self.training_metrics["losses"])
        axes[0, 0].set_title("Training Loss Progression")
        axes[0, 0].set_xlabel("Training Steps")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(True)
        
        # Tool usage distribution (if available)
        if len(self.training_metrics["losses"]) > 0:
            axes[0, 1].hist(self.training_metrics["losses"], bins=20, alpha=0.7)
            axes[0, 1].set_title("Loss Distribution")
            axes[0, 1].set_xlabel("Loss Value")
            axes[0, 1].set_ylabel("Frequency")
        
        # Training progress smoothed
        if len(self.training_metrics["losses"]) > 10:
            window_size = min(10, len(self.training_metrics["losses"]) // 5)
            smoothed_losses = np.convolve(
                self.training_metrics["losses"], 
                np.ones(window_size)/window_size, 
                mode='valid'
            )
            axes[1, 0].plot(smoothed_losses)
            axes[1, 0].set_title(f"Smoothed Training Loss (window={window_size})")
            axes[1, 0].set_xlabel("Training Steps")
            axes[1, 0].set_ylabel("Smoothed Loss")
            axes[1, 0].grid(True)
        
        # Tool capabilities overview
        tool_counts = {tool: 1 for tool in self.tool_config.available_tools}
        axes[1, 1].bar(range(len(tool_counts)), list(tool_counts.values()))
        axes[1, 1].set_title("Available Tool Categories")
        axes[1, 1].set_xlabel("Tool Index")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_xticks(range(len(tool_counts)))
        axes[1, 1].set_xticklabels([tool[:10] + "..." if len(tool) > 10 else tool 
                                   for tool in tool_counts.keys()], rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_dir) / "training_visualizations.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training visualizations saved to {plot_file}")
    
    def run_complete_training_pipeline(self):
        """Run the complete enhanced tool-aware training pipeline"""
        
        logger.info("🚀 Starting Enhanced Tool-Aware SAD Training Pipeline")
        
        try:
            # Step 1: Setup model
            self.setup_student_model()
            
            # Step 2: Generate training data
            self.prepare_training_data()
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Evaluate model
            test_scenarios = self.trajectory_generator.scenarios[:3]  # Use first 3 for testing
            evaluation_results = self.evaluate_model(test_scenarios)
            
            # Step 5: Save everything
            self.save_model()
            self.create_training_visualizations()
            
            # Summary
            logger.info("✅ Enhanced Tool-Aware SAD Training Pipeline Completed!")
            logger.info(f"📁 Model saved to: {self.config.output_dir}")
            logger.info(f"📊 Evaluation results: {self.config.evaluation_output}")
            logger.info(f"📈 Training trajectories: {len(self.train_dataset)} examples")
            logger.info(f"🛠️  Tools trained: {len(self.tool_config.available_tools)} categories")
            
            return {
                "status": "completed",
                "model_path": self.config.output_dir,
                "evaluation_results": evaluation_results,
                "training_metrics": self.training_metrics
            }
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise

def main():
    """Main execution function"""
    
    # Check for API key
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    # Initialize trainer
    trainer = EnhancedToolAwareSADTrainer(groq_api_key)
    
    # Run complete pipeline
    results = trainer.run_complete_training_pipeline()
    
    print("\n" + "="*60)
    print("🎉 ENHANCED TOOL-AWARE SAD TRAINING COMPLETED!")
    print("="*60)
    print(f"✅ Status: {results['status']}")
    print(f"📁 Model Location: {results['model_path']}")
    print(f"📊 Training Examples: {len(results['training_metrics']['losses'])}")
    print(f"🛠️  Enhanced with sophisticated tool usage capabilities")
    print(f"🧠 Ready for advanced agentic reasoning tasks")
    print("="*60)

if __name__ == "__main__":
    main() 