#!/usr/bin/env python3
"""
Efficient Tool-Aware SAD Trainer (Memory Optimized)
===================================================

This trainer performs tool-aware SAD training using:
✅ SGLang server for student model inference (memory efficient)
✅ Groq API for teacher demonstrations with rate limiting  
✅ Iterative improvement approach with quality metrics
✅ Comprehensive tool usage pattern training
✅ Before/after evaluation to measure improvements
✅ Training data generation and analysis

Memory-efficient approach:
- Uses existing SGLang server for student inference
- Teacher model via Groq API with smart rate limiting
- Focuses on quality trajectory generation and analysis
- Creates comprehensive training datasets for future use
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# API clients
from groq import Groq
from openai import OpenAI
import requests

console = Console()

@dataclass
class RateLimitConfig:
    """Rate limit configuration for Groq API."""
    requests_per_minute: int = 30
    tokens_per_minute: int = 6000
    requests_per_day: int = 1000
    tokens_per_day: int = 500000

class RateLimiter:
    """Advanced rate limiter for API calls."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = deque()
        self.token_times = deque()
        self.daily_requests = 0
        self.daily_tokens = 0
        self.last_reset = time.time()
    
    def _cleanup_old_records(self):
        """Remove records older than 1 minute."""
        current_time = time.time()
        one_minute_ago = current_time - 60
        
        while self.request_times and self.request_times[0] < one_minute_ago:
            self.request_times.popleft()
        while self.token_times and self.token_times[0][0] < one_minute_ago:
            self.token_times.popleft()
            
        if current_time - self.last_reset > 86400:
            self.daily_requests = 0
            self.daily_tokens = 0
            self.last_reset = current_time
    
    def get_wait_time(self, estimated_tokens: int = 500) -> float:
        """Calculate wait time before making a request."""
        self._cleanup_old_records()
        current_time = time.time()
        wait_times = []
        
        # Request rate limit
        if len(self.request_times) >= self.config.requests_per_minute:
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                wait_times.append(wait_time)
        
        # Token rate limit
        minute_tokens = sum(tokens for _, tokens in self.token_times)
        if minute_tokens + estimated_tokens > self.config.tokens_per_minute:
            if self.token_times:
                oldest_token_time = self.token_times[0][0]
                wait_time = 60 - (current_time - oldest_token_time)
                if wait_time > 0:
                    wait_times.append(wait_time)
        
        return max(wait_times) if wait_times else 0
    
    def record_request(self, tokens_used: int):
        """Record a successful request."""
        current_time = time.time()
        self.request_times.append(current_time)
        self.token_times.append((current_time, tokens_used))
        self.daily_requests += 1
        self.daily_tokens += tokens_used

@dataclass
class ToolUsageConfig:
    """Configuration for tool usage patterns."""
    
    available_tools: List[str] = field(default_factory=lambda: [
        "terminal_command", "web_search", "file_operations", "code_analyzer",
        "system_monitor", "git_operations", "database_query", "api_client", 
        "data_processor", "documentation_search"
    ])
    
    tool_patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        "debugging": ["terminal_command", "file_operations", "code_analyzer", "system_monitor"],
        "research": ["web_search", "documentation_search", "file_operations"],
        "development": ["file_operations", "git_operations", "terminal_command", "code_analyzer"],
        "data_analysis": ["data_processor", "file_operations", "system_monitor", "database_query"],
        "system_admin": ["terminal_command", "system_monitor", "file_operations"],
        "api_integration": ["api_client", "file_operations", "code_analyzer", "terminal_command"]
    })

@dataclass
class EfficientSADConfig:
    """Configuration for efficient SAD training."""
    
    # API settings
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_model: str = "qwen3-30b-a3b"
    
    # Training parameters
    num_scenarios: int = 12
    variations_per_scenario: int = 3
    improvement_iterations: int = 4
    max_tokens: int = 1500
    temperature: float = 0.3
    
    # Output settings
    output_dir: str = "/workspace/persistent/efficient_tool_sad"
    training_data_file: str = "/workspace/persistent/efficient_tool_sad/training_data.json"
    evaluation_file: str = "/workspace/persistent/efficient_tool_sad/evaluation_results.json"

class EfficientToolSADTrainer:
    """Memory-efficient tool-aware SAD trainer."""
    
    def __init__(self):
        self.config = EfficientSADConfig()
        self.tool_config = ToolUsageConfig()
        self.rate_limiter = RateLimiter(RateLimitConfig())
        
        # Initialize clients
        self.groq_client = None
        self.sglang_client = None
        
        # Training data and metrics
        self.training_trajectories = []
        self.evaluation_results = []
        self.improvement_metrics = []
        
        # Tool scenarios
        self.scenarios = self._create_tool_scenarios()
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def _create_tool_scenarios(self) -> List[Dict]:
        """Create comprehensive tool usage scenarios."""
        return [
            {
                "id": "debug_memory_leak",
                "pattern": "debugging",
                "task": "Debug a Python application with memory leaks",
                "context": "Production app using excessive memory, causing server crashes",
                "required_tools": ["system_monitor", "terminal_command", "code_analyzer", "file_operations"],
                "complexity": "high"
            },
            {
                "id": "api_performance_issue", 
                "pattern": "debugging",
                "task": "Fix slow API response times in microservices",
                "context": "API endpoints responding slowly, affecting user experience",
                "required_tools": ["system_monitor", "api_client", "code_analyzer", "terminal_command"],
                "complexity": "high"
            },
            {
                "id": "research_ml_architecture",
                "pattern": "research", 
                "task": "Research and evaluate new transformer architectures",
                "context": "Need to implement state-of-the-art model for NLP project",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "high"
            },
            {
                "id": "security_best_practices",
                "pattern": "research",
                "task": "Research API security best practices and implementations",
                "context": "Building secure API for financial application",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "medium"
            },
            {
                "id": "auth_system_development",
                "pattern": "development",
                "task": "Build comprehensive user authentication system",
                "context": "Developing secure multi-factor authentication for web app",
                "required_tools": ["file_operations", "git_operations", "code_analyzer", "terminal_command"],
                "complexity": "high"
            },
            {
                "id": "ci_cd_pipeline",
                "pattern": "development",
                "task": "Set up automated CI/CD pipeline with testing",
                "context": "Implementing continuous deployment for microservices",
                "required_tools": ["git_operations", "terminal_command", "file_operations", "code_analyzer"],
                "complexity": "high"
            },
            {
                "id": "data_pipeline_optimization",
                "pattern": "data_analysis", 
                "task": "Optimize large-scale data processing pipeline",
                "context": "ETL pipeline processing terabytes daily, needs optimization",
                "required_tools": ["data_processor", "database_query", "system_monitor", "file_operations"],
                "complexity": "high"
            },
            {
                "id": "real_time_analytics",
                "pattern": "data_analysis",
                "task": "Build real-time analytics dashboard",
                "context": "Creating live monitoring dashboard for business metrics",
                "required_tools": ["data_processor", "database_query", "file_operations", "system_monitor"],
                "complexity": "medium"
            },
            {
                "id": "server_optimization",
                "pattern": "system_admin",
                "task": "Optimize server performance and resource usage",
                "context": "Production servers experiencing high load and slow responses",
                "required_tools": ["system_monitor", "terminal_command", "file_operations"],
                "complexity": "medium"
            },
            {
                "id": "monitoring_setup", 
                "pattern": "system_admin",
                "task": "Implement comprehensive system monitoring",
                "context": "Setting up proactive monitoring for cloud infrastructure",
                "required_tools": ["terminal_command", "system_monitor", "file_operations"],
                "complexity": "high"
            },
            {
                "id": "microservices_integration",
                "pattern": "api_integration",
                "task": "Debug microservices communication issues", 
                "context": "Inter-service communication failing in distributed system",
                "required_tools": ["api_client", "system_monitor", "code_analyzer", "terminal_command"],
                "complexity": "high"
            },
            {
                "id": "third_party_api_integration",
                "pattern": "api_integration", 
                "task": "Integrate with multiple third-party APIs reliably",
                "context": "Building payment processing with multiple providers",
                "required_tools": ["api_client", "file_operations", "code_analyzer", "terminal_command"],
                "complexity": "medium"
            }
        ]
    
    def setup_clients(self) -> bool:
        """Setup API clients."""
        try:
            # Setup Groq client
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                console.print("[red]❌ GROQ_API_KEY not found in environment[/red]")
                return False
            
            self.groq_client = Groq(api_key=groq_api_key)
            
            # Setup SGLang client
            self.sglang_client = OpenAI(
                base_url=self.config.sglang_base_url,
                api_key="EMPTY"
            )
            
            # Test SGLang connection
            try:
                health_response = requests.get("http://localhost:30000/health", timeout=5)
                if health_response.status_code == 200:
                    console.print("[green]✅ SGLang server connection verified[/green]")
                else:
                    console.print("[red]❌ SGLang server not responding[/red]")
                    return False
            except:
                console.print("[red]❌ SGLang server not available[/red]")
                return False
            
            console.print("[green]✅ API clients setup successful[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to setup clients: {e}[/red]")
            return False
    
    def generate_tool_prompt(self, scenario: Dict) -> str:
        """Generate comprehensive tool usage prompt."""
        
        tools_desc = "\n".join([
            f"• {tool}: {self._get_tool_description(tool)}" 
            for tool in scenario['required_tools']
        ])
        
        prompt = f"""
You are a senior software engineer working on: {scenario['task']}

CONTEXT: {scenario['context']}

AVAILABLE TOOLS:
{tools_desc}

REQUIREMENTS:
1. Use systematic problem-solving approach
2. Strategically select tools and explain your reasoning
3. Show realistic tool calls with parameters
4. Analyze tool outputs and determine next steps
5. Demonstrate multi-step reasoning with tool integration
6. Include error handling and validation
7. Provide comprehensive solutions

FORMAT YOUR RESPONSE:
Use this structure for each tool interaction:

[TOOL_CALL]
Tool: tool_name
Parameters: {{"param": "value"}}
Reasoning: Why using this tool and expected outcome
[/TOOL_CALL]

[TOOL_OUTPUT]
[Realistic simulated output]
[/TOOL_OUTPUT]

[ANALYSIS]
Analysis of results and next steps
[/ANALYSIS]

Demonstrate expert-level problem-solving with sophisticated tool usage.
"""
        return prompt.strip()
    
    def _get_tool_description(self, tool: str) -> str:
        """Get description for a tool."""
        descriptions = {
            "terminal_command": "Execute shell commands and system operations",
            "web_search": "Search for current information and documentation",
            "file_operations": "Read, write, edit, and manage files",
            "code_analyzer": "Analyze code for bugs and performance issues",
            "system_monitor": "Monitor system resources and performance",
            "git_operations": "Version control and repository management",
            "database_query": "Query and manage database operations",
            "api_client": "Make HTTP requests and API interactions",
            "data_processor": "Process and analyze datasets",
            "documentation_search": "Search technical documentation"
        }
        return descriptions.get(tool, "General purpose tool")
    
    def get_teacher_response(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Get response from teacher model with rate limiting."""
        for attempt in range(max_retries):
            try:
                # Check rate limits
                wait_time = self.rate_limiter.get_wait_time(estimated_tokens=1000)
                if wait_time > 0:
                    console.print(f"[yellow]⏳ Rate limit reached, waiting {wait_time:.1f}s...[/yellow]")
                    time.sleep(wait_time)
                
                # Enhanced system prompt
                system_prompt = """
You are a world-class senior software engineer and technical lead with expertise in:
- Advanced debugging and performance optimization
- Complex system architecture and design
- Professional development workflows and best practices
- Strategic tool usage for maximum efficiency
- Multi-step problem-solving methodologies

When solving problems:
1. Think systematically like a senior engineer
2. Use tools strategically and explain your reasoning clearly
3. Show realistic, detailed tool outputs demonstrating expertise
4. Build solutions incrementally with proper validation
5. Include error handling, edge cases, and production considerations
6. Demonstrate sophisticated reasoning and decision-making
7. Provide actionable insights and recommendations

Use [TOOL_CALL], [TOOL_OUTPUT], and [ANALYSIS] tags to structure responses clearly.
"""
                
                response = self.groq_client.chat.completions.create(
                    model=self.config.teacher_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature
                )
                
                content = response.choices[0].message.content
                tokens_used = response.usage.total_tokens
                self.rate_limiter.record_request(tokens_used)
                
                return content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    console.print(f"[red]❌ Failed to get teacher response: {e}[/red]")
                    return None
                else:
                    wait_time = 2 ** attempt
                    console.print(f"[yellow]⚠️ Attempt {attempt + 1} failed, retrying in {wait_time}s...[/yellow]")
                    time.sleep(wait_time)
        
        return None
    
    def get_student_response(self, prompt: str) -> Optional[str]:
        """Get response from student model via SGLang."""
        try:
            response = self.sglang_client.chat.completions.create(
                model=self.config.sglang_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Handle DeepSeek-R1 format
            content = response.choices[0].message.content
            if content is None:
                content = getattr(response.choices[0].message, 'reasoning_content', '')
            
            return content
            
        except Exception as e:
            console.print(f"[red]❌ Error getting student response: {e}[/red]")
            return None
    
    def analyze_response_quality(self, teacher_response: str, student_response: str, scenario: Dict) -> Dict:
        """Analyze quality differences between teacher and student responses."""
        
        def extract_tool_usage(response: str) -> Dict:
            tool_calls = len(re.findall(r'\[TOOL_CALL\]', response, re.IGNORECASE))
            tool_outputs = len(re.findall(r'\[TOOL_OUTPUT\]', response, re.IGNORECASE))
            analysis_blocks = len(re.findall(r'\[ANALYSIS\]', response, re.IGNORECASE))
            
            tools_mentioned = 0
            required_tools_used = 0
            for tool in self.tool_config.available_tools:
                if tool in response.lower() or tool.replace("_", " ") in response.lower():
                    tools_mentioned += 1
                    if tool in scenario['required_tools']:
                        required_tools_used += 1
            
            return {
                "tool_calls": tool_calls,
                "tool_outputs": tool_outputs,
                "analysis_blocks": analysis_blocks,
                "tools_mentioned": tools_mentioned,
                "required_tools_used": required_tools_used,
                "has_structured_format": tool_calls > 0 and tool_outputs > 0,
                "reasoning_indicators": len(re.findall(r'\b(because|therefore|since|thus|hence|consequently)\b', response.lower())),
                "response_length": len(response.split())
            }
        
        teacher_analysis = extract_tool_usage(teacher_response)
        student_analysis = extract_tool_usage(student_response)
        
        # Calculate improvement potential
        improvement_potential = {
            "tool_usage_gap": teacher_analysis["required_tools_used"] - student_analysis["required_tools_used"],
            "structure_gap": int(teacher_analysis["has_structured_format"]) - int(student_analysis["has_structured_format"]),
            "reasoning_gap": teacher_analysis["reasoning_indicators"] - student_analysis["reasoning_indicators"],
            "overall_quality_gap": (
                (teacher_analysis["tool_calls"] - student_analysis["tool_calls"]) +
                (teacher_analysis["analysis_blocks"] - student_analysis["analysis_blocks"]) +
                (teacher_analysis["required_tools_used"] - student_analysis["required_tools_used"])
            ) / 3
        }
        
        return {
            "teacher": teacher_analysis,
            "student": student_analysis,
            "improvement_potential": improvement_potential,
            "scenario": scenario
        }
    
    def generate_training_trajectories(self) -> bool:
        """Generate comprehensive training trajectories."""
        console.print("[blue]📚 Generating tool-aware training trajectories...[/blue]")
        
        total_examples = len(self.scenarios) * self.config.variations_per_scenario
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting trajectories...", total=total_examples)
            
            for scenario in self.scenarios:
                for variation in range(self.config.variations_per_scenario):
                    # Create scenario variation
                    if variation > 0:
                        modified_scenario = scenario.copy()
                        modified_scenario["task"] = f"Variation {variation}: {scenario['task']}"
                        modified_scenario["context"] = f"Alternative approach: {scenario['context']}"
                    else:
                        modified_scenario = scenario
                    
                    # Generate prompt
                    prompt = self.generate_tool_prompt(modified_scenario)
                    
                    # Get teacher and student responses
                    teacher_response = self.get_teacher_response(prompt)
                    student_response = self.get_student_response(prompt)
                    
                    if teacher_response and student_response:
                        # Analyze quality
                        quality_analysis = self.analyze_response_quality(
                            teacher_response, student_response, modified_scenario
                        )
                        
                        trajectory = {
                            "id": f"{scenario['id']}_v{variation}",
                            "scenario": modified_scenario,
                            "prompt": prompt,
                            "teacher_response": teacher_response,
                            "student_response": student_response,
                            "quality_analysis": quality_analysis,
                            "timestamp": datetime.now().isoformat()
                        }
                        
                        self.training_trajectories.append(trajectory)
                    
                    progress.update(task, advance=1)
                    time.sleep(1)  # Be respectful to APIs
        
        console.print(f"[green]✅ Generated {len(self.training_trajectories)} training trajectories[/green]")
        return len(self.training_trajectories) > 0
    
    def run_iterative_improvement(self) -> bool:
        """Run iterative improvement process."""
        console.print("[blue]🔄 Starting iterative improvement process...[/blue]")
        
        # Sort trajectories by improvement potential
        sorted_trajectories = sorted(
            self.training_trajectories,
            key=lambda x: x["quality_analysis"]["improvement_potential"]["overall_quality_gap"],
            reverse=True
        )
        
        # Select top candidates for improvement
        improvement_candidates = sorted_trajectories[:5]
        
        for iteration in range(self.config.improvement_iterations):
            console.print(f"[yellow]📈 Improvement iteration {iteration + 1}/{self.config.improvement_iterations}[/yellow]")
            
            iteration_improvements = []
            
            for candidate in improvement_candidates:
                # Create improvement prompt
                improvement_prompt = f"""
Based on this expert demonstration:
{candidate['teacher_response'][:800]}...

Improve your approach to: {candidate['scenario']['task']}

Focus on:
1. Using more sophisticated tool combinations
2. Adding deeper analysis and reasoning
3. Including proper error handling
4. Following expert problem-solving patterns

{candidate['prompt']}
"""
                
                # Get improved student response
                improved_response = self.get_student_response(improvement_prompt)
                
                if improved_response:
                    # Analyze improvement
                    new_quality = self.analyze_response_quality(
                        candidate['teacher_response'], improved_response, candidate['scenario']
                    )
                    
                    improvement_score = (
                        new_quality["improvement_potential"]["overall_quality_gap"] - 
                        candidate["quality_analysis"]["improvement_potential"]["overall_quality_gap"]
                    )
                    
                    iteration_improvements.append({
                        "candidate_id": candidate["id"],
                        "improvement_score": improvement_score,
                        "original_quality": candidate["quality_analysis"],
                        "improved_quality": new_quality,
                        "improved_response": improved_response
                    })
                
                time.sleep(2)  # Rate limiting
            
            self.improvement_metrics.append({
                "iteration": iteration + 1,
                "improvements": iteration_improvements,
                "average_improvement": np.mean([imp["improvement_score"] for imp in iteration_improvements if imp["improvement_score"] is not None])
            })
            
            console.print(f"[green]✅ Iteration {iteration + 1} completed[/green]")
        
        return True
    
    def create_comprehensive_evaluation(self):
        """Create comprehensive evaluation report."""
        console.print("[blue]📊 Creating comprehensive evaluation...[/blue]")
        
        # Overall statistics
        total_trajectories = len(self.training_trajectories)
        avg_improvement_potential = np.mean([
            traj["quality_analysis"]["improvement_potential"]["overall_quality_gap"]
            for traj in self.training_trajectories
        ])
        
        # Pattern analysis
        pattern_stats = defaultdict(list)
        for traj in self.training_trajectories:
            pattern = traj["scenario"]["pattern"]
            pattern_stats[pattern].append(traj["quality_analysis"]["improvement_potential"]["overall_quality_gap"])
        
        pattern_analysis = {
            pattern: {
                "count": len(scores),
                "avg_improvement_potential": np.mean(scores),
                "max_improvement_potential": np.max(scores),
                "min_improvement_potential": np.min(scores)
            }
            for pattern, scores in pattern_stats.items()
        }
        
        # Tool usage analysis
        tool_usage_stats = defaultdict(int)
        for traj in self.training_trajectories:
            for tool in traj["scenario"]["required_tools"]:
                tool_usage_stats[tool] += 1
        
        evaluation_results = {
            "summary": {
                "total_trajectories": total_trajectories,
                "avg_improvement_potential": avg_improvement_potential,
                "completion_timestamp": datetime.now().isoformat()
            },
            "pattern_analysis": pattern_analysis,
            "tool_usage_analysis": dict(tool_usage_stats),
            "improvement_metrics": self.improvement_metrics,
            "detailed_trajectories": self.training_trajectories
        }
        
        # Save evaluation results
        with open(self.config.evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        console.print(f"[green]💾 Evaluation saved to {self.config.evaluation_file}[/green]")
        
        return evaluation_results
    
    def create_visualizations(self):
        """Create training visualizations."""
        if not self.training_trajectories:
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Improvement potential by pattern
        patterns = list(set(traj["scenario"]["pattern"] for traj in self.training_trajectories))
        pattern_scores = [
            np.mean([
                traj["quality_analysis"]["improvement_potential"]["overall_quality_gap"]
                for traj in self.training_trajectories
                if traj["scenario"]["pattern"] == pattern
            ])
            for pattern in patterns
        ]
        
        axes[0, 0].bar(patterns, pattern_scores)
        axes[0, 0].set_title("Average Improvement Potential by Pattern")
        axes[0, 0].set_xlabel("Pattern Type")
        axes[0, 0].set_ylabel("Improvement Score")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Tool usage frequency
        tool_counts = defaultdict(int)
        for traj in self.training_trajectories:
            for tool in traj["scenario"]["required_tools"]:
                tool_counts[tool] += 1
        
        tools = list(tool_counts.keys())
        counts = list(tool_counts.values())
        axes[0, 1].bar(tools, counts)
        axes[0, 1].set_title("Tool Usage Frequency")
        axes[0, 1].set_xlabel("Tool")
        axes[0, 1].set_ylabel("Usage Count")
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Improvement over iterations
        if self.improvement_metrics:
            iterations = [metric["iteration"] for metric in self.improvement_metrics]
            avg_improvements = [metric["average_improvement"] for metric in self.improvement_metrics]
            
            axes[1, 0].plot(iterations, avg_improvements, 'bo-')
            axes[1, 0].set_title("Average Improvement Over Iterations")
            axes[1, 0].set_xlabel("Iteration")
            axes[1, 0].set_ylabel("Average Improvement Score")
            axes[1, 0].grid(True)
        
        # Quality gap distribution
        quality_gaps = [
            traj["quality_analysis"]["improvement_potential"]["overall_quality_gap"]
            for traj in self.training_trajectories
        ]
        axes[1, 1].hist(quality_gaps, bins=15, alpha=0.7)
        axes[1, 1].set_title("Distribution of Quality Gaps")
        axes[1, 1].set_xlabel("Quality Gap Score")
        axes[1, 1].set_ylabel("Frequency")
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_dir) / "training_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]📊 Visualizations saved to {plot_file}[/green]")
    
    def save_training_data(self):
        """Save training data for future use."""
        training_data = {
            "metadata": {
                "total_trajectories": len(self.training_trajectories),
                "scenarios_covered": len(self.scenarios),
                "variations_per_scenario": self.config.variations_per_scenario,
                "generation_timestamp": datetime.now().isoformat()
            },
            "trajectories": self.training_trajectories
        }
        
        with open(self.config.training_data_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        console.print(f"[green]💾 Training data saved to {self.config.training_data_file}[/green]")
    
    def run_complete_pipeline(self):
        """Run the complete efficient SAD training pipeline."""
        console.print("[bold blue]🚀 Starting Efficient Tool-Aware SAD Training Pipeline[/bold blue]")
        
        try:
            # Step 1: Setup clients
            if not self.setup_clients():
                return False
            
            # Step 2: Generate training trajectories
            if not self.generate_training_trajectories():
                return False
            
            # Step 3: Run iterative improvement
            if not self.run_iterative_improvement():
                return False
            
            # Step 4: Save training data
            self.save_training_data()
            
            # Step 5: Create evaluation
            evaluation = self.create_comprehensive_evaluation()
            
            # Step 6: Create visualizations
            self.create_visualizations()
            
            # Summary
            console.print("\n" + "="*80)
            console.print("[bold green]🎉 EFFICIENT TOOL-AWARE SAD TRAINING COMPLETED![/bold green]")
            console.print("="*80)
            console.print(f"✅ Training trajectories: {len(self.training_trajectories)}")
            console.print(f"📊 Tool patterns covered: {len(set(traj['scenario']['pattern'] for traj in self.training_trajectories))}")
            console.print(f"🛠️  Unique tools trained: {len(set(tool for traj in self.training_trajectories for tool in traj['scenario']['required_tools']))}")
            console.print(f"📈 Improvement iterations: {self.config.improvement_iterations}")
            console.print(f"💾 Results saved to: {self.config.output_dir}")
            console.print(f"🧠 Ready for advanced tool-aware reasoning")
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
    trainer = EfficientToolSADTrainer()
    success = trainer.run_complete_pipeline()
    
    if success:
        console.print("[bold green]🎉 Training completed successfully![/bold green]")
    else:
        console.print("[bold red]❌ Training failed![/bold red]")

if __name__ == "__main__":
    main() 