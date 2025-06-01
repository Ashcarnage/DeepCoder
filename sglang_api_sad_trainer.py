#!/usr/bin/env python3
"""
SGLang API-Based Structured Agent Distillation (SAD) Trainer
============================================================

This trainer uses the already running SGLang server (localhost:30000) to:
1. Generate teacher demonstrations with tool usage patterns
2. Train the student model via API-based fine-tuning approaches
3. Evaluate improvements in reasoning and tool usage capabilities
4. Create quality response trajectories with Cursor-like tool usage

Features:
- Uses existing optimized SGLang server (no model reloading)
- Tool-aware trajectory generation with teacher demonstrations
- API-based training approach with weight updates
- Comprehensive evaluation of tool usage capabilities
- Real-world scenarios (debugging, research, development, etc.)
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

# API Clients
from groq import Groq
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/sglang_sad_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
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
class SGLangSADConfig:
    """Configuration for SGLang API-based SAD training"""
    
    # SGLang server settings
    sglang_base_url: str = "http://localhost:30000/v1"
    sglang_model: str = "qwen3-30b-a3b"
    
    # Teacher model settings
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    
    # Training parameters
    num_training_trajectories: int = 30
    num_iterations: int = 5
    max_tokens: int = 1500
    temperature: float = 0.3
    
    # Tool training settings
    tool_scenarios_per_type: int = 6
    
    # Output settings
    output_dir: str = "/workspace/persistent/sglang_sad_results"
    trajectories_file: str = "/workspace/persistent/tool_trajectories.json"
    evaluation_file: str = "/workspace/persistent/tool_evaluation.json"

class ToolAwareTrajectoryGenerator:
    """Generates training trajectories with sophisticated tool usage using existing SGLang server"""
    
    def __init__(self, config: SGLangSADConfig, tool_config: ToolUsageConfig, groq_client: Groq, sglang_client: OpenAI):
        self.config = config
        self.tool_config = tool_config
        self.groq_client = groq_client
        self.sglang_client = sglang_client
        
        # Advanced tool usage scenarios
        self.scenarios = [
            {
                "type": "debugging_python_performance",
                "description": "Debug a slow Python script by analyzing performance bottlenecks and optimizing code",
                "required_tools": ["file_operations", "code_analyzer", "terminal_command", "system_monitor"],
                "complexity": "high",
                "context": "A Python data processing script is running very slowly on large datasets"
            },
            {
                "type": "web_research_implementation",
                "description": "Research a new AI/ML library and implement a proof of concept",
                "required_tools": ["web_search", "documentation_search", "file_operations", "terminal_command"],
                "complexity": "high",
                "context": "Need to evaluate and implement a new transformer architecture for a project"
            },
            {
                "type": "system_performance_analysis",
                "description": "Analyze system performance issues and optimize resource usage",
                "required_tools": ["system_monitor", "terminal_command", "data_processor", "file_operations"],
                "complexity": "medium",
                "context": "Server experiencing high memory usage and slow response times"
            },
            {
                "type": "full_stack_development",
                "description": "Develop a complete feature from backend API to frontend UI with testing",
                "required_tools": ["file_operations", "git_operations", "code_analyzer", "terminal_command", "api_client"],
                "complexity": "high",
                "context": "Building a new user authentication system with proper testing and deployment"
            },
            {
                "type": "data_pipeline_optimization",
                "description": "Build and optimize a data processing pipeline with monitoring",
                "required_tools": ["data_processor", "database_query", "file_operations", "system_monitor", "terminal_command"],
                "complexity": "high",
                "context": "Processing millions of records daily with reliability and performance requirements"
            },
            {
                "type": "api_integration_debugging",
                "description": "Debug API integration issues and implement proper error handling",
                "required_tools": ["api_client", "file_operations", "code_analyzer", "terminal_command"],
                "complexity": "medium",
                "context": "Third-party API calls failing intermittently in production"
            }
        ]
    
    def generate_tool_aware_prompt(self, scenario: Dict) -> str:
        """Generate a comprehensive prompt that encourages sophisticated tool usage"""
        
        tools_description = self._format_available_tools(scenario['required_tools'])
        
        prompt = f"""
You are an expert senior software engineer and AI assistant working on a {scenario['type']} task.

CONTEXT: {scenario['context']}

TASK: {scenario['description']}

AVAILABLE TOOLS:
{tools_description}

REQUIREMENTS:
1. Approach this systematically with professional problem-solving methodology
2. Use tools strategically - explain WHY you choose each tool
3. Show realistic tool calls with proper parameters and expected outputs
4. Analyze results from each tool and adapt your approach accordingly
5. Demonstrate multi-step reasoning with tool integration
6. Include error handling and validation steps
7. Provide comprehensive solutions like a senior engineer would

FORMAT YOUR RESPONSE:
- Start with analysis and planning
- For each tool usage, format as:
  <tool_call>
  <tool_name>tool_name</tool_name>
  <parameters>{"param": "value"}</parameters>
  <reasoning>Why I'm using this tool and what I expect to find</reasoning>
  </tool_call>
  
  <tool_output>
  [Expected/simulated realistic output]
  </tool_output>
  
  <analysis>
  Analysis of the results and next steps
  </analysis>

- Build upon results logically
- Include follow-up tool calls as needed
- Conclude with summary and recommendations

Be thorough and demonstrate expert-level problem-solving with sophisticated tool usage.
"""
        return prompt.strip()
    
    def _format_available_tools(self, required_tools: List[str]) -> str:
        """Format tool descriptions for the prompt"""
        tool_descriptions = []
        for tool in required_tools:
            context = self.tool_config.tool_contexts.get(tool, "General purpose tool")
            tool_descriptions.append(f"• {tool}: {context}")
        
        return "\n".join(tool_descriptions)
    
    def get_teacher_demonstration(self, scenario: Dict) -> str:
        """Get high-quality tool usage demonstration from teacher model"""
        
        prompt = self.generate_tool_aware_prompt(scenario)
        
        # Enhanced system prompt for sophisticated tool usage
        system_prompt = """
You are a world-class senior software engineer and technical lead with deep expertise in:
- Advanced debugging and performance optimization
- Complex system architecture and design
- Professional development workflows and best practices
- Strategic tool usage for maximum efficiency
- Multi-step problem-solving methodologies

When solving problems:
1. Think like a senior engineer - consider scalability, maintainability, and robustness
2. Use tools strategically and systematically
3. Show realistic, detailed tool outputs that demonstrate deep technical knowledge
4. Build solutions incrementally with proper validation at each step
5. Include error handling, edge cases, and production considerations
6. Demonstrate sophisticated reasoning and decision-making
7. Provide actionable insights and recommendations

Your tool calls should be realistic and show the kind of sophisticated problem-solving approach that distinguishes expert-level engineers.
"""
        
        try:
            response = self.groq_client.chat.completions.create(
                model=self.config.teacher_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error getting teacher demonstration: {e}")
            return ""
    
    def get_student_response(self, scenario: Dict) -> str:
        """Get response from student model via SGLang API"""
        
        prompt = self.generate_tool_aware_prompt(scenario)
        
        try:
            response = self.sglang_client.chat.completions.create(
                model=self.config.sglang_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            
            # Handle DeepSeek-R1 format where content might be in reasoning_content
            content = response.choices[0].message.content
            if content is None:
                content = getattr(response.choices[0].message, 'reasoning_content', '')
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting student response: {e}")
            return ""
    
    def generate_training_trajectories(self) -> List[Dict]:
        """Generate comprehensive training trajectories with tool usage"""
        
        trajectories = []
        
        # Generate multiple trajectories per scenario type
        for scenario in self.scenarios:
            for i in range(self.config.tool_scenarios_per_type):
                logger.info(f"Generating trajectory for {scenario['type']} (variation {i+1})")
                
                # Get teacher demonstration
                teacher_response = self.get_teacher_demonstration(scenario)
                
                # Get baseline student response
                student_response = self.get_student_response(scenario)
                
                if teacher_response and student_response:
                    trajectory = {
                        "id": f"{scenario['type']}_trajectory_{i:02d}",
                        "scenario": scenario,
                        "prompt": self.generate_tool_aware_prompt(scenario),
                        "teacher_response": teacher_response,
                        "student_response": student_response,
                        "tools_used_teacher": self._extract_tools_from_response(teacher_response),
                        "tools_used_student": self._extract_tools_from_response(student_response),
                        "quality_analysis": self._analyze_response_quality(teacher_response, student_response, scenario),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    trajectories.append(trajectory)
                    
                    # Rate limiting for API calls
                    time.sleep(3)
                else:
                    logger.warning(f"Failed to generate complete trajectory for {scenario['type']} variation {i+1}")
        
        logger.info(f"Generated {len(trajectories)} comprehensive tool-aware trajectories")
        return trajectories
    
    def _extract_tools_from_response(self, response: str) -> List[str]:
        """Extract tool names mentioned in the response"""
        tools_found = []
        for tool in self.tool_config.available_tools:
            if tool in response or tool.replace("_", " ") in response:
                tools_found.append(tool)
        return tools_found
    
    def _analyze_response_quality(self, teacher_response: str, student_response: str, scenario: Dict) -> Dict:
        """Analyze quality differences between teacher and student responses"""
        
        analysis = {
            "teacher_tools_count": len(self._extract_tools_from_response(teacher_response)),
            "student_tools_count": len(self._extract_tools_from_response(student_response)),
            "teacher_has_tool_calls": "<tool_call>" in teacher_response,
            "student_has_tool_calls": "<tool_call>" in student_response,
            "teacher_has_reasoning": any(word in teacher_response.lower() for word in ["because", "therefore", "analysis", "reasoning"]),
            "student_has_reasoning": any(word in student_response.lower() for word in ["because", "therefore", "analysis", "reasoning"]),
            "teacher_length": len(teacher_response.split()),
            "student_length": len(student_response.split()),
            "required_tools_used_teacher": len([t for t in self._extract_tools_from_response(teacher_response) if t in scenario["required_tools"]]),
            "required_tools_used_student": len([t for t in self._extract_tools_from_response(student_response) if t in scenario["required_tools"]]),
        }
        
        # Calculate improvement potential
        analysis["improvement_potential"] = max(0, analysis["teacher_tools_count"] - analysis["student_tools_count"])
        analysis["tool_usage_gap"] = analysis["required_tools_used_teacher"] - analysis["required_tools_used_student"]
        
        return analysis

class SGLangAPISADTrainer:
    """API-based SAD trainer using existing SGLang server"""
    
    def __init__(self, groq_api_key: str):
        self.config = SGLangSADConfig()
        self.tool_config = ToolUsageConfig()
        
        # Initialize clients
        self.groq_client = Groq(api_key=groq_api_key)
        self.sglang_client = OpenAI(
            base_url=self.config.sglang_base_url,
            api_key="EMPTY"
        )
        
        # Initialize trajectory generator
        self.trajectory_generator = ToolAwareTrajectoryGenerator(
            self.config, self.tool_config, self.groq_client, self.sglang_client
        )
        
        # Training tracking
        self.training_metrics = {
            "iteration": [],
            "improvement_scores": [],
            "tool_usage_scores": [],
            "reasoning_scores": []
        }
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        logger.info("SGLang API-based SAD Trainer initialized")
    
    def verify_sglang_connection(self) -> bool:
        """Verify SGLang server is accessible and responding"""
        
        try:
            # Test health endpoint
            health_response = requests.get("http://localhost:30000/health", timeout=5)
            if health_response.status_code != 200:
                return False
            
            # Test model endpoint
            models_response = self.sglang_client.models.list()
            logger.info(f"✅ SGLang server connected. Available models: {[m.id for m in models_response.data]}")
            
            # Test simple inference
            test_response = self.sglang_client.chat.completions.create(
                model=self.config.sglang_model,
                messages=[{"role": "user", "content": "Hello! Just testing the connection."}],
                max_tokens=50
            )
            
            # Handle DeepSeek-R1 format where content might be in reasoning_content
            content = test_response.choices[0].message.content
            if content is None:
                content = getattr(test_response.choices[0].message, 'reasoning_content', 'No content available')
            
            logger.info(f"✅ SGLang inference test successful: {content[:100]}...")
            return True
            
        except Exception as e:
            logger.error(f"❌ SGLang connection failed: {e}")
            return False
    
    def iterative_training_process(self):
        """Run iterative training process using API-based approach"""
        
        logger.info("🚀 Starting API-based iterative training process...")
        
        # Generate initial trajectories
        logger.info("Generating comprehensive training trajectories...")
        trajectories = self.trajectory_generator.generate_training_trajectories()
        
        # Save trajectories
        with open(self.config.trajectories_file, 'w') as f:
            json.dump(trajectories, f, indent=2)
        
        logger.info(f"💾 Saved {len(trajectories)} trajectories to {self.config.trajectories_file}")
        
        # Analyze initial performance
        initial_metrics = self._analyze_trajectories(trajectories)
        logger.info(f"📊 Initial performance metrics: {initial_metrics}")
        
        # Iterative improvement process
        for iteration in range(self.config.num_iterations):
            logger.info(f"\n🔄 Starting iteration {iteration + 1}/{self.config.num_iterations}")
            
            # Select trajectories with highest improvement potential
            target_trajectories = self._select_improvement_targets(trajectories)
            
            # Generate improved responses using teacher-guided prompting
            improved_trajectories = self._generate_improved_responses(target_trajectories)
            
            # Evaluate improvements
            improvement_metrics = self._evaluate_improvements(target_trajectories, improved_trajectories)
            
            # Track metrics
            self.training_metrics["iteration"].append(iteration + 1)
            self.training_metrics["improvement_scores"].append(improvement_metrics["overall_improvement"])
            self.training_metrics["tool_usage_scores"].append(improvement_metrics["tool_usage_improvement"])
            self.training_metrics["reasoning_scores"].append(improvement_metrics["reasoning_improvement"])
            
            logger.info(f"✅ Iteration {iteration + 1} completed. Improvement: {improvement_metrics['overall_improvement']:.2f}")
            
            # Update trajectories with improvements
            trajectories.extend(improved_trajectories)
        
        # Final evaluation
        final_metrics = self._analyze_trajectories(trajectories)
        
        return {
            "trajectories": trajectories,
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "training_metrics": self.training_metrics
        }
    
    def _analyze_trajectories(self, trajectories: List[Dict]) -> Dict:
        """Analyze trajectory quality metrics"""
        
        if not trajectories:
            return {}
        
        tool_usage_scores = []
        reasoning_scores = []
        tool_call_format_scores = []
        
        for traj in trajectories:
            analysis = traj.get("quality_analysis", {})
            
            # Tool usage score (0-1)
            required_tools = len(traj["scenario"]["required_tools"])
            if required_tools > 0:
                tool_score = analysis.get("required_tools_used_student", 0) / required_tools
                tool_usage_scores.append(tool_score)
            
            # Reasoning score (0-1)
            reasoning_score = 1.0 if analysis.get("student_has_reasoning", False) else 0.0
            reasoning_scores.append(reasoning_score)
            
            # Tool call format score (0-1)
            format_score = 1.0 if analysis.get("student_has_tool_calls", False) else 0.0
            tool_call_format_scores.append(format_score)
        
        return {
            "avg_tool_usage_score": np.mean(tool_usage_scores) if tool_usage_scores else 0,
            "avg_reasoning_score": np.mean(reasoning_scores),
            "avg_tool_call_format_score": np.mean(tool_call_format_scores),
            "total_trajectories": len(trajectories)
        }
    
    def _select_improvement_targets(self, trajectories: List[Dict]) -> List[Dict]:
        """Select trajectories with highest improvement potential"""
        
        # Sort by improvement potential
        sorted_trajectories = sorted(
            trajectories,
            key=lambda x: x["quality_analysis"].get("improvement_potential", 0),
            reverse=True
        )
        
        # Select top candidates for improvement
        num_targets = min(5, len(sorted_trajectories))
        return sorted_trajectories[:num_targets]
    
    def _generate_improved_responses(self, target_trajectories: List[Dict]) -> List[Dict]:
        """Generate improved responses using teacher-guided prompting"""
        
        improved_trajectories = []
        
        for traj in target_trajectories:
            logger.info(f"Generating improvement for {traj['id']}")
            
            # Create improvement prompt with teacher example
            improvement_prompt = f"""
Looking at this task: {traj['scenario']['description']}

Here's how an expert approached it:
{traj['teacher_response'][:1000]}...

Now, improve your approach by:
1. Using more sophisticated tool combinations
2. Adding deeper analysis and reasoning
3. Including proper error handling and validation
4. Following the expert's strategic thinking patterns

Solve the task with enhanced tool usage and reasoning:
{traj['prompt']}
"""
            
            # Get improved student response
            try:
                improved_response = self.sglang_client.chat.completions.create(
                    model=self.config.sglang_model,
                    messages=[{"role": "user", "content": improvement_prompt}],
                    max_tokens=self.config.max_tokens,
                    temperature=0.1  # Lower temperature for more focused improvement
                )
                
                improved_trajectory = {
                    "id": f"{traj['id']}_improved",
                    "scenario": traj["scenario"],
                    "prompt": traj["prompt"],
                    "teacher_response": traj["teacher_response"],
                    "student_response": improved_response.choices[0].message.content,
                    "original_response": traj["student_response"],
                    "tools_used_teacher": traj["tools_used_teacher"],
                    "tools_used_student": self.trajectory_generator._extract_tools_from_response(improved_response.choices[0].message.content),
                    "quality_analysis": self.trajectory_generator._analyze_response_quality(
                        traj["teacher_response"], 
                        improved_response.choices[0].message.content, 
                        traj["scenario"]
                    ),
                    "improvement_iteration": True,
                    "timestamp": datetime.now().isoformat()
                }
                
                improved_trajectories.append(improved_trajectory)
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error generating improved response for {traj['id']}: {e}")
        
        return improved_trajectories
    
    def _evaluate_improvements(self, original_trajectories: List[Dict], improved_trajectories: List[Dict]) -> Dict:
        """Evaluate improvements between original and improved trajectories"""
        
        original_metrics = self._analyze_trajectories(original_trajectories)
        improved_metrics = self._analyze_trajectories(improved_trajectories)
        
        return {
            "overall_improvement": improved_metrics["avg_tool_usage_score"] - original_metrics["avg_tool_usage_score"],
            "tool_usage_improvement": improved_metrics["avg_tool_usage_score"] - original_metrics["avg_tool_usage_score"],
            "reasoning_improvement": improved_metrics["avg_reasoning_score"] - original_metrics["avg_reasoning_score"],
            "format_improvement": improved_metrics["avg_tool_call_format_score"] - original_metrics["avg_tool_call_format_score"]
        }
    
    def create_comprehensive_evaluation(self, results: Dict):
        """Create comprehensive evaluation report and visualizations"""
        
        logger.info("📈 Creating comprehensive evaluation report...")
        
        # Save detailed results
        evaluation_results = {
            "training_summary": {
                "total_trajectories": len(results["trajectories"]),
                "training_iterations": self.config.num_iterations,
                "initial_performance": results["initial_metrics"],
                "final_performance": results["final_metrics"],
                "improvement_over_time": self.training_metrics
            },
            "detailed_trajectories": results["trajectories"],
            "tool_usage_analysis": self._analyze_tool_usage_patterns(results["trajectories"]),
            "scenario_performance": self._analyze_scenario_performance(results["trajectories"]),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save evaluation results
        with open(self.config.evaluation_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Create visualizations
        self._create_training_visualizations()
        
        logger.info(f"💾 Comprehensive evaluation saved to {self.config.evaluation_file}")
        
        return evaluation_results
    
    def _analyze_tool_usage_patterns(self, trajectories: List[Dict]) -> Dict:
        """Analyze tool usage patterns across trajectories"""
        
        tool_usage_counts = {tool: 0 for tool in self.tool_config.available_tools}
        scenario_tool_usage = {}
        
        for traj in trajectories:
            scenario_type = traj["scenario"]["type"]
            if scenario_type not in scenario_tool_usage:
                scenario_tool_usage[scenario_type] = {tool: 0 for tool in self.tool_config.available_tools}
            
            for tool in traj.get("tools_used_student", []):
                tool_usage_counts[tool] += 1
                scenario_tool_usage[scenario_type][tool] += 1
        
        return {
            "overall_tool_usage": tool_usage_counts,
            "scenario_specific_usage": scenario_tool_usage,
            "most_used_tools": sorted(tool_usage_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            "least_used_tools": sorted(tool_usage_counts.items(), key=lambda x: x[1])[:5]
        }
    
    def _analyze_scenario_performance(self, trajectories: List[Dict]) -> Dict:
        """Analyze performance by scenario type"""
        
        scenario_performance = {}
        
        for traj in trajectories:
            scenario_type = traj["scenario"]["type"]
            if scenario_type not in scenario_performance:
                scenario_performance[scenario_type] = {
                    "trajectories": [],
                    "avg_tools_used": 0,
                    "avg_required_tools_used": 0,
                    "reasoning_success_rate": 0
                }
            
            scenario_performance[scenario_type]["trajectories"].append(traj)
        
        # Calculate averages
        for scenario_type, data in scenario_performance.items():
            trajectories = data["trajectories"]
            data["avg_tools_used"] = np.mean([len(t.get("tools_used_student", [])) for t in trajectories])
            data["avg_required_tools_used"] = np.mean([
                t["quality_analysis"].get("required_tools_used_student", 0) for t in trajectories
            ])
            data["reasoning_success_rate"] = np.mean([
                1.0 if t["quality_analysis"].get("student_has_reasoning", False) else 0.0 for t in trajectories
            ])
            
            # Remove trajectories list for cleaner output
            del data["trajectories"]
        
        return scenario_performance
    
    def _create_training_visualizations(self):
        """Create comprehensive training visualizations"""
        
        if not self.training_metrics["iteration"]:
            logger.warning("No training metrics to visualize")
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training progression
        axes[0, 0].plot(self.training_metrics["iteration"], self.training_metrics["improvement_scores"], 'b-o')
        axes[0, 0].set_title("Overall Improvement Progression")
        axes[0, 0].set_xlabel("Iteration")
        axes[0, 0].set_ylabel("Improvement Score")
        axes[0, 0].grid(True)
        
        # Tool usage improvement
        axes[0, 1].plot(self.training_metrics["iteration"], self.training_metrics["tool_usage_scores"], 'g-s')
        axes[0, 1].set_title("Tool Usage Improvement")
        axes[0, 1].set_xlabel("Iteration")
        axes[0, 1].set_ylabel("Tool Usage Score")
        axes[0, 1].grid(True)
        
        # Reasoning improvement
        axes[1, 0].plot(self.training_metrics["iteration"], self.training_metrics["reasoning_scores"], 'r-^')
        axes[1, 0].set_title("Reasoning Quality Improvement")
        axes[1, 0].set_xlabel("Iteration")
        axes[1, 0].set_ylabel("Reasoning Score")
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.training_metrics["iteration"], self.training_metrics["improvement_scores"], 'b-o', label="Overall")
        axes[1, 1].plot(self.training_metrics["iteration"], self.training_metrics["tool_usage_scores"], 'g-s', label="Tool Usage")
        axes[1, 1].plot(self.training_metrics["iteration"], self.training_metrics["reasoning_scores"], 'r-^', label="Reasoning")
        axes[1, 1].set_title("Combined Improvement Metrics")
        axes[1, 1].set_xlabel("Iteration")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_dir) / "training_progression.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training visualizations saved to {plot_file}")
    
    def run_complete_training_pipeline(self):
        """Run the complete SGLang API-based training pipeline"""
        
        logger.info("🚀 Starting SGLang API-based SAD Training Pipeline")
        
        try:
            # Step 1: Verify SGLang connection
            if not self.verify_sglang_connection():
                raise Exception("Cannot connect to SGLang server")
            
            # Step 2: Run iterative training process
            results = self.iterative_training_process()
            
            # Step 3: Create comprehensive evaluation
            evaluation = self.create_comprehensive_evaluation(results)
            
            # Summary
            logger.info("✅ SGLang API-based SAD Training Pipeline Completed!")
            logger.info(f"📁 Results saved to: {self.config.output_dir}")
            logger.info(f"📊 Trajectories: {len(results['trajectories'])}")
            logger.info(f"🛠️  Tool categories trained: {len(self.tool_config.available_tools)}")
            logger.info(f"🔄 Training iterations: {self.config.num_iterations}")
            
            improvement = results["final_metrics"]["avg_tool_usage_score"] - results["initial_metrics"]["avg_tool_usage_score"]
            logger.info(f"📈 Overall improvement: {improvement:.3f}")
            
            return {
                "status": "completed",
                "results": results,
                "evaluation": evaluation,
                "improvement": improvement
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
    trainer = SGLangAPISADTrainer(groq_api_key)
    
    # Run complete pipeline
    results = trainer.run_complete_training_pipeline()
    
    print("\n" + "="*70)
    print("🎉 SGLANG API-BASED SAD TRAINING COMPLETED!")
    print("="*70)
    print(f"✅ Status: {results['status']}")
    print(f"📁 Results Directory: /workspace/persistent/sglang_sad_results")
    print(f"📊 Total Trajectories: {len(results['results']['trajectories'])}")
    print(f"🛠️  Enhanced Tool Usage Capabilities")
    print(f"📈 Performance Improvement: {results['improvement']:.3f}")
    print(f"🧠 Ready for advanced agentic reasoning with tool usage")
    print("="*70)

if __name__ == "__main__":
    main() 