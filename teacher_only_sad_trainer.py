#!/usr/bin/env python3
"""
Teacher-Only Tool-Aware SAD Training Data Generator
===================================================

This trainer generates comprehensive tool-aware training data using:
✅ Groq API for teacher demonstrations with rate limiting
✅ Comprehensive tool usage pattern scenarios
✅ High-quality training trajectories for future SAD training
✅ Detailed analysis and evaluation metrics
✅ Export-ready training datasets

Approach:
- Focus on generating high-quality teacher demonstrations
- Create comprehensive tool usage scenarios
- Generate training data that can be used for later SAD training
- Analyze tool usage patterns and complexity
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
class TeacherOnlyConfig:
    """Configuration for teacher-only training data generation."""
    
    # API settings
    teacher_model: str = "deepseek-r1-distill-llama-70b"
    
    # Training parameters
    num_scenarios: int = 15
    variations_per_scenario: int = 4
    max_tokens: int = 2000
    temperature: float = 0.3
    
    # Output settings
    output_dir: str = "/workspace/persistent/teacher_tool_training"
    training_data_file: str = "/workspace/persistent/teacher_tool_training/comprehensive_training_data.json"
    analysis_file: str = "/workspace/persistent/teacher_tool_training/tool_analysis.json"

class TeacherOnlySADTrainer:
    """Teacher-only tool-aware training data generator."""
    
    def __init__(self):
        self.config = TeacherOnlyConfig()
        self.rate_limiter = RateLimiter(RateLimitConfig())
        
        # Initialize client
        self.groq_client = None
        
        # Training data
        self.training_trajectories = []
        
        # Tool scenarios
        self.scenarios = self._create_comprehensive_scenarios()
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def _create_comprehensive_scenarios(self) -> List[Dict]:
        """Create comprehensive tool usage scenarios."""
        return [
            # Advanced Debugging Scenarios
            {
                "id": "memory_leak_debugging",
                "pattern": "debugging",
                "task": "Debug a Python application with severe memory leaks affecting production",
                "context": "Production app consuming 8GB+ memory, causing OOM crashes every 2 hours",
                "required_tools": ["system_monitor", "terminal_command", "code_analyzer", "file_operations"],
                "complexity": "high",
                "specific_challenges": ["Memory profiling", "Leak detection", "Performance optimization"]
            },
            {
                "id": "distributed_system_debugging",
                "pattern": "debugging", 
                "task": "Debug intermittent failures in distributed microservices architecture",
                "context": "Microservices randomly failing with timeout errors, affecting user transactions",
                "required_tools": ["system_monitor", "api_client", "terminal_command", "code_analyzer"],
                "complexity": "high",
                "specific_challenges": ["Service mesh debugging", "Network latency analysis", "Distributed tracing"]
            },
            {
                "id": "database_performance_debugging",
                "pattern": "debugging",
                "task": "Debug slow database queries causing application bottlenecks",
                "context": "API response times increased 10x due to database performance issues",
                "required_tools": ["database_query", "system_monitor", "code_analyzer", "terminal_command"],
                "complexity": "high",
                "specific_challenges": ["Query optimization", "Index analysis", "Connection pooling"]
            },
            
            # Advanced Research Scenarios
            {
                "id": "ai_architecture_research",
                "pattern": "research",
                "task": "Research and evaluate cutting-edge transformer architectures for production deployment",
                "context": "Need to implement state-of-the-art LLM architecture for large-scale inference",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "high",
                "specific_challenges": ["Model efficiency", "Deployment strategies", "Performance benchmarking"]
            },
            {
                "id": "security_compliance_research",
                "pattern": "research",
                "task": "Research comprehensive security compliance for financial services API",
                "context": "Building payment processing API requiring SOC2, PCI-DSS compliance",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "high",
                "specific_challenges": ["Regulatory compliance", "Security frameworks", "Audit requirements"]
            },
            {
                "id": "scalability_architecture_research",
                "pattern": "research",
                "task": "Research cloud-native architecture patterns for billion-user scale",
                "context": "Designing system to handle 1B+ users with 99.99% uptime requirements",
                "required_tools": ["web_search", "documentation_search", "file_operations"],
                "complexity": "high",
                "specific_challenges": ["Auto-scaling", "Global distribution", "Disaster recovery"]
            },
            
            # Advanced Development Scenarios
            {
                "id": "zero_trust_auth_system",
                "pattern": "development",
                "task": "Build zero-trust authentication system with advanced threat detection",
                "context": "Implementing next-gen auth for enterprise with ML-based anomaly detection",
                "required_tools": ["file_operations", "git_operations", "code_analyzer", "terminal_command"],
                "complexity": "high",
                "specific_challenges": ["Behavioral analysis", "Risk scoring", "Adaptive authentication"]
            },
            {
                "id": "edge_computing_platform",
                "pattern": "development",
                "task": "Develop edge computing platform for real-time AI inference",
                "context": "Building distributed edge platform for sub-10ms AI model inference",
                "required_tools": ["file_operations", "terminal_command", "code_analyzer", "system_monitor"],
                "complexity": "high",
                "specific_challenges": ["Edge orchestration", "Model optimization", "Network efficiency"]
            },
            {
                "id": "blockchain_integration_platform",
                "pattern": "development",
                "task": "Build enterprise blockchain integration platform",
                "context": "Developing secure, scalable blockchain integration for supply chain",
                "required_tools": ["file_operations", "git_operations", "code_analyzer", "api_client"],
                "complexity": "high",
                "specific_challenges": ["Smart contracts", "Consensus mechanisms", "Cross-chain interoperability"]
            },
            
            # Advanced Data Analysis Scenarios
            {
                "id": "real_time_ml_pipeline",
                "pattern": "data_analysis",
                "task": "Build real-time ML pipeline processing 1M+ events per second",
                "context": "Creating streaming ML pipeline for fraud detection with <100ms latency",
                "required_tools": ["data_processor", "database_query", "system_monitor", "terminal_command"],
                "complexity": "high",
                "specific_challenges": ["Stream processing", "Model serving", "Feature engineering"]
            },
            {
                "id": "predictive_analytics_platform",
                "pattern": "data_analysis",
                "task": "Develop predictive analytics platform for supply chain optimization",
                "context": "Building ML platform to predict and prevent supply chain disruptions",
                "required_tools": ["data_processor", "database_query", "file_operations", "code_analyzer"],
                "complexity": "high",
                "specific_challenges": ["Time series forecasting", "Anomaly detection", "Multi-modal data fusion"]
            },
            
            # Advanced System Administration Scenarios
            {
                "id": "kubernetes_optimization",
                "pattern": "system_admin",
                "task": "Optimize Kubernetes cluster for cost and performance at enterprise scale",
                "context": "Managing 1000+ node K8s cluster, need 40% cost reduction with better performance",
                "required_tools": ["terminal_command", "system_monitor", "file_operations"],
                "complexity": "high",
                "specific_challenges": ["Resource optimization", "Auto-scaling", "Multi-cluster management"]
            },
            {
                "id": "hybrid_cloud_infrastructure",
                "pattern": "system_admin",
                "task": "Design and implement hybrid cloud infrastructure with disaster recovery",
                "context": "Building resilient hybrid cloud spanning AWS, Azure, on-premise data centers",
                "required_tools": ["terminal_command", "system_monitor", "file_operations"],
                "complexity": "high",
                "specific_challenges": ["Multi-cloud networking", "Data replication", "Failover orchestration"]
            },
            
            # Advanced API Integration Scenarios
            {
                "id": "event_driven_microservices",
                "pattern": "api_integration",
                "task": "Build event-driven microservices architecture with CQRS and event sourcing",
                "context": "Implementing complex business workflows with eventual consistency",
                "required_tools": ["api_client", "file_operations", "code_analyzer", "database_query"],
                "complexity": "high",
                "specific_challenges": ["Event sourcing", "CQRS patterns", "Saga orchestration"]
            },
            {
                "id": "api_gateway_federation",
                "pattern": "api_integration",
                "task": "Implement federated API gateway for enterprise service mesh",
                "context": "Building unified API layer for 500+ microservices across multiple teams",
                "required_tools": ["api_client", "terminal_command", "code_analyzer", "system_monitor"],
                "complexity": "high",
                "specific_challenges": ["Service discovery", "Rate limiting", "Circuit breakers"]
            },
            
            # Cross-functional Complex Scenarios
            {
                "id": "ml_ops_platform",
                "pattern": "development",
                "task": "Build comprehensive MLOps platform with automated ML pipeline",
                "context": "Creating end-to-end ML platform from data ingestion to model deployment",
                "required_tools": ["file_operations", "terminal_command", "data_processor", "system_monitor"],
                "complexity": "high",
                "specific_challenges": ["Model versioning", "A/B testing", "Automated retraining"]
            }
        ]
    
    def setup_client(self) -> bool:
        """Setup Groq client."""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                console.print("[red]❌ GROQ_API_KEY not found in environment[/red]")
                return False
            
            self.groq_client = Groq(api_key=groq_api_key)
            console.print("[green]✅ Groq client setup successful[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Failed to setup Groq client: {e}[/red]")
            return False
    
    def generate_comprehensive_tool_prompt(self, scenario: Dict) -> str:
        """Generate comprehensive tool usage prompt."""
        
        tools_desc = "\n".join([
            f"• {tool}: {self._get_tool_description(tool)}" 
            for tool in scenario['required_tools']
        ])
        
        challenges_desc = "\n".join([
            f"• {challenge}" for challenge in scenario.get('specific_challenges', [])
        ])
        
        prompt = f"""
You are a world-class senior software engineer and architect working on: {scenario['task']}

CONTEXT: {scenario['context']}

SPECIFIC TECHNICAL CHALLENGES:
{challenges_desc}

AVAILABLE TOOLS:
{tools_desc}

REQUIREMENTS:
1. Use systematic, enterprise-grade problem-solving approach
2. Strategically select tools and provide detailed reasoning for each choice
3. Show realistic tool calls with comprehensive parameters
4. Analyze tool outputs thoroughly and determine sophisticated next steps
5. Demonstrate multi-step reasoning with advanced tool integration
6. Include comprehensive error handling, monitoring, and validation
7. Consider scalability, security, and maintainability throughout
8. Provide production-ready solutions with proper documentation

FORMAT YOUR RESPONSE:
Use this detailed structure for each tool interaction:

[TOOL_CALL]
Tool: tool_name
Parameters: {{"param1": "value1", "param2": "value2", "flags": ["--option1", "--option2"]}}
Reasoning: Detailed explanation of why this tool is chosen, what specific problem it solves, and expected outcomes
Context: How this fits into the overall solution strategy
[/TOOL_CALL]

[TOOL_OUTPUT]
[Comprehensive, realistic simulated output with actual values, error messages, metrics, etc.]
[/TOOL_OUTPUT]

[ANALYSIS]
Detailed analysis of the results including:
- What the output reveals about the problem
- Performance implications and bottlenecks identified
- Next steps based on findings
- Risk assessment and mitigation strategies
- Optimization opportunities
[/ANALYSIS]

[FOLLOW_UP_ACTIONS]
- Immediate next steps
- Long-term strategies
- Monitoring and alerting recommendations
- Documentation and knowledge sharing plans
[/FOLLOW_UP_ACTIONS]

Demonstrate expert-level, production-grade problem-solving with sophisticated tool usage patterns.
Show the depth of thinking and systematic approach of a senior architect.
"""
        return prompt.strip()
    
    def _get_tool_description(self, tool: str) -> str:
        """Get comprehensive description for a tool."""
        descriptions = {
            "terminal_command": "Execute shell commands, system operations, process management, and infrastructure automation",
            "web_search": "Search for current information, documentation, best practices, and emerging technologies",
            "file_operations": "Read, write, edit, and manage files, configurations, logs, and code repositories",
            "code_analyzer": "Analyze code for bugs, performance issues, security vulnerabilities, and architectural patterns",
            "system_monitor": "Monitor system resources, performance metrics, application health, and infrastructure status",
            "git_operations": "Version control, repository management, branching strategies, and collaborative development",
            "database_query": "Query, analyze, and manage database operations, performance tuning, and data modeling",
            "api_client": "Make HTTP requests, test APIs, integrate services, and manage external service communications",
            "data_processor": "Process, analyze, and transform datasets, perform statistical analysis, and data pipeline management",
            "documentation_search": "Search technical documentation, API references, architectural guides, and knowledge bases"
        }
        return descriptions.get(tool, "General purpose engineering tool")
    
    def get_teacher_response(self, prompt: str, max_retries: int = 3) -> Optional[str]:
        """Get comprehensive response from teacher model with rate limiting."""
        for attempt in range(max_retries):
            try:
                # Check rate limits
                wait_time = self.rate_limiter.get_wait_time(estimated_tokens=1500)
                if wait_time > 0:
                    console.print(f"[yellow]⏳ Rate limit reached, waiting {wait_time:.1f}s...[/yellow]")
                    time.sleep(wait_time)
                
                # Comprehensive system prompt for expert-level responses
                system_prompt = """
You are a distinguished principal engineer and technical architect with 20+ years of experience across:
- Large-scale distributed systems design and implementation
- Advanced debugging and performance optimization at enterprise scale
- Complex system architecture and design patterns
- Professional development workflows and DevOps best practices
- Strategic tool usage for maximum efficiency and reliability
- Multi-step problem-solving methodologies for complex technical challenges
- Security, compliance, and regulatory requirements
- Team leadership and technical mentoring

Your expertise includes:
- Microservices and service mesh architectures
- Cloud-native technologies and container orchestration
- Data engineering and ML/AI system design
- High-performance computing and optimization
- Database design and distributed data systems
- API design and integration patterns
- Infrastructure as code and automation
- Monitoring, observability, and reliability engineering

When solving complex problems:
1. Think systematically like a principal engineer with deep architectural understanding
2. Use tools strategically with comprehensive reasoning and context awareness
3. Show realistic, detailed tool outputs that demonstrate deep technical expertise
4. Build solutions incrementally with proper validation, testing, and monitoring
5. Include comprehensive error handling, edge cases, security considerations, and production readiness
6. Demonstrate sophisticated reasoning, decision-making, and trade-off analysis
7. Provide actionable insights, recommendations, and implementation strategies
8. Consider scalability, maintainability, observability, and operational excellence
9. Think about team collaboration, documentation, and knowledge transfer

Use [TOOL_CALL], [TOOL_OUTPUT], [ANALYSIS], and [FOLLOW_UP_ACTIONS] tags to structure responses clearly.
Provide production-grade solutions that demonstrate the depth and sophistication expected from a principal engineer.
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
    
    def analyze_response_complexity(self, response: str, scenario: Dict) -> Dict:
        """Analyze the complexity and quality of the teacher response."""
        
        # Tool usage analysis
        tool_calls = len(re.findall(r'\[TOOL_CALL\]', response, re.IGNORECASE))
        tool_outputs = len(re.findall(r'\[TOOL_OUTPUT\]', response, re.IGNORECASE))
        analysis_blocks = len(re.findall(r'\[ANALYSIS\]', response, re.IGNORECASE))
        follow_up_blocks = len(re.findall(r'\[FOLLOW_UP_ACTIONS\]', response, re.IGNORECASE))
        
        # Content analysis
        technical_terms = len(re.findall(r'\b(architecture|scalability|performance|optimization|monitoring|observability|reliability|security|compliance|infrastructure|deployment|orchestration|automation|integration|microservices|distributed|fault.tolerant|load.balancing|caching|database|indexing|replication|sharding|consensus|eventual.consistency|circuit.breaker|bulkhead|timeout|retry|exponential.backoff)\b', response.lower()))
        
        reasoning_indicators = len(re.findall(r'\b(because|therefore|since|thus|hence|consequently|given|considering|due to|as a result|this leads to|furthermore|moreover|additionally|however|nevertheless|although|while|whereas)\b', response.lower()))
        
        production_considerations = len(re.findall(r'\b(production|enterprise|scale|scalability|high.availability|disaster.recovery|backup|monitoring|alerting|logging|metrics|observability|security|compliance|audit|governance|risk|sla|slo|rpo|rto)\b', response.lower()))
        
        # Tool coverage
        required_tools_mentioned = sum(1 for tool in scenario['required_tools'] if tool in response.lower() or tool.replace("_", " ") in response.lower())
        
        # Response structure quality
        has_comprehensive_structure = (tool_calls > 0 and tool_outputs > 0 and 
                                     analysis_blocks > 0 and follow_up_blocks > 0)
        
        # Calculate complexity score
        complexity_score = (
            (tool_calls * 2) +
            (analysis_blocks * 3) +
            (follow_up_blocks * 2) +
            (technical_terms * 0.5) +
            (production_considerations * 1.5) +
            (required_tools_mentioned * 2) +
            (reasoning_indicators * 0.3)
        ) / 20  # Normalize to 0-10 scale
        
        return {
            "tool_calls": tool_calls,
            "tool_outputs": tool_outputs,
            "analysis_blocks": analysis_blocks,
            "follow_up_blocks": follow_up_blocks,
            "technical_terms": technical_terms,
            "reasoning_indicators": reasoning_indicators,
            "production_considerations": production_considerations,
            "required_tools_mentioned": required_tools_mentioned,
            "total_required_tools": len(scenario['required_tools']),
            "tool_coverage_ratio": required_tools_mentioned / len(scenario['required_tools']) if scenario['required_tools'] else 0,
            "has_comprehensive_structure": has_comprehensive_structure,
            "complexity_score": complexity_score,
            "response_length": len(response.split()),
            "estimated_reading_time": len(response.split()) / 200  # minutes
        }
    
    def generate_comprehensive_training_data(self) -> bool:
        """Generate comprehensive training data with expert teacher responses."""
        console.print("[blue]📚 Generating comprehensive tool-aware training data...[/blue]")
        
        total_examples = len(self.scenarios) * self.config.variations_per_scenario
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Collecting expert demonstrations...", total=total_examples)
            
            for scenario in self.scenarios:
                for variation in range(self.config.variations_per_scenario):
                    # Create scenario variation
                    if variation > 0:
                        modified_scenario = scenario.copy()
                        modified_scenario["task"] = f"Advanced Variation {variation}: {scenario['task']}"
                        modified_scenario["context"] = f"Extended scenario: {scenario['context']} with additional complexity requirements"
                        modified_scenario["specific_challenges"] = scenario.get('specific_challenges', []) + [f"Variation-specific challenge {variation}"]
                    else:
                        modified_scenario = scenario
                    
                    # Generate comprehensive prompt
                    prompt = self.generate_comprehensive_tool_prompt(modified_scenario)
                    
                    # Get teacher response
                    teacher_response = self.get_teacher_response(prompt)
                    
                    if teacher_response:
                        # Analyze response complexity
                        complexity_analysis = self.analyze_response_complexity(teacher_response, modified_scenario)
                        
                        trajectory = {
                            "id": f"{scenario['id']}_v{variation}",
                            "scenario": modified_scenario,
                            "prompt": prompt,
                            "teacher_response": teacher_response,
                            "complexity_analysis": complexity_analysis,
                            "metadata": {
                                "generation_timestamp": datetime.now().isoformat(),
                                "model_used": self.config.teacher_model,
                                "temperature": self.config.temperature,
                                "max_tokens": self.config.max_tokens
                            }
                        }
                        
                        self.training_trajectories.append(trajectory)
                        
                        # Log progress
                        console.print(f"[green]✅ Generated {scenario['id']}_v{variation} (complexity: {complexity_analysis['complexity_score']:.2f})[/green]")
                    
                    progress.update(task, advance=1)
                    time.sleep(1.5)  # Be respectful to API limits
        
        console.print(f"[green]✅ Generated {len(self.training_trajectories)} comprehensive training trajectories[/green]")
        return len(self.training_trajectories) > 0
    
    def create_comprehensive_analysis(self):
        """Create comprehensive analysis of the generated training data."""
        console.print("[blue]📊 Creating comprehensive training data analysis...[/blue]")
        
        if not self.training_trajectories:
            return None
        
        # Overall statistics
        total_trajectories = len(self.training_trajectories)
        avg_complexity = np.mean([traj["complexity_analysis"]["complexity_score"] for traj in self.training_trajectories])
        avg_tool_coverage = np.mean([traj["complexity_analysis"]["tool_coverage_ratio"] for traj in self.training_trajectories])
        
        # Pattern analysis
        pattern_stats = defaultdict(list)
        for traj in self.training_trajectories:
            pattern = traj["scenario"]["pattern"]
            pattern_stats[pattern].append(traj["complexity_analysis"]["complexity_score"])
        
        pattern_analysis = {
            pattern: {
                "count": len(scores),
                "avg_complexity": np.mean(scores),
                "max_complexity": np.max(scores),
                "min_complexity": np.min(scores),
                "std_complexity": np.std(scores)
            }
            for pattern, scores in pattern_stats.items()
        }
        
        # Tool usage analysis
        tool_usage_stats = defaultdict(int)
        tool_coverage_stats = defaultdict(list)
        for traj in self.training_trajectories:
            for tool in traj["scenario"]["required_tools"]:
                tool_usage_stats[tool] += 1
                tool_coverage_stats[tool].append(traj["complexity_analysis"]["tool_coverage_ratio"])
        
        # Complexity distribution
        complexity_scores = [traj["complexity_analysis"]["complexity_score"] for traj in self.training_trajectories]
        complexity_distribution = {
            "high_complexity": len([s for s in complexity_scores if s >= 7.0]),
            "medium_complexity": len([s for s in complexity_scores if 4.0 <= s < 7.0]),
            "low_complexity": len([s for s in complexity_scores if s < 4.0])
        }
        
        # Quality metrics
        quality_metrics = {
            "avg_tool_calls": np.mean([traj["complexity_analysis"]["tool_calls"] for traj in self.training_trajectories]),
            "avg_analysis_blocks": np.mean([traj["complexity_analysis"]["analysis_blocks"] for traj in self.training_trajectories]),
            "avg_technical_terms": np.mean([traj["complexity_analysis"]["technical_terms"] for traj in self.training_trajectories]),
            "avg_production_considerations": np.mean([traj["complexity_analysis"]["production_considerations"] for traj in self.training_trajectories]),
            "comprehensive_structure_ratio": np.mean([traj["complexity_analysis"]["has_comprehensive_structure"] for traj in self.training_trajectories])
        }
        
        analysis_results = {
            "generation_summary": {
                "total_trajectories": total_trajectories,
                "avg_complexity_score": avg_complexity,
                "avg_tool_coverage": avg_tool_coverage,
                "generation_timestamp": datetime.now().isoformat()
            },
            "pattern_analysis": pattern_analysis,
            "tool_usage_analysis": dict(tool_usage_stats),
            "complexity_distribution": complexity_distribution,
            "quality_metrics": quality_metrics,
            "detailed_trajectories": self.training_trajectories
        }
        
        # Save analysis results
        with open(self.config.analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        console.print(f"[green]💾 Analysis saved to {self.config.analysis_file}[/green]")
        return analysis_results
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the training data."""
        if not self.training_trajectories:
            return
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Complexity by pattern
        patterns = list(set(traj["scenario"]["pattern"] for traj in self.training_trajectories))
        pattern_complexity = [
            np.mean([
                traj["complexity_analysis"]["complexity_score"]
                for traj in self.training_trajectories
                if traj["scenario"]["pattern"] == pattern
            ])
            for pattern in patterns
        ]
        
        axes[0, 0].bar(patterns, pattern_complexity, color='skyblue')
        axes[0, 0].set_title("Average Complexity Score by Pattern", fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel("Pattern Type")
        axes[0, 0].set_ylabel("Complexity Score")
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Tool usage frequency
        tool_counts = defaultdict(int)
        for traj in self.training_trajectories:
            for tool in traj["scenario"]["required_tools"]:
                tool_counts[tool] += 1
        
        tools = list(tool_counts.keys())
        counts = list(tool_counts.values())
        axes[0, 1].barh(tools, counts, color='lightcoral')
        axes[0, 1].set_title("Tool Usage Frequency", fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel("Usage Count")
        
        # Complexity distribution
        complexity_scores = [traj["complexity_analysis"]["complexity_score"] for traj in self.training_trajectories]
        axes[0, 2].hist(complexity_scores, bins=15, alpha=0.7, color='lightgreen')
        axes[0, 2].set_title("Complexity Score Distribution", fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel("Complexity Score")
        axes[0, 2].set_ylabel("Frequency")
        
        # Tool coverage ratio
        coverage_ratios = [traj["complexity_analysis"]["tool_coverage_ratio"] for traj in self.training_trajectories]
        axes[1, 0].hist(coverage_ratios, bins=10, alpha=0.7, color='gold')
        axes[1, 0].set_title("Tool Coverage Ratio Distribution", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Coverage Ratio")
        axes[1, 0].set_ylabel("Frequency")
        
        # Technical terms vs complexity
        technical_terms = [traj["complexity_analysis"]["technical_terms"] for traj in self.training_trajectories]
        axes[1, 1].scatter(technical_terms, complexity_scores, alpha=0.6, color='purple')
        axes[1, 1].set_title("Technical Terms vs Complexity", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Technical Terms Count")
        axes[1, 1].set_ylabel("Complexity Score")
        
        # Response length distribution
        response_lengths = [traj["complexity_analysis"]["response_length"] for traj in self.training_trajectories]
        axes[1, 2].hist(response_lengths, bins=15, alpha=0.7, color='orange')
        axes[1, 2].set_title("Response Length Distribution", fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel("Response Length (words)")
        axes[1, 2].set_ylabel("Frequency")
        
        plt.tight_layout()
        
        # Save plot
        plot_file = Path(self.config.output_dir) / "comprehensive_training_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]📊 Visualizations saved to {plot_file}[/green]")
    
    def save_training_data(self):
        """Save comprehensive training data for future use."""
        
        # Prepare export format for different use cases
        export_data = {
            "metadata": {
                "total_trajectories": len(self.training_trajectories),
                "scenarios_covered": len(self.scenarios),
                "variations_per_scenario": self.config.variations_per_scenario,
                "generation_timestamp": datetime.now().isoformat(),
                "model_used": self.config.teacher_model,
                "average_complexity": np.mean([traj["complexity_analysis"]["complexity_score"] for traj in self.training_trajectories]),
                "quality_metrics": {
                    "high_complexity_count": len([t for t in self.training_trajectories if t["complexity_analysis"]["complexity_score"] >= 7.0]),
                    "comprehensive_structure_ratio": np.mean([t["complexity_analysis"]["has_comprehensive_structure"] for t in self.training_trajectories])
                }
            },
            "training_trajectories": self.training_trajectories,
            "export_formats": {
                "huggingface_format": [
                    {
                        "instruction": traj["prompt"],
                        "output": traj["teacher_response"],
                        "metadata": traj["metadata"]
                    }
                    for traj in self.training_trajectories
                ],
                "openai_format": [
                    {
                        "messages": [
                            {"role": "user", "content": traj["prompt"]},
                            {"role": "assistant", "content": traj["teacher_response"]}
                        ],
                        "metadata": traj["metadata"]
                    }
                    for traj in self.training_trajectories
                ]
            }
        }
        
        with open(self.config.training_data_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        console.print(f"[green]💾 Comprehensive training data saved to {self.config.training_data_file}[/green]")
        
        # Also save just the essential format for quick loading
        essential_data = {
            "trajectories": [
                {
                    "prompt": traj["prompt"],
                    "response": traj["teacher_response"],
                    "scenario": traj["scenario"]["pattern"],
                    "complexity": traj["complexity_analysis"]["complexity_score"]
                }
                for traj in self.training_trajectories
            ]
        }
        
        essential_file = Path(self.config.output_dir) / "essential_training_data.json"
        with open(essential_file, 'w') as f:
            json.dump(essential_data, f, indent=2)
        
        console.print(f"[green]💾 Essential training data saved to {essential_file}[/green]")
    
    def run_complete_pipeline(self):
        """Run the complete teacher-only training data generation pipeline."""
        console.print("[bold blue]🚀 Starting Comprehensive Teacher-Only Training Data Generation[/bold blue]")
        
        try:
            # Step 1: Setup client
            if not self.setup_client():
                return False
            
            # Step 2: Generate training data
            if not self.generate_comprehensive_training_data():
                return False
            
            # Step 3: Save training data
            self.save_training_data()
            
            # Step 4: Create analysis
            analysis = self.create_comprehensive_analysis()
            
            # Step 5: Create visualizations
            self.create_visualizations()
            
            # Summary
            console.print("\n" + "="*90)
            console.print("[bold green]🎉 COMPREHENSIVE TEACHER-ONLY TRAINING DATA GENERATION COMPLETED![/bold green]")
            console.print("="*90)
            console.print(f"✅ Training trajectories generated: {len(self.training_trajectories)}")
            console.print(f"📊 Tool patterns covered: {len(set(traj['scenario']['pattern'] for traj in self.training_trajectories))}")
            console.print(f"🛠️  Unique tools trained: {len(set(tool for traj in self.training_trajectories for tool in traj['scenario']['required_tools']))}")
            console.print(f"📈 Average complexity score: {np.mean([traj['complexity_analysis']['complexity_score'] for traj in self.training_trajectories]):.2f}")
            console.print(f"🏆 High-complexity scenarios: {len([t for t in self.training_trajectories if t['complexity_analysis']['complexity_score'] >= 7.0])}")
            console.print(f"💾 Results saved to: {self.config.output_dir}")
            console.print(f"📄 Ready for SAD training and fine-tuning")
            console.print("="*90)
            
            return True
            
        except Exception as e:
            console.print(f"[red]❌ Training data generation pipeline failed: {e}[/red]")
            return False

def main():
    """Main execution function."""
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]❌ GROQ_API_KEY environment variable not set[/red]")
        return
    
    # Initialize and run trainer
    trainer = TeacherOnlySADTrainer()
    success = trainer.run_complete_pipeline()
    
    if success:
        console.print("[bold green]🎉 Training data generation completed successfully![/bold green]")
    else:
        console.print("[bold red]❌ Training data generation failed![/bold red]")

if __name__ == "__main__":
    main() 