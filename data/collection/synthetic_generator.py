"""
Synthetic Agentic Data Generator

This module generates high-quality synthetic agentic conversation data using free/low-cost
API providers like Groq, Together AI, HuggingFace Inference API, and local models.
"""

import asyncio
import json
import logging
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator
from pathlib import Path
from dataclasses import dataclass
import aiohttp
import hashlib

from data.collection.base_collector import BaseCollector, DataItem, CollectionConfig


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation"""
    # API Provider settings
    provider: str = "groq"  # groq, together, huggingface, local
    api_key: Optional[str] = None
    model_name: str = "llama3-8b-8192"  # Default Groq model
    base_url: Optional[str] = None
    
    # Generation settings
    scenarios_per_batch: int = 5
    max_turns_per_conversation: int = 8
    temperature: float = 0.8
    max_tokens: int = 2048
    
    # Quality settings
    require_tool_usage: bool = True
    require_reasoning: bool = True
    min_conversation_length: int = 200
    
    # Scenario types
    scenario_types: List[str] = None
    
    def __post_init__(self):
        if self.scenario_types is None:
            self.scenario_types = [
                "tool_usage",
                "multi_step_reasoning", 
                "error_handling",
                "code_debugging",
                "data_analysis",
                "web_research",
                "api_integration",
                "problem_solving"
            ]


class APIProvider:
    """Base class for API providers"""
    
    def __init__(self, config: SyntheticConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate completion from messages"""
        raise NotImplementedError


class GroqProvider(APIProvider):
    """Groq API provider - Free tier with fast inference"""
    
    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate completion using Groq API"""
        
        if not self.config.api_key:
            raise ValueError("Groq API key required")
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        try:
            async with self.session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logging.error(f"Groq API error {response.status}: {error_text}")
                    return None
        except Exception as e:
            logging.error(f"Groq API request failed: {e}")
            return None


class TogetherProvider(APIProvider):
    """Together AI provider - $1 free credit"""
    
    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate completion using Together AI"""
        
        if not self.config.api_key:
            raise ValueError("Together AI API key required")
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        
        try:
            async with self.session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logging.error(f"Together AI error {response.status}: {error_text}")
                    return None
        except Exception as e:
            logging.error(f"Together AI request failed: {e}")
            return None


class HuggingFaceProvider(APIProvider):
    """HuggingFace Inference API provider - Free tier available"""
    
    def __init__(self, config: SyntheticConfig):
        super().__init__(config)
        self.base_url = f"https://api-inference.huggingface.co/models/{config.model_name}"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate_completion(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate completion using HuggingFace Inference API"""
        
        if not self.config.api_key:
            raise ValueError("HuggingFace API key required")
        
        # Convert messages to single prompt for HF
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_new_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "return_full_text": False
            }
        }
        
        try:
            async with self.session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)  # HF can be slower
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if isinstance(data, list) and len(data) > 0:
                        return data[0].get("generated_text", "")
                    return ""
                else:
                    error_text = await response.text()
                    logging.error(f"HuggingFace API error {response.status}: {error_text}")
                    return None
        except Exception as e:
            logging.error(f"HuggingFace API request failed: {e}")
            return None
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt for HuggingFace"""
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant: "
        return prompt


class SyntheticDataGenerator(BaseCollector):
    """Generate synthetic agentic conversation data"""
    
    # Agentic scenario templates
    SCENARIO_TEMPLATES = {
        "tool_usage": {
            "system_prompt": """You are an AI assistant that helps users accomplish tasks by using various tools and APIs. 
            You should demonstrate clear tool selection, usage, and result interpretation. Always explain your reasoning 
            for choosing specific tools and how you plan to use them.""",
            "scenarios": [
                "Help me analyze sales data from multiple CSV files",
                "I need to create a dashboard showing website traffic trends", 
                "Find and summarize recent research papers about machine learning",
                "Debug a Python script that's not working correctly",
                "Set up automated email notifications for system alerts",
                "Create a data pipeline to process user feedback",
                "Build a web scraper for product price monitoring",
                "Integrate a payment API into an e-commerce website"
            ]
        },
        "multi_step_reasoning": {
            "system_prompt": """You are an AI assistant that excels at breaking down complex problems into logical steps.
            You should demonstrate clear reasoning chains, show your thought process, and explain how each step 
            contributes to solving the overall problem.""",
            "scenarios": [
                "Plan a comprehensive marketing strategy for a new product launch",
                "Troubleshoot why our web application is running slowly",
                "Design a database schema for a social media platform",
                "Optimize our cloud infrastructure costs while maintaining performance",
                "Investigate and resolve customer complaints about billing issues",
                "Create a hiring process for a remote development team",
                "Develop a backup and disaster recovery plan",
                "Analyze and improve our application's security posture"
            ]
        },
        "error_handling": {
            "system_prompt": """You are an AI assistant that specializes in identifying, diagnosing, and fixing errors.
            You should demonstrate systematic debugging approaches, error analysis, and solution implementation.""",
            "scenarios": [
                "My API is returning 500 errors intermittently",
                "Users are reporting that the login system isn't working",
                "The database connection keeps timing out",
                "Our deployment pipeline is failing at the testing stage",
                "Email notifications stopped working after the last update",
                "The mobile app crashes when users try to upload images",
                "Our analytics dashboard shows incorrect data",
                "The payment processing system is declining valid cards"
            ]
        },
        "code_debugging": {
            "system_prompt": """You are an AI assistant that helps developers debug and fix code issues.
            You should analyze code systematically, identify root causes, and provide working solutions.""",
            "scenarios": [
                "This Python function is not returning the expected results",
                "My JavaScript code has a memory leak that I can't locate",
                "The SQL query is running too slowly in production",
                "My React component isn't re-rendering when props change",
                "The unit tests are failing after refactoring the code",
                "There's a race condition in my concurrent Go program",
                "My machine learning model predictions are inconsistent",
                "The API integration is failing with authentication errors"
            ]
        },
        "data_analysis": {
            "system_prompt": """You are an AI assistant that helps with data analysis and interpretation.
            You should demonstrate data exploration, statistical analysis, and insight generation.""",
            "scenarios": [
                "Analyze customer churn patterns in our subscription data",
                "Find correlations between marketing spend and user acquisition",
                "Identify seasonal trends in our e-commerce sales data",
                "Detect anomalies in our server performance metrics",
                "Segment users based on their behavior patterns",
                "Predict inventory needs for the next quarter",
                "Analyze A/B test results for our new feature",
                "Create insights from customer feedback surveys"
            ]
        },
        "web_research": {
            "system_prompt": """You are an AI assistant that helps with research and information gathering.
            You should demonstrate systematic research approaches, source evaluation, and synthesis.""",
            "scenarios": [
                "Research competitors in the AI automation space",
                "Find the best practices for implementing microservices",
                "Investigate recent trends in cybersecurity threats",
                "Research regulatory requirements for data privacy",
                "Compare cloud providers for our infrastructure needs",
                "Study user experience best practices for mobile apps",
                "Research emerging technologies that could benefit our business",
                "Analyze industry benchmarks for our performance metrics"
            ]
        },
        "api_integration": {
            "system_prompt": """You are an AI assistant that helps with API integration and development.
            You should demonstrate API exploration, integration planning, and implementation.""",
            "scenarios": [
                "Integrate Stripe payment processing into our application",
                "Connect our app to the Salesforce CRM API",
                "Set up webhook endpoints to receive real-time updates",
                "Implement OAuth authentication with Google",
                "Build a data sync between our database and external services",
                "Create API rate limiting and error handling",
                "Design RESTful endpoints for our mobile app",
                "Integrate third-party analytics and tracking services"
            ]
        },
        "problem_solving": {
            "system_prompt": """You are an AI assistant that excels at general problem-solving and decision-making.
            You should demonstrate analytical thinking, option evaluation, and solution recommendation.""",
            "scenarios": [
                "Our team is struggling with remote collaboration efficiency",
                "We need to reduce our customer support response times",
                "How can we improve our application's user onboarding",
                "Our development team needs better code review processes",
                "We want to implement CI/CD but don't know where to start",
                "How should we structure our data for better analytics",
                "We need to improve our application's performance monitoring",
                "What's the best way to handle user permissions and roles"
            ]
        }
    }
    
    # Available tools that the AI can reference
    AVAILABLE_TOOLS = [
        {
            "name": "web_search",
            "description": "Search the internet for current information",
            "parameters": ["query", "num_results"]
        },
        {
            "name": "code_analyzer",
            "description": "Analyze code for bugs, performance issues, and best practices",
            "parameters": ["code", "language", "analysis_type"]
        },
        {
            "name": "data_processor", 
            "description": "Process and analyze datasets",
            "parameters": ["data_file", "operation", "filters"]
        },
        {
            "name": "api_client",
            "description": "Make requests to external APIs",
            "parameters": ["endpoint", "method", "headers", "payload"]
        },
        {
            "name": "file_manager",
            "description": "Read, write, and manipulate files",
            "parameters": ["file_path", "operation", "content"]
        },
        {
            "name": "database_query",
            "description": "Execute database queries and operations",
            "parameters": ["query", "database", "parameters"]
        },
        {
            "name": "system_monitor",
            "description": "Monitor system performance and resources",
            "parameters": ["metric", "duration", "threshold"]
        },
        {
            "name": "email_sender",
            "description": "Send emails and notifications",
            "parameters": ["recipients", "subject", "body", "attachments"]
        }
    ]
    
    def __init__(self, config: CollectionConfig, synthetic_config: SyntheticConfig):
        super().__init__(config, "synthetic")
        self.synthetic_config = synthetic_config
        self.provider = self._create_provider()
        self.scenario_count = 0
    
    def _create_provider(self) -> APIProvider:
        """Create appropriate API provider"""
        if self.synthetic_config.provider == "groq":
            return GroqProvider(self.synthetic_config)
        elif self.synthetic_config.provider == "together":
            return TogetherProvider(self.synthetic_config)
        elif self.synthetic_config.provider == "huggingface":
            return HuggingFaceProvider(self.synthetic_config)
        else:
            raise ValueError(f"Unsupported provider: {self.synthetic_config.provider}")
    
    async def get_item_ids(self) -> Iterator[str]:
        """Generate synthetic scenario IDs"""
        item_ids = []
        
        for scenario_type in self.synthetic_config.scenario_types:
            if scenario_type in self.SCENARIO_TEMPLATES:
                scenarios = self.SCENARIO_TEMPLATES[scenario_type]["scenarios"]
                for i, scenario in enumerate(scenarios):
                    # Generate multiple variations per scenario
                    # Use pipe separator to avoid conflicts with scenario type names
                    for variation in range(3):  # 3 variations per scenario
                        item_id = f"{scenario_type}|{i}|{variation}"
                        item_ids.append(item_id)
        
        self.logger.info(f"Generated {len(item_ids)} synthetic scenario IDs")
        return iter(item_ids)
    
    async def __aenter__(self):
        """Async context manager entry"""
        await super().__aenter__()
        # Initialize provider session
        await self.provider.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        # Close provider session
        if hasattr(self.provider, '__aexit__'):
            await self.provider.__aexit__(exc_type, exc_val, exc_tb)
        await super().__aexit__(exc_type, exc_val, exc_tb)
    
    async def fetch_item(self, item_id: str) -> Optional[DataItem]:
        """Generate a synthetic conversation"""
        
        # Parse item ID (using pipe separator)
        parts = item_id.split('|')
        if len(parts) != 3:
            return None
        
        scenario_type, scenario_idx, variation = parts
        
        if scenario_type not in self.SCENARIO_TEMPLATES:
            return None
        
        try:
            scenario_idx = int(scenario_idx)
            variation = int(variation)
        except ValueError:
            return None
        
        template = self.SCENARIO_TEMPLATES[scenario_type]
        scenarios = template["scenarios"]
        
        if scenario_idx >= len(scenarios):
            return None
        
        base_scenario = scenarios[scenario_idx]
        
        try:
            # Generate conversation (provider context managed at collector level)
            conversation = await self._generate_conversation(
                scenario_type, base_scenario, template["system_prompt"], variation
            )
            
            if not conversation:
                return None
            
            # Create DataItem
            item = DataItem(
                source="synthetic",
                item_id=item_id,
                content=conversation,
                quality_score=0.0,  # Will be assessed later
                timestamp=datetime.now(),
                metadata={
                    "scenario_type": scenario_type,
                    "scenario_index": scenario_idx,
                    "variation": variation,
                    "provider": self.synthetic_config.provider,
                    "model": self.synthetic_config.model_name
                }
            )
            
            return item
                
        except Exception as e:
            self.logger.error(f"Error generating conversation for {item_id}: {e}")
            return None
    
    async def _generate_conversation(
        self, 
        scenario_type: str, 
        base_scenario: str, 
        system_prompt: str, 
        variation: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a full agentic conversation"""
        
        # Create scenario variation
        user_prompt = self._create_scenario_variation(base_scenario, scenario_type, variation)
        
        # Start conversation
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        conversation_turns = []
        agentic_patterns = []
        
        # Add initial user turn
        conversation_turns.append({
            "role": "user",
            "content": user_prompt
        })
        
        # Generate assistant response with tool usage
        for turn in range(self.synthetic_config.max_turns_per_conversation // 2):
            
            # Generate assistant response
            assistant_prompt = self._create_assistant_prompt(scenario_type, turn == 0)
            messages.append({"role": "user", "content": assistant_prompt})
            
            assistant_response = await self.provider.generate_completion(messages)
            if not assistant_response:
                break
            
            # Process response for tool usage and patterns
            processed_response = self._process_assistant_response(
                assistant_response, scenario_type, agentic_patterns
            )
            
            conversation_turns.append(processed_response)
            messages.append({"role": "assistant", "content": assistant_response})
            
            # Sometimes add follow-up user question
            if turn < (self.synthetic_config.max_turns_per_conversation // 2) - 1 and random.random() < 0.7:
                follow_up = await self._generate_follow_up(messages, scenario_type)
                if follow_up:
                    conversation_turns.append({
                        "role": "user",
                        "content": follow_up
                    })
                    messages.append({"role": "user", "content": follow_up})
        
        if not conversation_turns:
            return None
        
        # Determine complexity
        complexity = self._assess_complexity(conversation_turns, agentic_patterns)
        
        return {
            "conversation_type": f"synthetic_{scenario_type}",
            "turns": conversation_turns,
            "agentic_patterns": list(set(agentic_patterns)),
            "domain": scenario_type,
            "complexity": complexity,
            "scenario_description": base_scenario,
            "available_tools": random.sample(self.AVAILABLE_TOOLS, random.randint(3, 6))
        }
    
    def _create_scenario_variation(self, base_scenario: str, scenario_type: str, variation: int) -> str:
        """Create variations of base scenarios"""
        
        variations = {
            0: base_scenario,  # Original
            1: f"I'm having trouble with {base_scenario.lower()}. Can you help me understand the best approach?",
            2: f"My team needs guidance on {base_scenario.lower()}. What would you recommend?"
        }
        
        return variations.get(variation, base_scenario)
    
    def _create_assistant_prompt(self, scenario_type: str, is_first_turn: bool) -> str:
        """Create prompts for assistant responses"""
        
        if is_first_turn:
            prompts = [
                "Please provide a detailed step-by-step solution. Use available tools when necessary and explain your reasoning.",
                "I'll help you solve this systematically. Let me break this down and use the appropriate tools.",
                "This requires a structured approach. I'll analyze the situation and use relevant tools to help you."
            ]
        else:
            prompts = [
                "Continue with the next steps, using additional tools if needed.",
                "Let me dig deeper into this and provide more specific guidance.",
                "Based on the results so far, here's what we should do next."
            ]
        
        return random.choice(prompts)
    
    def _process_assistant_response(
        self, 
        response: str, 
        scenario_type: str, 
        agentic_patterns: List[str]
    ) -> Dict[str, Any]:
        """Process assistant response and extract agentic patterns"""
        
        turn = {
            "role": "assistant",
            "content": response
        }
        
        # Detect agentic patterns in response
        response_lower = response.lower()
        
        # Tool usage patterns
        if any(tool["name"] in response_lower for tool in self.AVAILABLE_TOOLS):
            agentic_patterns.append("tool_calling")
            turn["tool_calls"] = self._extract_tool_calls(response)
        
        # Reasoning patterns
        if any(phrase in response_lower for phrase in ["step", "first", "then", "next", "because", "reasoning"]):
            agentic_patterns.append("step_by_step_reasoning")
        
        # Problem decomposition
        if any(phrase in response_lower for phrase in ["break down", "analyze", "separate", "divide"]):
            agentic_patterns.append("problem_decomposition")
        
        # Error handling
        if any(phrase in response_lower for phrase in ["error", "issue", "problem", "troubleshoot", "debug"]):
            agentic_patterns.append("error_handling")
        
        # Planning
        if any(phrase in response_lower for phrase in ["plan", "strategy", "approach", "method"]):
            agentic_patterns.append("planning")
        
        return turn
    
    def _extract_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """Extract tool calls from assistant response"""
        
        tool_calls = []
        response_lower = response.lower()
        
        for tool in self.AVAILABLE_TOOLS:
            if tool["name"] in response_lower:
                # Create synthetic tool call
                tool_call = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "arguments": {}
                    }
                }
                
                # Add some realistic parameters
                for param in tool["parameters"][:2]:  # Use first 2 params
                    tool_call["function"]["arguments"][param] = f"example_{param}"
                
                tool_calls.append(tool_call)
        
        return tool_calls
    
    async def _generate_follow_up(self, messages: List[Dict[str, str]], scenario_type: str) -> Optional[str]:
        """Generate follow-up questions from user"""
        
        follow_ups = [
            "Can you explain that step in more detail?",
            "What if that approach doesn't work?",
            "Are there any alternative methods?",
            "How would you handle errors in this process?",
            "What are the potential risks with this solution?",
            "Can you show me a specific example?",
            "How would this scale for larger datasets?",
            "What tools would be best for this task?"
        ]
        
        return random.choice(follow_ups)
    
    def _assess_complexity(self, turns: List[Dict[str, Any]], patterns: List[str]) -> str:
        """Assess conversation complexity"""
        
        # Count indicators
        tool_usage = sum(1 for turn in turns if "tool_calls" in turn)
        reasoning_indicators = len([p for p in patterns if p in ["step_by_step_reasoning", "problem_decomposition"]])
        content_length = sum(len(turn.get("content", "")) for turn in turns)
        
        if tool_usage >= 2 and reasoning_indicators >= 2 and content_length > 1000:
            return "high"
        elif tool_usage >= 1 or reasoning_indicators >= 1 or content_length > 500:
            return "medium"
        else:
            return "low"
    
    async def assess_quality(self, item: DataItem) -> float:
        """Assess quality of synthetic conversation"""
        
        content = item.content
        turns = content.get("turns", [])
        agentic_patterns = content.get("agentic_patterns", [])
        complexity = content.get("complexity", "low")
        
        score = 0.0
        
        # Basic conversation structure
        if len(turns) >= 2:
            score += 0.2
        if len(turns) >= 4:
            score += 0.1
        
        # Agentic patterns
        high_value_patterns = [
            "tool_calling", "step_by_step_reasoning", "problem_decomposition",
            "error_handling", "planning"
        ]
        pattern_score = len([p for p in agentic_patterns if p in high_value_patterns]) * 0.15
        score += min(pattern_score, 0.4)
        
        # Tool usage
        has_tool_calls = any("tool_calls" in turn for turn in turns)
        if has_tool_calls:
            score += 0.2
        
        # Content quality
        total_length = sum(len(turn.get("content", "")) for turn in turns)
        if total_length > 300:
            score += 0.1
        if total_length > 800:
            score += 0.1
        
        # Complexity bonus
        if complexity == "high":
            score += 0.1
        elif complexity == "medium":
            score += 0.05
        
        return min(score, 1.0)


async def collect_synthetic_data(
    config: CollectionConfig,
    synthetic_config: SyntheticConfig,
    max_items: Optional[int] = None
) -> Dict[str, Any]:
    """Collect synthetic agentic data"""
    
    generator = SyntheticDataGenerator(config, synthetic_config)
    
    async with generator:
        metrics = await generator.collect(max_items)
        
        return {
            "source": "synthetic",
            "metrics": metrics,
            "provider": synthetic_config.provider,
            "model": synthetic_config.model_name,
            "scenario_types": synthetic_config.scenario_types,
            "output_directory": str(generator.output_dir)
        } 