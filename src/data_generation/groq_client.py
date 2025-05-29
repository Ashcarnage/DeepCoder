"""
Groq API Client for DeepSeek R1 Teacher Model Integration
Handles API calls with retry logic, rate limiting, and error handling.
"""

import os
import time
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging
from groq import Groq
from groq.types.chat import ChatCompletion
import yaml
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@dataclass
class GroqConfig:
    """Configuration for Groq API client"""
    api_key: str
    model: str = "deepseek-chat"
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_delay: float = 0.1
    timeout: int = 60

class GroqAPIError(Exception):
    """Custom exception for Groq API errors"""
    pass

class GroqRateLimitError(GroqAPIError):
    """Exception for rate limit errors"""
    pass

class GroqClient:
    """
    Robust Groq API client with error handling and retry logic
    """
    
    def __init__(self, config: Optional[GroqConfig] = None):
        """Initialize Groq client with configuration"""
        if config is None:
            config = self._load_config()
        
        self.config = config
        self.client = Groq(api_key=config.api_key)
        self.last_request_time = 0
        
        logger.info(f"Initialized Groq client with model: {config.model}")
    
    def _load_config(self) -> GroqConfig:
        """Load configuration from environment and config files"""
        # Load from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE_REPLACE_THIS":
            raise ValueError(
                "GROQ_API_KEY not found in environment variables. "
                "Please set it in your .env file or environment."
            )
        
        # Load additional config from yaml
        config_path = "configs/config.yaml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                # Updated to match the YAML structure
                teacher_config = yaml_config.get('models', {}).get('teacher', {})
        else:
            teacher_config = {}
        
        # Prioritize environment variables, then config file, then defaults
        model_name = (
            os.getenv('TEACHER_MODEL') or 
            teacher_config.get('name') or 
            'deepseek-r1-distill-llama-70b'
        )
        
        return GroqConfig(
            api_key=api_key,
            model=model_name,
            max_tokens=teacher_config.get('max_tokens', 4096),
            temperature=teacher_config.get('temperature', 0.7),
            top_p=teacher_config.get('top_p', 0.9),
            max_retries=teacher_config.get('max_retries', 3),
            retry_delay=teacher_config.get('retry_delay', 1.0),
            rate_limit_delay=teacher_config.get('rate_limit_delay', 0.1),
            timeout=teacher_config.get('timeout', 60)
        )
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.config.rate_limit_delay:
            sleep_time = self.config.rate_limit_delay - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((GroqRateLimitError, GroqAPIError))
    )
    def _make_request(self, messages: List[Dict[str, str]], **kwargs) -> ChatCompletion:
        """Make a request to Groq API with retry logic"""
        self._enforce_rate_limit()
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                timeout=self.config.timeout
            )
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            if "rate limit" in error_msg or "429" in error_msg:
                logger.warning(f"Rate limit hit, retrying: {e}")
                raise GroqRateLimitError(f"Rate limit exceeded: {e}")
            elif "timeout" in error_msg:
                logger.warning(f"Request timeout, retrying: {e}")
                raise GroqAPIError(f"Request timeout: {e}")
            else:
                logger.error(f"Groq API error: {e}")
                raise GroqAPIError(f"API error: {e}")
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response from teacher model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            response = self._make_request(messages, **kwargs)
            
            return {
                'content': response.choices[0].message.content,
                'model': response.model,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason,
                'created': response.created
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def generate_reasoning_trajectory(
        self, 
        problem: str, 
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a reasoning trajectory for a coding problem
        
        Args:
            problem: The coding problem statement
            system_prompt: Optional system prompt override
            
        Returns:
            Dictionary containing the reasoning trajectory
        """
        if system_prompt is None:
            system_prompt = self._get_default_reasoning_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem}
        ]
        
        return self.generate_response(messages)
    
    def _get_default_reasoning_prompt(self) -> str:
        """Get default system prompt for reasoning trajectory generation"""
        return """You are DeepSeek R1, an advanced AI assistant specialized in coding and problem-solving.

When given a coding problem, you should:
1. Think through the problem step by step
2. Consider multiple approaches
3. Reason about edge cases and constraints
4. Provide clear explanations for your thought process
5. Generate clean, efficient code

Format your response with clear reasoning sections followed by the final code solution.
Use <thinking> tags to show your internal reasoning process.
Use <action> tags to indicate concrete steps or code implementations.
Use <observation> tags to reflect on results or intermediate steps.

Be thorough in your reasoning but concise in your final solution."""
    
    async def generate_batch_responses(
        self, 
        batch_messages: List[List[Dict[str, str]]], 
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for a batch of message sets
        
        Args:
            batch_messages: List of message lists
            **kwargs: Additional parameters for API calls
            
        Returns:
            List of response dictionaries
        """
        responses = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(
                f"Generating {len(batch_messages)} responses...", 
                total=len(batch_messages)
            )
            
            for messages in batch_messages:
                try:
                    response = self.generate_response(messages, **kwargs)
                    responses.append(response)
                    progress.advance(task)
                    
                except Exception as e:
                    logger.error(f"Failed to generate response for batch item: {e}")
                    responses.append({
                        'content': None,
                        'error': str(e),
                        'model': self.config.model,
                        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                    })
                    progress.advance(task)
        
        return responses
    
    def test_connection(self) -> bool:
        """Test the connection to Groq API"""
        try:
            test_messages = [
                {"role": "user", "content": "Hello, please respond with 'Connection successful'"}
            ]
            
            response = self.generate_response(test_messages, max_tokens=50)
            
            if response.get('content'):
                logger.info("✓ Groq API connection successful")
                return True
            else:
                logger.error("✗ Groq API connection failed - no content in response")
                return False
                
        except Exception as e:
            logger.error(f"✗ Groq API connection failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model': self.config.model,
            'max_tokens': self.config.max_tokens,
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'api_configured': bool(self.config.api_key and 
                                 self.config.api_key != "YOUR_GROQ_API_KEY_HERE_REPLACE_THIS")
        }

# Convenience function for easy import
def create_groq_client(config: Optional[GroqConfig] = None) -> GroqClient:
    """Create and return a configured Groq client"""
    return GroqClient(config)

if __name__ == "__main__":
    # Test the client
    try:
        client = create_groq_client()
        console.print("[green]✓ Groq client created successfully[/green]")
        
        if client.test_connection():
            console.print("[green]✓ API connection test passed[/green]")
            
            # Test reasoning trajectory generation
            test_problem = "Write a Python function to find the maximum element in a list."
            response = client.generate_reasoning_trajectory(test_problem)
            
            console.print(f"[blue]Model:[/blue] {response.get('model')}")
            console.print(f"[blue]Tokens used:[/blue] {response.get('usage', {}).get('total_tokens', 0)}")
            console.print(f"[blue]Response preview:[/blue] {response.get('content', '')[:200]}...")
            
        else:
            console.print("[red]✗ API connection test failed[/red]")
            
    except Exception as e:
        console.print(f"[red]✗ Error testing Groq client: {e}[/red]")
        console.print("[yellow]Make sure to set your GROQ_API_KEY in the .env file[/yellow]") 