"""
Student Model Integration for DeepCoder

This module provides integration with the Qwen3-30B-A3B student model through SGLang
server API calls. It handles model interactions, response processing, and maintains
compatibility with the existing agent framework.
"""

import asyncio
import json
import logging
import time
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from openai import OpenAI

# Use try/except for flexible importing
try:
    from .sglang_manager import SGLangManager, load_config
except ImportError:
    from sglang_manager import SGLangManager, load_config

logger = logging.getLogger(__name__)


@dataclass
class StudentModelConfig:
    """Configuration for student model integration."""
    
    # Model identification
    model_name: str = "Qwen/Qwen3-30B-A3B"
    served_model_name: str = "qwen3-30b-a3b"
    
    # Generation parameters
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    presence_penalty: float = 0.0
    max_tokens: int = 32768
    
    # Thinking mode configuration
    enable_thinking: bool = True
    thinking_timeout: int = 300
    
    # Connection settings
    timeout: int = 120
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance settings
    context_length: int = 32768
    max_concurrent_requests: int = 4


@dataclass
class StudentResponse:
    """Structured response from student model."""
    
    content: str
    thinking: str = ""
    full_response: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    generation_time: float = 0.0
    success: bool = True
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'content': self.content,
            'thinking': self.thinking,
            'full_response': self.full_response,
            'usage': self.usage,
            'model': self.model,
            'generation_time': self.generation_time,
            'success': self.success,
            'error_message': self.error_message
        }


class StudentModel:
    """
    Student model client using SGLang server API.
    
    This class provides a high-level interface for interacting with the Qwen3 student
    model through SGLang server, handling conversation management, thinking mode,
    and response processing.
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize student model client.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.student_config = self._build_student_config()
        
        # SGLang server manager
        self.sglang_manager = SGLangManager(config_path)
        self.client: Optional[OpenAI] = None
        
        # Connection state
        self.is_connected = False
        self.last_health_check = 0
        self.health_check_interval = 30  # seconds
        
        logger.info(f"Initialized student model: {self.student_config.model_name}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        return load_config(self.config_path)
    
    def _build_student_config(self) -> StudentModelConfig:
        """Build student model configuration from loaded config."""
        # Extract student model config
        models_config = self.config.get('models', {})
        student_config = models_config.get('student', {})
        sglang_config = self.config.get('sglang', {})
        client_config = sglang_config.get('client', {})
        
        return StudentModelConfig(
            model_name=student_config.get('name', 'Qwen/Qwen3-30B-A3B'),
            served_model_name=sglang_config.get('server', {}).get('served_model_name', 'qwen3-30b-a3b'),
            temperature=client_config.get('temperature', 0.6),
            top_p=client_config.get('top_p', 0.95),
            top_k=client_config.get('top_k', 20),
            min_p=client_config.get('min_p', 0.0),
            presence_penalty=client_config.get('presence_penalty', 0.0),
            max_tokens=client_config.get('max_tokens', 32768),
            enable_thinking=client_config.get('enable_thinking', True),
            thinking_timeout=client_config.get('thinking_timeout', 300),
            timeout=client_config.get('timeout', 120),
            max_retries=client_config.get('max_retries', 3),
            context_length=sglang_config.get('server', {}).get('context_length', 32768)
        )
    
    def start_server(self, timeout: int = 600) -> bool:
        """Start SGLang server and establish connection.
        
        Args:
            timeout: Maximum time to wait for server startup
            
        Returns:
            True if server started and connected successfully
        """
        logger.info("Starting SGLang server for student model...")
        
        # Start SGLang server
        if not self.sglang_manager.start(timeout):
            logger.error("Failed to start SGLang server")
            return False
        
        # Establish client connection
        if not self._connect():
            logger.error("Failed to connect to SGLang server")
            return False
        
        logger.info("Student model server started and connected successfully")
        return True
    
    def stop_server(self) -> bool:
        """Stop SGLang server and cleanup connections.
        
        Returns:
            True if stopped successfully
        """
        logger.info("Stopping student model server...")
        
        self.is_connected = False
        self.client = None
        
        return self.sglang_manager.stop()
    
    def _connect(self) -> bool:
        """Establish connection to SGLang server.
        
        Returns:
            True if connection established successfully
        """
        try:
            self.client = self.sglang_manager.get_client()
            if self.client is None:
                return False
            
            # Test connection with a simple request
            test_response = self._test_connection()
            if test_response:
                self.is_connected = True
                self.last_health_check = time.time()
                logger.info("Successfully connected to student model")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to connect to student model: {e}")
            return False
    
    def _test_connection(self) -> bool:
        """Test connection to SGLang server.
        
        Returns:
            True if connection test successful
        """
        try:
            # Simple test request
            test_messages = [
                {"role": "user", "content": "Hello! Please respond with 'Test successful.'"}
            ]
            
            response = self.client.chat.completions.create(
                model=self.student_config.served_model_name,
                messages=test_messages,
                max_tokens=10,
                temperature=0.1
            )
            
            return response.choices[0].message.content is not None
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if student model is healthy and responsive.
        
        Returns:
            True if model is healthy
        """
        current_time = time.time()
        
        # Check if recent health check is available
        if (current_time - self.last_health_check) < self.health_check_interval:
            return self.is_connected
        
        # Perform new health check
        if not self.is_connected or not self.sglang_manager.is_healthy():
            self.is_connected = False
            return False
        
        # Test with simple request
        try:
            test_successful = self._test_connection()
            self.is_connected = test_successful
            self.last_health_check = current_time
            return test_successful
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.is_connected = False
            return False
    
    def generate_response(self, 
                         messages: List[Dict[str, str]],
                         enable_thinking: Optional[bool] = None,
                         **kwargs) -> StudentResponse:
        """Generate response using student model.
        
        Args:
            messages: List of conversation messages
            enable_thinking: Whether to enable thinking mode (overrides config)
            **kwargs: Additional generation parameters
            
        Returns:
            StudentResponse object with generated content
        """
        if not self.is_connected or not self.is_healthy():
            return StudentResponse(
                content="",
                success=False,
                error_message="Student model not connected or unhealthy"
            )
        
        start_time = time.time()
        
        try:
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(enable_thinking, **kwargs)
            gen_params['messages'] = messages
            
            # Generate response with retries
            response = self._generate_with_retries(gen_params)
            
            generation_time = time.time() - start_time
            
            # Process response
            return self._process_response(response, generation_time)
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return StudentResponse(
                content="",
                success=False,
                error_message=f"Generation failed: {str(e)}",
                generation_time=time.time() - start_time
            )
    
    def _prepare_generation_params(self, 
                                 enable_thinking: Optional[bool] = None,
                                 **kwargs) -> Dict[str, Any]:
        """Prepare generation parameters.
        
        Args:
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional parameters
            
        Returns:
            Generation parameters dictionary
        """
        params = {
            'model': self.student_config.served_model_name,
            'temperature': kwargs.get('temperature', self.student_config.temperature),
            'top_p': kwargs.get('top_p', self.student_config.top_p),
            'top_k': kwargs.get('top_k', self.student_config.top_k),
            'presence_penalty': kwargs.get('presence_penalty', self.student_config.presence_penalty),
            'max_tokens': kwargs.get('max_tokens', self.student_config.max_tokens),
        }
        
        # Add thinking mode parameter
        thinking_enabled = enable_thinking if enable_thinking is not None else self.student_config.enable_thinking
        if thinking_enabled:
            params['extra_body'] = {'enable_thinking': True}
        
        return params
    
    def _generate_with_retries(self, gen_params: Dict[str, Any]) -> Any:
        """Generate response with retry logic.
        
        Args:
            gen_params: Generation parameters
            
        Returns:
            OpenAI chat completion response
        """
        last_error = None
        
        for attempt in range(self.student_config.max_retries):
            try:
                response = self.client.chat.completions.create(**gen_params)
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.student_config.max_retries - 1:
                    time.sleep(self.student_config.retry_delay * (attempt + 1))
        
        raise last_error
    
    def _process_response(self, response: Any, generation_time: float) -> StudentResponse:
        """Process OpenAI response into StudentResponse.
        
        Args:
            response: OpenAI chat completion response
            generation_time: Time taken for generation
            
        Returns:
            Processed StudentResponse object
        """
        try:
            # Extract basic content
            content = response.choices[0].message.content or ""
            
            # Parse thinking content if present
            thinking_content = ""
            final_content = content
            
            if content and "<think>" in content:
                try:
                    start_idx = content.find("<think>") + 7
                    end_idx = content.find("</think>")
                    if start_idx > 6 and end_idx > start_idx:
                        thinking_content = content[start_idx:end_idx].strip()
                        final_content = content[end_idx + 8:].strip()
                except Exception:
                    # Fallback to original content if parsing fails
                    pass
            
            # Extract usage information
            usage = {}
            if response.usage:
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            
            return StudentResponse(
                content=final_content,
                thinking=thinking_content,
                full_response=content,
                usage=usage,
                model=response.model,
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            return StudentResponse(
                content="",
                success=False,
                error_message=f"Response processing failed: {str(e)}",
                generation_time=generation_time
            )
    
    def generate_agent_response(self, 
                              system_prompt: str,
                              user_input: str,
                              conversation_history: Optional[List[Dict[str, str]]] = None,
                              enable_thinking: bool = True) -> StudentResponse:
        """Generate agent-style response with proper formatting.
        
        Args:
            system_prompt: System prompt for agent behavior
            user_input: User input/problem
            conversation_history: Previous conversation turns
            enable_thinking: Whether to enable thinking mode
            
        Returns:
            StudentResponse with agent-formatted output
        """
        # Build conversation messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        # Generate response
        return self.generate_response(
            messages=messages,
            enable_thinking=enable_thinking
        )
    
    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


class StudentModelManager:
    """High-level manager for student model operations."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize student model manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.student_model: Optional[StudentModel] = None
        
    def initialize(self, timeout: int = 600) -> bool:
        """Initialize student model and start server.
        
        Args:
            timeout: Maximum time to wait for initialization
            
        Returns:
            True if initialization successful
        """
        logger.info("Initializing student model...")
        
        try:
            self.student_model = StudentModel(self.config_path)
            success = self.student_model.start_server(timeout)
            
            if success:
                logger.info("Student model initialized successfully")
            else:
                logger.error("Failed to initialize student model")
                
            return success
            
        except Exception as e:
            logger.error(f"Student model initialization failed: {e}")
            return False
    
    def shutdown(self) -> bool:
        """Shutdown student model and cleanup resources.
        
        Returns:
            True if shutdown successful
        """
        if self.student_model:
            result = self.student_model.stop_server()
            self.student_model = None
            return result
        return True
    
    def is_ready(self) -> bool:
        """Check if student model is ready for use.
        
        Returns:
            True if model is ready
        """
        return (self.student_model is not None and 
                self.student_model.is_healthy())
    
    def generate(self, *args, **kwargs) -> StudentResponse:
        """Generate response using student model.
        
        Returns:
            StudentResponse object
        """
        if not self.is_ready():
            return StudentResponse(
                content="",
                success=False,
                error_message="Student model not ready"
            )
        
        return self.student_model.generate_response(*args, **kwargs)


def create_student_model(config_path: str = "configs/config.yaml") -> StudentModel:
    """Factory function to create student model instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        StudentModel instance
    """
    return StudentModel(config_path)


def create_student_model_manager(config_path: str = "configs/config.yaml") -> StudentModelManager:
    """Factory function to create student model manager.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        StudentModelManager instance
    """
    return StudentModelManager(config_path)


if __name__ == "__main__":
    # CLI interface for testing student model
    import argparse
    
    parser = argparse.ArgumentParser(description="Student Model CLI")
    parser.add_argument("command", choices=["test", "chat", "benchmark"],
                      help="Command to execute")
    parser.add_argument("--config", default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--thinking", action="store_true",
                      help="Enable thinking mode")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.command == "test":
        # Test student model
        with create_student_model(args.config) as model:
            response = model.generate_response(
                messages=[{"role": "user", "content": "Hello! Can you solve 2+2?"}],
                enable_thinking=args.thinking
            )
            print(f"Response: {response.content}")
            if response.thinking:
                print(f"Thinking: {response.thinking}")
    
    elif args.command == "chat":
        # Interactive chat
        with create_student_model(args.config) as model:
            print("Student Model Chat (type 'quit' to exit)")
            while True:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                
                response = model.generate_response(
                    messages=[{"role": "user", "content": user_input}],
                    enable_thinking=args.thinking
                )
                
                print(f"Student: {response.content}")
                if response.thinking and args.thinking:
                    print(f"Thinking: {response.thinking}")
    
    elif args.command == "benchmark":
        # Simple benchmark
        with create_student_model(args.config) as model:
            test_cases = [
                "What is 2+2?",
                "Write a Python function to calculate factorial",
                "Explain how to sort a list in Python"
            ]
            
            for i, test in enumerate(test_cases, 1):
                print(f"\nTest {i}: {test}")
                response = model.generate_response(
                    messages=[{"role": "user", "content": test}],
                    enable_thinking=args.thinking
                )
                print(f"Response: {response.content}")
                print(f"Time: {response.generation_time:.2f}s") 