"""
SGLang Server Manager for Qwen3 MOE Model

This module provides functionality to manage SGLang server instances for serving
the Qwen3-30B-A3B model with MOE optimization. It includes server lifecycle
management, health checks, and client configuration.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
import yaml
from typing import Any, Dict, Optional, Union
import psutil
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return {}


class SGLangServer:
    """Manages SGLang server instance for Qwen3 MOE model."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize SGLang server manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.sglang_config = self.config.get('sglang', {})
        self.server_config = self.sglang_config.get('server', {})
        self.client_config = self.sglang_config.get('client', {})
        
        self.process: Optional[subprocess.Popen] = None
        self.client: Optional[OpenAI] = None
        self.is_running = False
        
        # Server configuration
        self.host = self.server_config.get('host', '127.0.0.1')
        self.port = self.server_config.get('port', 30000)
        self.model_path = self.server_config.get('model_path', '/workspace/persistent/models/qwen3-30b-a3b')
        
        # Validate model path
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
    
    def _build_server_command(self) -> list:
        """Build SGLang server launch command.
        
        Returns:
            Command list for subprocess
        """
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--reasoning-parser", self.server_config.get('reasoning_parser', 'qwen3')
        ]
        
        # Add optional parameters
        optional_params = {
            'tp_size': '--tp-size',
            'dp_size': '--dp-size', 
            'max_running_requests': '--max-running-requests',
            'max_total_tokens': '--max-total-tokens',
            'max_prefill_tokens': '--max-prefill-tokens',
            'context_length': '--context-length',
            'mem_fraction_static': '--mem-fraction-static',
            'chunked_prefill_size': '--chunked-prefill-size',
            'tensor_parallel_size': '--tensor-parallel-size',
            'served_model_name': '--served-model-name',
            'tokenizer_mode': '--tokenizer-mode'
        }
        
        for config_key, cli_arg in optional_params.items():
            value = self.server_config.get(config_key)
            if value is not None:
                cmd.extend([cli_arg, str(value)])
        
        # Add boolean flags
        boolean_flags = {
            'enable_flashinfer': '--enable-flashinfer',
            'trust_remote_code': '--trust-remote-code',
            'disable_regex_jump_forward': '--disable-regex-jump-forward',
            'disable_custom_all_reduce': '--disable-custom-all-reduce'
        }
        
        for config_key, cli_flag in boolean_flags.items():
            if self.server_config.get(config_key, False):
                cmd.append(cli_flag)
        
        # Add chat template if specified
        chat_template = self.server_config.get('chat_template')
        if chat_template:
            cmd.extend(['--chat-template', chat_template])
        
        logger.info(f"SGLang server command: {' '.join(cmd)}")
        return cmd
    
    def start_server(self, timeout: int = 300) -> bool:
        """Start SGLang server.
        
        Args:
            timeout: Maximum time to wait for server startup
            
        Returns:
            True if server started successfully, False otherwise
        """
        if self.is_running:
            logger.warning("SGLang server is already running")
            return True
        
        logger.info("Starting SGLang server...")
        
        try:
            # Build command
            cmd = self._build_server_command()
            
            # Set environment variables
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU
            
            # Start server process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for server to be ready
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.check_health():
                    self.is_running = True
                    logger.info(f"SGLang server started successfully on {self.host}:{self.port}")
                    return True
                
                # Check if process crashed
                if self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"SGLang server crashed during startup:")
                    logger.error(f"STDOUT: {stdout.decode()}")
                    logger.error(f"STDERR: {stderr.decode()}")
                    return False
                
                time.sleep(5)
            
            logger.error(f"SGLang server failed to start within {timeout} seconds")
            self.stop_server()
            return False
            
        except Exception as e:
            logger.error(f"Failed to start SGLang server: {e}")
            return False
    
    def stop_server(self) -> bool:
        """Stop SGLang server gracefully.
        
        Returns:
            True if server stopped successfully, False otherwise
        """
        if not self.is_running or not self.process:
            logger.info("SGLang server is not running")
            return True
        
        logger.info("Stopping SGLang server...")
        
        try:
            # Send SIGTERM for graceful shutdown
            if os.name != 'nt':
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            else:
                self.process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing SGLang server")
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait()
            
            self.is_running = False
            self.process = None
            logger.info("SGLang server stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop SGLang server: {e}")
            return False
    
    def check_health(self) -> bool:
        """Check if SGLang server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            url = f"http://{self.host}:{self.port}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_server_info(self) -> Optional[Dict[str, Any]]:
        """Get server information.
        
        Returns:
            Server info dict or None if unavailable
        """
        try:
            url = f"http://{self.host}:{self.port}/v1/models"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return None
    
    def get_client(self) -> OpenAI:
        """Get or create OpenAI client for SGLang server.
        
        Returns:
            Configured OpenAI client
        """
        if self.client is None:
            self.client = OpenAI(
                api_key=self.client_config.get('api_key', 'EMPTY'),
                base_url=self.client_config.get('base_url', f"http://{self.host}:{self.port}/v1"),
                timeout=self.client_config.get('timeout', 120),
                max_retries=self.client_config.get('max_retries', 3)
            )
        return self.client
    
    def generate_response(self, 
                         messages: list,
                         enable_thinking: Optional[bool] = None,
                         **kwargs) -> Dict[str, Any]:
        """Generate response using SGLang server.
        
        Args:
            messages: List of message dicts
            enable_thinking: Whether to enable thinking mode
            **kwargs: Additional generation parameters
            
        Returns:
            Response dict with content and thinking
        """
        if not self.is_running:
            raise RuntimeError("SGLang server is not running")
        
        client = self.get_client()
        
        # Prepare generation parameters
        gen_params = {
            'model': self.server_config.get('served_model_name', 'qwen3-30b-a3b'),
            'messages': messages,
            'temperature': kwargs.get('temperature', self.client_config.get('temperature', 0.6)),
            'top_p': kwargs.get('top_p', self.client_config.get('top_p', 0.95)),
            'top_k': kwargs.get('top_k', self.client_config.get('top_k', 20)),
            'presence_penalty': kwargs.get('presence_penalty', self.client_config.get('presence_penalty', 0.0)),
            'max_tokens': kwargs.get('max_tokens', self.client_config.get('max_tokens', 32768)),
        }
        
        # Add thinking mode parameter
        if enable_thinking is not None:
            gen_params['extra_body'] = {'enable_thinking': enable_thinking}
        elif self.client_config.get('enable_thinking'):
            gen_params['extra_body'] = {'enable_thinking': True}
        
        try:
            response = client.chat.completions.create(**gen_params)
            
            # Extract content and thinking
            content = response.choices[0].message.content
            thinking_content = ""
            final_content = content
            
            # Parse thinking content if present
            if content and "<think>" in content:
                try:
                    start_idx = content.find("<think>") + 7
                    end_idx = content.find("</think>")
                    if start_idx > 6 and end_idx > start_idx:
                        thinking_content = content[start_idx:end_idx].strip()
                        final_content = content[end_idx + 8:].strip()
                except Exception:
                    pass  # Fallback to original content
            
            return {
                'content': final_content,
                'thinking': thinking_content,
                'full_response': content,
                'usage': response.usage.dict() if response.usage else {},
                'model': response.model
            }
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()


class SGLangManager:
    """High-level manager for SGLang server instances."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize SGLang manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.server: Optional[SGLangServer] = None
        
    def start(self, timeout: int = 300) -> bool:
        """Start SGLang server.
        
        Args:
            timeout: Maximum time to wait for server startup
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.server and self.server.is_running:
            logger.info("SGLang server is already running")
            return True
        
        self.server = SGLangServer(self.config_path)
        return self.server.start_server(timeout)
    
    def stop(self) -> bool:
        """Stop SGLang server.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.server:
            return True
        
        result = self.server.stop_server()
        self.server = None
        return result
    
    def restart(self, timeout: int = 300) -> bool:
        """Restart SGLang server.
        
        Args:
            timeout: Maximum time to wait for server startup
            
        Returns:
            True if restarted successfully, False otherwise
        """
        self.stop()
        time.sleep(5)  # Brief pause between stop and start
        return self.start(timeout)
    
    def is_healthy(self) -> bool:
        """Check if SGLang server is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        return self.server.check_health() if self.server else False
    
    def get_client(self) -> Optional[OpenAI]:
        """Get OpenAI client for SGLang server.
        
        Returns:
            OpenAI client or None if server not running
        """
        return self.server.get_client() if self.server and self.server.is_running else None
    
    def generate(self, messages: list, **kwargs) -> Dict[str, Any]:
        """Generate response using SGLang server.
        
        Args:
            messages: List of message dicts
            **kwargs: Additional generation parameters
            
        Returns:
            Response dict
        """
        if not self.server or not self.server.is_running:
            raise RuntimeError("SGLang server is not running")
        
        return self.server.generate_response(messages, **kwargs)


def main():
    """CLI interface for SGLang server management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SGLang Server Manager")
    parser.add_argument("command", choices=["start", "stop", "restart", "status"],
                      help="Command to execute")
    parser.add_argument("--config", default="configs/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--timeout", type=int, default=300,
                      help="Timeout for server startup")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    manager = SGLangManager(args.config)
    
    if args.command == "start":
        success = manager.start(args.timeout)
        exit(0 if success else 1)
    elif args.command == "stop":
        success = manager.stop()
        exit(0 if success else 1)
    elif args.command == "restart":
        success = manager.restart(args.timeout)
        exit(0 if success else 1)
    elif args.command == "status":
        if manager.is_healthy():
            print("SGLang server is running and healthy")
            exit(0)
        else:
            print("SGLang server is not running or unhealthy")
            exit(1)


if __name__ == "__main__":
    main() 