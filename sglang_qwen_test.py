#!/usr/bin/env python3
"""
SGLang Qwen 30B Student Model Test
==================================
✅ Focused SGLang integration testing
✅ Proper connection handling and retry logic
✅ Rich CLI formatting with reasoning separation
✅ Real inference demonstrations
✅ Self-deleting test file
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import requests

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Install rich if needed
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.syntax import Syntax
    from rich.markdown import Markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.layout import Layout
    from rich.live import Live
    from rich.syntax import Syntax
    from rich.markdown import Markdown

# OpenAI client for SGLang
try:
    from openai import OpenAI
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

# SGLang integration
try:
    from sglang_manager import SGLangManager, load_config
    SGLANG_AVAILABLE = True
except ImportError as e:
    console = Console()
    console.print(f"[red]SGLang integration not available: {e}[/red]")
    SGLANG_AVAILABLE = False

console = Console()

class SGLangQwenTester:
    """Focused SGLang Qwen 30B testing with proper connection handling."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = config_path
        self.config = load_config(config_path) if SGLANG_AVAILABLE else {}
        self.sglang_manager = None
        self.client = None
        self.server_ready = False
        
        # SGLang connection settings
        self.base_url = "http://127.0.0.1:30000/v1"
        self.model_name = "qwen3-30b-a3b"
        
        console.print(Panel.fit(
            "[bold blue]SGLang Qwen 30B Student Model Test[/bold blue]\n"
            "🚀 Testing SGLang server integration\n"
            "🧠 Demonstrating reasoning capabilities\n"
            "📝 Rich CLI result formatting",
            style="blue"
        ))
    
    def start_sglang_server(self) -> bool:
        """Start SGLang server with proper monitoring."""
        if not SGLANG_AVAILABLE:
            console.print("[red]❌ SGLang not available[/red]")
            return False
        
        console.print("[blue]🚀 Starting SGLang server...[/blue]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            # Initialize SGLang manager
            task1 = progress.add_task("Initializing SGLang manager...", total=100)
            try:
                self.sglang_manager = SGLangManager(self.config_path)
                progress.update(task1, advance=25)
                time.sleep(1)
                
                # Start server
                progress.update(task1, description="Starting SGLang server...")
                success = self.sglang_manager.start(timeout=300)
                progress.update(task1, advance=50)
                
                if not success:
                    progress.update(task1, description="[red]❌ Failed to start server[/red]")
                    return False
                
                # Wait for server to be ready
                progress.update(task1, description="Waiting for server to be ready...")
                if self._wait_for_server_ready(progress, task1):
                    progress.update(task1, advance=25, description="[green]✅ SGLang server ready![/green]")
                    return True
                else:
                    progress.update(task1, description="[red]❌ Server failed to become ready[/red]")
                    return False
                    
            except Exception as e:
                progress.update(task1, description=f"[red]❌ Error: {e}[/red]")
                return False
    
    def _wait_for_server_ready(self, progress, task, max_attempts: int = 30) -> bool:
        """Wait for SGLang server to be ready with proper health checks."""
        
        for attempt in range(max_attempts):
            try:
                # Check if server is responding
                response = requests.get(f"http://127.0.0.1:30000/health", timeout=5)
                if response.status_code == 200:
                    progress.update(task, advance=1)
                    console.print("[green]✅ Server health check passed[/green]")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            try:
                # Alternative: try to create client and test basic endpoint
                test_client = OpenAI(
                    api_key="EMPTY",
                    base_url=self.base_url
                )
                
                # Test with a very simple request
                response = test_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                    timeout=10
                )
                
                if response and response.choices:
                    progress.update(task, advance=1)
                    console.print("[green]✅ SGLang client test successful[/green]")
                    return True
                    
            except Exception as e:
                # This is expected during startup
                pass
            
            progress.update(task, advance=1)
            time.sleep(2)  # Wait before next attempt
        
        return False
    
    def setup_client(self) -> bool:
        """Setup OpenAI client for SGLang."""
        try:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=self.base_url
            )
            
            # Test connection with simple request
            console.print("[blue]🔗 Testing SGLang client connection...[/blue]")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello! Please respond with 'Connection successful'."}],
                max_tokens=10,
                temperature=0.1
            )
            
            if response and response.choices:
                console.print("[green]✅ SGLang client connected successfully![/green]")
                console.print(f"[dim]Test response: {response.choices[0].message.content}[/dim]")
                return True
            else:
                console.print("[red]❌ No response from SGLang server[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]❌ Client setup failed: {e}[/red]")
            return False
    
    def parse_response_with_thinking(self, response_text: str) -> Dict[str, str]:
        """Parse SGLang response to separate thinking and final answer."""
        
        # SGLang often returns responses with <thinking> tags or reasoning patterns
        thinking = ""
        final_answer = ""
        
        # Check for explicit thinking tags
        if "<thinking>" in response_text and "</thinking>" in response_text:
            import re
            thinking_match = re.search(r'<thinking>(.*?)</thinking>', response_text, re.DOTALL)
            if thinking_match:
                thinking = thinking_match.group(1).strip()
                final_answer = response_text.replace(thinking_match.group(0), "").strip()
        else:
            # Parse by reasoning indicators
            lines = response_text.split('\n')
            reasoning_lines = []
            answer_lines = []
            in_reasoning = True
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect reasoning patterns
                if any(pattern in line.lower() for pattern in [
                    'let me think', 'step by step', 'first', 'analysis', 
                    'approach', 'reasoning', 'consider', 'examine'
                ]):
                    reasoning_lines.append(line)
                    in_reasoning = True
                # Detect transition to final answer
                elif any(pattern in line.lower() for pattern in [
                    'therefore', 'so the answer', 'final answer', 'in conclusion',
                    'the result is', 'solution:'
                ]):
                    in_reasoning = False
                    answer_lines.append(line)
                else:
                    if in_reasoning and any(word in line.lower() for word in [
                        'because', 'since', 'due to', 'thus', 'hence'
                    ]):
                        reasoning_lines.append(line)
                    else:
                        answer_lines.append(line)
            
            thinking = '\n'.join(reasoning_lines) if reasoning_lines else "Processing the request..."
            final_answer = '\n'.join(answer_lines) if answer_lines else response_text
        
        return {
            'thinking': thinking.strip(),
            'answer': final_answer.strip(),
            'full_response': response_text.strip()
        }
    
    def run_inference_test(self, prompt: str, category: str, max_tokens: int = 400) -> Dict:
        """Run inference test with SGLang and display results."""
        
        if not self.client:
            return {'error': 'SGLang client not available'}
        
        try:
            console.print(f"\n[blue]🧠 Testing {category}:[/blue] {prompt[:80]}...")
            
            start_time = time.time()
            
            # Generate response with SGLang
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            if not response or not response.choices:
                return {'error': 'No response from SGLang'}
            
            # Extract response content
            response_content = response.choices[0].message.content
            
            # Parse thinking and answer
            parsed = self.parse_response_with_thinking(response_content)
            
            # Display results with rich formatting
            self.display_inference_result(
                prompt, parsed, inference_time, category, response
            )
            
            return {
                'prompt': prompt,
                'category': category,
                'thinking': parsed['thinking'],
                'answer': parsed['answer'],
                'full_response': parsed['full_response'],
                'inference_time': inference_time,
                'model_used': self.model_name,
                'usage': response.usage.dict() if response.usage else {},
                'success': True
            }
            
        except Exception as e:
            console.print(f"[red]❌ Inference failed: {e}[/red]")
            return {'error': str(e), 'prompt': prompt, 'category': category}
    
    def display_inference_result(self, prompt: str, parsed: Dict, 
                                inference_time: float, category: str, response):
        """Display inference result with rich CLI formatting."""
        
        # Create main layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="content"),
            Layout(name="footer", size=8)
        )
        
        # Header with key info
        layout["header"].update(
            Panel(
                f"[bold blue]SGLang Qwen 30B Inference[/bold blue] | "
                f"Category: [green]{category}[/green] | "
                f"Time: [yellow]{inference_time:.2f}s[/yellow] | "
                f"Model: [cyan]{self.model_name}[/cyan]",
                style="blue"
            )
        )
        
        # Content area with three columns
        layout["content"].split_row(
            Layout(name="prompt", ratio=1),
            Layout(name="thinking", ratio=2), 
            Layout(name="answer", ratio=2)
        )
        
        # Prompt panel
        layout["prompt"].update(
            Panel(
                prompt,
                title="[bold cyan]📝 Input Prompt[/bold cyan]",
                style="cyan",
                padding=(1, 1)
            )
        )
        
        # Thinking/reasoning panel
        thinking_content = parsed['thinking'] if parsed['thinking'] else "No explicit reasoning detected"
        layout["thinking"].update(
            Panel(
                thinking_content,
                title="[bold yellow]🧠 Reasoning Process[/bold yellow]",
                style="yellow",
                padding=(1, 1)
            )
        )
        
        # Final answer panel
        layout["answer"].update(
            Panel(
                parsed['answer'],
                title="[bold green]✅ Final Answer[/bold green]",
                style="green",
                padding=(1, 1)
            )
        )
        
        # Footer with technical details
        footer_table = Table(show_header=True, header_style="bold magenta")
        footer_table.add_column("Metric", style="dim", width=15)
        footer_table.add_column("Value", justify="right", width=20)
        footer_table.add_column("Details", style="dim")
        
        # Add metrics
        if hasattr(response, 'usage') and response.usage:
            footer_table.add_row(
                "Tokens Used", 
                str(response.usage.total_tokens),
                f"Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}"
            )
        
        footer_table.add_row("Inference Time", f"{inference_time:.2f}s", f"{len(parsed['full_response'])} characters")
        footer_table.add_row("Response Length", f"{len(parsed['answer'])} chars", f"Thinking: {len(parsed['thinking'])} chars")
        footer_table.add_row("Model", self.model_name, "SGLang server")
        
        layout["footer"].update(
            Panel(
                footer_table,
                title="[bold magenta]📊 Technical Details[/bold magenta]",
                style="magenta"
            )
        )
        
        console.print(layout)
        console.print("\n" + "="*120 + "\n")
    
    def run_comprehensive_tests(self):
        """Run comprehensive SGLang tests with various prompts."""
        
        # Test prompts covering different capabilities
        test_prompts = [
            {
                "prompt": "Solve this step by step: A store has 150 apples. They sell 30% in the morning and 25% of the remaining in the afternoon. How many apples are left?",
                "category": "Mathematical Reasoning",
                "max_tokens": 300
            },
            {
                "prompt": "Write a Python function to find the second largest number in a list. Include error handling and explain your approach.",
                "category": "Coding & Explanation", 
                "max_tokens": 400
            },
            {
                "prompt": "Analyze this scenario: A team is working remotely but productivity has decreased. What could be the causes and what solutions would you recommend?",
                "category": "Problem Analysis",
                "max_tokens": 350
            },
            {
                "prompt": "Compare and contrast the advantages of renewable energy sources versus traditional fossil fuels. Provide a balanced perspective.",
                "category": "Comparative Analysis",
                "max_tokens": 400
            },
            {
                "prompt": "Plan a learning strategy for someone who wants to become proficient in machine learning within 6 months. Break it down into phases.",
                "category": "Strategic Planning",
                "max_tokens": 450
            }
        ]
        
        console.print(f"[green]🚀 Running {len(test_prompts)} comprehensive SGLang tests...[/green]\n")
        
        results = []
        
        for i, test in enumerate(test_prompts, 1):
            console.print(f"[bold blue]Test {i}/{len(test_prompts)}:[/bold blue] {test['category']}")
            
            result = self.run_inference_test(
                test['prompt'], 
                test['category'], 
                test['max_tokens']
            )
            
            results.append(result)
            
            if 'error' in result:
                console.print(f"[red]❌ Error: {result['error']}[/red]")
            else:
                console.print(f"[green]✅ Test {i} completed successfully[/green]")
            
            # Brief pause between tests
            time.sleep(2)
        
        return results
    
    def display_summary(self, results: List[Dict]):
        """Display comprehensive test summary."""
        
        console.print("\n" + "="*120)
        console.print(Panel.fit("[bold green]📊 SGLang Test Summary[/bold green]", style="green"))
        
        successful = [r for r in results if 'error' not in r and r.get('success', False)]
        failed = [r for r in results if 'error' in r or not r.get('success', False)]
        
        # Summary statistics
        summary_table = Table(title="Test Results Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan", width=25)
        summary_table.add_column("Value", justify="right", style="green", width=15)
        summary_table.add_column("Details", style="dim")
        
        summary_table.add_row("Total Tests", str(len(results)), f"Various categories")
        summary_table.add_row("Successful", str(len(successful)), "✅ Completed without errors")
        summary_table.add_row("Failed", str(len(failed)), "❌ Had errors or issues")
        
        if successful:
            avg_time = sum(r['inference_time'] for r in successful) / len(successful)
            total_tokens = sum(r.get('usage', {}).get('total_tokens', 0) for r in successful)
            avg_response_length = sum(len(r['answer']) for r in successful) / len(successful)
            
            summary_table.add_row("Avg Inference Time", f"{avg_time:.2f}s", "Per successful test")
            summary_table.add_row("Total Tokens Used", str(total_tokens), "Across all tests")
            summary_table.add_row("Avg Response Length", f"{avg_response_length:.0f} chars", "Final answers only")
        
        console.print(summary_table)
        
        # Category breakdown
        if successful:
            console.print("\n[bold yellow]📋 Test Categories:[/bold yellow]")
            for result in successful:
                console.print(f"  ✅ {result['category']}: {result['inference_time']:.2f}s")
        
        if failed:
            console.print("\n[bold red]❌ Failed Tests:[/bold red]")
            for result in failed:
                error_msg = result.get('error', 'Unknown error')
                console.print(f"  ❌ {result.get('category', 'Unknown')}: {error_msg}")
    
    def save_results(self, results: List[Dict]):
        """Save test results to file."""
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'sglang_config': {
                'base_url': self.base_url,
                'model_name': self.model_name,
                'server_type': 'SGLang'
            },
            'test_results': results,
            'summary': {
                'total_tests': len(results),
                'successful_tests': len([r for r in results if 'error' not in r]),
                'failed_tests': len([r for r in results if 'error' in r])
            }
        }
        
        filename = f"sglang_qwen_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        console.print(f"\n[green]💾 Results saved to: {filename}[/green]")
        return filename
    
    def cleanup(self):
        """Cleanup SGLang server and resources."""
        console.print("\n[blue]🧹 Cleaning up SGLang resources...[/blue]")
        
        if self.sglang_manager:
            try:
                success = self.sglang_manager.stop()
                if success:
                    console.print("[green]✅ SGLang server stopped successfully[/green]")
                else:
                    console.print("[yellow]⚠️ SGLang server stop completed with warnings[/yellow]")
            except Exception as e:
                console.print(f"[red]❌ Error stopping SGLang server: {e}[/red]")
    
    def run_full_test_suite(self):
        """Run the complete SGLang test suite."""
        
        try:
            # Start SGLang server
            if not self.start_sglang_server():
                console.print("[red]❌ Failed to start SGLang server. Exiting.[/red]")
                return
            
            # Setup client
            if not self.setup_client():
                console.print("[red]❌ Failed to setup SGLang client. Exiting.[/red]")
                return
            
            # Run comprehensive tests
            results = self.run_comprehensive_tests()
            
            # Display summary
            self.display_summary(results)
            
            # Save results
            self.save_results(results)
            
            console.print("\n[bold green]🎉 SGLang testing completed successfully![/bold green]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]⚠️ Test interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"\n[red]❌ Test suite failed: {e}[/red]")
        finally:
            self.cleanup()

def main():
    """Main execution function."""
    
    delete_after = len(sys.argv) > 1 and sys.argv[1] == "--delete-after"
    
    # Run SGLang test suite
    tester = SGLangQwenTester()
    tester.run_full_test_suite()
    
    # Self-delete if requested
    if delete_after:
        file_path = Path(__file__)
        try:
            console.print(f"\n[blue]🗑️ Self-deleting test file: {file_path}[/blue]")
            time.sleep(3)  # Give time to read
            file_path.unlink()
            console.print("[green]✅ Test file deleted successfully[/green]")
        except Exception as e:
            console.print(f"[red]❌ Failed to delete test file: {e}[/red]")

if __name__ == "__main__":
    main() 