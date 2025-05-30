#!/usr/bin/env python3
"""
Simple SGLang Qwen 30B Direct Test
==================================
✅ Direct SGLang API testing
✅ Debug response parsing
✅ Show raw outputs
✅ Build up to proper formatting
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Install rich if needed
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax

# OpenAI client
try:
    from openai import OpenAI
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

console = Console()

def test_sglang_direct():
    """Direct test of SGLang with debugging."""
    
    console.print(Panel.fit(
        "[bold blue]Simple SGLang Qwen 30B Test[/bold blue]\n"
        "🔍 Testing direct API calls\n"
        "🐛 Debugging response parsing\n"
        "📊 Raw output display",
        style="blue"
    ))
    
    # Check if SGLang server is running
    console.print("[blue]🔍 Checking if SGLang server is running...[/blue]")
    
    try:
        import requests
        response = requests.get("http://127.0.0.1:30000/health", timeout=5)
        if response.status_code == 200:
            console.print("[green]✅ SGLang server is running[/green]")
        else:
            console.print(f"[yellow]⚠️ Server responded with status {response.status_code}[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Server not reachable: {e}[/red]")
        console.print("[blue]🚀 Starting SGLang server...[/blue]")
        
        # Try to start server
        try:
            from sglang_manager import SGLangManager, load_config
            manager = SGLangManager("configs/config.yaml")
            success = manager.start(timeout=300)
            if success:
                console.print("[green]✅ SGLang server started[/green]")
                time.sleep(10)  # Give it time to fully load
            else:
                console.print("[red]❌ Failed to start SGLang server[/red]")
                return
        except Exception as e:
            console.print(f"[red]❌ Error starting server: {e}[/red]")
            return
    
    # Create OpenAI client
    console.print("[blue]🔗 Creating SGLang client...[/blue]")
    
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://127.0.0.1:30000/v1"
    )
    
    # Test prompts
    test_prompts = [
        "Hello! Can you respond with a simple greeting?",
        "What is 2 + 2? Please think step by step.",
        "Write a simple Python function to add two numbers.",
        "Explain why the sky is blue in one paragraph.",
        "Solve this: If I have 10 apples and eat 3, how many do I have left? Show your reasoning."
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        console.print(f"\n[bold blue]Test {i}/5:[/bold blue] {prompt[:60]}...")
        
        try:
            # Make API call
            console.print("[dim]Making API call...[/dim]")
            
            response = client.chat.completions.create(
                model="qwen3-30b-a3b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
                stream=False
            )
            
            console.print("[dim]Response received![/dim]")
            
            # Debug: Print raw response structure
            console.print("\n[yellow]📋 Raw Response Structure:[/yellow]")
            console.print(f"Type: {type(response)}")
            console.print(f"Has choices: {hasattr(response, 'choices')}")
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                console.print(f"First choice type: {type(choice)}")
                console.print(f"Has message: {hasattr(choice, 'message')}")
                
                if hasattr(choice, 'message'):
                    message = choice.message
                    console.print(f"Message type: {type(message)}")
                    console.print(f"Has content: {hasattr(message, 'content')}")
                    console.print(f"Content value: {getattr(message, 'content', 'NO CONTENT ATTR')}")
                    console.print(f"Content type: {type(getattr(message, 'content', None))}")
                    
                    # Try to get content
                    content = getattr(message, 'content', None)
                    
                    if content is not None:
                        console.print("\n[green]✅ Response Content:[/green]")
                        
                        # Display in a nice panel
                        display_response(prompt, content, i)
                        
                    else:
                        console.print("[red]❌ Content is None[/red]")
                        
                        # Try alternative ways to get content
                        console.print("[yellow]🔍 Trying alternative content access...[/yellow]")
                        
                        # Check all attributes
                        attrs = [attr for attr in dir(message) if not attr.startswith('_')]
                        console.print(f"Message attributes: {attrs}")
                        
                        # Try different attribute names
                        for attr in ['text', 'content', 'response', 'output']:
                            if hasattr(message, attr):
                                val = getattr(message, attr)
                                console.print(f"  {attr}: {val} (type: {type(val)})")
                else:
                    console.print("[red]❌ Choice has no message attribute[/red]")
            else:
                console.print("[red]❌ Response has no choices[/red]")
            
            # Check usage info
            if hasattr(response, 'usage') and response.usage:
                console.print(f"\n[cyan]📊 Usage: {response.usage}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]❌ Error: {e}[/red]")
            console.print(f"[red]Error type: {type(e)}[/red]")
            import traceback
            console.print(f"[red]Traceback: {traceback.format_exc()}[/red]")
        
        time.sleep(2)  # Brief pause between tests

def display_response(prompt, content, test_num):
    """Display response with nice formatting."""
    
    # Parse content to separate reasoning and answer
    lines = content.split('\n')
    reasoning_lines = []
    answer_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Simple heuristic to separate reasoning from final answer
        if any(word in line.lower() for word in ['step', 'first', 'because', 'since', 'let me', 'think']):
            reasoning_lines.append(line)
        else:
            answer_lines.append(line)
    
    reasoning = '\n'.join(reasoning_lines) if reasoning_lines else "No explicit reasoning detected"
    answer = '\n'.join(answer_lines) if answer_lines else content
    
    # Create display table
    table = Table(title=f"Test {test_num} Results", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan", width=15)
    table.add_column("Content", style="white", width=80)
    
    table.add_row("Prompt", prompt[:200] + "..." if len(prompt) > 200 else prompt)
    table.add_row("Reasoning", reasoning[:300] + "..." if len(reasoning) > 300 else reasoning)
    table.add_row("Answer", answer[:300] + "..." if len(answer) > 300 else answer)
    table.add_row("Full Response", content[:200] + "..." if len(content) > 200 else content)
    
    console.print(table)
    
    # Also show raw content in a panel
    console.print("\n[bold green]📝 Full Raw Response:[/bold green]")
    console.print(Panel(content, style="green", padding=(1, 2)))

def main():
    """Main execution."""
    
    delete_after = len(sys.argv) > 1 and sys.argv[1] == "--delete-after"
    
    try:
        test_sglang_direct()
        console.print("\n[bold green]🎉 Testing completed![/bold green]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Test interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Test failed: {e}[/red]")
    finally:
        # Self-delete if requested
        if delete_after:
            file_path = Path(__file__)
            try:
                console.print(f"\n[blue]🗑️ Self-deleting: {file_path}[/blue]")
                time.sleep(2)
                file_path.unlink()
                console.print("[green]✅ File deleted[/green]")
            except Exception as e:
                console.print(f"[red]❌ Delete failed: {e}[/red]")

if __name__ == "__main__":
    main() 