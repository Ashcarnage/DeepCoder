#!/usr/bin/env python3
"""
Quick test for SAD training results with DeepSeek-R1 format handling
"""

from openai import OpenAI
from rich.console import Console
from rich.panel import Panel

console = Console()

def test_model():
    """Quick test of the model with proper response extraction."""
    
    client = OpenAI(base_url="http://localhost:30000/v1", api_key="EMPTY")
    
    test_prompt = "Debug a Python memory leak consuming 8GB RAM. Available tools: system_monitor, terminal_command, code_analyzer. Use systematic approach."
    
    try:
        response = client.chat.completions.create(
            model="qwen3-30b-a3b",
            messages=[{"role": "user", "content": test_prompt}],
            max_tokens=400,
            temperature=0.3
        )
        
        # Handle DeepSeek-R1 format
        content = response.choices[0].message.content
        if content is None:
            content = getattr(response.choices[0].message, 'reasoning_content', '') or ""
        
        console.print(Panel(
            f"[bold yellow]Prompt:[/bold yellow] {test_prompt}\n\n"
            f"[bold green]Response:[/bold green] {content}",
            title="🔍 Quick SAD Test Result",
            border_style="cyan"
        ))
        
        # Analyze for tool usage
        tools_mentioned = ["system_monitor", "terminal_command", "code_analyzer"]
        tool_count = sum(1 for tool in tools_mentioned if tool.lower() in content.lower())
        
        console.print(f"\n[bold]Analysis:[/bold]")
        console.print(f"• Response length: {len(content.split())} words")
        console.print(f"• Tools mentioned: {tool_count}/{len(tools_mentioned)}")
        console.print(f"• Contains reasoning: {'✅' if any(word in content.lower() for word in ['step', 'approach', 'systematic', 'analysis']) else '❌'}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    test_model() 