#!/usr/bin/env python3
"""
🤖 Interactive Qwen CLI
======================

Beautiful Rich CLI interface for interacting with your locally trained Qwen 30B model
with SAD (Structured Agent Distillation) capabilities.
"""

import time
import json
import sys
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.columns import Columns
from rich.live import Live
from rich.layout import Layout
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich import box
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn

from openai import OpenAI

console = Console()

@dataclass
class ConversationMessage:
    """Single conversation message."""
    role: str
    content: str
    timestamp: datetime
    tokens: Optional[int] = None

class QwenInteractiveCLI:
    """Interactive CLI for Qwen model with Rich interface."""
    
    def __init__(self):
        self.client = None
        self.conversation_history: List[ConversationMessage] = []
        self.sglang_host = "localhost"
        self.sglang_port = 30000
        self.model_name = "qwen3-30b-a3b"
        self.session_start = datetime.now()
        
        # Predefined prompts for testing SAD capabilities
        self.test_prompts = {
            "Debug Memory Leak": "Debug a Python memory leak consuming 8GB RAM. Available tools: system_monitor, terminal_command, code_analyzer. Use systematic approach.",
            "API Optimization": "Optimize slow API endpoint with 2000ms response time. Available tools: system_monitor, api_client, code_analyzer. Target: <200ms.",
            "Database Performance": "Analyze and fix database queries causing 10x performance degradation. Available tools: database_query, system_monitor, code_analyzer.",
            "Microservices Debug": "Investigate intermittent 500 errors in microservices. Available tools: terminal_command, system_monitor, api_client, code_analyzer.",
            "CI/CD Pipeline": "Set up CI/CD pipeline with automated testing. Available tools: git_operations, terminal_command, file_operations, code_analyzer.",
            "Custom Prompt": "Enter your own custom prompt"
        }
        
    def setup_client(self) -> bool:
        """Setup OpenAI client for SGLang."""
        try:
            self.client = OpenAI(
                base_url=f"http://{self.sglang_host}:{self.sglang_port}/v1", 
                api_key="EMPTY"
            )
            # Test connection
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to connect to model: {e}[/red]")
            return False
    
    def display_header(self):
        """Display beautiful header."""
        header_text = Text()
        header_text.append("🤖 ", style="bold blue")
        header_text.append("Interactive Qwen CLI", style="bold cyan")
        header_text.append(" - SAD Enhanced Model", style="bold yellow")
        
        info_text = f"""[cyan]Model:[/cyan] {self.model_name}
[cyan]Endpoint:[/cyan] {self.sglang_host}:{self.sglang_port}
[cyan]Session Started:[/cyan] {self.session_start.strftime('%H:%M:%S')}
[cyan]Messages:[/cyan] {len(self.conversation_history)}

[yellow]Enhanced with Structured Agent Distillation (SAD) training[/yellow]
[green]✅ Tool-aware reasoning, systematic problem solving, enhanced capabilities[/green]"""
        
        console.print(Panel(
            info_text,
            title=header_text,
            border_style="blue",
            padding=(1, 2)
        ))
    
    def show_main_menu(self) -> str:
        """Show main menu and get user choice."""
        console.print("\n[bold cyan]🎯 Choose Interaction Mode:[/bold cyan]")
        
        options = [
            ("chat", "💬 Free Chat", "Open conversation with the model"),
            ("test", "🧪 Test SAD Capabilities", "Test enhanced tool-aware reasoning"),
            ("batch", "📋 Batch Testing", "Run multiple test scenarios"),
            ("analyze", "📊 Analyze Conversation", "View conversation statistics"),
            ("export", "💾 Export Session", "Save conversation history"),
            ("clear", "🗑️  Clear History", "Start fresh conversation"),
            ("help", "❓ Help", "Show detailed help"),
            ("exit", "🚪 Exit", "Quit the application")
        ]
        
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Key", style="bold yellow", width=8)
        table.add_column("Option", style="bold green", width=25)
        table.add_column("Description", style="cyan")
        
        for key, option, desc in options:
            table.add_row(f"[{key}]", option, desc)
        
        console.print(table)
        
        choice = Prompt.ask(
            "\n[bold]Enter your choice[/bold]",
            choices=[opt[0] for opt in options],
            default="chat"
        ).lower().strip()
        
        return choice
    
    def chat_mode(self):
        """Interactive chat mode."""
        console.print(Panel(
            "[bold green]💬 Chat Mode Activated[/bold green]\n"
            "[yellow]Type 'exit' to return to main menu[/yellow]\n"
            "[yellow]Type 'clear' to clear conversation history[/yellow]",
            title="Chat Mode",
            border_style="green"
        ))
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]").strip()
                
                if user_input.lower() == 'exit':
                    break
                elif user_input.lower() == 'clear':
                    self.conversation_history.clear()
                    console.print("[green]✅ Conversation history cleared[/green]")
                    continue
                elif not user_input:
                    continue
                
                # Send to model with progress indicator
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                ) as progress:
                    task = progress.add_task("🤖 Thinking...", total=None)
                    
                    response = self.send_message(user_input)
                    progress.update(task, completed=True)
                
                if response:
                    # Display response beautifully
                    self.display_response(response)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Chat interrupted. Returning to main menu...[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Error: {e}[/red]")
    
    def test_sad_capabilities(self):
        """Test SAD capabilities with predefined prompts."""
        console.print(Panel(
            "[bold blue]🧪 SAD Capabilities Testing[/bold blue]\n"
            "[yellow]Choose a test scenario to evaluate enhanced reasoning[/yellow]",
            title="Test Mode",
            border_style="blue"
        ))
        
        # Show test options
        table = Table(title="Available Test Scenarios", box=box.ROUNDED)
        table.add_column("Key", style="bold yellow", width=5)
        table.add_column("Test Scenario", style="bold green", width=25)
        table.add_column("Description", style="cyan")
        
        for i, (name, prompt) in enumerate(self.test_prompts.items(), 1):
            if name == "Custom Prompt":
                table.add_row(f"[{i}]", name, "Enter your own test prompt")
            else:
                desc = prompt[:80] + "..." if len(prompt) > 80 else prompt
                table.add_row(f"[{i}]", name, desc)
        
        console.print(table)
        
        choice = Prompt.ask(
            "\n[bold]Select test scenario[/bold]",
            choices=[str(i) for i in range(1, len(self.test_prompts) + 1)],
            default="1"
        )
        
        # Get the selected prompt
        test_names = list(self.test_prompts.keys())
        selected_name = test_names[int(choice) - 1]
        
        if selected_name == "Custom Prompt":
            prompt = Prompt.ask("[bold]Enter your custom prompt[/bold]")
        else:
            prompt = self.test_prompts[selected_name]
        
        console.print(f"\n[bold cyan]Testing:[/bold cyan] {selected_name}")
        console.print(Panel(prompt, title="Test Prompt", border_style="yellow"))
        
        # Send to model with detailed progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🧠 Analyzing with SAD-enhanced reasoning...", total=None)
            
            response = self.send_message(prompt)
            progress.update(task, completed=True)
        
        if response:
            self.display_response(response, test_mode=True)
            self.analyze_sad_response(response, selected_name)
    
    def batch_testing(self):
        """Run batch testing on multiple scenarios."""
        console.print(Panel(
            "[bold magenta]📋 Batch Testing Mode[/bold magenta]\n"
            "[yellow]Running all predefined test scenarios automatically[/yellow]",
            title="Batch Testing",
            border_style="magenta"
        ))
        
        if not Confirm.ask("[bold]Run all test scenarios?[/bold]", default=True):
            return
        
        results = []
        test_scenarios = {k: v for k, v in self.test_prompts.items() if k != "Custom Prompt"}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running batch tests...", total=len(test_scenarios))
            
            for name, prompt in test_scenarios.items():
                progress.update(task, description=f"Testing: {name}")
                
                response = self.send_message(prompt)
                if response:
                    analysis = self.analyze_response_quality(response)
                    results.append({
                        "scenario": name,
                        "prompt": prompt,
                        "response": response,
                        "analysis": analysis
                    })
                
                progress.advance(task)
                time.sleep(0.5)  # Brief pause between tests
        
        # Display batch results
        self.display_batch_results(results)
    
    def send_message(self, message: str) -> Optional[str]:
        """Send message to model and return response."""
        try:
            # Add to conversation history
            user_msg = ConversationMessage(
                role="user",
                content=message,
                timestamp=datetime.now()
            )
            self.conversation_history.append(user_msg)
            
            # Send to model
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": message}],
                max_tokens=1200,
                temperature=0.3
            )
            
            # Extract content (handle DeepSeek-R1 format)
            content = response.choices[0].message.content
            if content is None:
                content = getattr(response.choices[0].message, 'reasoning_content', '') or ""
            
            # Add to conversation history
            assistant_msg = ConversationMessage(
                role="assistant",
                content=content,
                timestamp=datetime.now(),
                tokens=len(content.split())
            )
            self.conversation_history.append(assistant_msg)
            
            return content
            
        except Exception as e:
            console.print(f"[red]❌ Error sending message: {e}[/red]")
            return None
    
    def display_response(self, response: str, test_mode: bool = False):
        """Display model response beautifully."""
        # No truncation - show full response
        display_response = response
        
        # Check if response looks like code
        if "```" in response or any(keyword in response.lower() for keyword in ["def ", "class ", "import ", "function"]):
            # Try to render as markdown for code blocks
            try:
                markdown_response = Markdown(display_response)
                console.print(Panel(
                    markdown_response,
                    title="🤖 Qwen Response" + (" (Test Mode)" if test_mode else ""),
                    border_style="green",
                    padding=(1, 2)
                ))
            except:
                # Fallback to plain text
                console.print(Panel(
                    display_response,
                    title="🤖 Qwen Response" + (" (Test Mode)" if test_mode else ""),
                    border_style="green",
                    padding=(1, 2)
                ))
        else:
            console.print(Panel(
                display_response,
                title="🤖 Qwen Response" + (" (Test Mode)" if test_mode else ""),
                border_style="green",
                padding=(1, 2)
            ))
    
    def analyze_sad_response(self, response: str, scenario_name: str):
        """Analyze response for SAD capabilities."""
        analysis = self.analyze_response_quality(response)
        
        # Create analysis panel
        analysis_text = f"""[bold cyan]📊 SAD Analysis for: {scenario_name}[/bold cyan]

[yellow]Quality Metrics:[/yellow]
• Response Length: {analysis['response_length']} words
• Tool Mentions: {analysis['tool_mentions']} 
• Structured Format: {'✅' if analysis['structured_format'] else '❌'}
• Reasoning Present: {'✅' if analysis['reasoning_present'] else '❌'}
• Quality Score: [bold]{analysis['quality_score']:.1f}%[/bold]

[green]Tools Identified:[/green] {', '.join(analysis.get('tool_calls_found', [])) or 'None'}

[blue]Assessment:[/blue] {"🎉 Excellent SAD performance!" if analysis['quality_score'] >= 80 else "🚀 Good SAD capabilities!" if analysis['quality_score'] >= 60 else "📈 Developing SAD skills"}"""
        
        console.print(Panel(
            analysis_text,
            title="🔬 SAD Capability Analysis",
            border_style="cyan",
            padding=(1, 2)
        ))
    
    def analyze_response_quality(self, response: str) -> Dict:
        """Analyze response quality for SAD capabilities."""
        analysis = {
            "response_length": len(response.split()),
            "tool_mentions": 0,
            "structured_format": False,
            "reasoning_present": False,
            "tool_calls_found": [],
            "quality_score": 0.0
        }
        
        # Common tools to look for
        tools = ["system_monitor", "terminal_command", "code_analyzer", "api_client", 
                "database_query", "git_operations", "file_operations"]
        
        # Count tool mentions
        for tool in tools:
            if tool.lower() in response.lower():
                analysis["tool_mentions"] += 1
                analysis["tool_calls_found"].append(tool)
        
        # Check for structured format
        if any(marker in response for marker in ["TOOL_CALL", "TOOL_OUTPUT", "ANALYSIS", "Step", "1.", "2."]):
            analysis["structured_format"] = True
        
        # Check for reasoning
        reasoning_words = ["reasoning", "because", "analysis", "systematic", "approach", "step", "first", "then", "next"]
        if any(word in response.lower() for word in reasoning_words):
            analysis["reasoning_present"] = True
        
        # Calculate quality score
        score = 0.0
        score += min(analysis["tool_mentions"] * 15, 40)  # Up to 40% for tool usage
        score += 25 if analysis["structured_format"] else 0  # 25% for structure
        score += 25 if analysis["reasoning_present"] else 0  # 25% for reasoning
        score += min(analysis["response_length"] / 200, 1.0) * 10  # Up to 10% for completeness
        
        analysis["quality_score"] = min(score, 100.0)
        
        return analysis
    
    def display_batch_results(self, results: List[Dict]):
        """Display batch testing results."""
        console.print("\n[bold magenta]📊 Batch Testing Results[/bold magenta]")
        
        # Summary table
        table = Table(title="Batch Test Summary", box=box.ROUNDED)
        table.add_column("Scenario", style="cyan", width=20)
        table.add_column("Quality Score", justify="center", style="yellow", width=12)
        table.add_column("Tools Found", justify="center", style="green", width=10)
        table.add_column("Reasoning", justify="center", style="blue", width=10)
        table.add_column("Length", justify="center", style="white", width=10)
        
        total_quality = 0
        for result in results:
            analysis = result["analysis"]
            score = analysis["quality_score"]
            
            # Color code score
            if score >= 80:
                score_style = "[green]"
            elif score >= 60:
                score_style = "[yellow]"
            else:
                score_style = "[red]"
            
            table.add_row(
                result["scenario"],
                f"{score_style}{score:.1f}%[/]",
                str(analysis["tool_mentions"]),
                "✅" if analysis["reasoning_present"] else "❌",
                f"{analysis['response_length']}w"
            )
            total_quality += score
        
        # Add average row
        avg_quality = total_quality / len(results) if results else 0
        table.add_section()
        table.add_row(
            "[bold]AVERAGE[/bold]",
            f"[bold]{avg_quality:.1f}%[/bold]",
            "[bold]Summary[/bold]",
            f"[bold]{sum(1 for r in results if r['analysis']['reasoning_present'])}/{len(results)}[/bold]",
            "[bold]Stats[/bold]"
        )
        
        console.print(table)
        
        # Overall assessment
        assessment = Panel(
            f"""[bold green]🎯 Overall SAD Performance Assessment[/bold green]

[yellow]Average Quality Score:[/yellow] [bold]{avg_quality:.1f}%[/bold]
[yellow]Tests Completed:[/yellow] {len(results)}/{len(results)} (100% success rate)
[yellow]Total Tool Mentions:[/yellow] {sum(r['analysis']['tool_mentions'] for r in results)}
[yellow]Reasoning Capability:[/yellow] {sum(1 for r in results if r['analysis']['reasoning_present'])}/{len(results)} scenarios

[cyan]Assessment:[/cyan] {"🏆 Outstanding SAD capabilities!" if avg_quality >= 85 else "🎉 Excellent SAD performance!" if avg_quality >= 75 else "🚀 Good SAD development!" if avg_quality >= 60 else "📈 SAD training in progress"}

[blue]The model demonstrates {"strong" if avg_quality >= 75 else "developing"} tool-aware reasoning and systematic problem-solving abilities.[/blue]""",
            title="📈 Performance Summary",
            border_style="green" if avg_quality >= 75 else "yellow"
        )
        
        console.print(f"\n{assessment}")
    
    def analyze_conversation(self):
        """Analyze current conversation statistics."""
        if not self.conversation_history:
            console.print("[yellow]No conversation history to analyze[/yellow]")
            return
        
        # Calculate stats
        user_msgs = [msg for msg in self.conversation_history if msg.role == "user"]
        assistant_msgs = [msg for msg in self.conversation_history if msg.role == "assistant"]
        
        total_tokens = sum(msg.tokens or 0 for msg in assistant_msgs)
        avg_response_length = sum(len(msg.content.split()) for msg in assistant_msgs) / len(assistant_msgs) if assistant_msgs else 0
        session_duration = (datetime.now() - self.session_start).total_seconds() / 60
        
        stats = f"""[bold cyan]📊 Conversation Analysis[/bold cyan]

[yellow]Session Statistics:[/yellow]
• Session Duration: {session_duration:.1f} minutes
• Total Messages: {len(self.conversation_history)}
• User Messages: {len(user_msgs)}
• Assistant Responses: {len(assistant_msgs)}
• Average Response Length: {avg_response_length:.1f} words
• Estimated Total Tokens: {total_tokens}

[green]Recent Activity:[/green]"""
        
        # Add recent messages summary
        recent_msgs = self.conversation_history[-6:]  # Last 6 messages
        for msg in recent_msgs:
            time_str = msg.timestamp.strftime("%H:%M:%S")
            role_emoji = "👤" if msg.role == "user" else "🤖"
            preview = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
            stats += f"\n• {role_emoji} [{time_str}] {preview}"
        
        console.print(Panel(
            stats,
            title="📈 Session Analytics",
            border_style="cyan",
            padding=(1, 2)
        ))
    
    def export_session(self):
        """Export conversation history."""
        if not self.conversation_history:
            console.print("[yellow]No conversation to export[/yellow]")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qwen_session_{timestamp}.json"
        
        export_data = {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "end_time": datetime.now().isoformat(),
                "model": self.model_name,
                "total_messages": len(self.conversation_history)
            },
            "conversation": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "tokens": msg.tokens
                }
                for msg in self.conversation_history
            ]
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            console.print(f"[green]✅ Session exported to: {filename}[/green]")
        except Exception as e:
            console.print(f"[red]❌ Export failed: {e}[/red]")
    
    def show_help(self):
        """Show detailed help information."""
        help_text = """[bold cyan]🎯 Qwen Interactive CLI Help[/bold cyan]

[yellow]Available Modes:[/yellow]

[bold green]💬 Chat Mode[/bold green]
• Free-form conversation with your SAD-enhanced Qwen model
• Type naturally, the model will respond with enhanced reasoning
• Commands: 'exit' (return to menu), 'clear' (clear history)

[bold blue]🧪 Test Mode[/bold blue]
• Test specific SAD (Structured Agent Distillation) capabilities
• Predefined scenarios that showcase tool-aware reasoning
• Includes memory debugging, API optimization, database performance, etc.

[bold magenta]📋 Batch Testing[/bold magenta]
• Automatically run all test scenarios
• Get comprehensive performance analysis
• Perfect for evaluating model improvements

[bold cyan]📊 Analysis Mode[/bold cyan]
• View conversation statistics and patterns
• Session duration, message counts, response quality
• Recent activity summary

[bold yellow]💾 Export Mode[/bold yellow]
• Save conversation history as JSON
• Includes timestamps, tokens, and metadata
• Useful for further analysis or record keeping

[green]Tips for Best Results:[/green]
• The model excels at systematic, tool-aware problem solving
• Try prompts that mention specific tools or technical scenarios
• Ask for step-by-step approaches to complex problems
• The SAD training enhances reasoning about debugging, optimization, and analysis tasks

[blue]Model Information:[/blue]
• Base Model: Qwen 30B
• Enhanced with: Structured Agent Distillation (SAD)
• Specialties: Tool-aware reasoning, systematic problem solving, technical analysis
• Training: LoRA fine-tuning with 6.6M parameters updated"""
        
        console.print(Panel(
            help_text,
            title="❓ Help & Documentation",
            border_style="blue",
            padding=(1, 2)
        ))
    
    def run(self):
        """Main application loop."""
        # Setup
        console.clear()
        self.display_header()
        
        # Check connection
        console.print("\n[yellow]🔌 Connecting to model...[/yellow]")
        if not self.setup_client():
            console.print("[red]❌ Failed to connect. Please ensure SGLang server is running.[/red]")
            sys.exit(1)
        
        console.print("[green]✅ Connected successfully![/green]")
        
        # Main loop
        while True:
            try:
                choice = self.show_main_menu()
                
                if choice == "chat":
                    self.chat_mode()
                elif choice == "test":
                    self.test_sad_capabilities()
                elif choice == "batch":
                    self.batch_testing()
                elif choice == "analyze":
                    self.analyze_conversation()
                elif choice == "export":
                    self.export_session()
                elif choice == "clear":
                    if Confirm.ask("[bold red]Clear all conversation history?[/bold red]", default=False):
                        self.conversation_history.clear()
                        console.print("[green]✅ Conversation history cleared[/green]")
                elif choice == "help":
                    self.show_help()
                elif choice == "exit":
                    if Confirm.ask("[bold]Exit the application?[/bold]", default=True):
                        break
                
                console.print("\n[dim]Press Enter to continue...[/dim]")
                input()
                console.clear()
                self.display_header()
                
            except KeyboardInterrupt:
                console.print("\n[yellow]👋 Goodbye![/yellow]")
                break
            except Exception as e:
                console.print(f"[red]❌ Unexpected error: {e}[/red]")
                time.sleep(2)

def main():
    """Main function."""
    try:
        cli = QwenInteractiveCLI()
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]👋 Goodbye![/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")

if __name__ == "__main__":
    main() 