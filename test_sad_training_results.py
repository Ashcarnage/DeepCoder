#!/usr/bin/env python3
"""
🎯 SAD Training Results Tester
=============================

Beautiful Rich CLI to test and demonstrate the results of SAD training
with before/after comparisons and detailed analysis.
"""

import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.columns import Columns
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import box

from openai import OpenAI

console = Console()

@dataclass
class TestScenario:
    """Test scenario for SAD training evaluation."""
    name: str
    prompt: str
    expected_tools: List[str]
    description: str

class SADTestingSuite:
    """Beautiful testing suite for SAD training results."""
    
    def __init__(self):
        self.sglang_client = None
        self.sglang_port = 30000
        self.sglang_host = "localhost"
        
        # Test scenarios targeting tool usage
        self.test_scenarios = [
            TestScenario(
                name="Memory Leak Debug",
                prompt="Debug a Python memory leak consuming 8GB RAM. Available tools: system_monitor, terminal_command, code_analyzer. Use systematic approach.",
                expected_tools=["system_monitor", "terminal_command", "code_analyzer"],
                description="Complex debugging requiring multiple tool coordination"
            ),
            TestScenario(
                name="API Optimization", 
                prompt="Optimize slow API endpoint with 2000ms response time. Available tools: system_monitor, api_client, code_analyzer. Target: <200ms.",
                expected_tools=["system_monitor", "api_client", "code_analyzer"],
                description="Performance optimization with specific metrics"
            ),
            TestScenario(
                name="Database Performance",
                prompt="Analyze and fix database queries causing 10x performance degradation. Available tools: database_query, system_monitor, code_analyzer.",
                expected_tools=["database_query", "system_monitor", "code_analyzer"],
                description="Database troubleshooting requiring analysis tools"
            ),
            TestScenario(
                name="Microservices Debug",
                prompt="Investigate intermittent 500 errors in microservices. Available tools: terminal_command, system_monitor, api_client, code_analyzer.",
                expected_tools=["terminal_command", "system_monitor", "api_client"],
                description="Distributed systems debugging scenario"
            ),
            TestScenario(
                name="CI/CD Pipeline",
                prompt="Set up CI/CD pipeline with automated testing. Available tools: git_operations, terminal_command, file_operations, code_analyzer.",
                expected_tools=["git_operations", "terminal_command", "file_operations"],
                description="DevOps automation and pipeline setup"
            )
        ]
        
    def wait_for_server(self, timeout: int = 120) -> bool:
        """Wait for SGLang server to be ready with beautiful progress."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🚀 Waiting for SGLang server to be ready...", total=timeout)
            
            for i in range(timeout):
                try:
                    response = requests.get(f"http://{self.sglang_host}:{self.sglang_port}/health", timeout=2)
                    if response.status_code == 200:
                        progress.update(task, completed=timeout)
                        console.print("[green]✅ SGLang server is ready![/green]")
                        return True
                except:
                    pass
                
                progress.update(task, advance=1)
                time.sleep(1)
            
            console.print("[red]❌ SGLang server not ready[/red]")
            return False
    
    def setup_client(self) -> bool:
        """Setup OpenAI client for SGLang."""
        try:
            self.sglang_client = OpenAI(
                base_url=f"http://{self.sglang_host}:{self.sglang_port}/v1", 
                api_key="EMPTY"
            )
            return True
        except Exception as e:
            console.print(f"[red]❌ Failed to setup client: {e}[/red]")
            return False
    
    def analyze_response(self, response: str, expected_tools: List[str]) -> Dict:
        """Analyze response quality and tool usage."""
        analysis = {
            "response_length": len(response.split()),
            "tool_mentions": 0,
            "structured_format": False,
            "reasoning_present": False,
            "tool_calls_found": [],
            "quality_score": 0.0
        }
        
        # Count tool mentions
        for tool in expected_tools:
            if tool.lower() in response.lower():
                analysis["tool_mentions"] += 1
                analysis["tool_calls_found"].append(tool)
        
        # Check for structured format
        if any(marker in response for marker in ["TOOL_CALL", "TOOL_OUTPUT", "ANALYSIS"]):
            analysis["structured_format"] = True
        
        # Check for reasoning
        reasoning_words = ["reasoning", "because", "analysis", "systematic", "approach", "step"]
        if any(word in response.lower() for word in reasoning_words):
            analysis["reasoning_present"] = True
        
        # Calculate quality score
        score = 0.0
        score += (analysis["tool_mentions"] / len(expected_tools)) * 40  # 40% for tool usage
        score += 20 if analysis["structured_format"] else 0  # 20% for structure
        score += 20 if analysis["reasoning_present"] else 0  # 20% for reasoning
        score += min(analysis["response_length"] / 100, 1.0) * 20  # 20% for completeness
        
        analysis["quality_score"] = min(score, 100.0)
        
        return analysis
    
    def test_scenario(self, scenario: TestScenario) -> Dict:
        """Test a single scenario and return results."""
        try:
            response = self.sglang_client.chat.completions.create(
                model="qwen3-30b-a3b",
                messages=[{"role": "user", "content": scenario.prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            # Handle DeepSeek-R1 format where content might be in reasoning_content
            content = response.choices[0].message.content
            if content is None:
                # Try reasoning_content for DeepSeek-R1 format
                content = getattr(response.choices[0].message, 'reasoning_content', '') or ""
            
            analysis = self.analyze_response(content, scenario.expected_tools)
            
            return {
                "scenario": scenario,
                "response": content,
                "analysis": analysis,
                "success": True
            }
            
        except Exception as e:
            return {
                "scenario": scenario,
                "response": f"Error: {str(e)}",
                "analysis": {"quality_score": 0.0, "tool_mentions": 0},
                "success": False
            }
    
    def create_results_table(self, results: List[Dict]) -> Table:
        """Create beautiful results table."""
        table = Table(title="🎯 SAD Training Results Analysis", box=box.ROUNDED)
        
        table.add_column("Scenario", style="cyan", width=20)
        table.add_column("Quality Score", justify="center", style="yellow", width=12)
        table.add_column("Tools Used", justify="center", style="green", width=10)
        table.add_column("Structured", justify="center", style="blue", width=10)
        table.add_column("Reasoning", justify="center", style="magenta", width=10)
        table.add_column("Response Length", justify="center", style="white", width=15)
        
        total_quality = 0
        total_tools = 0
        structured_count = 0
        reasoning_count = 0
        
        for result in results:
            if result["success"]:
                analysis = result["analysis"]
                scenario = result["scenario"]
                
                # Quality score with color coding
                score = analysis["quality_score"]
                if score >= 80:
                    score_style = "[green]"
                elif score >= 60:
                    score_style = "[yellow]"
                else:
                    score_style = "[red]"
                
                # Tool usage
                tools_used = f"{analysis['tool_mentions']}/{len(scenario.expected_tools)}"
                
                # Boolean indicators
                structured = "✅" if analysis["structured_format"] else "❌"
                reasoning = "✅" if analysis["reasoning_present"] else "❌"
                
                table.add_row(
                    scenario.name,
                    f"{score_style}{score:.1f}%[/]",
                    tools_used,
                    structured,
                    reasoning,
                    f"{analysis['response_length']} words"
                )
                
                total_quality += score
                total_tools += analysis["tool_mentions"]
                if analysis["structured_format"]:
                    structured_count += 1
                if analysis["reasoning_present"]:
                    reasoning_count += 1
        
        # Add summary row
        avg_quality = total_quality / len(results) if results else 0
        table.add_section()
        table.add_row(
            "[bold]AVERAGE[/bold]",
            f"[bold]{avg_quality:.1f}%[/bold]",
            f"[bold]{total_tools}[/bold]",
            f"[bold]{structured_count}/{len(results)}[/bold]",
            f"[bold]{reasoning_count}/{len(results)}[/bold]",
            "[bold]Summary[/bold]"
        )
        
        return table
    
    def show_detailed_response(self, result: Dict) -> Panel:
        """Show detailed response for a scenario."""
        scenario = result["scenario"]
        response = result["response"]
        analysis = result["analysis"]
        
        # Truncate response for display
        display_response = response[:500] + "..." if len(response) > 500 else response
        
        content = f"""[bold cyan]{scenario.description}[/bold cyan]

[yellow]Prompt:[/yellow] {scenario.prompt}

[green]Response:[/green] {display_response}

[blue]Analysis:[/blue]
• Quality Score: {analysis['quality_score']:.1f}%
• Tool Mentions: {analysis['tool_mentions']}
• Structured Format: {'✅' if analysis['structured_format'] else '❌'}
• Reasoning Present: {'✅' if analysis['reasoning_present'] else '❌'}
• Tools Found: {', '.join(analysis.get('tool_calls_found', []))}"""
        
        return Panel(content, title=f"📋 {scenario.name}", border_style="cyan")
    
    def run_complete_test_suite(self):
        """Run the complete testing suite with beautiful output."""
        # Header
        console.print(Panel.fit(
            "[bold blue]🎯 SAD Training Results Testing Suite[/bold blue]\n"
            "[yellow]Testing Qwen 30B with LoRA-trained SAD capabilities[/yellow]",
            border_style="blue"
        ))
        
        # Step 1: Wait for server
        console.print("\n[bold]Step 1: Server Readiness Check[/bold]")
        if not self.wait_for_server():
            console.print("[red]❌ Cannot proceed without server[/red]")
            return
        
        # Step 2: Setup client
        console.print("\n[bold]Step 2: Client Setup[/bold]")
        if not self.setup_client():
            console.print("[red]❌ Cannot setup client[/red]")
            return
        console.print("[green]✅ Client setup successful[/green]")
        
        # Step 3: Run tests
        console.print("\n[bold]Step 3: Running Test Scenarios[/bold]")
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("🧪 Running test scenarios...", total=len(self.test_scenarios))
            
            for scenario in self.test_scenarios:
                progress.update(task, description=f"Testing: {scenario.name}")
                result = self.test_scenario(scenario)
                results.append(result)
                progress.advance(task)
                time.sleep(0.5)  # Brief pause between tests
        
        # Step 4: Show results
        console.print("\n[bold]Step 4: Results Analysis[/bold]")
        
        # Summary table
        table = self.create_results_table(results)
        console.print(table)
        
        # Calculate overall metrics
        successful_tests = [r for r in results if r["success"]]
        if successful_tests:
            avg_quality = sum(r["analysis"]["quality_score"] for r in successful_tests) / len(successful_tests)
            total_tool_usage = sum(r["analysis"]["tool_mentions"] for r in successful_tests)
            structured_responses = sum(1 for r in successful_tests if r["analysis"]["structured_format"])
            reasoning_responses = sum(1 for r in successful_tests if r["analysis"]["reasoning_present"])
            
            # Overall summary
            summary_panel = Panel(
                f"""[bold green]🏆 Overall Performance Summary[/bold green]

[yellow]Average Quality Score:[/yellow] {avg_quality:.1f}%
[yellow]Total Tool Mentions:[/yellow] {total_tool_usage}
[yellow]Structured Responses:[/yellow] {structured_responses}/{len(successful_tests)} ({structured_responses/len(successful_tests)*100:.1f}%)
[yellow]Reasoning Present:[/yellow] {reasoning_responses}/{len(successful_tests)} ({reasoning_responses/len(successful_tests)*100:.1f}%)

[cyan]Test Results:[/cyan] {len(successful_tests)}/{len(results)} scenarios completed successfully

[bold]{"🎉 Excellent Performance!" if avg_quality >= 80 else "🚀 Good Progress!" if avg_quality >= 60 else "📈 Room for Improvement"}[/bold]""",
                title="📊 SAD Training Assessment",
                border_style="green" if avg_quality >= 80 else "yellow" if avg_quality >= 60 else "red"
            )
            console.print(f"\n{summary_panel}")
        
        # Detailed responses
        console.print("\n[bold]Step 5: Detailed Response Analysis[/bold]")
        for i, result in enumerate(results[:3]):  # Show first 3 detailed
            if result["success"]:
                console.print(f"\n{self.show_detailed_response(result)}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"/workspace/persistent/sad_test_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "summary": {
                    "avg_quality": avg_quality if successful_tests else 0,
                    "total_tests": len(results),
                    "successful_tests": len(successful_tests),
                    "tool_usage_total": total_tool_usage if successful_tests else 0,
                    "structured_responses": structured_responses if successful_tests else 0,
                    "reasoning_responses": reasoning_responses if successful_tests else 0
                },
                "detailed_results": results
            }, f, indent=2, default=str)
        
        console.print(f"\n[green]💾 Results saved to: {results_file}[/green]")
        
        # Final message
        console.print(Panel.fit(
            "[bold green]🎯 SAD Training Test Complete![/bold green]\n"
            "[yellow]The model has been tested for tool usage, reasoning, and structured responses.[/yellow]",
            border_style="green"
        ))

def main():
    """Main function to run the testing suite."""
    tester = SADTestingSuite()
    tester.run_complete_test_suite()

if __name__ == "__main__":
    main() 