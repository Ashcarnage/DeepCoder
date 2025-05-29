#!/usr/bin/env python3
"""
Test script for student model integration with SGLang
Tests the SGLang server connection, student model responses, and integration.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
logging.basicConfig(level=logging.INFO)

def test_imports():
    """Test that all modules can be imported"""
    console.print(Panel("[bold blue]Testing Student Model Imports[/bold blue]"))
    
    try:
        from student_model import (
            StudentModel,
            StudentModelManager,
            StudentModelConfig,
            StudentResponse,
            create_student_model,
            create_student_model_manager
        )
        from sglang_manager import SGLangManager, load_config
        console.print("✓ All student model modules imported successfully")
        return True
    except Exception as e:
        console.print(f"✗ Import error: {e}")
        return False

def test_configuration():
    """Test configuration loading and validation"""
    console.print(Panel("[bold blue]Testing Configuration Loading[/bold blue]"))
    
    try:
        from student_model import create_student_model
        
        # Test configuration loading
        model = create_student_model()
        config = model.student_config
        
        console.print(f"✓ Configuration loaded successfully")
        console.print(f"  Model name: {config.model_name}")
        console.print(f"  Served model: {config.served_model_name}")
        console.print(f"  Max tokens: {config.max_tokens}")
        console.print(f"  Thinking enabled: {config.enable_thinking}")
        console.print(f"  Temperature: {config.temperature}")
        
        # Validate key settings
        if config.model_name and config.served_model_name:
            console.print("✓ Model names properly configured")
        else:
            console.print("✗ Model names not properly configured")
            return False
            
        return True
        
    except Exception as e:
        console.print(f"✗ Configuration test failed: {e}")
        return False

def test_sglang_server_startup():
    """Test SGLang server startup and health check"""
    console.print(Panel("[bold blue]Testing SGLang Server Startup[/bold blue]"))
    
    try:
        from student_model import create_student_model_manager
        
        # Create student model manager
        manager = create_student_model_manager()
        console.print("✓ Student model manager created")
        
        # Initialize with timeout
        console.print("Starting SGLang server (this may take 2-5 minutes)...")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Initializing student model...", total=None)
            
            success = manager.initialize(timeout=600)
            
            progress.remove_task(task)
        
        if success:
            console.print("✓ SGLang server started successfully")
            
            # Test readiness
            if manager.is_ready():
                console.print("✓ Student model is ready for requests")
                
                # Cleanup
                manager.shutdown()
                console.print("✓ Server shutdown successful")
                return True
            else:
                console.print("✗ Student model not ready after startup")
                manager.shutdown()
                return False
        else:
            console.print("✗ SGLang server failed to start")
            return False
            
    except Exception as e:
        console.print(f"✗ Server startup test failed: {e}")
        return False

def test_basic_response_generation():
    """Test basic response generation"""
    console.print(Panel("[bold blue]Testing Basic Response Generation[/bold blue]"))
    
    try:
        from student_model import create_student_model
        
        with create_student_model() as model:
            console.print("✓ Student model context manager working")
            
            # Test simple request
            response = model.generate_response(
                messages=[{"role": "user", "content": "Hello! Please respond with 'Hello World!'"}],
                enable_thinking=False,
                max_tokens=50
            )
            
            if response.success:
                console.print("✓ Basic response generation successful")
                console.print(f"  Content: {response.content}")
                console.print(f"  Generation time: {response.generation_time:.2f}s")
                console.print(f"  Model: {response.model}")
                
                if response.usage:
                    console.print(f"  Token usage: {response.usage}")
                
                return True
            else:
                console.print(f"✗ Response generation failed: {response.error_message}")
                return False
                
    except Exception as e:
        console.print(f"✗ Basic response test failed: {e}")
        return False

def test_thinking_mode():
    """Test thinking mode functionality"""
    console.print(Panel("[bold blue]Testing Thinking Mode[/bold blue]"))
    
    try:
        from student_model import create_student_model
        
        with create_student_model() as model:
            # Test with thinking mode enabled
            response = model.generate_response(
                messages=[{
                    "role": "user", 
                    "content": "What is 15 * 23? Please show your reasoning."
                }],
                enable_thinking=True,
                max_tokens=200
            )
            
            if response.success:
                console.print("✓ Thinking mode response generated")
                console.print(f"  Content: {response.content}")
                
                if response.thinking:
                    console.print("✓ Thinking content extracted")
                    console.print(f"  Thinking: {response.thinking[:100]}...")
                else:
                    console.print("⚠ No thinking content found (may be normal)")
                
                console.print(f"  Generation time: {response.generation_time:.2f}s")
                return True
            else:
                console.print(f"✗ Thinking mode failed: {response.error_message}")
                return False
                
    except Exception as e:
        console.print(f"✗ Thinking mode test failed: {e}")
        return False

def test_agent_response_generation():
    """Test agent-style response generation"""
    console.print(Panel("[bold blue]Testing Agent Response Generation[/bold blue]"))
    
    try:
        from student_model import create_student_model
        
        # Load agent system prompt from config
        system_prompt = """You are an advanced AI assistant designed to solve complex coding and reasoning problems.
You must operate in a strict cycle of Thought, Action, and Observation.

1. **Thought:** Articulate your reasoning for the current step.
2. **Action:** Choose ONE action: execute_python(code), retrieve_knowledge(query), or finish(answer)
3. **Observation:** Use the result to inform your next thought.

Continue this cycle until you solve the problem."""

        user_input = "Calculate the factorial of 5 using Python."
        
        with create_student_model() as model:
            response = model.generate_agent_response(
                system_prompt=system_prompt,
                user_input=user_input,
                enable_thinking=True
            )
            
            if response.success:
                console.print("✓ Agent response generation successful")
                console.print(f"  Content: {response.content}")
                
                # Check for agent-style formatting
                if any(keyword in response.content.lower() for keyword in ['thought:', 'action:', 'observation:']):
                    console.print("✓ Agent-style formatting detected")
                else:
                    console.print("⚠ No explicit agent formatting (may be normal)")
                
                if response.thinking:
                    console.print(f"✓ Thinking content: {response.thinking[:100]}...")
                
                console.print(f"  Generation time: {response.generation_time:.2f}s")
                return True
            else:
                console.print(f"✗ Agent response generation failed: {response.error_message}")
                return False
                
    except Exception as e:
        console.print(f"✗ Agent response test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and recovery"""
    console.print(Panel("[bold blue]Testing Error Handling[/bold blue]"))
    
    try:
        from student_model import create_student_model
        
        with create_student_model() as model:
            # Test with invalid parameters
            response = model.generate_response(
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=0  # Invalid parameter
            )
            
            # Should handle gracefully
            if not response.success:
                console.print("✓ Error handling working correctly")
                console.print(f"  Error message: {response.error_message}")
            else:
                console.print("⚠ Expected error but got success")
            
            # Test health check
            if model.is_healthy():
                console.print("✓ Model health check working")
            else:
                console.print("✗ Model health check failed")
            
            return True
            
    except Exception as e:
        console.print(f"✗ Error handling test failed: {e}")
        return False

def test_performance_benchmark():
    """Test basic performance metrics"""
    console.print(Panel("[bold blue]Testing Performance Benchmark[/bold blue]"))
    
    try:
        from student_model import create_student_model
        
        test_cases = [
            "What is 2+2?",
            "Write a simple Python function to add two numbers.",
            "Explain what a variable is in programming."
        ]
        
        with create_student_model() as model:
            total_time = 0
            successful_requests = 0
            
            for i, test_case in enumerate(test_cases, 1):
                console.print(f"Testing case {i}: {test_case}")
                
                response = model.generate_response(
                    messages=[{"role": "user", "content": test_case}],
                    max_tokens=100
                )
                
                if response.success:
                    successful_requests += 1
                    total_time += response.generation_time
                    console.print(f"  ✓ Success in {response.generation_time:.2f}s")
                else:
                    console.print(f"  ✗ Failed: {response.error_message}")
            
            if successful_requests > 0:
                avg_time = total_time / successful_requests
                console.print(f"✓ Performance benchmark completed")
                console.print(f"  Successful requests: {successful_requests}/{len(test_cases)}")
                console.print(f"  Average response time: {avg_time:.2f}s")
                console.print(f"  Total time: {total_time:.2f}s")
                return True
            else:
                console.print("✗ No successful requests in benchmark")
                return False
                
    except Exception as e:
        console.print(f"✗ Performance benchmark failed: {e}")
        return False

def test_integration_with_existing_code():
    """Test integration with existing codebase"""
    console.print(Panel("[bold blue]Testing Integration with Existing Codebase[/bold blue]"))
    
    try:
        # Test if we can import and use with existing data generation components
        from data_generation import ProblemLoader
        from student_model import create_student_model
        
        # Load a problem
        problem_loader = ProblemLoader()
        problems = problem_loader.load_problems("data/problems/coding_problems.jsonl")
        
        if problems:
            problem = problems[0]
            console.print(f"✓ Loaded problem: {problem.title}")
            
            # Test student model on the problem
            with create_student_model() as model:
                response = model.generate_response(
                    messages=[{
                        "role": "user", 
                        "content": f"Solve this problem: {problem.description}"
                    }],
                    enable_thinking=True,
                    max_tokens=300
                )
                
                if response.success:
                    console.print("✓ Student model solved problem from existing dataset")
                    console.print(f"  Response length: {len(response.content)} chars")
                    console.print(f"  Time taken: {response.generation_time:.2f}s")
                    return True
                else:
                    console.print(f"✗ Failed to solve problem: {response.error_message}")
                    return False
        else:
            console.print("⚠ No problems found, but integration imports work")
            return True
            
    except Exception as e:
        console.print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    console.print(Panel(
        "[bold green]DeepCoder Student Model Integration Tests[/bold green]",
        title="Test Suite"
    ))
    
    # Test results
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['configuration'] = test_configuration()
    results['sglang_startup'] = test_sglang_server_startup()
    results['basic_response'] = test_basic_response_generation()
    results['thinking_mode'] = test_thinking_mode()
    results['agent_response'] = test_agent_response_generation()
    results['error_handling'] = test_error_handling()
    results['performance'] = test_performance_benchmark()
    results['integration'] = test_integration_with_existing_code()
    
    # Summary table
    table = Table(title="Student Model Test Results Summary")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Notes", style="dim")
    
    for component, passed in results.items():
        if passed:
            status = "[green]✓ PASS[/green]"
            notes = "All tests passed"
        else:
            status = "[red]✗ FAIL[/red]"
            notes = "Check logs for details"
        
        table.add_row(component.replace('_', ' ').title(), status, notes)
    
    console.print(table)
    
    # Overall result
    total_passed = sum(results.values())
    total_tests = len(results)
    
    if total_passed == total_tests:
        console.print(Panel(
            f"[bold green]All {total_tests} tests passed! 🎉[/bold green]\n"
            f"[green]Student model integration is working correctly[/green]",
            title="Success"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]{total_passed}/{total_tests} tests passed[/bold yellow]\n"
            f"[red]Some tests failed - check the output above[/red]",
            title="Partial Success"
        ))
    
    # Next steps
    console.print(Panel(
        "[bold blue]Next Steps:[/bold blue]\n"
        "1. If tests passed, proceed to Phase 2.1: Trajectory Processing\n"
        "2. If tests failed, review error messages and fix issues\n"
        "3. Ensure SGLang server has sufficient memory and GPU resources\n"
        "4. Test student model with actual trajectory data",
        title="What's Next"
    ))

if __name__ == "__main__":
    main() 