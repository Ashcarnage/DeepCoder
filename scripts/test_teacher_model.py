#!/usr/bin/env python3
"""
Test script for teacher model integration
Tests the Groq client, agent framework, and trajectory generation.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
logging.basicConfig(level=logging.INFO)

def test_imports():
    """Test that all modules can be imported"""
    console.print(Panel("[bold blue]Testing Module Imports[/bold blue]"))
    
    try:
        from data_generation import (
            create_groq_client,
            create_agent_framework, 
            create_trajectory_generator,
            ProblemLoader,
            GenerationConfig
        )
        console.print("✓ All modules imported successfully")
        return True
    except Exception as e:
        console.print(f"✗ Import error: {e}")
        return False

def test_groq_client():
    """Test Groq API client"""
    console.print(Panel("[bold blue]Testing Groq Client[/bold blue]"))
    
    try:
        from data_generation import create_groq_client
        
        # Test client creation
        client = create_groq_client()
        console.print("✓ Groq client created successfully")
        
        # Test model info
        model_info = client.get_model_info()
        console.print(f"✓ Model: {model_info['model']}")
        console.print(f"✓ API configured: {model_info['api_configured']}")
        
        if not model_info['api_configured']:
            console.print("[yellow]⚠️ API key not configured - skipping connection test[/yellow]")
            return False
        
        # Test connection
        if client.test_connection():
            console.print("✓ API connection successful")
            
            # Test simple generation
            test_response = client.generate_response([
                {"role": "user", "content": "Say 'Hello from DeepSeek R1!'"}
            ], max_tokens=50)
            
            console.print(f"✓ Test response: {test_response['content'][:100]}...")
            console.print(f"✓ Tokens used: {test_response['usage']['total_tokens']}")
            
            return True
        else:
            console.print("✗ API connection failed")
            return False
            
    except Exception as e:
        console.print(f"✗ Error testing Groq client: {e}")
        return False

def test_problem_loader():
    """Test problem loader"""
    console.print(Panel("[bold blue]Testing Problem Loader[/bold blue]"))
    
    try:
        from data_generation import ProblemLoader
        
        # Create problem loader
        loader = ProblemLoader()
        console.print("✓ Problem loader created successfully")
        
        # Load problems (will create sample problems if none exist)
        problems = loader.load_problems("data/problems/coding_problems.jsonl")
        console.print(f"✓ Loaded {len(problems)} problems")
        
        # Display stats
        loader.display_stats()
        
        # Test sampling
        sample = loader.sample_problems(3, difficulty="easy")
        console.print(f"✓ Sampled {len(sample)} easy problems")
        
        return True
        
    except Exception as e:
        console.print(f"✗ Error testing problem loader: {e}")
        return False

def test_agent_framework():
    """Test agent framework"""
    console.print(Panel("[bold blue]Testing Agent Framework[/bold blue]"))
    
    try:
        from data_generation import create_groq_client, create_agent_framework
        
        # Create client (mock if no API key)
        try:
            client = create_groq_client()
            has_api = client.get_model_info()['api_configured']
        except:
            console.print("[yellow]⚠️ No API key - using mock client[/yellow]")
            has_api = False
        
        if not has_api:
            console.print("[yellow]Skipping agent framework test (requires API key)[/yellow]")
            return False
        
        # Create agent framework
        agent = create_agent_framework(client)
        console.print("✓ Agent framework created successfully")
        
        # Test simple problem
        test_problem = "Write a Python function to add two numbers."
        
        console.print(f"✓ Testing with problem: {test_problem}")
        
        # Generate trajectory (with small max_steps for testing)
        trajectory = agent.generate_trajectory(test_problem, max_steps=3)
        
        console.print(f"✓ Generated trajectory with {len(trajectory.steps)} steps")
        console.print(f"✓ Success: {trajectory.success}")
        console.print(f"✓ Tokens used: {trajectory.total_tokens}")
        console.print(f"✓ Execution time: {trajectory.execution_time:.2f}s")
        
        # Display trajectory
        agent.display_trajectory(trajectory)
        
        return True
        
    except Exception as e:
        console.print(f"✗ Error testing agent framework: {e}")
        return False

def test_trajectory_generator():
    """Test trajectory generator"""
    console.print(Panel("[bold blue]Testing Trajectory Generator[/bold blue]"))
    
    try:
        from data_generation import create_groq_client, create_trajectory_generator, GenerationConfig
        
        # Check if API key is available
        try:
            client = create_groq_client()
            has_api = client.get_model_info()['api_configured']
        except:
            has_api = False
        
        if not has_api:
            console.print("[yellow]Skipping trajectory generator test (requires API key)[/yellow]")
            return False
        
        # Create small test config
        test_config = GenerationConfig(
            num_trajectories=2,
            batch_size=1,
            max_workers=1,
            output_dir="data/test_trajectories",
            max_steps_per_trajectory=3
        )
        
        # Create generator
        generator = create_trajectory_generator(test_config)
        console.print("✓ Trajectory generator created successfully")
        
        # Generate trajectories
        console.print("✓ Starting test trajectory generation...")
        stats = generator.generate_trajectories()
        
        console.print(f"✓ Generated {stats.total_generated} trajectories")
        console.print(f"✓ Success rate: {stats.success_rate:.2%}")
        console.print(f"✓ Total tokens: {stats.total_tokens}")
        
        return True
        
    except Exception as e:
        console.print(f"✗ Error testing trajectory generator: {e}")
        return False

def main():
    """Run all tests"""
    console.print(Panel(
        "[bold green]DeepCoder Teacher Model Integration Tests[/bold green]",
        title="Test Suite"
    ))
    
    # Test results
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['problem_loader'] = test_problem_loader()
    results['groq_client'] = test_groq_client()
    results['agent_framework'] = test_agent_framework()
    results['trajectory_generator'] = test_trajectory_generator()
    
    # Summary table
    table = Table(title="Test Results Summary")
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
            f"[bold green]All {total_tests} tests passed! 🎉[/bold green]",
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
        "1. Set GROQ_API_KEY in .env file if not already done\n"
        "2. Run full trajectory generation with: python -m src.data_generation.trajectory_generator\n"
        "3. Proceed to Phase 2: Data preprocessing and SAD implementation",
        title="What's Next"
    ))

if __name__ == "__main__":
    main() 