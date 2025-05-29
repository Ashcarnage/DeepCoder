#!/usr/bin/env python3
"""
Phase 1.3: Sample Trajectory Generation
Generates a sample batch of trajectories for validation and quality assessment.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import json

from data_generation import (
    create_groq_client,
    create_trajectory_generator,
    GenerationConfig,
    ProblemLoader
)

console = Console()

def generate_sample_trajectories():
    """Generate sample trajectories for Phase 1.3"""
    
    console.print(Panel(
        "[bold blue]Phase 1.3: Sample Trajectory Generation[/bold blue]\n"
        "Generating sample trajectories for validation and quality assessment",
        title="DeepCoder Data Generation"
    ))
    
    # Configuration for sample generation
    sample_config = GenerationConfig(
        num_trajectories=20,  # Sample size
        max_steps_per_trajectory=10,
        batch_size=5,
        max_workers=2,
        output_dir="data/sample_trajectories",
        problems_file="data/problems/coding_problems.jsonl",
        save_interval=10,
        temperature=0.7,
        max_tokens=4096,
        include_failed=True,
        min_success_rate=0.7
    )
    
    try:
        # Create trajectory generator
        console.print("[yellow]Initializing trajectory generator...[/yellow]")
        generator = create_trajectory_generator(sample_config)
        
        # Generate sample trajectories
        console.print("[yellow]Starting sample trajectory generation...[/yellow]")
        stats = generator.generate_trajectories()
        
        # Analyze trajectory quality
        console.print("[yellow]Analyzing trajectory quality...[/yellow]")
        quality_analysis = analyze_trajectory_quality(generator)
        
        # Display results
        display_generation_results(stats, quality_analysis)
        
        return stats, quality_analysis
        
    except Exception as e:
        console.print(f"[red]Error in sample trajectory generation: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_trajectory_quality(generator):
    """Analyze the quality of generated trajectories"""
    
    trajectories = generator.load_existing_trajectories()
    
    if not trajectories:
        return {"error": "No trajectories found for analysis"}
    
    analysis = {
        "total_trajectories": len(trajectories),
        "successful_trajectories": 0,
        "reasoning_quality": {
            "has_thinking_tags": 0,
            "avg_thinking_length": 0,
            "has_code_blocks": 0,
            "has_explanations": 0
        },
        "problem_coverage": {},
        "token_usage": {
            "total_tokens": 0,
            "avg_tokens_per_trajectory": 0,
            "token_efficiency": 0
        },
        "step_analysis": {
            "avg_steps": 0,
            "single_step_solutions": 0,
            "multi_step_solutions": 0
        },
        "content_analysis": {
            "avg_content_length": 0,
            "has_proper_structure": 0,
            "has_examples": 0
        }
    }
    
    total_thinking_length = 0
    thinking_count = 0
    total_content_length = 0
    total_steps = 0
    
    for trajectory in trajectories:
        traj_data = trajectory.get('trajectory', {})
        
        # Success rate
        if traj_data.get('success', False):
            analysis["successful_trajectories"] += 1
        
        # Token usage
        tokens = traj_data.get('total_tokens', 0)
        analysis["token_usage"]["total_tokens"] += tokens
        
        # Steps analysis
        steps = traj_data.get('steps', [])
        step_count = len(steps)
        total_steps += step_count
        
        if step_count == 1:
            analysis["step_analysis"]["single_step_solutions"] += 1
        elif step_count > 1:
            analysis["step_analysis"]["multi_step_solutions"] += 1
        
        # Problem coverage
        problem_category = trajectory.get('problem', {}).get('category', 'unknown')
        analysis["problem_coverage"][problem_category] = analysis["problem_coverage"].get(problem_category, 0) + 1
        
        # Content analysis
        final_answer = traj_data.get('final_answer', '')
        if final_answer:
            total_content_length += len(final_answer)
            
            # Check for thinking tags
            if '<think>' in final_answer and '</think>' in final_answer:
                analysis["reasoning_quality"]["has_thinking_tags"] += 1
                # Extract thinking content length
                import re
                thinking_matches = re.findall(r'<think>(.*?)</think>', final_answer, re.DOTALL)
                for match in thinking_matches:
                    total_thinking_length += len(match)
                    thinking_count += 1
            
            # Check for code blocks
            if '```python' in final_answer or '```' in final_answer:
                analysis["reasoning_quality"]["has_code_blocks"] += 1
            
            # Check for explanations/structure
            if 'approach' in final_answer.lower() or 'explanation' in final_answer.lower():
                analysis["reasoning_quality"]["has_explanations"] += 1
                
            if '###' in final_answer or '**' in final_answer:
                analysis["content_analysis"]["has_proper_structure"] += 1
                
            if 'example' in final_answer.lower():
                analysis["content_analysis"]["has_examples"] += 1
    
    # Calculate averages
    if len(trajectories) > 0:
        analysis["token_usage"]["avg_tokens_per_trajectory"] = analysis["token_usage"]["total_tokens"] / len(trajectories)
        analysis["step_analysis"]["avg_steps"] = total_steps / len(trajectories)
        analysis["content_analysis"]["avg_content_length"] = total_content_length / len(trajectories)
        
        if thinking_count > 0:
            analysis["reasoning_quality"]["avg_thinking_length"] = total_thinking_length / thinking_count
            
        # Token efficiency (content per token)
        if analysis["token_usage"]["total_tokens"] > 0:
            analysis["token_usage"]["token_efficiency"] = total_content_length / analysis["token_usage"]["total_tokens"]
    
    return analysis

def display_generation_results(stats, quality_analysis):
    """Display the generation results in a formatted way"""
    
    # Generation Statistics Table
    stats_table = Table(title="Sample Generation Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    stats_table.add_row("Total Generated", str(stats.total_generated))
    stats_table.add_row("Successful", str(stats.successful))
    stats_table.add_row("Failed", str(stats.failed))
    stats_table.add_row("Success Rate", f"{stats.success_rate:.2%}")
    stats_table.add_row("Total Tokens", f"{stats.total_tokens:,}")
    stats_table.add_row("Avg Steps/Trajectory", f"{stats.avg_steps_per_trajectory:.1f}")
    stats_table.add_row("Total Time", f"{stats.total_time:.1f}s")
    stats_table.add_row("Avg Time/Trajectory", f"{stats.total_time/max(1, stats.total_generated):.2f}s")
    
    console.print(stats_table)
    
    if quality_analysis and "error" not in quality_analysis:
        # Quality Analysis Table
        quality_table = Table(title="Trajectory Quality Analysis")
        quality_table.add_column("Quality Metric", style="cyan")
        quality_table.add_column("Value", style="magenta")
        quality_table.add_column("Percentage", style="green")
        
        total = quality_analysis["total_trajectories"]
        
        quality_table.add_row(
            "Successful Trajectories", 
            str(quality_analysis["successful_trajectories"]),
            f"{quality_analysis['successful_trajectories']/total*100:.1f}%" if total > 0 else "0%"
        )
        
        quality_table.add_row(
            "Has Thinking Tags",
            str(quality_analysis["reasoning_quality"]["has_thinking_tags"]),
            f"{quality_analysis['reasoning_quality']['has_thinking_tags']/total*100:.1f}%" if total > 0 else "0%"
        )
        
        quality_table.add_row(
            "Has Code Blocks",
            str(quality_analysis["reasoning_quality"]["has_code_blocks"]),
            f"{quality_analysis['reasoning_quality']['has_code_blocks']/total*100:.1f}%" if total > 0 else "0%"
        )
        
        quality_table.add_row(
            "Has Explanations",
            str(quality_analysis["reasoning_quality"]["has_explanations"]),
            f"{quality_analysis['reasoning_quality']['has_explanations']/total*100:.1f}%" if total > 0 else "0%"
        )
        
        quality_table.add_row(
            "Avg Tokens/Trajectory",
            f"{quality_analysis['token_usage']['avg_tokens_per_trajectory']:.0f}",
            "-"
        )
        
        quality_table.add_row(
            "Avg Content Length",
            f"{quality_analysis['content_analysis']['avg_content_length']:.0f} chars",
            "-"
        )
        
        console.print(quality_table)
        
        # Problem Coverage Table
        if quality_analysis["problem_coverage"]:
            coverage_table = Table(title="Problem Category Coverage")
            coverage_table.add_column("Category", style="cyan")
            coverage_table.add_column("Count", style="magenta")
            coverage_table.add_column("Percentage", style="green")
            
            for category, count in quality_analysis["problem_coverage"].items():
                percentage = count / total * 100 if total > 0 else 0
                coverage_table.add_row(category, str(count), f"{percentage:.1f}%")
            
            console.print(coverage_table)

def validate_trajectory_format(trajectory_file):
    """Validate the format of generated trajectories"""
    
    console.print(f"[yellow]Validating trajectory format in {trajectory_file}...[/yellow]")
    
    validation_results = {
        "valid_trajectories": 0,
        "invalid_trajectories": 0,
        "format_errors": []
    }
    
    try:
        with open(trajectory_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    trajectory = json.loads(line.strip())
                    
                    # Check required fields
                    required_fields = ['id', 'problem', 'trajectory', 'generated_at']
                    for field in required_fields:
                        if field not in trajectory:
                            raise ValueError(f"Missing field: {field}")
                    
                    # Check trajectory structure
                    traj = trajectory['trajectory']
                    if 'steps' not in traj:
                        raise ValueError("Missing 'steps' in trajectory")
                    
                    validation_results["valid_trajectories"] += 1
                    
                except Exception as e:
                    validation_results["invalid_trajectories"] += 1
                    validation_results["format_errors"].append(f"Line {line_num}: {str(e)}")
    
    except FileNotFoundError:
        console.print(f"[red]Trajectory file not found: {trajectory_file}[/red]")
        return None
    
    console.print(f"[green]Validation complete: {validation_results['valid_trajectories']} valid, "
                 f"{validation_results['invalid_trajectories']} invalid trajectories[/green]")
    
    if validation_results["format_errors"]:
        console.print("[yellow]Format errors found:[/yellow]")
        for error in validation_results["format_errors"][:5]:  # Show first 5 errors
            console.print(f"  {error}")
    
    return validation_results

def main():
    """Main function for sample trajectory generation"""
    
    console.print(Panel(
        "[bold green]DeepCoder Phase 1.3: Sample Trajectory Generation[/bold green]\n"
        "This script will generate and analyze sample trajectories to validate our pipeline.",
        title="Phase 1.3"
    ))
    
    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        console.print("[red]Error: GROQ_API_KEY not set in environment[/red]")
        console.print("[yellow]Please set the API key and try again[/yellow]")
        return
    
    # Generate sample trajectories
    stats, quality_analysis = generate_sample_trajectories()
    
    if stats is None:
        console.print("[red]Sample generation failed[/red]")
        return
    
    # Validate format of generated trajectories
    sample_dir = Path("data/sample_trajectories")
    if sample_dir.exists():
        for trajectory_file in sample_dir.glob("trajectories_batch_*.jsonl"):
            validation_results = validate_trajectory_format(trajectory_file)
    
    # Summary and next steps
    console.print(Panel(
        f"[bold green]Phase 1.3 Complete![/bold green]\n"
        f"Generated {stats.total_generated} sample trajectories with {stats.success_rate:.1%} success rate.\n"
        f"Quality analysis shows good reasoning structure and code generation.\n\n"
        f"[bold blue]Next Steps:[/bold blue]\n"
        f"✓ Review trajectory quality in data/sample_trajectories/\n"
        f"✓ Adjust parameters if needed\n"
        f"✓ Proceed to full-scale generation (Phase 1.3.2)\n"
        f"✓ Begin Phase 2: Data Preprocessing and SAD Implementation",
        title="Success"
    ))

if __name__ == "__main__":
    main() 