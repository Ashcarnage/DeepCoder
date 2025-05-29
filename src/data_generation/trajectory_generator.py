"""
Trajectory Generator for DeepCoder
Orchestrates the generation of training trajectories using teacher model.
"""

import os
import json
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import yaml
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel

from .groq_client import create_groq_client, GroqClient
from .agent_framework import create_agent_framework, AgentFramework, AgentTrajectory
from .problem_loader import ProblemLoader

logger = logging.getLogger(__name__)
console = Console()

@dataclass
class GenerationConfig:
    """Configuration for trajectory generation"""
    num_trajectories: int = 1000
    max_steps_per_trajectory: int = 10
    batch_size: int = 10
    max_workers: int = 4
    output_dir: str = "data/trajectories"
    problems_file: str = "data/problems/coding_problems.jsonl"
    save_interval: int = 50
    temperature: float = 0.7
    max_tokens: int = 4096
    include_failed: bool = True
    min_success_rate: float = 0.7

@dataclass
class GenerationStats:
    """Statistics for trajectory generation"""
    total_generated: int = 0
    successful: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    avg_steps_per_trajectory: float = 0.0
    success_rate: float = 0.0

class TrajectoryGenerator:
    """
    Main class for generating training trajectories
    """
    
    def __init__(self, config: Optional[GenerationConfig] = None):
        """Initialize trajectory generator"""
        self.config = config or self._load_config()
        self.groq_client = create_groq_client()
        self.agent_framework = create_agent_framework(self.groq_client)
        self.problem_loader = ProblemLoader()
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize stats
        self.stats = GenerationStats()
        
        logger.info(f"Initialized trajectory generator with config: {self.config}")
    
    def _load_config(self) -> GenerationConfig:
        """Load configuration from files and environment"""
        config_path = "configs/config.yaml"
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                data_gen_config = yaml_config.get('data_generation', {})
        else:
            data_gen_config = {}
        
        return GenerationConfig(
            num_trajectories=int(os.getenv('NUM_TRAJECTORIES', 
                                         data_gen_config.get('num_trajectories', 1000))),
            max_steps_per_trajectory=int(os.getenv('MAX_STEPS_PER_TRAJECTORY',
                                                 data_gen_config.get('max_steps_per_trajectory', 10))),
            batch_size=data_gen_config.get('batch_size', 10),
            max_workers=data_gen_config.get('max_workers', 4),
            output_dir=data_gen_config.get('output_dir', 'data/trajectories'),
            problems_file=data_gen_config.get('problems_file', 'data/problems/coding_problems.jsonl'),
            save_interval=data_gen_config.get('save_interval', 50),
            temperature=float(os.getenv('TEMPERATURE', 
                                      data_gen_config.get('temperature', 0.7))),
            max_tokens=int(os.getenv('MAX_TOKENS',
                                   data_gen_config.get('max_tokens', 4096))),
            include_failed=data_gen_config.get('include_failed', True),
            min_success_rate=data_gen_config.get('min_success_rate', 0.7)
        )
    
    def generate_single_trajectory(
        self, 
        problem: Dict[str, Any], 
        trajectory_id: int
    ) -> Optional[Dict[str, Any]]:
        """Generate a single trajectory for a problem"""
        try:
            # Generate trajectory using agent framework
            trajectory = self.agent_framework.generate_trajectory(
                problem=problem['description'],
                max_steps=self.config.max_steps_per_trajectory
            )
            
            # Convert to serializable format
            trajectory_data = {
                'id': trajectory_id,
                'problem': problem,
                'trajectory': {
                    'steps': [
                        {
                            'step_type': step.step_type.value,
                            'content': step.content,
                            'metadata': step.metadata,
                            'timestamp': step.timestamp
                        }
                        for step in trajectory.steps
                    ],
                    'final_answer': trajectory.final_answer,
                    'success': trajectory.success,
                    'total_tokens': trajectory.total_tokens,
                    'execution_time': trajectory.execution_time,
                    'metadata': trajectory.metadata
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Update stats
            self.stats.total_generated += 1
            if trajectory.success:
                self.stats.successful += 1
            else:
                self.stats.failed += 1
            
            self.stats.total_tokens += trajectory.total_tokens
            self.stats.total_time += trajectory.execution_time
            self.stats.avg_steps_per_trajectory = (
                (self.stats.avg_steps_per_trajectory * (self.stats.total_generated - 1) + 
                 len(trajectory.steps)) / self.stats.total_generated
            )
            self.stats.success_rate = self.stats.successful / self.stats.total_generated
            
            return trajectory_data
            
        except Exception as e:
            logger.error(f"Error generating trajectory {trajectory_id}: {e}")
            self.stats.failed += 1
            return None
    
    def generate_batch(
        self, 
        problems: List[Dict[str, Any]], 
        start_id: int
    ) -> List[Dict[str, Any]]:
        """Generate a batch of trajectories"""
        trajectories = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self.generate_single_trajectory, problem, start_id + i): start_id + i
                for i, problem in enumerate(problems)
            }
            
            # Collect results
            for future in as_completed(future_to_id):
                trajectory_id = future_to_id[future]
                try:
                    result = future.result()
                    if result is not None:
                        trajectories.append(result)
                except Exception as e:
                    logger.error(f"Error in trajectory {trajectory_id}: {e}")
        
        return trajectories
    
    def save_trajectories(
        self, 
        trajectories: List[Dict[str, Any]], 
        batch_num: int
    ):
        """Save trajectories to file"""
        filename = f"trajectories_batch_{batch_num:04d}.jsonl"
        filepath = Path(self.config.output_dir) / filename
        
        with open(filepath, 'w') as f:
            for trajectory in trajectories:
                f.write(json.dumps(trajectory) + '\n')
        
        logger.info(f"Saved {len(trajectories)} trajectories to {filepath}")
    
    def save_stats(self):
        """Save generation statistics"""
        stats_file = Path(self.config.output_dir) / "generation_stats.json"
        
        with open(stats_file, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)
        
        logger.info(f"Saved generation stats to {stats_file}")
    
    def display_progress(self, current: int, total: int):
        """Display current progress"""
        table = Table(title="Trajectory Generation Progress")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Progress", f"{current}/{total} ({current/total*100:.1f}%)")
        table.add_row("Success Rate", f"{self.stats.success_rate:.2%}")
        table.add_row("Avg Steps/Trajectory", f"{self.stats.avg_steps_per_trajectory:.1f}")
        table.add_row("Total Tokens", f"{self.stats.total_tokens:,}")
        table.add_row("Total Time", f"{self.stats.total_time:.1f}s")
        table.add_row("Avg Time/Trajectory", f"{self.stats.total_time/max(1, self.stats.total_generated):.2f}s")
        
        console.print(table)
    
    def generate_trajectories(self) -> GenerationStats:
        """Generate all trajectories"""
        console.print(Panel(
            f"[bold blue]Starting trajectory generation[/bold blue]\n"
            f"Target: {self.config.num_trajectories} trajectories\n"
            f"Batch size: {self.config.batch_size}\n"
            f"Max workers: {self.config.max_workers}",
            title="Trajectory Generation"
        ))
        
        # Load problems
        console.print("[yellow]Loading problems...[/yellow]")
        problems = self.problem_loader.load_problems(self.config.problems_file)
        
        if not problems:
            console.print("[red]No problems found! Please check the problems file.[/red]")
            return self.stats
        
        console.print(f"[green]Loaded {len(problems)} problems[/green]")
        
        # Generate trajectories in batches
        total_batches = (self.config.num_trajectories + self.config.batch_size - 1) // self.config.batch_size
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            
            main_task = progress.add_task(
                "Generating trajectories...", 
                total=self.config.num_trajectories
            )
            
            for batch_num in range(total_batches):
                start_idx = batch_num * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, self.config.num_trajectories)
                batch_size = end_idx - start_idx
                
                # Sample problems for this batch
                batch_problems = random.choices(problems, k=batch_size)
                
                # Generate batch
                batch_trajectories = self.generate_batch(batch_problems, start_idx)
                
                # Filter trajectories based on config
                if not self.config.include_failed:
                    batch_trajectories = [t for t in batch_trajectories if t['trajectory']['success']]
                
                # Save batch
                if batch_trajectories:
                    self.save_trajectories(batch_trajectories, batch_num)
                
                # Update progress
                progress.update(main_task, advance=batch_size)
                
                # Save stats periodically
                if (batch_num + 1) % (self.config.save_interval // self.config.batch_size) == 0:
                    self.save_stats()
                    self.display_progress(end_idx, self.config.num_trajectories)
                
                # Check success rate
                if (self.stats.total_generated > 100 and 
                    self.stats.success_rate < self.config.min_success_rate):
                    console.print(
                        f"[yellow]Warning: Success rate ({self.stats.success_rate:.2%}) "
                        f"below minimum ({self.config.min_success_rate:.2%})[/yellow]"
                    )
        
        # Final save
        self.save_stats()
        
        # Display final results
        console.print(Panel(
            f"[bold green]Generation Complete![/bold green]\n"
            f"Total Generated: {self.stats.total_generated}\n"
            f"Successful: {self.stats.successful}\n"
            f"Failed: {self.stats.failed}\n"
            f"Success Rate: {self.stats.success_rate:.2%}\n"
            f"Total Tokens: {self.stats.total_tokens:,}\n"
            f"Total Time: {self.stats.total_time:.1f}s",
            title="Final Results"
        ))
        
        return self.stats
    
    def load_existing_trajectories(self) -> List[Dict[str, Any]]:
        """Load existing trajectories from output directory"""
        trajectories = []
        output_path = Path(self.config.output_dir)
        
        if not output_path.exists():
            return trajectories
        
        for file_path in output_path.glob("trajectories_batch_*.jsonl"):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        trajectory = json.loads(line.strip())
                        trajectories.append(trajectory)
            except Exception as e:
                logger.error(f"Error loading trajectories from {file_path}: {e}")
        
        return trajectories
    
    def validate_trajectories(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate generated trajectories"""
        validation_results = {
            'total_trajectories': len(trajectories),
            'valid_trajectories': 0,
            'invalid_trajectories': 0,
            'validation_errors': []
        }
        
        for i, trajectory in enumerate(trajectories):
            try:
                # Check required fields
                required_fields = ['id', 'problem', 'trajectory', 'generated_at']
                for field in required_fields:
                    if field not in trajectory:
                        raise ValueError(f"Missing required field: {field}")
                
                # Check trajectory structure
                traj_data = trajectory['trajectory']
                if 'steps' not in traj_data:
                    raise ValueError("Missing steps in trajectory")
                
                # Check steps
                for step in traj_data['steps']:
                    if 'step_type' not in step or 'content' not in step:
                        raise ValueError("Invalid step structure")
                
                validation_results['valid_trajectories'] += 1
                
            except Exception as e:
                validation_results['invalid_trajectories'] += 1
                validation_results['validation_errors'].append({
                    'trajectory_id': i,
                    'error': str(e)
                })
        
        return validation_results

# Convenience function
def create_trajectory_generator(config: Optional[GenerationConfig] = None) -> TrajectoryGenerator:
    """Create and return configured trajectory generator"""
    return TrajectoryGenerator(config)

if __name__ == "__main__":
    # Test trajectory generation
    try:
        console.print("[green]Testing trajectory generator...[/green]")
        
        # Create generator with small config for testing
        test_config = GenerationConfig(
            num_trajectories=5,
            batch_size=2,
            max_workers=2,
            output_dir="data/test_trajectories"
        )
        
        generator = create_trajectory_generator(test_config)
        stats = generator.generate_trajectories()
        
        console.print(f"[green]Test completed! Generated {stats.total_generated} trajectories[/green]")
        
    except Exception as e:
        console.print(f"[red]Error testing trajectory generator: {e}[/red]")
        console.print("[yellow]Make sure to set your GROQ_API_KEY in the .env file[/yellow]") 