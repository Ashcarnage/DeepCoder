#!/usr/bin/env python3
"""
Demonstration Script for Trajectory Processing (Phase 2.1)

Shows the Structured Agent Distillation (SAD) trajectory processing pipeline:
1. Loading and parsing different trajectory formats
2. Span detection and classification (reasoning vs action)
3. Token-level processing and alignment
4. Quality filtering and validation
5. Output generation for SAD training

Usage:
    python scripts/demo_trajectory_processing.py
    python scripts/demo_trajectory_processing.py --sample-size 10
    python scripts/demo_trajectory_processing.py --config configs/config.yaml
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from trajectory_processing import (
        TrajectoryProcessor, 
        TrajectoryProcessingConfig,
        TrajectoryParser,
        SpanDetector,
        TrajectoryTokenizer,
        TokenSpan,
        SpanType,
        ProcessedTrajectory,
        create_trajectory_processor,
        create_trajectory_processing_config
    )
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    import yaml
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed and the module path is correct")
    sys.exit(1)

console = Console()


class TrajectoryProcessingDemo:
    """Demonstration of the trajectory processing system"""
    
    def __init__(self, config_path: str = "configs/config.yaml", sample_size: int = 5):
        self.config_path = config_path
        self.sample_size = sample_size
        
        console.print("[bold blue]🚀 Trajectory Processing Demo (Phase 2.1)[/bold blue]")
        console.print("Demonstrating Structured Agent Distillation (SAD) Implementation")
        console.print(f"Configuration: {config_path}")
        console.print(f"Sample size: {sample_size}")
    
    def create_diverse_sample_trajectories(self) -> List[Dict[str, Any]]:
        """Create diverse sample trajectories showcasing different formats and patterns"""
        return [
            # Standard agent trajectory with steps
            {
                "id": "demo_1",
                "trajectory": {
                    "steps": [
                        {
                            "step_type": "reasoning",
                            "content": "<think>I need to solve a coding problem about finding prime numbers. Let me break this down: I need to check if a number is prime by testing divisibility up to its square root.</think>",
                            "metadata": {"confidence": 0.9},
                            "timestamp": "2024-01-01T10:00:00"
                        },
                        {
                            "step_type": "action",
                            "content": "execute_python('def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1))')",
                            "metadata": {"tool": "python_executor"},
                            "timestamp": "2024-01-01T10:00:01"
                        },
                        {
                            "step_type": "observation",
                            "content": "Function defined successfully. Let me test it with some examples.",
                            "metadata": {"execution_time": 0.1},
                            "timestamp": "2024-01-01T10:00:02"
                        },
                        {
                            "step_type": "action",
                            "content": "execute_python('print([n for n in range(2, 20) if is_prime(n)])')",
                            "metadata": {"tool": "python_executor"},
                            "timestamp": "2024-01-01T10:00:03"
                        },
                        {
                            "step_type": "observation",
                            "content": "Output: [2, 3, 5, 7, 11, 13, 17, 19]",
                            "metadata": {"execution_time": 0.05},
                            "timestamp": "2024-01-01T10:00:04"
                        }
                    ],
                    "final_answer": "The prime number function is working correctly. It efficiently checks primality by testing divisibility up to the square root.",
                    "success": True,
                    "total_tokens": 320,
                    "execution_time": 4.2
                }
            },
            
            # Text-based trajectory with thinking tags
            {
                "id": "demo_2",
                "text": """<thinking>
This is a mathematical optimization problem. I need to find the minimum value of a quadratic function.
For f(x) = ax² + bx + c, the minimum occurs at x = -b/(2a) when a > 0.
Let me work through this step by step.
</thinking>

To solve this optimization problem, I'll use calculus to find the critical points.

Given f(x) = 2x² - 8x + 3, I need to find the minimum value.

First, I'll find the derivative:
f'(x) = 4x - 8

Setting f'(x) = 0:
4x - 8 = 0
x = 2

Since the second derivative f''(x) = 4 > 0, this is indeed a minimum.

The minimum value is f(2) = 2(4) - 8(2) + 3 = 8 - 16 + 3 = -5.

Therefore, the minimum value of the function is -5, occurring at x = 2."""
            },
            
            # Content-based trajectory with mixed patterns
            {
                "id": "demo_3",
                "content": """Thought: I need to implement a binary search algorithm. This is a classic divide-and-conquer approach.

Let me think about the implementation:
1. Compare the target with the middle element
2. If equal, we found it
3. If target is smaller, search left half
4. If target is larger, search right half

Action: I'll implement this recursively.

```python
def binary_search(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, left, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, right)

# Test the function
test_array = [1, 3, 5, 7, 9, 11, 13, 15]
result = binary_search(test_array, 7)
print(f"Index of 7: {result}")
```

Observation: Function works correctly. Found 7 at index 3 in the sorted array.

The binary search implementation is complete and tested. Time complexity is O(log n)."""
            },
            
            # SGLang-style trajectory with multiple thinking blocks
            {
                "id": "demo_4",
                "trajectory": {
                    "steps": [
                        {
                            "step_type": "reasoning",
                            "content": "<think>This is a dynamic programming problem - the classic coin change problem. I need to find the minimum number of coins to make a target amount.\n\nI can use either a top-down (memoization) or bottom-up (tabulation) approach. Let me go with bottom-up as it's usually more efficient.</think>",
                            "metadata": {},
                            "timestamp": "2024-01-01T11:00:00"
                        },
                        {
                            "step_type": "action",
                            "content": "I'll implement the dynamic programming solution:\n\n```python\ndef min_coins(coins, amount):\n    dp = [float('inf')] * (amount + 1)\n    dp[0] = 0\n    \n    for i in range(1, amount + 1):\n        for coin in coins:\n            if coin <= i:\n                dp[i] = min(dp[i], dp[i - coin] + 1)\n    \n    return dp[amount] if dp[amount] != float('inf') else -1\n```",
                            "metadata": {},
                            "timestamp": "2024-01-01T11:00:01"
                        },
                        {
                            "step_type": "reasoning",
                            "content": "<think>Now I should test this with a concrete example. Let's use coins [1, 3, 4] and target amount 6. The optimal solution should be two coins: 3 + 3 = 6.</think>",
                            "metadata": {},
                            "timestamp": "2024-01-01T11:00:02"
                        },
                        {
                            "step_type": "action",
                            "content": "execute_python('coins = [1, 3, 4]; amount = 6; result = min_coins(coins, amount); print(f\"Minimum coins for {amount}: {result}\")')",
                            "metadata": {},
                            "timestamp": "2024-01-01T11:00:03"
                        },
                        {
                            "step_type": "observation",
                            "content": "Minimum coins for 6: 2",
                            "metadata": {},
                            "timestamp": "2024-01-01T11:00:04"
                        }
                    ],
                    "final_answer": "finish('The dynamic programming solution correctly finds the minimum number of coins needed. For amount 6 with coins [1,3,4], it returns 2 coins (3+3).')",
                    "success": True,
                    "total_tokens": 450,
                    "execution_time": 5.1
                }
            },
            
            # Complex reasoning trajectory
            {
                "id": "demo_5",
                "text": """<thinking>
This is an interesting graph theory problem. I need to find the shortest path between two nodes in a weighted graph. 

There are several algorithms I could use:
1. Dijkstra's algorithm - good for non-negative weights
2. Bellman-Ford - handles negative weights  
3. Floyd-Warshall - all pairs shortest paths

Since the problem doesn't mention negative weights, Dijkstra's is the best choice. It has O((V + E) log V) time complexity with a priority queue.

Let me implement this step by step.
</thinking>

To solve this shortest path problem, I'll implement Dijkstra's algorithm.

First, let me set up the data structures:

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start, end):
    # Initialize distances and priority queue
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    pq = [(0, start)]
    visited = set()
    previous = {}
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node in visited:
            continue
            
        visited.add(current_node)
        
        if current_node == end:
            break
            
        for neighbor, weight in graph[current_node]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    node = end
    while node in previous:
        path.append(node)
        node = previous[node]
    path.append(start)
    path.reverse()
    
    return distances[end], path

# Test with a sample graph
graph = {
    'A': [('B', 4), ('C', 2)],
    'B': [('C', 1), ('D', 5)],
    'C': [('D', 8), ('E', 10)],
    'D': [('E', 2)],
    'E': []
}

distance, path = dijkstra(graph, 'A', 'E')
print(f"Shortest distance from A to E: {distance}")
print(f"Path: {' -> '.join(path)}")
```

The algorithm successfully finds the shortest path. For the given graph, the shortest distance from A to E is 11, following the path A -> C -> D -> E."""
            }
        ][:self.sample_size]  # Limit to requested sample size
    
    def demonstrate_configuration(self):
        """Demonstrate configuration loading and customization"""
        console.print("\n[bold cyan]📋 Configuration System[/bold cyan]")
        
        # Show default configuration
        default_config = create_trajectory_processing_config()
        
        config_table = Table(title="Trajectory Processing Configuration")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="magenta")
        config_table.add_column("Description", style="dim")
        
        config_items = [
            ("model_name", default_config.model_name, "Target model for tokenization"),
            ("max_length", str(default_config.max_length), "Maximum sequence length"),
            ("min_reasoning_ratio", f"{default_config.min_reasoning_ratio:.2f}", "Minimum reasoning tokens ratio"),
            ("min_action_ratio", f"{default_config.min_action_ratio:.2f}", "Minimum action tokens ratio"),
            ("confidence_threshold", f"{default_config.confidence_threshold:.2f}", "Span detection confidence threshold"),
        ]
        
        for param, value, desc in config_items:
            config_table.add_row(param, value, desc)
        
        console.print(config_table)
        
        # Show pattern configuration
        patterns_panel = Panel(
            f"[bold]Thinking Patterns:[/bold] {len(default_config.thinking_patterns)}\n"
            f"[bold]Action Patterns:[/bold] {len(default_config.action_patterns)}\n"
            f"[bold]Observation Patterns:[/bold] {len(default_config.observation_patterns)}",
            title="Span Detection Patterns",
            border_style="green"
        )
        console.print(patterns_panel)
    
    def demonstrate_trajectory_parsing(self, trajectories: List[Dict[str, Any]]):
        """Demonstrate trajectory parsing capabilities"""
        console.print("\n[bold cyan]🔍 Trajectory Parsing[/bold cyan]")
        
        config = create_trajectory_processing_config()
        parser = TrajectoryParser(config)
        
        parsing_results = Table(title="Trajectory Parsing Results")
        parsing_results.add_column("ID", style="cyan")
        parsing_results.add_column("Format", style="yellow")
        parsing_results.add_column("Text Length", style="magenta")
        parsing_results.add_column("Segments", style="green")
        parsing_results.add_column("Has Thinking", style="blue")
        parsing_results.add_column("Has Actions", style="red")
        parsing_results.add_column("Quality Score", style="bright_magenta")
        
        for traj in trajectories:
            parsed = parser.parse_trajectory(traj)
            
            # Determine format
            if 'trajectory' in traj and 'steps' in traj['trajectory']:
                format_type = "Agent Steps"
            elif 'text' in traj:
                format_type = "Text Block"
            elif 'content' in traj:
                format_type = "Content Field"
            else:
                format_type = "Unknown"
            
            parsing_results.add_row(
                str(traj.get('id', 'N/A')),
                format_type,
                str(len(parsed['text_content'])),
                str(len(parsed['segments'])),
                "✅" if parsed['has_thinking'] else "❌",
                "✅" if parsed['has_actions'] else "❌",
                f"{parsed['quality_score']:.2f}"
            )
        
        console.print(parsing_results)
        
        # Show detailed parsing for first trajectory
        if trajectories:
            first_traj = trajectories[0]
            first_parsed = parser.parse_trajectory(first_traj)
            
            console.print(f"\n[bold]Detailed Analysis of Trajectory {first_traj.get('id', 'N/A')}:[/bold]")
            
            segments_table = Table(title="Detected Segments")
            segments_table.add_column("Type", style="cyan")
            segments_table.add_column("Length", style="magenta")
            segments_table.add_column("Confidence", style="green")
            segments_table.add_column("Preview", style="dim")
            
            for segment in first_parsed['segments']:
                preview = segment['text'][:50] + "..." if len(segment['text']) > 50 else segment['text']
                segments_table.add_row(
                    segment['type'].title(),
                    str(segment['length']),
                    f"{segment['confidence']:.2f}",
                    preview.replace('\n', ' ')
                )
            
            console.print(segments_table)
    
    def demonstrate_span_detection(self, trajectories: List[Dict[str, Any]]):
        """Demonstrate span detection and classification"""
        console.print("\n[bold cyan]🎯 Span Detection & Classification[/bold cyan]")
        
        config = create_trajectory_processing_config()
        parser = TrajectoryParser(config)
        
        # Show pattern matching examples
        pattern_examples = [
            ("<think>Complex reasoning here</think>", "Thinking Tag"),
            ("execute_python('print(1)')", "Python Execution"), 
            ("Thought: I need to consider...", "Agent Thought"),
            ("Action: Let me implement...", "Agent Action"),
            ("Observation: The result is...", "Agent Observation"),
            ("```python\ncode here\n```", "Code Block"),
        ]
        
        pattern_table = Table(title="Pattern Classification Examples")
        pattern_table.add_column("Text Sample", style="cyan")
        pattern_table.add_column("Pattern Type", style="yellow")
        pattern_table.add_column("Classified As", style="green")
        pattern_table.add_column("Confidence", style="magenta")
        
        for text, pattern_type in pattern_examples:
            segment_type, confidence = parser._classify_segment(text)
            pattern_table.add_row(
                text[:30] + "..." if len(text) > 30 else text,
                pattern_type,
                segment_type.title(),
                f"{confidence:.2f}"
            )
        
        console.print(pattern_table)
        
        # Show span type distribution
        all_segments = []
        for traj in trajectories:
            parsed = parser.parse_trajectory(traj)
            all_segments.extend(parsed['segments'])
        
        if all_segments:
            type_counts = {}
            for segment in all_segments:
                seg_type = segment['type']
                type_counts[seg_type] = type_counts.get(seg_type, 0) + 1
            
            distribution_table = Table(title="Span Type Distribution")
            distribution_table.add_column("Span Type", style="cyan")
            distribution_table.add_column("Count", style="magenta")
            distribution_table.add_column("Percentage", style="green")
            
            total_segments = len(all_segments)
            for span_type, count in sorted(type_counts.items()):
                percentage = (count / total_segments) * 100
                distribution_table.add_row(
                    span_type.title(),
                    str(count),
                    f"{percentage:.1f}%"
                )
            
            console.print(distribution_table)
    
    def demonstrate_tokenization_concepts(self):
        """Demonstrate tokenization concepts without requiring the actual model"""
        console.print("\n[bold cyan]🔤 Tokenization & Token Alignment[/bold cyan]")
        
        # Show concept with mock data
        sample_text = "Thought: I need to solve this problem. Action: execute_python('print(1)')"
        
        # Simulate tokenization
        mock_tokens = sample_text.split()  # Simple word tokenization for demo
        mock_input_ids = list(range(len(mock_tokens)))
        
        # Simulate span detection
        reasoning_spans = [(0, 8)]  # "Thought: I need to solve this problem."
        action_spans = [(8, 12)]   # "Action: execute_python('print(1)')"
        
        # Create masks
        reasoning_mask = [1 if any(start <= i < end for start, end in reasoning_spans) else 0 for i in range(len(mock_tokens))]
        action_mask = [1 if any(start <= i < end for start, end in action_spans) else 0 for i in range(len(mock_tokens))]
        
        tokenization_table = Table(title="Token-Level Processing Example")
        tokenization_table.add_column("Token ID", style="cyan")
        tokenization_table.add_column("Token", style="yellow")
        tokenization_table.add_column("Reasoning", style="green")
        tokenization_table.add_column("Action", style="red")
        tokenization_table.add_column("Span Type", style="magenta")
        
        for i, (token_id, token) in enumerate(zip(mock_input_ids, mock_tokens)):
            reasoning = "✅" if reasoning_mask[i] else "❌"
            action = "✅" if action_mask[i] else "❌"
            
            if reasoning_mask[i]:
                span_type = "REASONING"
            elif action_mask[i]:
                span_type = "ACTION"
            else:
                span_type = "OTHER"
            
            tokenization_table.add_row(
                str(token_id),
                token,
                reasoning,
                action,
                span_type
            )
        
        console.print(tokenization_table)
        
        # Show processing statistics
        stats_panel = Panel(
            f"[bold]Total Tokens:[/bold] {len(mock_tokens)}\n"
            f"[bold]Reasoning Tokens:[/bold] {sum(reasoning_mask)} ({sum(reasoning_mask)/len(mock_tokens)*100:.1f}%)\n"
            f"[bold]Action Tokens:[/bold] {sum(action_mask)} ({sum(action_mask)/len(mock_tokens)*100:.1f}%)\n"
            f"[bold]Other Tokens:[/bold] {len(mock_tokens) - sum(reasoning_mask) - sum(action_mask)} ({(len(mock_tokens) - sum(reasoning_mask) - sum(action_mask))/len(mock_tokens)*100:.1f}%)",
            title="Token Statistics",
            border_style="blue"
        )
        console.print(stats_panel)
    
    def demonstrate_quality_filtering(self, trajectories: List[Dict[str, Any]]):
        """Demonstrate quality filtering criteria"""
        console.print("\n[bold cyan]✅ Quality Filtering[/bold cyan]")
        
        config = create_trajectory_processing_config()
        parser = TrajectoryParser(config)
        
        quality_table = Table(title="Quality Assessment")
        quality_table.add_column("Trajectory ID", style="cyan")
        quality_table.add_column("Quality Score", style="magenta")
        quality_table.add_column("Reasoning %", style="green")
        quality_table.add_column("Action %", style="red")
        quality_table.add_column("Status", style="bold")
        
        for traj in trajectories:
            parsed = parser.parse_trajectory(traj)
            quality_score = parsed['quality_score']
            
            # Simulate token ratios for demo
            segments = parsed['segments']
            total_length = sum(seg['length'] for seg in segments)
            
            if total_length > 0:
                reasoning_ratio = sum(seg['length'] for seg in segments if seg['type'] == 'reasoning') / total_length
                action_ratio = sum(seg['length'] for seg in segments if seg['type'] == 'action') / total_length
            else:
                reasoning_ratio = 0
                action_ratio = 0
            
            # Apply quality criteria
            passes_quality = (
                quality_score >= 0.3 and
                reasoning_ratio >= config.min_reasoning_ratio and
                action_ratio >= config.min_action_ratio
            )
            
            status = "✅ PASS" if passes_quality else "❌ FAIL"
            
            quality_table.add_row(
                str(traj.get('id', 'N/A')),
                f"{quality_score:.2f}",
                f"{reasoning_ratio:.1%}",
                f"{action_ratio:.1%}",
                status
            )
        
        console.print(quality_table)
        
        # Show quality criteria
        criteria_panel = Panel(
            f"[bold]Quality Criteria:[/bold]\n"
            f"• Minimum Quality Score: {config.min_quality_score if hasattr(config, 'min_quality_score') else 0.3}\n"
            f"• Minimum Reasoning Ratio: {config.min_reasoning_ratio:.1%}\n"
            f"• Minimum Action Ratio: {config.min_action_ratio:.1%}\n"
            f"• Maximum Other Ratio: {config.max_other_ratio:.1%}",
            title="Filtering Thresholds",
            border_style="yellow"
        )
        console.print(criteria_panel)
    
    def demonstrate_output_format(self, trajectories: List[Dict[str, Any]]):
        """Demonstrate the output format for SAD training"""
        console.print("\n[bold cyan]📤 SAD Training Output Format[/bold cyan]")
        
        # Simulate processed trajectory structure
        sample_processed = {
            "trajectory_id": "demo_1",
            "input_ids": [1, 15, 284, 25, 934, 28, 391, 847, 29, 847],
            "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "reasoning_mask": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "action_mask": [0, 0, 0, 0, 1, 1, 1, 1, 0, 0],
            "spans": [
                {
                    "span_type": "reasoning",
                    "start_token": 0,
                    "end_token": 4,
                    "text": "I need to think about this",
                    "confidence": 0.95
                },
                {
                    "span_type": "action", 
                    "start_token": 4,
                    "end_token": 8,
                    "text": "execute_python('code')",
                    "confidence": 0.89
                }
            ],
            "metadata": {
                "num_reasoning_tokens": 4,
                "num_action_tokens": 4,
                "num_spans": 2,
                "quality_score": 0.82,
                "has_thinking": True
            }
        }
        
        # Display as formatted JSON
        json_syntax = Syntax(
            json.dumps(sample_processed, indent=2),
            "json",
            theme="monokai",
            line_numbers=True
        )
        
        console.print(Panel(
            json_syntax,
            title="Processed Trajectory Format (Sample)",
            border_style="green"
        ))
        
        # Show key features
        features_table = Table(title="SAD Training Features")
        features_table.add_column("Component", style="cyan")
        features_table.add_column("Purpose", style="yellow")
        features_table.add_column("Shape/Type", style="magenta")
        
        features = [
            ("input_ids", "Tokenized sequence for model input", "List[int]"),
            ("attention_mask", "Valid token positions", "List[int] (0/1)"),
            ("reasoning_mask", "Reasoning token positions", "List[int] (0/1)"),
            ("action_mask", "Action token positions", "List[int] (0/1)"),
            ("spans", "Detailed span information", "List[TokenSpan]"),
            ("metadata", "Quality and processing info", "Dict[str, Any]")
        ]
        
        for component, purpose, shape in features:
            features_table.add_row(component, purpose, shape)
        
        console.print(features_table)
    
    def run_demonstration(self):
        """Run the complete demonstration"""
        console.print("\n[bold green]🎯 Starting Trajectory Processing Demonstration[/bold green]")
        
        # Create sample trajectories
        trajectories = self.create_diverse_sample_trajectories()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            demos = [
                ("Configuration System", self.demonstrate_configuration),
                ("Trajectory Parsing", lambda: self.demonstrate_trajectory_parsing(trajectories)),
                ("Span Detection", lambda: self.demonstrate_span_detection(trajectories)),
                ("Tokenization Concepts", self.demonstrate_tokenization_concepts),
                ("Quality Filtering", lambda: self.demonstrate_quality_filtering(trajectories)),
                ("Output Format", lambda: self.demonstrate_output_format(trajectories))
            ]
            
            task = progress.add_task("Running demonstrations...", total=len(demos))
            
            for demo_name, demo_func in demos:
                progress.update(task, description=f"Demonstrating {demo_name}")
                demo_func()
                progress.advance(task)
        
        # Final summary
        console.print("\n[bold blue]🎉 Demonstration Complete![/bold blue]")
        
        summary_panel = Panel(
            "[bold]Phase 2.1 Trajectory Processing System[/bold]\n\n"
            "✅ [green]Configuration System[/green] - Flexible parameter management\n"
            "✅ [green]Multi-format Parsing[/green] - Handles diverse trajectory formats\n"
            "✅ [green]Intelligent Span Detection[/green] - Reasoning vs Action classification\n"
            "✅ [green]Token-level Processing[/green] - Precise alignment for SAD training\n"
            "✅ [green]Quality Filtering[/green] - Ensures high-quality training data\n"
            "✅ [green]SAD-Ready Output[/green] - Optimized for distillation training\n\n"
            "[bold cyan]Ready for Phase 2.2: SAD Loss Implementation[/bold cyan]",
            title="System Status",
            border_style="green"
        )
        console.print(summary_panel)


def main():
    """Main function for CLI interface"""
    parser = argparse.ArgumentParser(description="Demonstrate Trajectory Processing System (Phase 2.1)")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Configuration file path")
    parser.add_argument("--sample-size", type=int, default=5,
                       help="Number of sample trajectories to use")
    
    args = parser.parse_args()
    
    # Run demonstration
    demo = TrajectoryProcessingDemo(args.config, args.sample_size)
    demo.run_demonstration()


if __name__ == "__main__":
    main() 