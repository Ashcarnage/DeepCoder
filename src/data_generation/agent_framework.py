"""
Agent Framework for DeepSeek R1 Teacher Model
Implements Thought-Action-Observation cycles with tool integration.
"""

import re
import json
import traceback
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from abc import ABC, abstractmethod
import subprocess
import tempfile
import os
import sys
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr
import RestrictedPython
from RestrictedPython import compile_restricted
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()

class StepType(Enum):
    """Types of steps in agent trajectory"""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"

@dataclass
class AgentStep:
    """Single step in agent trajectory"""
    step_type: StepType
    content: str
    metadata: Optional[Dict[str, Any]] = None
    timestamp: Optional[float] = None

@dataclass
class AgentTrajectory:
    """Complete agent trajectory for a problem"""
    problem: str
    steps: List[AgentStep]
    final_answer: Optional[str] = None
    success: bool = False
    total_tokens: int = 0
    execution_time: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

class Tool(ABC):
    """Abstract base class for agent tools"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description"""
        pass
    
    @abstractmethod
    def execute(self, code: str, **kwargs) -> Dict[str, Any]:
        """Execute tool with given input"""
        pass

class PythonExecutor(Tool):
    """Safe Python code execution tool"""
    
    @property
    def name(self) -> str:
        return "python_executor"
    
    @property
    def description(self) -> str:
        return "Execute Python code safely and return results"
    
    def __init__(self, timeout: int = 30, max_output_length: int = 10000):
        self.timeout = timeout
        self.max_output_length = max_output_length
        
        # Safe builtins for restricted execution
        self.safe_builtins = {
            '__builtins__': {
                'len': len, 'range': range, 'enumerate': enumerate,
                'zip': zip, 'map': map, 'filter': filter, 'sorted': sorted,
                'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'int': int, 'float': float, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
                'print': print, 'type': type, 'isinstance': isinstance,
                'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
                'all': all, 'any': any, 'round': round, 'pow': pow,
            }
        }
    
    def execute(self, code: str, **kwargs) -> Dict[str, Any]:
        """Execute Python code safely"""
        try:
            # Capture stdout and stderr
            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            # Compile with RestrictedPython for safety
            compiled_code = compile_restricted(code, '<string>', 'exec')
            
            if compiled_code is None:
                return {
                    'success': False,
                    'output': '',
                    'error': 'Code compilation failed - potentially unsafe code detected',
                    'execution_time': 0.0
                }
            
            # Create execution environment
            exec_globals = self.safe_builtins.copy()
            exec_locals = {}
            
            # Execute code with output capture
            import time
            start_time = time.time()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled_code, exec_globals, exec_locals)
            
            execution_time = time.time() - start_time
            
            # Get outputs
            stdout_output = stdout_capture.getvalue()
            stderr_output = stderr_capture.getvalue()
            
            # Truncate output if too long
            if len(stdout_output) > self.max_output_length:
                stdout_output = stdout_output[:self.max_output_length] + "\n... (output truncated)"
            
            return {
                'success': True,
                'output': stdout_output,
                'error': stderr_output if stderr_output else None,
                'execution_time': execution_time,
                'variables': {k: str(v) for k, v in exec_locals.items() if not k.startswith('_')}
            }
            
        except Exception as e:
            return {
                'success': False,
                'output': '',
                'error': f"Execution error: {str(e)}",
                'execution_time': 0.0
            }

class KnowledgeRetriever(Tool):
    """Tool for retrieving relevant knowledge/documentation"""
    
    @property
    def name(self) -> str:
        return "knowledge_retriever"
    
    @property
    def description(self) -> str:
        return "Retrieve relevant documentation or knowledge for coding problems"
    
    def __init__(self):
        # Simple knowledge base - in production this could be a vector database
        self.knowledge_base = {
            'algorithms': {
                'sorting': 'Common sorting algorithms: bubble sort, merge sort, quick sort, heap sort',
                'searching': 'Common searching algorithms: linear search, binary search, hash tables',
                'graph': 'Graph algorithms: DFS, BFS, Dijkstra, A*, topological sort',
                'dynamic_programming': 'DP patterns: memoization, tabulation, optimal substructure'
            },
            'data_structures': {
                'arrays': 'Arrays: fixed size, O(1) access, O(n) search',
                'linked_lists': 'Linked lists: dynamic size, O(n) access, O(1) insertion/deletion',
                'trees': 'Trees: hierarchical structure, BST, AVL, Red-Black trees',
                'graphs': 'Graphs: vertices and edges, directed/undirected, weighted/unweighted'
            },
            'python_stdlib': {
                'collections': 'collections module: Counter, defaultdict, deque, namedtuple',
                'itertools': 'itertools module: permutations, combinations, product, chain',
                'heapq': 'heapq module: heap operations, priority queues',
                'bisect': 'bisect module: binary search utilities'
            }
        }
    
    def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """Retrieve knowledge based on query"""
        query_lower = query.lower()
        relevant_info = []
        
        # Simple keyword matching - could be improved with embeddings
        for category, items in self.knowledge_base.items():
            for topic, info in items.items():
                if any(keyword in query_lower for keyword in [topic, category]):
                    relevant_info.append(f"{category}.{topic}: {info}")
        
        if not relevant_info:
            relevant_info = ["No specific knowledge found for this query."]
        
        return {
            'success': True,
            'knowledge': relevant_info,
            'query': query
        }

class AgentFramework:
    """
    Agent framework implementing Thought-Action-Observation cycles
    """
    
    def __init__(self, groq_client, tools: Optional[List[Tool]] = None):
        """Initialize agent framework"""
        self.groq_client = groq_client
        self.tools = tools or [PythonExecutor(), KnowledgeRetriever()]
        self.tool_map = {tool.name: tool for tool in self.tools}
        
        logger.info(f"Initialized agent framework with {len(self.tools)} tools")
    
    def _parse_response(self, response: str) -> List[AgentStep]:
        """Parse agent response into structured steps"""
        steps = []
        
        # Patterns for different step types
        patterns = {
            StepType.THOUGHT: r'<thinking>(.*?)</thinking>',
            StepType.ACTION: r'<action>(.*?)</action>',
            StepType.OBSERVATION: r'<observation>(.*?)</observation>'
        }
        
        # Find all matches for each pattern
        for step_type, pattern in patterns.items():
            matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
            for match in matches:
                steps.append(AgentStep(
                    step_type=step_type,
                    content=match.strip()
                ))
        
        # If no structured tags found, treat entire response as thought
        if not steps:
            steps.append(AgentStep(
                step_type=StepType.THOUGHT,
                content=response.strip()
            ))
        
        return steps
    
    def _execute_action(self, action_content: str) -> Dict[str, Any]:
        """Execute an action step"""
        # Try to parse action as tool call
        tool_pattern = r'(\w+):\s*(.*)'
        match = re.match(tool_pattern, action_content.strip(), re.DOTALL)
        
        if match:
            tool_name, tool_input = match.groups()
            
            if tool_name in self.tool_map:
                tool = self.tool_map[tool_name]
                return tool.execute(tool_input.strip())
            else:
                return {
                    'success': False,
                    'error': f"Unknown tool: {tool_name}",
                    'available_tools': list(self.tool_map.keys())
                }
        else:
            # Default to Python execution
            python_tool = self.tool_map.get('python_executor')
            if python_tool:
                return python_tool.execute(action_content)
            else:
                return {
                    'success': False,
                    'error': "No Python executor available"
                }
    
    def generate_trajectory(
        self, 
        problem: str, 
        max_steps: int = 10,
        system_prompt: Optional[str] = None
    ) -> AgentTrajectory:
        """Generate complete agent trajectory for a problem"""
        import time
        start_time = time.time()
        
        if system_prompt is None:
            system_prompt = self._get_agent_system_prompt()
        
        trajectory = AgentTrajectory(
            problem=problem,
            steps=[],
            metadata={'tools_used': [], 'step_count': 0}
        )
        
        # Initial conversation context
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {problem}"}
        ]
        
        total_tokens = 0
        
        try:
            for step_num in range(max_steps):
                # Generate response from teacher model
                response = self.groq_client.generate_response(messages)
                total_tokens += response.get('usage', {}).get('total_tokens', 0)
                
                # Parse response into steps
                parsed_steps = self._parse_response(response['content'])
                
                # Process each step
                for step in parsed_steps:
                    trajectory.steps.append(step)
                    
                    # If it's an action step, execute it
                    if step.step_type == StepType.ACTION:
                        execution_result = self._execute_action(step.content)
                        
                        # Create observation step
                        if execution_result['success']:
                            obs_content = f"Execution successful:\n{execution_result.get('output', '')}"
                            if execution_result.get('variables'):
                                obs_content += f"\nVariables: {execution_result['variables']}"
                        else:
                            obs_content = f"Execution failed:\n{execution_result.get('error', 'Unknown error')}"
                        
                        observation = AgentStep(
                            step_type=StepType.OBSERVATION,
                            content=obs_content,
                            metadata=execution_result
                        )
                        trajectory.steps.append(observation)
                        
                        # Add observation to conversation
                        messages.append({
                            "role": "assistant", 
                            "content": response['content']
                        })
                        messages.append({
                            "role": "user", 
                            "content": f"Observation: {obs_content}"
                        })
                        
                        # Track tool usage
                        if 'tools_used' not in trajectory.metadata:
                            trajectory.metadata['tools_used'] = []
                        trajectory.metadata['tools_used'].append(step.content[:50])
                
                # Check if we have a final answer
                if self._has_final_answer(response['content']):
                    trajectory.final_answer = self._extract_final_answer(response['content'])
                    trajectory.success = True
                    break
                
                # If no actions were taken, we might be done
                if not any(s.step_type == StepType.ACTION for s in parsed_steps):
                    # Try to extract final answer
                    trajectory.final_answer = response['content']
                    trajectory.success = True
                    break
        
        except Exception as e:
            logger.error(f"Error generating trajectory: {e}")
            trajectory.metadata['error'] = str(e)
            trajectory.success = False
        
        trajectory.total_tokens = total_tokens
        trajectory.execution_time = time.time() - start_time
        trajectory.metadata['step_count'] = len(trajectory.steps)
        
        return trajectory
    
    def _get_agent_system_prompt(self) -> str:
        """Get system prompt for agent framework"""
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools
        ])
        
        return f"""You are DeepSeek R1, an advanced AI assistant with access to tools for solving coding problems.

Available tools:
{tool_descriptions}

When solving problems, use this structured approach:

1. <thinking>Your reasoning process</thinking>
2. <action>tool_name: input_for_tool</action>
3. <observation>Results will be provided here</observation>

Repeat thinking-action-observation cycles as needed.

For Python code execution, use:
<action>python_executor: your_python_code</action>

For knowledge retrieval, use:
<action>knowledge_retriever: your_query</action>

Always think step by step and use tools to verify your solutions.
Provide a clear final answer when you're confident in your solution."""
    
    def _has_final_answer(self, content: str) -> bool:
        """Check if response contains a final answer"""
        final_indicators = [
            'final answer', 'solution:', 'answer:', 'result:',
            'the answer is', 'therefore', 'in conclusion'
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in final_indicators)
    
    def _extract_final_answer(self, content: str) -> str:
        """Extract final answer from response"""
        # Look for code blocks or explicit answers
        code_pattern = r'```python\n(.*?)\n```'
        code_match = re.search(code_pattern, content, re.DOTALL)
        
        if code_match:
            return code_match.group(1).strip()
        
        # Look for explicit answer sections
        answer_pattern = r'(?:final answer|solution|answer):\s*(.*?)(?:\n\n|\Z)'
        answer_match = re.search(answer_pattern, content, re.IGNORECASE | re.DOTALL)
        
        if answer_match:
            return answer_match.group(1).strip()
        
        # Return last paragraph as fallback
        paragraphs = content.split('\n\n')
        return paragraphs[-1].strip() if paragraphs else content.strip()
    
    def display_trajectory(self, trajectory: AgentTrajectory):
        """Display trajectory in a nice format"""
        console.print(Panel(f"[bold blue]Problem:[/bold blue] {trajectory.problem}"))
        
        for i, step in enumerate(trajectory.steps):
            if step.step_type == StepType.THOUGHT:
                console.print(f"\n[bold yellow]💭 Thought {i+1}:[/bold yellow]")
                console.print(step.content)
            elif step.step_type == StepType.ACTION:
                console.print(f"\n[bold green]🔧 Action {i+1}:[/bold green]")
                console.print(Syntax(step.content, "python", theme="monokai"))
            elif step.step_type == StepType.OBSERVATION:
                console.print(f"\n[bold cyan]👁️ Observation {i+1}:[/bold cyan]")
                console.print(step.content)
        
        if trajectory.final_answer:
            console.print(f"\n[bold magenta]🎯 Final Answer:[/bold magenta]")
            console.print(trajectory.final_answer)
        
        console.print(f"\n[dim]Success: {trajectory.success} | "
                     f"Steps: {len(trajectory.steps)} | "
                     f"Tokens: {trajectory.total_tokens} | "
                     f"Time: {trajectory.execution_time:.2f}s[/dim]")

# Convenience function
def create_agent_framework(groq_client, tools: Optional[List[Tool]] = None) -> AgentFramework:
    """Create and return configured agent framework"""
    return AgentFramework(groq_client, tools)

if __name__ == "__main__":
    # Test the agent framework
    from groq_client import create_groq_client
    
    try:
        # Create client and agent
        client = create_groq_client()
        agent = create_agent_framework(client)
        
        # Test problem
        test_problem = """
        Write a Python function that takes a list of integers and returns the two numbers 
        that add up to a specific target sum. You can assume there's exactly one solution.
        
        Example: nums = [2, 7, 11, 15], target = 9
        Output: [2, 7] (because 2 + 7 = 9)
        """
        
        console.print("[green]Testing agent framework...[/green]")
        trajectory = agent.generate_trajectory(test_problem, max_steps=5)
        
        agent.display_trajectory(trajectory)
        
    except Exception as e:
        console.print(f"[red]Error testing agent framework: {e}[/red]")
        console.print("[yellow]Make sure to set your GROQ_API_KEY in the .env file[/yellow]") 