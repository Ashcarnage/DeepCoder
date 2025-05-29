"""
Data Generation Module for DeepCoder
Handles teacher model integration and trajectory generation.
"""

from .groq_client import GroqClient, GroqConfig, create_groq_client
from .agent_framework import (
    AgentFramework, 
    AgentStep, 
    AgentTrajectory, 
    StepType,
    Tool,
    PythonExecutor,
    KnowledgeRetriever,
    create_agent_framework
)
from .trajectory_generator import (
    TrajectoryGenerator,
    GenerationConfig,
    GenerationStats,
    create_trajectory_generator
)
from .problem_loader import ProblemLoader, Problem

__all__ = [
    # Groq client
    'GroqClient',
    'GroqConfig', 
    'create_groq_client',
    
    # Agent framework
    'AgentFramework',
    'AgentStep',
    'AgentTrajectory',
    'StepType',
    'Tool',
    'PythonExecutor',
    'KnowledgeRetriever',
    'create_agent_framework',
    
    # Trajectory generation
    'TrajectoryGenerator',
    'GenerationConfig',
    'GenerationStats',
    'create_trajectory_generator',
    
    # Problem loading
    'ProblemLoader',
    'Problem'
] 