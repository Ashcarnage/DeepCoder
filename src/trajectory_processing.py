"""
Trajectory Processing for Structured Agent Distillation (SAD)

This module implements Phase 2.1 of the DeepCoder pipeline, processing teacher
trajectories for student model training using Structured Agent Distillation.

Key components:
1. Trajectory parsing and tokenization for SGLang/Qwen3
2. Span identification (reasoning vs action)
3. Token alignment and sequence preparation for SAD loss
4. Support for thinking mode and agent-style responses
"""

import os
import json
import logging
import re
import yaml
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import torch
from transformers import AutoTokenizer
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel

logger = logging.getLogger(__name__)
console = Console()


class SpanType(Enum):
    """Types of spans in agent trajectories"""
    REASONING = "reasoning"  # [REASON] spans - thinking, analysis, planning
    ACTION = "action"       # [ACT] spans - tool calls, code execution, decisions
    OBSERVATION = "observation"  # Observation results from actions
    OTHER = "other"         # Non-classified content


@dataclass
class TokenSpan:
    """Represents a tokenized span with type and position information"""
    span_type: SpanType
    start_token: int
    end_token: int  # exclusive
    text: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return self.end_token - self.start_token
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'span_type': self.span_type.value,
            'start_token': self.start_token,
            'end_token': self.end_token,
            'text': self.text,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


@dataclass
class ProcessedTrajectory:
    """A trajectory processed for SAD training"""
    trajectory_id: str
    input_ids: List[int]
    attention_mask: List[int]
    reasoning_mask: List[int]  # 1 for reasoning tokens, 0 otherwise
    action_mask: List[int]     # 1 for action tokens, 0 otherwise
    spans: List[TokenSpan]
    original_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.input_ids)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trajectory_id': self.trajectory_id,
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'reasoning_mask': self.reasoning_mask,
            'action_mask': self.action_mask,
            'spans': [span.to_dict() for span in self.spans],
            'original_text': self.original_text,
            'metadata': self.metadata
        }


@dataclass
class TrajectoryProcessingConfig:
    """Configuration for trajectory processing"""
    # Model and tokenization
    model_name: str = "Qwen/Qwen3-30B-A3B"
    max_length: int = 32768
    context_length: int = 32768
    
    # Span detection patterns
    thinking_patterns: List[str] = field(default_factory=lambda: [
        r'<think>(.*?)</think>',
        r'<thinking>(.*?)</thinking>',
        r'Let me think(.*?)(?=\n\n|\n[A-Z]|$)',
        r'I need to(.*?)(?=\n\n|\n[A-Z]|$)',
        r'First, I(.*?)(?=\n\n|\n[A-Z]|$)',
    ])
    
    action_patterns: List[str] = field(default_factory=lambda: [
        r'execute_python\((.*?)\)',
        r'```python(.*?)```',
        r'```(.*?)```',
        r'Action:(.*?)(?=\n\n|Observation:|$)',
        r'I will(.*?)(?=\n\n|\n[A-Z]|$)',
        r'Let me(.*?)(?=\n\n|\n[A-Z]|$)',
    ])
    
    observation_patterns: List[str] = field(default_factory=lambda: [
        r'Observation:(.*?)(?=\n\n|Thought:|Action:|$)',
        r'Result:(.*?)(?=\n\n|\n[A-Z]|$)',
        r'Output:(.*?)(?=\n\n|\n[A-Z]|$)',
    ])
    
    # Processing settings
    enable_span_overlap: bool = True
    confidence_threshold: float = 0.7
    max_span_length: int = 1024
    padding: str = "max_length"
    truncation: bool = True
    
    # Quality filtering
    min_reasoning_ratio: float = 0.1  # At least 10% reasoning tokens
    min_action_ratio: float = 0.05    # At least 5% action tokens
    max_other_ratio: float = 0.8      # At most 80% unclassified tokens


class TrajectoryParser:
    """Parses raw trajectories and extracts structured information"""
    
    def __init__(self, config: TrajectoryProcessingConfig):
        self.config = config
        self.thinking_regexes = [re.compile(p, re.DOTALL | re.IGNORECASE) 
                                for p in config.thinking_patterns]
        self.action_regexes = [re.compile(p, re.DOTALL | re.IGNORECASE) 
                              for p in config.action_patterns]
        self.observation_regexes = [re.compile(p, re.DOTALL | re.IGNORECASE) 
                                   for p in config.observation_patterns]
    
    def parse_trajectory(self, trajectory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a raw trajectory into structured format"""
        try:
            # Extract text content from trajectory
            text_content = self._extract_text_content(trajectory_data)
            
            # Detect agent-style segments
            segments = self._detect_agent_segments(text_content)
            
            # Extract thinking content if present
            thinking_content = self._extract_thinking_content(text_content)
            
            return {
                'text_content': text_content,
                'segments': segments,
                'thinking_content': thinking_content,
                'has_thinking': len(thinking_content) > 0,
                'has_actions': any(seg['type'] == 'action' for seg in segments),
                'quality_score': self._calculate_quality_score(segments, thinking_content)
            }
            
        except Exception as e:
            logger.error(f"Error parsing trajectory: {e}")
            return {
                'text_content': '',
                'segments': [],
                'thinking_content': '',
                'has_thinking': False,
                'has_actions': False,
                'quality_score': 0.0,
                'error': str(e)
            }
    
    def _extract_text_content(self, trajectory_data: Dict[str, Any]) -> str:
        """Extract text content from trajectory data structure"""
        content_parts = []
        
        # Handle different trajectory formats
        if 'trajectory' in trajectory_data:
            traj = trajectory_data['trajectory']
            
            # Extract from steps
            if 'steps' in traj:
                for step in traj['steps']:
                    if isinstance(step, dict) and 'content' in step:
                        content_parts.append(step['content'])
            
            # Extract final answer
            if 'final_answer' in traj and traj['final_answer']:
                content_parts.append(traj['final_answer'])
        
        # Handle direct content
        elif 'content' in trajectory_data:
            content_parts.append(trajectory_data['content'])
        
        # Handle text field
        elif 'text' in trajectory_data:
            content_parts.append(trajectory_data['text'])
        
        return '\n\n'.join(content_parts)
    
    def _detect_agent_segments(self, text: str) -> List[Dict[str, Any]]:
        """Detect agent-style segments (thought, action, observation)"""
        segments = []
        
        # Split by common agent delimiters
        agent_splits = re.split(r'\n(?=(?:Thought:|Action:|Observation:))', text)
        
        for segment in agent_splits:
            segment = segment.strip()
            if not segment:
                continue
            
            segment_type = 'other'
            confidence = 0.5
            
            # Classify segment type
            if segment.lower().startswith(('thought:', 'thinking:')):
                segment_type = 'reasoning'
                confidence = 0.9
            elif segment.lower().startswith('action:'):
                segment_type = 'action'
                confidence = 0.9
            elif segment.lower().startswith('observation:'):
                segment_type = 'observation'
                confidence = 0.9
            else:
                # Use pattern matching for classification
                segment_type, confidence = self._classify_segment(segment)
            
            segments.append({
                'type': segment_type,
                'text': segment,
                'confidence': confidence,
                'length': len(segment)
            })
        
        return segments
    
    def _classify_segment(self, text: str) -> Tuple[str, float]:
        """Classify a text segment using pattern matching"""
        max_confidence = 0.0
        best_type = 'other'
        
        # Check reasoning patterns
        for regex in self.thinking_regexes:
            if regex.search(text):
                confidence = 0.8
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_type = 'reasoning'
        
        # Check action patterns
        for regex in self.action_regexes:
            if regex.search(text):
                confidence = 0.8
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_type = 'action'
        
        # Check observation patterns
        for regex in self.observation_regexes:
            if regex.search(text):
                confidence = 0.8
                if confidence > max_confidence:
                    max_confidence = confidence
                    best_type = 'observation'
        
        return best_type, max_confidence
    
    def _extract_thinking_content(self, text: str) -> str:
        """Extract thinking/reasoning content from <think> tags and similar"""
        thinking_parts = []
        
        # Extract from <think> tags
        think_matches = re.findall(r'<think>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
        thinking_parts.extend(think_matches)
        
        # Extract from <thinking> tags  
        thinking_matches = re.findall(r'<thinking>(.*?)</thinking>', text, re.DOTALL | re.IGNORECASE)
        thinking_parts.extend(thinking_matches)
        
        return '\n\n'.join(thinking_parts).strip()
    
    def _calculate_quality_score(self, segments: List[Dict[str, Any]], thinking_content: str) -> float:
        """Calculate a quality score for the trajectory"""
        if not segments:
            return 0.0
        
        # Count segment types
        reasoning_count = sum(1 for s in segments if s['type'] == 'reasoning')
        action_count = sum(1 for s in segments if s['type'] == 'action')
        total_segments = len(segments)
        
        # Base score from segment diversity
        diversity_score = min(1.0, (reasoning_count + action_count) / total_segments)
        
        # Bonus for thinking content
        thinking_bonus = 0.2 if thinking_content else 0.0
        
        # Bonus for balanced reasoning/action
        if reasoning_count > 0 and action_count > 0:
            balance_bonus = 0.3
        else:
            balance_bonus = 0.0
        
        return min(1.0, diversity_score + thinking_bonus + balance_bonus)


class SpanDetector:
    """Detects and classifies spans in tokenized text"""
    
    def __init__(self, config: TrajectoryProcessingConfig, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.parser = TrajectoryParser(config)
    
    def detect_spans(self, text: str, input_ids: List[int]) -> List[TokenSpan]:
        """Detect spans in tokenized text"""
        # Parse text into segments
        parsed = self.parser.parse_trajectory({'text': text})
        segments = parsed['segments']
        
        if not segments:
            # Fallback: classify entire text as reasoning
            return [TokenSpan(
                span_type=SpanType.REASONING,
                start_token=0,
                end_token=len(input_ids),
                text=text,
                confidence=0.5
            )]
        
        spans = []
        current_pos = 0
        
        for segment in segments:
            segment_text = segment['text']
            segment_type_str = segment['type']
            confidence = segment['confidence']
            
            # Convert to SpanType
            if segment_type_str == 'reasoning':
                span_type = SpanType.REASONING
            elif segment_type_str == 'action':
                span_type = SpanType.ACTION
            elif segment_type_str == 'observation':
                span_type = SpanType.OBSERVATION
            else:
                span_type = SpanType.OTHER
            
            # Find segment position in full text
            segment_start = text.find(segment_text, current_pos)
            if segment_start == -1:
                logger.warning(f"Could not find segment in text: {segment_text[:50]}...")
                continue
            
            segment_end = segment_start + len(segment_text)
            
            # Convert character positions to token positions
            token_start = self._char_to_token_position(text, segment_start, input_ids)
            token_end = self._char_to_token_position(text, segment_end, input_ids)
            
            if token_start is not None and token_end is not None and token_end > token_start:
                spans.append(TokenSpan(
                    span_type=span_type,
                    start_token=token_start,
                    end_token=token_end,
                    text=segment_text,
                    confidence=confidence,
                    metadata={'segment_index': len(spans)}
                ))
            
            current_pos = segment_end
        
        # Fill gaps with OTHER spans if needed
        spans = self._fill_span_gaps(spans, len(input_ids), text)
        
        # Sort spans by start position
        spans.sort(key=lambda s: s.start_token)
        
        return spans
    
    def _char_to_token_position(self, text: str, char_pos: int, input_ids: List[int]) -> Optional[int]:
        """Convert character position to token position"""
        try:
            # Tokenize prefix to find position
            prefix = text[:char_pos]
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            return len(prefix_tokens)
        except Exception as e:
            logger.warning(f"Error converting char to token position: {e}")
            return None
    
    def _fill_span_gaps(self, spans: List[TokenSpan], total_tokens: int, text: str) -> List[TokenSpan]:
        """Fill gaps between spans with OTHER spans"""
        if not spans:
            return [TokenSpan(
                span_type=SpanType.OTHER,
                start_token=0,
                end_token=total_tokens,
                text=text,
                confidence=0.5
            )]
        
        filled_spans = []
        current_pos = 0
        
        for span in sorted(spans, key=lambda s: s.start_token):
            # Fill gap before span
            if span.start_token > current_pos:
                filled_spans.append(TokenSpan(
                    span_type=SpanType.OTHER,
                    start_token=current_pos,
                    end_token=span.start_token,
                    text="",  # Will be filled if needed
                    confidence=0.3
                ))
            
            filled_spans.append(span)
            current_pos = span.end_token
        
        # Fill final gap
        if current_pos < total_tokens:
            filled_spans.append(TokenSpan(
                span_type=SpanType.OTHER,
                start_token=current_pos,
                end_token=total_tokens,
                text="",
                confidence=0.3
            ))
        
        return filled_spans


class TrajectoryTokenizer:
    """Tokenizes trajectories for SAD training"""
    
    def __init__(self, config: TrajectoryProcessingConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.span_detector = SpanDetector(config, self.tokenizer)
        
        logger.info(f"Initialized tokenizer for {config.model_name}")
        logger.info(f"Vocab size: {self.tokenizer.vocab_size}")
        logger.info(f"Max length: {config.max_length}")
    
    def tokenize_trajectory(self, trajectory_data: Dict[str, Any]) -> Optional[ProcessedTrajectory]:
        """Tokenize a trajectory for SAD training"""
        try:
            # Parse trajectory
            parser = TrajectoryParser(self.config)
            parsed = parser.parse_trajectory(trajectory_data)
            
            if parsed.get('error'):
                logger.warning(f"Parsing error: {parsed['error']}")
                return None
            
            text_content = parsed['text_content']
            if not text_content.strip():
                logger.warning("Empty text content in trajectory")
                return None
            
            # Tokenize text
            encoding = self.tokenizer(
                text_content,
                max_length=self.config.max_length,
                padding=self.config.padding,
                truncation=self.config.truncation,
                return_tensors="pt"
            )
            
            input_ids = encoding['input_ids'][0].tolist()
            attention_mask = encoding['attention_mask'][0].tolist()
            
            # Detect spans
            spans = self.span_detector.detect_spans(text_content, input_ids)
            
            # Create span masks
            reasoning_mask = [0] * len(input_ids)
            action_mask = [0] * len(input_ids)
            
            for span in spans:
                if span.span_type == SpanType.REASONING:
                    for i in range(span.start_token, min(span.end_token, len(input_ids))):
                        reasoning_mask[i] = 1
                elif span.span_type == SpanType.ACTION:
                    for i in range(span.start_token, min(span.end_token, len(input_ids))):
                        action_mask[i] = 1
            
            # Quality check
            if not self._passes_quality_check(reasoning_mask, action_mask):
                logger.warning("Trajectory failed quality check")
                return None
            
            # Create processed trajectory
            trajectory_id = trajectory_data.get('id', f"traj_{hash(text_content) % 1000000}")
            
            processed = ProcessedTrajectory(
                trajectory_id=str(trajectory_id),
                input_ids=input_ids,
                attention_mask=attention_mask,
                reasoning_mask=reasoning_mask,
                action_mask=action_mask,
                spans=spans,
                original_text=text_content,
                metadata={
                    'num_reasoning_tokens': sum(reasoning_mask),
                    'num_action_tokens': sum(action_mask),
                    'num_spans': len(spans),
                    'quality_score': parsed.get('quality_score', 0.0),
                    'has_thinking': parsed.get('has_thinking', False),
                    'text_length': len(text_content),
                    'token_length': len(input_ids)
                }
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Error tokenizing trajectory: {e}")
            return None
    
    def _passes_quality_check(self, reasoning_mask: List[int], action_mask: List[int]) -> bool:
        """Check if trajectory meets quality thresholds"""
        total_tokens = len(reasoning_mask)
        reasoning_ratio = sum(reasoning_mask) / total_tokens
        action_ratio = sum(action_mask) / total_tokens
        other_ratio = 1 - reasoning_ratio - action_ratio
        
        return (reasoning_ratio >= self.config.min_reasoning_ratio and
                action_ratio >= self.config.min_action_ratio and
                other_ratio <= self.config.max_other_ratio)


class TrajectoryProcessor:
    """Main processor for converting trajectories to SAD training format"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize trajectory processor"""
        self.config_path = config_path
        self.config = self._load_config()
        self.tokenizer = TrajectoryTokenizer(self.config)
        
        # Create output directories
        self.output_dir = Path(self.config.output_dir if hasattr(self.config, 'output_dir') 
                              else "data/processed_trajectories")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trajectory processor")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _load_config(self) -> TrajectoryProcessingConfig:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Extract trajectory processing config
            processing_config = yaml_config.get('trajectory_processing', {})
            
            return TrajectoryProcessingConfig(
                model_name=processing_config.get('model_name', 'Qwen/Qwen3-30B-A3B'),
                max_length=processing_config.get('max_length', 32768),
                context_length=processing_config.get('context_length', 32768),
                confidence_threshold=processing_config.get('confidence_threshold', 0.7),
                min_reasoning_ratio=processing_config.get('min_reasoning_ratio', 0.1),
                min_action_ratio=processing_config.get('min_action_ratio', 0.05),
                max_other_ratio=processing_config.get('max_other_ratio', 0.8)
            )
            
        except Exception as e:
            logger.warning(f"Error loading config: {e}, using defaults")
            return TrajectoryProcessingConfig()
    
    def process_trajectory_file(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """Process a single trajectory file"""
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if output_file is None:
            output_file = self.output_dir / f"processed_{input_path.stem}.jsonl"
        
        console.print(f"[yellow]Processing trajectory file: {input_file}[/yellow]")
        
        stats = {
            'total_trajectories': 0,
            'processed_successfully': 0,
            'failed_processing': 0,
            'failed_quality_check': 0,
            'total_tokens': 0,
            'total_reasoning_tokens': 0,
            'total_action_tokens': 0
        }
        
        # Process trajectories
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line_num, line in enumerate(infile, 1):
                try:
                    trajectory_data = json.loads(line.strip())
                    stats['total_trajectories'] += 1
                    
                    # Process trajectory
                    processed = self.tokenizer.tokenize_trajectory(trajectory_data)
                    
                    if processed is not None:
                        # Write processed trajectory
                        outfile.write(json.dumps(processed.to_dict()) + '\n')
                        
                        # Update stats
                        stats['processed_successfully'] += 1
                        stats['total_tokens'] += len(processed.input_ids)
                        stats['total_reasoning_tokens'] += sum(processed.reasoning_mask)
                        stats['total_action_tokens'] += sum(processed.action_mask)
                    else:
                        stats['failed_processing'] += 1
                        
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    stats['failed_processing'] += 1
                    continue
        
        # Calculate additional stats
        if stats['processed_successfully'] > 0:
            stats['avg_tokens_per_trajectory'] = stats['total_tokens'] / stats['processed_successfully']
            stats['reasoning_token_ratio'] = stats['total_reasoning_tokens'] / stats['total_tokens']
            stats['action_token_ratio'] = stats['total_action_tokens'] / stats['total_tokens']
        
        console.print(f"[green]Processing complete: {output_file}[/green]")
        self._display_processing_stats(stats)
        
        return stats
    
    def process_trajectory_directory(self, input_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process all trajectory files in a directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        if output_dir is None:
            output_dir = self.output_dir / "batch_processed"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all trajectory files
        trajectory_files = list(input_path.glob("*.jsonl"))
        
        if not trajectory_files:
            console.print(f"[yellow]No .jsonl files found in {input_dir}[/yellow]")
            return {}
        
        console.print(f"[yellow]Found {len(trajectory_files)} trajectory files to process[/yellow]")
        
        # Process all files
        combined_stats = {
            'total_trajectories': 0,
            'processed_successfully': 0,
            'failed_processing': 0,
            'failed_quality_check': 0,
            'total_tokens': 0,
            'total_reasoning_tokens': 0,
            'total_action_tokens': 0,
            'files_processed': 0
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Processing trajectory files...", total=len(trajectory_files))
            
            for traj_file in trajectory_files:
                try:
                    output_file = output_path / f"processed_{traj_file.stem}.jsonl"
                    file_stats = self.process_trajectory_file(str(traj_file), str(output_file))
                    
                    # Combine stats
                    for key in combined_stats:
                        if key != 'files_processed':
                            combined_stats[key] += file_stats.get(key, 0)
                    
                    combined_stats['files_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"Error processing file {traj_file}: {e}")
                
                progress.update(task, advance=1)
        
        # Calculate final stats
        if combined_stats['processed_successfully'] > 0:
            combined_stats['avg_tokens_per_trajectory'] = (
                combined_stats['total_tokens'] / combined_stats['processed_successfully']
            )
            combined_stats['reasoning_token_ratio'] = (
                combined_stats['total_reasoning_tokens'] / combined_stats['total_tokens']
            )
            combined_stats['action_token_ratio'] = (
                combined_stats['total_action_tokens'] / combined_stats['total_tokens']
            )
        
        console.print(f"[green]Batch processing complete![/green]")
        self._display_processing_stats(combined_stats)
        
        return combined_stats
    
    def _display_processing_stats(self, stats: Dict[str, Any]):
        """Display processing statistics"""
        table = Table(title="Trajectory Processing Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Percentage", style="green")
        
        total = stats.get('total_trajectories', 0)
        
        table.add_row(
            "Total Trajectories",
            str(total),
            "100.0%"
        )
        
        table.add_row(
            "Successfully Processed",
            str(stats.get('processed_successfully', 0)),
            f"{stats.get('processed_successfully', 0)/max(1, total)*100:.1f}%"
        )
        
        table.add_row(
            "Failed Processing",
            str(stats.get('failed_processing', 0)),
            f"{stats.get('failed_processing', 0)/max(1, total)*100:.1f}%"
        )
        
        if 'avg_tokens_per_trajectory' in stats:
            table.add_row(
                "Avg Tokens/Trajectory",
                f"{stats['avg_tokens_per_trajectory']:.1f}",
                "-"
            )
        
        if 'reasoning_token_ratio' in stats:
            table.add_row(
                "Reasoning Token Ratio",
                f"{stats['reasoning_token_ratio']:.3f}",
                f"{stats['reasoning_token_ratio']*100:.1f}%"
            )
        
        if 'action_token_ratio' in stats:
            table.add_row(
                "Action Token Ratio",
                f"{stats['action_token_ratio']:.3f}",
                f"{stats['action_token_ratio']*100:.1f}%"
            )
        
        console.print(table)
    
    def validate_processed_trajectories(self, file_path: str) -> Dict[str, Any]:
        """Validate processed trajectory format"""
        validation_results = {
            'valid_trajectories': 0,
            'invalid_trajectories': 0,
            'validation_errors': [],
            'total_checked': 0
        }
        
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    validation_results['total_checked'] += 1
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Check required fields
                        required_fields = [
                            'trajectory_id', 'input_ids', 'attention_mask',
                            'reasoning_mask', 'action_mask', 'spans', 'original_text'
                        ]
                        
                        for field in required_fields:
                            if field not in data:
                                raise ValueError(f"Missing required field: {field}")
                        
                        # Check array lengths match
                        length = len(data['input_ids'])
                        if (len(data['attention_mask']) != length or
                            len(data['reasoning_mask']) != length or
                            len(data['action_mask']) != length):
                            raise ValueError("Inconsistent array lengths")
                        
                        # Check span validity
                        for i, span in enumerate(data['spans']):
                            if (span['start_token'] < 0 or 
                                span['end_token'] > length or
                                span['start_token'] >= span['end_token']):
                                raise ValueError(f"Invalid span {i}: {span}")
                        
                        validation_results['valid_trajectories'] += 1
                        
                    except Exception as e:
                        validation_results['invalid_trajectories'] += 1
                        validation_results['validation_errors'].append(
                            f"Line {line_num}: {str(e)}"
                        )
        
        except FileNotFoundError:
            console.print(f"[red]File not found: {file_path}[/red]")
            return validation_results
        
        console.print(f"[green]Validation complete: {validation_results['valid_trajectories']} valid, "
                     f"{validation_results['invalid_trajectories']} invalid[/green]")
        
        return validation_results


# Factory functions
def create_trajectory_processor(config_path: str = "configs/config.yaml") -> TrajectoryProcessor:
    """Create trajectory processor instance"""
    return TrajectoryProcessor(config_path)


def create_trajectory_processing_config(**kwargs) -> TrajectoryProcessingConfig:
    """Create trajectory processing configuration"""
    return TrajectoryProcessingConfig(**kwargs)


if __name__ == "__main__":
    # CLI interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Trajectory Processing for SAD")
    parser.add_argument("command", choices=["process", "validate", "test"],
                       help="Command to execute")
    parser.add_argument("--input", required=True,
                       help="Input file or directory")
    parser.add_argument("--output", 
                       help="Output file or directory")
    parser.add_argument("--config", default="configs/config.yaml",
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    if args.command == "process":
        processor = create_trajectory_processor(args.config)
        
        input_path = Path(args.input)
        if input_path.is_file():
            stats = processor.process_trajectory_file(args.input, args.output)
        elif input_path.is_dir():
            stats = processor.process_trajectory_directory(args.input, args.output)
        else:
            console.print(f"[red]Invalid input path: {args.input}[/red]")
            exit(1)
        
        console.print(f"[green]Processing complete! Stats: {stats}[/green]")
    
    elif args.command == "validate":
        processor = create_trajectory_processor(args.config)
        results = processor.validate_processed_trajectories(args.input)
        console.print(f"[green]Validation results: {results}[/green]")
    
    elif args.command == "test":
        # Quick test functionality
        processor = create_trajectory_processor(args.config)
        console.print("[green]Trajectory processor initialized successfully![/green]") 