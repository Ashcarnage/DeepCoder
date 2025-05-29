# Phase 2.2: Structured Agent Distillation (SAD) Loss Implementation
# Advanced loss computation system with multiple weighting strategies and optimizations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
import math
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
from contextlib import contextmanager

# Import trajectory processing components
from trajectory_processing import TrajectoryProcessor, SpanType, ProcessedTrajectory

logger = logging.getLogger(__name__)

class LossType(Enum):
    """Types of loss computation strategies for SAD"""
    STANDARD_KL = "standard_kl"  # Classic KL divergence
    WASSERSTEIN = "wasserstein"  # Wasserstein distance based
    FOCAL_KL = "focal_kl"  # Focal loss weighted KL divergence
    ADAPTIVE_KL = "adaptive_kl"  # Adaptive temperature KL divergence
    SYMMETRIC_KL = "symmetric_kl"  # Symmetric KL divergence
    ALPHA_DIVERGENCE = "alpha_divergence"  # Alpha divergence family
    
class WeightingStrategy(Enum):
    """Span weighting strategies for reasoning vs action tokens"""
    UNIFORM = "uniform"  # Equal weights
    REASONING_HEAVY = "reasoning_heavy"  # Higher weight for reasoning
    ACTION_HEAVY = "action_heavy"  # Higher weight for actions
    ADAPTIVE = "adaptive"  # Dynamic adaptation based on performance
    CURRICULUM = "curriculum"  # Curriculum learning progression
    CONFIDENCE_BASED = "confidence_based"  # Based on teacher confidence
    DIFFICULTY_AWARE = "difficulty_aware"  # Based on token difficulty

@dataclass
class SADLossConfig:
    """Configuration for SAD loss computation"""
    # Core loss parameters
    loss_type: LossType = LossType.ADAPTIVE_KL
    weighting_strategy: WeightingStrategy = WeightingStrategy.ADAPTIVE
    temperature: float = 3.0
    min_temperature: float = 1.0
    max_temperature: float = 10.0
    
    # Span-specific weights
    reasoning_weight: float = 2.0
    action_weight: float = 1.5
    observation_weight: float = 1.0
    other_weight: float = 0.8
    
    # Advanced loss parameters
    alpha_divergence_alpha: float = 1.5  # For alpha divergence
    focal_gamma: float = 2.0  # For focal loss
    label_smoothing: float = 0.1
    gradient_clip_value: float = 5.0
    
    # Optimization features
    use_mixed_precision: bool = True
    use_gradient_accumulation: bool = True
    accumulation_steps: int = 4
    use_span_balancing: bool = True
    
    # Regularization
    entropy_regularization: float = 0.01
    confidence_penalty: float = 0.05
    span_consistency_weight: float = 0.1
    
    # Adaptive features
    adaptive_temperature: bool = True
    adaptive_weights: bool = True
    curriculum_learning: bool = True
    difficulty_estimation: bool = True
    
    # Numerical stability
    epsilon: float = 1e-8
    max_logit_value: float = 50.0
    numerical_stability: bool = True
    
    # Performance monitoring
    compute_span_metrics: bool = True
    track_convergence: bool = True
    log_detailed_metrics: bool = False

class SpanWeightComputer:
    """Advanced span weight computation with multiple strategies"""
    
    def __init__(self, config: SADLossConfig):
        self.config = config
        self.strategy = config.weighting_strategy
        self.step_count = 0
        self.performance_history = []
        self.weight_history = []
        
    def compute_weights(self, 
                       spans: List[Tuple[int, int, SpanType]], 
                       teacher_probs: torch.Tensor,
                       student_probs: torch.Tensor,
                       step: int = None) -> torch.Tensor:
        """Compute weights for each token based on span type and strategy"""
        
        if step is not None:
            self.step_count = step
            
        seq_len = teacher_probs.size(0)  # Fix: get sequence length from first dimension
        weights = torch.ones(seq_len, dtype=torch.float32, device=teacher_probs.device)
        
        # Apply base span weights
        base_weights = self._get_base_weights()
        for start, end, span_type in spans:
            if start < seq_len and end <= seq_len:
                # Only apply span weights if not using uniform strategy
                if self.strategy != WeightingStrategy.UNIFORM:
                    weights[start:end] = base_weights[span_type.value]
        
        # Apply strategy-specific modifications
        if self.strategy == WeightingStrategy.UNIFORM:
            # For uniform strategy, keep weights as 1.0 (no modification needed)
            pass
        elif self.strategy == WeightingStrategy.ADAPTIVE:
            weights = self._apply_adaptive_weighting(weights, spans, teacher_probs, student_probs)
        elif self.strategy == WeightingStrategy.CURRICULUM:
            weights = self._apply_curriculum_weighting(weights, spans)
        elif self.strategy == WeightingStrategy.CONFIDENCE_BASED:
            weights = self._apply_confidence_weighting(weights, teacher_probs)
        elif self.strategy == WeightingStrategy.DIFFICULTY_AWARE:
            weights = self._apply_difficulty_weighting(weights, teacher_probs, student_probs)
            
        # Normalize weights to maintain loss scale
        weights = weights / weights.mean()
        
        return weights
    
    def _get_base_weights(self) -> Dict[str, float]:
        """Get base weights for each span type"""
        return {
            SpanType.REASONING.value: self.config.reasoning_weight,
            SpanType.ACTION.value: self.config.action_weight,
            SpanType.OBSERVATION.value: self.config.observation_weight,
            SpanType.OTHER.value: self.config.other_weight
        }
    
    def _apply_adaptive_weighting(self, weights: torch.Tensor, spans: List, 
                                teacher_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
        """Apply adaptive weighting based on current performance"""
        
        # Compute KL divergence for each span type
        span_kl = {}
        for start, end, span_type in spans:
            if start < len(weights) and end <= len(weights):
                span_teacher = teacher_probs[start:end]
                span_student = student_probs[start:end]
                kl = F.kl_div(torch.log(span_student + self.config.epsilon), 
                             span_teacher, reduction='mean')
                span_kl[span_type.value] = kl.item()
        
        # Adjust weights based on relative performance
        if span_kl:
            max_kl = max(span_kl.values())
            for start, end, span_type in spans:
                if start < len(weights) and end <= len(weights):
                    relative_difficulty = span_kl.get(span_type.value, max_kl) / max_kl
                    adaptation_factor = 1.0 + 0.5 * relative_difficulty
                    weights[start:end] *= adaptation_factor
        
        return weights
    
    def _apply_curriculum_weighting(self, weights: torch.Tensor, spans: List) -> torch.Tensor:
        """Apply curriculum learning progression"""
        
        if not self.config.curriculum_learning:
            return weights
            
        # Progress from simple (actions) to complex (reasoning)
        progress = min(1.0, self.step_count / 10000)  # Full curriculum in 10k steps
        
        for start, end, span_type in spans:
            if start < len(weights) and end <= len(weights):
                if span_type == SpanType.REASONING:
                    # Gradually increase reasoning weight
                    curriculum_weight = 0.5 + 1.5 * progress
                    weights[start:end] *= curriculum_weight
                elif span_type == SpanType.ACTION:
                    # Start with high action weight, gradually decrease
                    curriculum_weight = 2.0 - 1.0 * progress
                    weights[start:end] *= curriculum_weight
        
        return weights
    
    def _apply_confidence_weighting(self, weights: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        """Weight based on teacher confidence (entropy)"""
        
        # Compute entropy for each position
        entropy = -torch.sum(teacher_probs * torch.log(teacher_probs + self.config.epsilon), dim=-1)
        
        # Higher weight for lower entropy (higher confidence)
        confidence_weights = 1.0 / (1.0 + entropy)
        confidence_weights = confidence_weights / confidence_weights.mean()
        
        # Ensure dimensions match
        if confidence_weights.size(0) != weights.size(0):
            confidence_weights = confidence_weights[:weights.size(0)]
        
        return weights * confidence_weights
    
    def _apply_difficulty_weighting(self, weights: torch.Tensor, 
                                  teacher_probs: torch.Tensor, student_probs: torch.Tensor) -> torch.Tensor:
        """Weight based on prediction difficulty"""
        
        # Compute prediction agreement
        teacher_pred = torch.argmax(teacher_probs, dim=-1)
        student_pred = torch.argmax(student_probs, dim=-1)
        agreement = (teacher_pred == student_pred).float()
        
        # Higher weight for disagreement (difficulty)
        difficulty_weights = 2.0 - agreement
        
        return weights * difficulty_weights

class AdvancedLossComputer:
    """Advanced loss computation with multiple divergence measures"""
    
    def __init__(self, config: SADLossConfig):
        self.config = config
        self.loss_type = config.loss_type
        
    def compute_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                    weights: torch.Tensor, temperature: float = None) -> torch.Tensor:
        """Compute loss based on configured strategy"""
        
        if temperature is None:
            temperature = self.config.temperature
            
        if self.loss_type == LossType.STANDARD_KL:
            return self._compute_kl_loss(teacher_logits, student_logits, weights, temperature)
        elif self.loss_type == LossType.WASSERSTEIN:
            return self._compute_wasserstein_loss(teacher_logits, student_logits, weights, temperature)
        elif self.loss_type == LossType.FOCAL_KL:
            return self._compute_focal_kl_loss(teacher_logits, student_logits, weights, temperature)
        elif self.loss_type == LossType.ADAPTIVE_KL:
            return self._compute_adaptive_kl_loss(teacher_logits, student_logits, weights, temperature)
        elif self.loss_type == LossType.SYMMETRIC_KL:
            return self._compute_symmetric_kl_loss(teacher_logits, student_logits, weights, temperature)
        elif self.loss_type == LossType.ALPHA_DIVERGENCE:
            return self._compute_alpha_divergence_loss(teacher_logits, student_logits, weights, temperature)
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
    
    def _compute_kl_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                        weights: torch.Tensor, temperature: float) -> torch.Tensor:
        """Standard KL divergence loss with improvements"""
        
        # Apply temperature scaling
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # Apply label smoothing
        if self.config.label_smoothing > 0:
            teacher_probs = self._apply_label_smoothing(teacher_probs)
        
        # Compute KL divergence
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        kl_loss = torch.sum(kl_loss, dim=-1)  # Sum over vocabulary
        
        # Apply weights and reduce
        weighted_loss = kl_loss * weights
        
        return weighted_loss.mean()
    
    def _compute_wasserstein_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                                 weights: torch.Tensor, temperature: float) -> torch.Tensor:
        """Wasserstein distance based loss (approximated via Sinkhorn)"""
        
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        
        # Simplified Wasserstein distance approximation
        # Full Sinkhorn algorithm would be more accurate but computationally expensive
        
        # Compute cost matrix (using vocabulary index distance as proxy)
        vocab_size = teacher_probs.size(-1)
        indices = torch.arange(vocab_size, device=teacher_probs.device).float()
        cost_matrix = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))
        cost_matrix = cost_matrix / cost_matrix.max()  # Normalize
        
        # Approximate Wasserstein distance
        wasserstein_loss = torch.sum(teacher_probs.unsqueeze(-1) * student_probs.unsqueeze(-2) * 
                                   cost_matrix.unsqueeze(0), dim=(-2, -1))
        
        # Apply weights
        weighted_loss = wasserstein_loss * weights
        
        return weighted_loss.mean()
    
    def _compute_focal_kl_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                              weights: torch.Tensor, temperature: float) -> torch.Tensor:
        """Focal loss weighted KL divergence"""
        
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # Compute base KL divergence
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='none')
        kl_loss = torch.sum(kl_loss, dim=-1)
        
        # Compute focal weight based on prediction confidence
        max_student_prob = torch.max(student_probs, dim=-1)[0]
        focal_weight = (1 - max_student_prob) ** self.config.focal_gamma
        
        # Apply focal weighting
        focal_kl_loss = focal_weight * kl_loss * weights
        
        return focal_kl_loss.mean()
    
    def _compute_adaptive_kl_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                                 weights: torch.Tensor, temperature: float) -> torch.Tensor:
        """Adaptive temperature KL divergence"""
        
        # Compute base KL loss
        base_loss = self._compute_kl_loss(teacher_logits, student_logits, weights, temperature)
        
        # Adaptive temperature based on agreement
        teacher_pred = torch.argmax(teacher_logits, dim=-1)
        student_pred = torch.argmax(student_logits, dim=-1)
        agreement = (teacher_pred == student_pred).float().mean()
        
        # Lower temperature for high agreement, higher for low agreement
        adaptive_temp = temperature * (2.0 - agreement)
        adaptive_temp = torch.clamp(adaptive_temp, self.config.min_temperature, self.config.max_temperature)
        
        # Recompute with adaptive temperature
        if adaptive_temp != temperature:
            adaptive_loss = self._compute_kl_loss(teacher_logits, student_logits, weights, adaptive_temp)
            return 0.7 * base_loss + 0.3 * adaptive_loss
        
        return base_loss
    
    def _compute_symmetric_kl_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                                  weights: torch.Tensor, temperature: float) -> torch.Tensor:
        """Symmetric KL divergence (Jensen-Shannon divergence)"""
        
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        
        # Compute symmetric KL
        m = 0.5 * (teacher_probs + student_probs)
        
        kl_teacher_m = F.kl_div(torch.log(m + self.config.epsilon), teacher_probs, reduction='none')
        kl_student_m = F.kl_div(torch.log(m + self.config.epsilon), student_probs, reduction='none')
        
        js_divergence = 0.5 * (torch.sum(kl_teacher_m, dim=-1) + torch.sum(kl_student_m, dim=-1))
        
        # Apply weights
        weighted_loss = js_divergence * weights
        
        return weighted_loss.mean()
    
    def _compute_alpha_divergence_loss(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                                      weights: torch.Tensor, temperature: float) -> torch.Tensor:
        """Alpha divergence family loss"""
        
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        
        alpha = self.config.alpha_divergence_alpha
        
        if abs(alpha - 1.0) < 1e-6:
            # Alpha = 1 is KL divergence
            return self._compute_kl_loss(teacher_logits, student_logits, weights, temperature)
        
        # Alpha divergence formula
        if alpha == 0:
            # Reverse KL
            alpha_div = torch.sum(student_probs * torch.log(student_probs / (teacher_probs + self.config.epsilon)), dim=-1)
        else:
            # General alpha divergence
            ratio = student_probs / (teacher_probs + self.config.epsilon)
            alpha_div = (1.0 / (alpha * (alpha - 1.0))) * torch.sum(
                teacher_probs * (torch.pow(ratio, alpha) - alpha * ratio + alpha - 1.0), dim=-1)
        
        # Apply weights
        weighted_loss = alpha_div * weights
        
        return weighted_loss.mean()
    
    def _apply_label_smoothing(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing to probability distribution"""
        
        smoothing = self.config.label_smoothing
        vocab_size = probs.size(-1)
        
        uniform_probs = torch.ones_like(probs) / vocab_size
        smoothed_probs = (1.0 - smoothing) * probs + smoothing * uniform_probs
        
        return smoothed_probs

class SADLoss(nn.Module):
    """
    Structured Agent Distillation Loss Module
    
    Advanced knowledge distillation loss for training student models to mimic
    teacher reasoning and action patterns with span-aware loss computation.
    """
    
    def __init__(self, config: SADLossConfig):
        super().__init__()
        self.config = config
        self.weight_computer = SpanWeightComputer(config)
        self.loss_computer = AdvancedLossComputer(config)
        
        # Training state
        self.step_count = 0
        self.loss_history = []
        self.span_metrics_history = []
        
        # Performance tracking
        self.best_loss = float('inf')
        self.convergence_steps = 0
        self.plateau_steps = 0
        
        # Numerical stability
        self.register_buffer('_epsilon', torch.tensor(config.epsilon))
        
        logger.info(f"Initialized SAD Loss with {config.loss_type.value} and {config.weighting_strategy.value}")
    
    def forward(self, 
                teacher_logits: torch.Tensor,
                student_logits: torch.Tensor,
                processed_trajectory: ProcessedTrajectory,
                target_tokens: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute SAD loss with span-aware weighting
        
        Args:
            teacher_logits: Teacher model logits [batch_size, seq_len, vocab_size]
            student_logits: Student model logits [batch_size, seq_len, vocab_size]
            processed_trajectory: Processed trajectory with span information
            target_tokens: Target token ids for additional supervision
            attention_mask: Attention mask for valid positions
            
        Returns:
            Dict containing loss components and metrics
        """
        
        self.step_count += 1
        
        # Input validation and preprocessing
        teacher_logits, student_logits = self._preprocess_logits(teacher_logits, student_logits)
        
        # Extract span information - adapt to actual ProcessedTrajectory structure
        spans = [(span.start_token, span.end_token, span.span_type) for span in processed_trajectory.spans]
        reasoning_mask = torch.tensor(processed_trajectory.reasoning_mask, device=teacher_logits.device)
        action_mask = torch.tensor(processed_trajectory.action_mask, device=teacher_logits.device)
        
        batch_size, seq_len, vocab_size = teacher_logits.shape
        
        # Initialize outputs
        outputs = {}
        total_loss = torch.tensor(0.0, device=teacher_logits.device, requires_grad=True)
        
        # Process each sequence in batch
        batch_losses = []
        span_metrics = []
        
        with autocast(enabled=self.config.use_mixed_precision):
            for batch_idx in range(batch_size):
                # Get sequence-specific data
                seq_teacher_logits = teacher_logits[batch_idx]  # [seq_len, vocab_size]
                seq_student_logits = student_logits[batch_idx]  # [seq_len, vocab_size]
                
                # Apply attention mask if provided
                if attention_mask is not None:
                    mask = attention_mask[batch_idx]
                    seq_len_actual = int(mask.sum().item())
                    seq_teacher_logits = seq_teacher_logits[:seq_len_actual]
                    seq_student_logits = seq_student_logits[:seq_len_actual]
                else:
                    seq_len_actual = seq_len
                
                # Compute probability distributions
                teacher_probs = F.softmax(seq_teacher_logits / self.config.temperature, dim=-1)
                student_probs = F.softmax(seq_student_logits / self.config.temperature, dim=-1)
                
                # Compute span weights
                seq_spans = [(s, min(e, seq_len_actual), t) for s, e, t in spans if s < seq_len_actual]
                weights = self.weight_computer.compute_weights(
                    seq_spans, teacher_probs, student_probs, self.step_count
                )
                
                # Compute primary loss
                primary_loss = self.loss_computer.compute_loss(
                    seq_teacher_logits, seq_student_logits, weights[:seq_len_actual]
                )
                
                # Compute auxiliary losses
                aux_losses = self._compute_auxiliary_losses(
                    seq_teacher_logits, seq_student_logits, seq_spans, 
                    target_tokens[batch_idx][:seq_len_actual] if target_tokens is not None else None
                )
                
                # Combine losses
                seq_loss = primary_loss
                for loss_name, loss_value in aux_losses.items():
                    seq_loss = seq_loss + loss_value
                    outputs[f'aux_{loss_name}'] = outputs.get(f'aux_{loss_name}', 0) + loss_value / batch_size
                
                batch_losses.append(seq_loss)
                
                # Compute span-specific metrics
                if self.config.compute_span_metrics:
                    metrics = self._compute_span_metrics(
                        seq_teacher_logits, seq_student_logits, seq_spans
                    )
                    span_metrics.append(metrics)
        
        # Aggregate batch losses
        total_loss = torch.stack(batch_losses).mean()
        
        # Apply gradient clipping if configured
        if self.config.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(
                [total_loss], self.config.gradient_clip_value
            )
        
        # Prepare outputs
        outputs.update({
            'loss': total_loss,
            'primary_loss': total_loss,
            'step': self.step_count,
            'temperature': self.config.temperature
        })
        
        # Add span metrics
        if span_metrics:
            aggregated_metrics = self._aggregate_span_metrics(span_metrics)
            outputs.update(aggregated_metrics)
        
        # Update training state
        self._update_training_state(total_loss.item(), outputs)
        
        # Log detailed metrics if enabled
        if self.config.log_detailed_metrics and self.step_count % 100 == 0:
            self._log_detailed_metrics(outputs)
        
        return outputs
    
    def _preprocess_logits(self, teacher_logits: torch.Tensor, 
                          student_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess logits for numerical stability"""
        
        if self.config.numerical_stability:
            # Clamp extreme values
            teacher_logits = torch.clamp(teacher_logits, 
                                       -self.config.max_logit_value, 
                                       self.config.max_logit_value)
            student_logits = torch.clamp(student_logits,
                                       -self.config.max_logit_value,
                                       self.config.max_logit_value)
        
        return teacher_logits, student_logits
    
    def _compute_auxiliary_losses(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                                 spans: List[Tuple[int, int, SpanType]], 
                                 target_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for regularization"""
        
        aux_losses = {}
        
        # Entropy regularization
        if self.config.entropy_regularization > 0:
            student_probs = F.softmax(student_logits, dim=-1)
            entropy = -torch.sum(student_probs * torch.log(student_probs + self._epsilon), dim=-1)
            # Use negative entropy as regularization (encourage higher entropy)
            aux_losses['entropy_reg'] = self.config.entropy_regularization * entropy.mean()
        
        # Confidence penalty (prevent overconfident predictions)
        if self.config.confidence_penalty > 0:
            student_probs = F.softmax(student_logits, dim=-1)
            max_probs = torch.max(student_probs, dim=-1)[0]
            confidence_penalty = self.config.confidence_penalty * torch.mean(max_probs ** 2)
            aux_losses['confidence_penalty'] = confidence_penalty
        
        # Span consistency regularization
        if self.config.span_consistency_weight > 0 and len(spans) > 1:
            consistency_loss = self._compute_span_consistency_loss(student_logits, spans)
            aux_losses['span_consistency'] = self.config.span_consistency_weight * consistency_loss
        
        # Target token supervision if available
        if target_tokens is not None:
            ce_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                target_tokens.view(-1),
                ignore_index=-100,
                reduction='mean'
            )
            aux_losses['target_supervision'] = 0.1 * ce_loss
        
        return aux_losses
    
    def _compute_span_consistency_loss(self, student_logits: torch.Tensor,
                                      spans: List[Tuple[int, int, SpanType]]) -> torch.Tensor:
        """Ensure consistency within span types"""
        
        span_losses = []
        span_types = {}
        
        # Group spans by type
        for start, end, span_type in spans:
            if span_type not in span_types:
                span_types[span_type] = []
            span_types[span_type].append((start, end))
        
        # Compute consistency loss for each span type
        for span_type, positions in span_types.items():
            if len(positions) > 1:
                span_probs = []
                for start, end in positions:
                    if start < len(student_logits) and end <= len(student_logits):
                        span_prob = F.softmax(student_logits[start:end], dim=-1).mean(dim=0)
                        span_probs.append(span_prob)
                
                if len(span_probs) > 1:
                    # Compute pairwise KL divergence between spans of same type
                    consistency_loss = 0
                    count = 0
                    for i in range(len(span_probs)):
                        for j in range(i + 1, len(span_probs)):
                            kl_ij = F.kl_div(torch.log(span_probs[i] + self._epsilon), 
                                           span_probs[j], reduction='sum')
                            kl_ji = F.kl_div(torch.log(span_probs[j] + self._epsilon), 
                                           span_probs[i], reduction='sum')
                            consistency_loss += 0.5 * (kl_ij + kl_ji)
                            count += 1
                    
                    if count > 0:
                        span_losses.append(consistency_loss / count)
        
        if span_losses:
            return torch.stack(span_losses).mean()
        else:
            return torch.tensor(0.0, device=student_logits.device)
    
    def _compute_span_metrics(self, teacher_logits: torch.Tensor, student_logits: torch.Tensor,
                             spans: List[Tuple[int, int, SpanType]]) -> Dict[str, float]:
        """Compute detailed metrics for each span type"""
        
        metrics = {}
        
        # Group spans by type
        span_groups = {}
        for start, end, span_type in spans:
            if span_type not in span_groups:
                span_groups[span_type] = []
            span_groups[span_type].append((start, end))
        
        # Compute metrics for each span type
        for span_type, positions in span_groups.items():
            if positions:
                span_teacher_logits = []
                span_student_logits = []
                
                for start, end in positions:
                    if start < len(teacher_logits) and end <= len(teacher_logits):
                        span_teacher_logits.append(teacher_logits[start:end])
                        span_student_logits.append(student_logits[start:end])
                
                if span_teacher_logits:
                    # Concatenate all spans of this type
                    span_teacher = torch.cat(span_teacher_logits, dim=0)
                    span_student = torch.cat(span_student_logits, dim=0)
                    
                    # Compute KL divergence
                    teacher_probs = F.softmax(span_teacher / self.config.temperature, dim=-1)
                    student_log_probs = F.log_softmax(span_student / self.config.temperature, dim=-1)
                    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction='mean')
                    
                    # Compute accuracy
                    teacher_pred = torch.argmax(span_teacher, dim=-1)
                    student_pred = torch.argmax(span_student, dim=-1)
                    accuracy = (teacher_pred == student_pred).float().mean()
                    
                    # Compute entropy
                    student_probs = F.softmax(span_student, dim=-1)
                    entropy = -torch.sum(student_probs * torch.log(student_probs + self._epsilon), dim=-1).mean()
                    
                    metrics[f'{span_type.value}_kl'] = kl_div.item()
                    metrics[f'{span_type.value}_accuracy'] = accuracy.item()
                    metrics[f'{span_type.value}_entropy'] = entropy.item()
                    metrics[f'{span_type.value}_count'] = len(span_teacher)
        
        return metrics
    
    def _aggregate_span_metrics(self, span_metrics_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Aggregate span metrics across batch"""
        
        aggregated = {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics in span_metrics_list:
            all_keys.update(metrics.keys())
        
        # Aggregate each metric
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in span_metrics_list]
            aggregated[f'span_metrics/{key}'] = torch.tensor(np.mean(values))
        
        return aggregated
    
    def _update_training_state(self, loss_value: float, outputs: Dict):
        """Update training state and convergence tracking"""
        
        self.loss_history.append(loss_value)
        
        # Track best loss
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.convergence_steps = 0
        else:
            self.convergence_steps += 1
        
        # Check for plateau
        if len(self.loss_history) >= 100:
            recent_avg = np.mean(self.loss_history[-50:])
            older_avg = np.mean(self.loss_history[-100:-50])
            
            if abs(recent_avg - older_avg) < 0.001:
                self.plateau_steps += 1
            else:
                self.plateau_steps = 0
        
        # Update adaptive parameters
        if self.config.adaptive_temperature and self.step_count % 100 == 0:
            self._update_adaptive_temperature(outputs)
        
        if self.config.adaptive_weights and self.step_count % 100 == 0:
            self._update_adaptive_weights(outputs)
    
    def _update_adaptive_temperature(self, outputs: Dict):
        """Update temperature based on performance"""
        
        if len(self.loss_history) < 100:
            return
        
        # Check loss trend
        recent_trend = np.mean(self.loss_history[-20:]) - np.mean(self.loss_history[-40:-20])
        
        if recent_trend > 0.01:  # Loss increasing
            self.config.temperature = min(self.config.max_temperature, 
                                        self.config.temperature * 1.05)
        elif recent_trend < -0.01:  # Loss decreasing
            self.config.temperature = max(self.config.min_temperature, 
                                        self.config.temperature * 0.95)
    
    def _update_adaptive_weights(self, outputs: Dict):
        """Update span weights based on performance"""
        
        # Check span-specific performance
        reasoning_kl = outputs.get('span_metrics/reasoning_kl', 0)
        action_kl = outputs.get('span_metrics/action_kl', 0)
        
        if reasoning_kl and action_kl:
            ratio = reasoning_kl / action_kl
            
            if ratio > 1.2:  # Reasoning performing worse
                self.config.reasoning_weight = min(3.0, self.config.reasoning_weight * 1.05)
            elif ratio < 0.8:  # Action performing worse
                self.config.action_weight = min(3.0, self.config.action_weight * 1.05)
    
    def _log_detailed_metrics(self, outputs: Dict):
        """Log detailed metrics for monitoring"""
        
        logger.info(f"Step {self.step_count}: Loss = {outputs['loss']:.4f}")
        logger.info(f"Temperature: {self.config.temperature:.2f}")
        logger.info(f"Convergence steps: {self.convergence_steps}")
        logger.info(f"Plateau steps: {self.plateau_steps}")
        
        # Log span metrics
        for key, value in outputs.items():
            if key.startswith('span_metrics/'):
                logger.debug(f"{key}: {value:.4f}")
    
    @contextmanager
    def evaluation_mode(self):
        """Context manager for evaluation mode"""
        original_mode = self.training
        self.eval()
        try:
            yield
        finally:
            self.train(original_mode)
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive training statistics"""
        
        stats = {
            'step_count': self.step_count,
            'best_loss': self.best_loss,
            'convergence_steps': self.convergence_steps,
            'plateau_steps': self.plateau_steps,
            'current_temperature': self.config.temperature,
            'current_reasoning_weight': self.config.reasoning_weight,
            'current_action_weight': self.config.action_weight
        }
        
        if self.loss_history:
            stats.update({
                'mean_loss_recent_100': np.mean(self.loss_history[-100:]),
                'loss_trend_recent_20': np.mean(self.loss_history[-20:]) - np.mean(self.loss_history[-40:-20]) if len(self.loss_history) >= 40 else 0,
                'loss_std_recent_100': np.std(self.loss_history[-100:]) if len(self.loss_history) >= 100 else 0
            })
        
        return stats
    
    def reset_training_state(self):
        """Reset training state for new training run"""
        
        self.step_count = 0
        self.loss_history = []
        self.span_metrics_history = []
        self.best_loss = float('inf')
        self.convergence_steps = 0
        self.plateau_steps = 0
        
        logger.info("Reset SAD Loss training state")

# Factory function for easy instantiation
def create_sad_loss(loss_type: str = "adaptive_kl", 
                   weighting_strategy: str = "adaptive",
                   **kwargs) -> SADLoss:
    """
    Factory function to create SAD loss with common configurations
    
    Args:
        loss_type: Type of loss computation
        weighting_strategy: Strategy for span weighting
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SADLoss instance
    """
    
    config = SADLossConfig(
        loss_type=LossType(loss_type),
        weighting_strategy=WeightingStrategy(weighting_strategy),
        **kwargs
    )
    
    return SADLoss(config)

# Predefined configurations for common use cases
def get_production_config() -> SADLossConfig:
    """Get production-ready configuration"""
    return SADLossConfig(
        loss_type=LossType.ADAPTIVE_KL,
        weighting_strategy=WeightingStrategy.ADAPTIVE,
        use_mixed_precision=True,
        numerical_stability=True,
        gradient_clip_value=1.0,
        temperature=2.0,
        reasoning_weight=2.5,
        action_weight=1.8
    )

def get_research_config() -> SADLossConfig:
    """Get research configuration with detailed logging"""
    return SADLossConfig(
        loss_type=LossType.WASSERSTEIN,
        weighting_strategy=WeightingStrategy.CURRICULUM,
        compute_span_metrics=True,
        log_detailed_metrics=True,
        track_convergence=True,
        curriculum_learning=True,
        adaptive_temperature=True
    )

def get_fast_config() -> SADLossConfig:
    """Get configuration optimized for speed"""
    return SADLossConfig(
        loss_type=LossType.STANDARD_KL,
        weighting_strategy=WeightingStrategy.UNIFORM,
        use_mixed_precision=True,
        compute_span_metrics=False,
        log_detailed_metrics=False,
        numerical_stability=False
    ) 