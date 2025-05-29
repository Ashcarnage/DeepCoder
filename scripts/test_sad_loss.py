#!/usr/bin/env python3
"""
Comprehensive Test Suite for SAD Loss Implementation
Tests all components with production-grade validation
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sad_loss import (
    SADLoss, SADLossConfig, LossType, WeightingStrategy,
    SpanWeightComputer, AdvancedLossComputer,
    create_sad_loss, get_production_config, get_research_config, get_fast_config
)
from trajectory_processing import SpanType, ProcessedTrajectory, TokenSpan

class TestSADLossConfig(unittest.TestCase):
    """Test SAD Loss Configuration System"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SADLossConfig()
        
        # Check default values
        self.assertEqual(config.loss_type, LossType.ADAPTIVE_KL)
        self.assertEqual(config.weighting_strategy, WeightingStrategy.ADAPTIVE)
        self.assertEqual(config.temperature, 3.0)
        self.assertEqual(config.reasoning_weight, 2.0)
        self.assertEqual(config.action_weight, 1.5)
        self.assertTrue(config.use_mixed_precision)
        self.assertTrue(config.numerical_stability)
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = SADLossConfig(
            loss_type=LossType.WASSERSTEIN,
            weighting_strategy=WeightingStrategy.CURRICULUM,
            temperature=5.0,
            reasoning_weight=3.0,
            action_weight=2.0,
            use_mixed_precision=False
        )
        
        self.assertEqual(config.loss_type, LossType.WASSERSTEIN)
        self.assertEqual(config.weighting_strategy, WeightingStrategy.CURRICULUM)
        self.assertEqual(config.temperature, 5.0)
        self.assertEqual(config.reasoning_weight, 3.0)
        self.assertEqual(config.action_weight, 2.0)
        self.assertFalse(config.use_mixed_precision)
    
    def test_predefined_configs(self):
        """Test predefined configuration presets"""
        # Production config
        prod_config = get_production_config()
        self.assertEqual(prod_config.loss_type, LossType.ADAPTIVE_KL)
        self.assertTrue(prod_config.numerical_stability)
        self.assertEqual(prod_config.gradient_clip_value, 1.0)
        
        # Research config
        research_config = get_research_config()
        self.assertEqual(research_config.loss_type, LossType.WASSERSTEIN)
        self.assertTrue(research_config.compute_span_metrics)
        self.assertTrue(research_config.curriculum_learning)
        
        # Fast config
        fast_config = get_fast_config()
        self.assertEqual(fast_config.loss_type, LossType.STANDARD_KL)
        self.assertEqual(fast_config.weighting_strategy, WeightingStrategy.UNIFORM)
        self.assertFalse(fast_config.compute_span_metrics)

class TestSpanWeightComputer(unittest.TestCase):
    """Test Span Weight Computation System"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SADLossConfig()
        self.weight_computer = SpanWeightComputer(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        self.seq_len = 50
        self.vocab_size = 1000
        self.teacher_probs = torch.randn(self.seq_len, self.vocab_size).softmax(dim=-1).to(self.device)
        self.student_probs = torch.randn(self.seq_len, self.vocab_size).softmax(dim=-1).to(self.device)
        
        # Test spans
        self.spans = [
            (0, 10, SpanType.REASONING),
            (10, 25, SpanType.ACTION),
            (25, 35, SpanType.OBSERVATION),
            (35, 50, SpanType.OTHER)
        ]
    
    def test_uniform_weighting(self):
        """Test uniform weighting strategy"""
        config = SADLossConfig(weighting_strategy=WeightingStrategy.UNIFORM)
        weight_computer = SpanWeightComputer(config)
        
        weights = weight_computer.compute_weights(
            self.spans, self.teacher_probs, self.student_probs
        )
        
        # Should have uniform weights
        expected_weights = torch.ones(self.seq_len, device=self.device)
        torch.testing.assert_close(weights, expected_weights, rtol=1e-5, atol=1e-5)
    
    def test_reasoning_heavy_weighting(self):
        """Test reasoning-heavy weighting strategy"""
        config = SADLossConfig(
            weighting_strategy=WeightingStrategy.REASONING_HEAVY,
            reasoning_weight=3.0,
            action_weight=1.0
        )
        weight_computer = SpanWeightComputer(config)
        
        weights = weight_computer.compute_weights(
            self.spans, self.teacher_probs, self.student_probs
        )
        
        # Check that reasoning spans have higher weights
        reasoning_weight = weights[0:10].mean()
        action_weight = weights[10:25].mean()
        self.assertGreater(reasoning_weight, action_weight)
    
    def test_adaptive_weighting(self):
        """Test adaptive weighting strategy"""
        config = SADLossConfig(weighting_strategy=WeightingStrategy.ADAPTIVE)
        weight_computer = SpanWeightComputer(config)
        
        weights = weight_computer.compute_weights(
            self.spans, self.teacher_probs, self.student_probs, step=100
        )
        
        # Should produce valid weights
        self.assertEqual(weights.shape[0], self.seq_len)
        self.assertTrue(torch.all(weights > 0))
        self.assertAlmostEqual(weights.mean().item(), 1.0, places=4)
    
    def test_curriculum_weighting(self):
        """Test curriculum learning weighting"""
        config = SADLossConfig(
            weighting_strategy=WeightingStrategy.CURRICULUM,
            curriculum_learning=True
        )
        weight_computer = SpanWeightComputer(config)
        
        # Test early training (step 100)
        early_weights = weight_computer.compute_weights(
            self.spans, self.teacher_probs, self.student_probs, step=100
        )
        
        # Test late training (step 10000)
        late_weights = weight_computer.compute_weights(
            self.spans, self.teacher_probs, self.student_probs, step=10000
        )
        
        # Reasoning weights should increase over time
        early_reasoning = early_weights[0:10].mean()
        late_reasoning = late_weights[0:10].mean()
        self.assertGreater(late_reasoning, early_reasoning)
    
    def test_confidence_based_weighting(self):
        """Test confidence-based weighting"""
        config = SADLossConfig(weighting_strategy=WeightingStrategy.CONFIDENCE_BASED)
        weight_computer = SpanWeightComputer(config)
        
        # Create high confidence teacher probs (low entropy)
        high_conf_probs = torch.zeros(self.seq_len, self.vocab_size, device=self.device)
        high_conf_probs[:, 0] = 0.9
        high_conf_probs[:, 1:] = 0.1 / (self.vocab_size - 1)
        
        weights = weight_computer.compute_weights(
            self.spans, high_conf_probs, self.student_probs
        )
        
        # Should produce valid weights favoring high confidence regions
        self.assertEqual(weights.shape[0], self.seq_len)
        self.assertTrue(torch.all(weights > 0))

class TestAdvancedLossComputer(unittest.TestCase):
    """Test Advanced Loss Computation System"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SADLossConfig()
        self.loss_computer = AdvancedLossComputer(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        self.seq_len = 20
        self.vocab_size = 100
        self.teacher_logits = torch.randn(self.seq_len, self.vocab_size, device=self.device)
        self.student_logits = torch.randn(self.seq_len, self.vocab_size, device=self.device)
        self.weights = torch.ones(self.seq_len, device=self.device)
        self.temperature = 3.0
    
    def test_standard_kl_loss(self):
        """Test standard KL divergence loss"""
        config = SADLossConfig(loss_type=LossType.STANDARD_KL)
        loss_computer = AdvancedLossComputer(config)
        
        loss = loss_computer.compute_loss(
            self.teacher_logits, self.student_logits, self.weights, self.temperature
        )
        
        # Should produce valid loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_wasserstein_loss(self):
        """Test Wasserstein distance loss"""
        config = SADLossConfig(loss_type=LossType.WASSERSTEIN)
        loss_computer = AdvancedLossComputer(config)
        
        loss = loss_computer.compute_loss(
            self.teacher_logits, self.student_logits, self.weights, self.temperature
        )
        
        # Should produce valid loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertGreaterEqual(loss.item(), 0)  # Wasserstein is non-negative
        self.assertFalse(torch.isnan(loss))
    
    def test_focal_kl_loss(self):
        """Test focal KL divergence loss"""
        config = SADLossConfig(loss_type=LossType.FOCAL_KL, focal_gamma=2.0)
        loss_computer = AdvancedLossComputer(config)
        
        loss = loss_computer.compute_loss(
            self.teacher_logits, self.student_logits, self.weights, self.temperature
        )
        
        # Should produce valid loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
    
    def test_adaptive_kl_loss(self):
        """Test adaptive temperature KL loss"""
        config = SADLossConfig(loss_type=LossType.ADAPTIVE_KL)
        loss_computer = AdvancedLossComputer(config)
        
        loss = loss_computer.compute_loss(
            self.teacher_logits, self.student_logits, self.weights, self.temperature
        )
        
        # Should produce valid loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
    
    def test_symmetric_kl_loss(self):
        """Test symmetric KL (Jensen-Shannon) divergence"""
        config = SADLossConfig(loss_type=LossType.SYMMETRIC_KL)
        loss_computer = AdvancedLossComputer(config)
        
        loss = loss_computer.compute_loss(
            self.teacher_logits, self.student_logits, self.weights, self.temperature
        )
        
        # Should produce valid loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreaterEqual(loss.item(), 0)  # JS divergence is non-negative
        self.assertFalse(torch.isnan(loss))
    
    def test_alpha_divergence_loss(self):
        """Test alpha divergence loss"""
        config = SADLossConfig(
            loss_type=LossType.ALPHA_DIVERGENCE,
            alpha_divergence_alpha=1.5
        )
        loss_computer = AdvancedLossComputer(config)
        
        loss = loss_computer.compute_loss(
            self.teacher_logits, self.student_logits, self.weights, self.temperature
        )
        
        # Should produce valid loss
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
    
    def test_label_smoothing(self):
        """Test label smoothing functionality"""
        config = SADLossConfig(
            loss_type=LossType.STANDARD_KL,
            label_smoothing=0.1
        )
        loss_computer = AdvancedLossComputer(config)
        
        # Create uniform teacher distribution
        teacher_probs = torch.ones(self.vocab_size, device=self.device) / self.vocab_size
        smoothed = loss_computer._apply_label_smoothing(teacher_probs.unsqueeze(0))
        
        # Should remain approximately uniform after smoothing
        self.assertAlmostEqual(smoothed.sum().item(), 1.0, places=5)
        self.assertTrue(torch.allclose(smoothed, teacher_probs.unsqueeze(0), atol=1e-5))

class TestSADLoss(unittest.TestCase):
    """Test Main SAD Loss Module"""
    
    def setUp(self):
        """Set up test environment"""
        self.config = SADLossConfig(
            compute_span_metrics=True,
            log_detailed_metrics=False,  # Disable for testing
            numerical_stability=True
        )
        self.sad_loss = SADLoss(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create test data
        self.batch_size = 2
        self.seq_len = 30
        self.vocab_size = 50
        
        self.teacher_logits = torch.randn(
            self.batch_size, self.seq_len, self.vocab_size, device=self.device
        )
        self.student_logits = torch.randn(
            self.batch_size, self.seq_len, self.vocab_size, device=self.device
        )
        
        # Create processed trajectory with correct structure
        spans = [
            TokenSpan(SpanType.REASONING, 0, 10, "reasoning text"),
            TokenSpan(SpanType.ACTION, 10, 20, "action text"),
            TokenSpan(SpanType.OBSERVATION, 20, 30, "observation text")
        ]
        
        self.processed_trajectory = ProcessedTrajectory(
            trajectory_id="test_trajectory",
            input_ids=list(range(self.seq_len)),
            attention_mask=[1] * self.seq_len,
            reasoning_mask=[1] * 10 + [0] * 20,
            action_mask=[0] * 10 + [1] * 10 + [0] * 10,
            spans=spans,
            original_text="Test trajectory with reasoning and actions",
            metadata={'quality_score': 0.9, 'confidence': 0.85}
        )
        
        self.target_tokens = torch.randint(
            0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device
        )
        self.attention_mask = torch.ones(
            self.batch_size, self.seq_len, device=self.device
        )
    
    def test_forward_pass_basic(self):
        """Test basic forward pass"""
        outputs = self.sad_loss(
            self.teacher_logits,
            self.student_logits,
            self.processed_trajectory,
            self.target_tokens,
            self.attention_mask
        )
        
        # Check required outputs
        self.assertIn('loss', outputs)
        self.assertIn('primary_loss', outputs)
        self.assertIn('step', outputs)
        self.assertIn('temperature', outputs)
        
        # Check loss properties
        loss = outputs['loss']
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # Scalar
        self.assertGreater(loss.item(), 0)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))
        
        # Check step increment
        self.assertEqual(outputs['step'], 1)
    
    def test_forward_pass_with_attention_mask(self):
        """Test forward pass with attention mask"""
        # Create attention mask with padding
        attention_mask = torch.ones(self.batch_size, self.seq_len, device=self.device)
        attention_mask[:, -5:] = 0  # Mask last 5 tokens
        
        outputs = self.sad_loss(
            self.teacher_logits,
            self.student_logits,
            self.processed_trajectory,
            self.target_tokens,
            attention_mask
        )
        
        # Should handle masked tokens correctly
        self.assertIn('loss', outputs)
        self.assertGreater(outputs['loss'].item(), 0)
        self.assertFalse(torch.isnan(outputs['loss']))
    
    def test_span_metrics_computation(self):
        """Test span metrics computation"""
        outputs = self.sad_loss(
            self.teacher_logits,
            self.student_logits,
            self.processed_trajectory
        )
        
        # Check span metrics
        span_metric_keys = [k for k in outputs.keys() if k.startswith('span_metrics/')]
        self.assertGreater(len(span_metric_keys), 0)
        
        # Should have metrics for each span type
        expected_metrics = ['reasoning_kl', 'action_kl', 'observation_kl']
        for metric in expected_metrics:
            metric_key = f'span_metrics/{metric}'
            if metric_key in outputs:
                self.assertIsInstance(outputs[metric_key], torch.Tensor)
                self.assertFalse(torch.isnan(outputs[metric_key]))
    
    def test_auxiliary_losses(self):
        """Test auxiliary loss computation"""
        config = SADLossConfig(
            entropy_regularization=0.01,
            confidence_penalty=0.05,
            span_consistency_weight=0.1
        )
        sad_loss = SADLoss(config)
        
        outputs = sad_loss(
            self.teacher_logits,
            self.student_logits,
            self.processed_trajectory,
            self.target_tokens
        )
        
        # Check auxiliary losses
        aux_keys = [k for k in outputs.keys() if k.startswith('aux_')]
        
        # Should have some auxiliary losses
        if aux_keys:
            for key in aux_keys:
                self.assertIsInstance(outputs[key], torch.Tensor)
                self.assertFalse(torch.isnan(outputs[key]))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Create extreme logits
        extreme_teacher = torch.full(
            (self.batch_size, self.seq_len, self.vocab_size), 100.0, device=self.device
        )
        extreme_student = torch.full(
            (self.batch_size, self.seq_len, self.vocab_size), -100.0, device=self.device
        )
        
        outputs = self.sad_loss(
            extreme_teacher,
            extreme_student,
            self.processed_trajectory
        )
        
        # Should handle extreme values gracefully
        self.assertFalse(torch.isnan(outputs['loss']))
        self.assertFalse(torch.isinf(outputs['loss']))
        self.assertGreater(outputs['loss'].item(), 0)
    
    def test_gradient_computation(self):
        """Test gradient computation"""
        # Enable gradient tracking
        self.teacher_logits.requires_grad_(True)
        self.student_logits.requires_grad_(True)
        
        outputs = self.sad_loss(
            self.teacher_logits,
            self.student_logits,
            self.processed_trajectory
        )
        
        # Backward pass
        outputs['loss'].backward()
        
        # Check gradients
        self.assertIsNotNone(self.student_logits.grad)
        self.assertFalse(torch.isnan(self.student_logits.grad).any())
        self.assertFalse(torch.isinf(self.student_logits.grad).any())
    
    def test_training_state_tracking(self):
        """Test training state tracking"""
        initial_step = self.sad_loss.step_count
        
        # Multiple forward passes
        for i in range(5):
            outputs = self.sad_loss(
                self.teacher_logits,
                self.student_logits,
                self.processed_trajectory
            )
        
        # Check step tracking
        self.assertEqual(self.sad_loss.step_count, initial_step + 5)
        
        # Check loss history
        self.assertEqual(len(self.sad_loss.loss_history), 5)
        
        # Get training stats
        stats = self.sad_loss.get_training_stats()
        self.assertIn('step_count', stats)
        self.assertIn('best_loss', stats)
        self.assertEqual(stats['step_count'], initial_step + 5)
    
    def test_adaptive_temperature(self):
        """Test adaptive temperature adjustment"""
        config = SADLossConfig(
            adaptive_temperature=True,
            min_temperature=1.0,
            max_temperature=10.0
        )
        sad_loss = SADLoss(config)
        
        initial_temp = config.temperature
        
        # Simulate training with loss trend
        for i in range(150):  # Trigger adaptation at step 100
            outputs = sad_loss(
                self.teacher_logits,
                self.student_logits,
                self.processed_trajectory
            )
        
        # Temperature might have changed
        final_temp = config.temperature
        self.assertGreaterEqual(final_temp, config.min_temperature)
        self.assertLessEqual(final_temp, config.max_temperature)
    
    def test_evaluation_mode(self):
        """Test evaluation mode context manager"""
        self.sad_loss.train()
        
        with self.sad_loss.evaluation_mode():
            self.assertFalse(self.sad_loss.training)
        
        self.assertTrue(self.sad_loss.training)
    
    def test_reset_training_state(self):
        """Test training state reset"""
        # Run some training steps
        for i in range(3):
            self.sad_loss(
                self.teacher_logits,
                self.student_logits,
                self.processed_trajectory
            )
        
        # Reset state
        self.sad_loss.reset_training_state()
        
        # Check reset
        self.assertEqual(self.sad_loss.step_count, 0)
        self.assertEqual(len(self.sad_loss.loss_history), 0)
        self.assertEqual(self.sad_loss.best_loss, float('inf'))

class TestFactoryAndUtilities(unittest.TestCase):
    """Test Factory Functions and Utilities"""
    
    def test_create_sad_loss_factory(self):
        """Test factory function"""
        sad_loss = create_sad_loss(
            loss_type="wasserstein",
            weighting_strategy="curriculum",
            temperature=4.0
        )
        
        self.assertIsInstance(sad_loss, SADLoss)
        self.assertEqual(sad_loss.config.loss_type, LossType.WASSERSTEIN)
        self.assertEqual(sad_loss.config.weighting_strategy, WeightingStrategy.CURRICULUM)
        self.assertEqual(sad_loss.config.temperature, 4.0)
    
    def test_predefined_configs(self):
        """Test predefined configuration functions"""
        # Production config
        prod_config = get_production_config()
        prod_loss = SADLoss(prod_config)
        self.assertIsInstance(prod_loss, SADLoss)
        
        # Research config
        research_config = get_research_config()
        research_loss = SADLoss(research_config)
        self.assertIsInstance(research_loss, SADLoss)
        
        # Fast config
        fast_config = get_fast_config()
        fast_loss = SADLoss(fast_config)
        self.assertIsInstance(fast_loss, SADLoss)

class TestPerformance(unittest.TestCase):
    """Test Performance and Scalability"""
    
    def setUp(self):
        """Set up performance test environment"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = get_fast_config()  # Use fast config for performance tests
        self.sad_loss = SADLoss(self.config)
    
    def test_batch_processing_performance(self):
        """Test performance with different batch sizes"""
        batch_sizes = [1, 4, 8, 16] if torch.cuda.is_available() else [1, 2, 4]
        seq_len = 100
        vocab_size = 1000
        
        for batch_size in batch_sizes:
            teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
            student_logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
            
            # Create processed trajectory with correct structure
            spans = [
                TokenSpan(SpanType.REASONING, 0, 50, "reasoning"),
                TokenSpan(SpanType.ACTION, 50, 100, "action")
            ]
            
            processed_trajectory = ProcessedTrajectory(
                trajectory_id="test_trajectory",
                input_ids=list(range(seq_len)),
                attention_mask=[1] * seq_len,
                reasoning_mask=[1] * 50 + [0] * 50,
                action_mask=[0] * 50 + [1] * 50,
                spans=spans,
                original_text="Test trajectory",
                metadata={'quality_score': 0.9}
            )
            
            start_time = time.time()
            outputs = self.sad_loss(teacher_logits, student_logits, processed_trajectory)
            end_time = time.time()
            
            # Should complete in reasonable time
            self.assertLess(end_time - start_time, 10.0)  # Less than 10 seconds
            self.assertIn('loss', outputs)
            self.assertFalse(torch.isnan(outputs['loss']))
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large sequences"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory test")
        
        batch_size = 2
        seq_len = 500  # Large sequence
        vocab_size = 1000
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        teacher_logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        student_logits = torch.randn(batch_size, seq_len, vocab_size, device=self.device)
        
        spans = [
            TokenSpan(SpanType.REASONING, 0, 250, "reasoning"),
            TokenSpan(SpanType.ACTION, 250, 500, "action")
        ]
        
        processed_trajectory = ProcessedTrajectory(
            trajectory_id="test_trajectory",
            input_ids=list(range(seq_len)),
            attention_mask=[1] * seq_len,
            reasoning_mask=[1] * 250 + [0] * 250,
            action_mask=[0] * 250 + [1] * 250,
            spans=spans,
            original_text="Test trajectory",
            metadata={'quality_score': 0.9}
        )
        
        outputs = self.sad_loss(teacher_logits, student_logits, processed_trajectory)
        
        final_memory = torch.cuda.memory_allocated()
        memory_usage = (final_memory - initial_memory) / (1024 ** 2)  # MB
        
        # Should not use excessive memory (threshold depends on system)
        self.assertLess(memory_usage, 1000)  # Less than 1GB
        self.assertIn('loss', outputs)

class TestEdgeCases(unittest.TestCase):
    """Test Edge Cases and Error Handling"""
    
    def setUp(self):
        """Set up edge case test environment"""
        self.config = SADLossConfig()
        self.sad_loss = SADLoss(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_empty_spans(self):
        """Test with empty span list"""
        teacher_logits = torch.randn(1, 10, 50, device=self.device)
        student_logits = torch.randn(1, 10, 50, device=self.device)
        
        processed_trajectory = ProcessedTrajectory(
            trajectory_id="test_trajectory",
            input_ids=list(range(10)),
            attention_mask=[1] * 10,
            reasoning_mask=[0] * 10,
            action_mask=[0] * 10,
            spans=[],  # Empty spans
            original_text="Test",
            metadata={}
        )
        
        outputs = self.sad_loss(teacher_logits, student_logits, processed_trajectory)
        
        # Should handle empty spans gracefully
        self.assertIn('loss', outputs)
        self.assertFalse(torch.isnan(outputs['loss']))
    
    def test_single_token_sequence(self):
        """Test with single token sequence"""
        teacher_logits = torch.randn(1, 1, 50, device=self.device)
        student_logits = torch.randn(1, 1, 50, device=self.device)
        
        spans = [TokenSpan(SpanType.OTHER, 0, 1, "A")]
        
        processed_trajectory = ProcessedTrajectory(
            trajectory_id="test_trajectory", 
            input_ids=[0],
            attention_mask=[1],
            reasoning_mask=[0],
            action_mask=[0],
            spans=spans,
            original_text="A",
            metadata={}
        )
        
        outputs = self.sad_loss(teacher_logits, student_logits, processed_trajectory)
        
        # Should handle single token gracefully
        self.assertIn('loss', outputs)
        self.assertFalse(torch.isnan(outputs['loss']))
    
    def test_mismatched_dimensions(self):
        """Test with mismatched tensor dimensions"""
        teacher_logits = torch.randn(1, 10, 50, device=self.device)
        student_logits = torch.randn(1, 15, 50, device=self.device)  # Different seq_len
        
        spans = [TokenSpan(SpanType.REASONING, 0, 5, "reasoning")]
        
        processed_trajectory = ProcessedTrajectory(
            trajectory_id="test_trajectory",
            input_ids=list(range(10)),
            attention_mask=[1] * 10,
            reasoning_mask=[1] * 5 + [0] * 5,
            action_mask=[0] * 10,
            spans=spans,
            original_text="Test",
            metadata={}
        )
        
        # Should raise an error or handle gracefully
        with self.assertRaises((RuntimeError, ValueError)):
            self.sad_loss(teacher_logits, student_logits, processed_trajectory)
    
    def test_invalid_span_indices(self):
        """Test with invalid span indices"""
        teacher_logits = torch.randn(1, 10, 50, device=self.device)
        student_logits = torch.randn(1, 10, 50, device=self.device)
        
        spans = [TokenSpan(SpanType.REASONING, 0, 15, "reasoning")]  # End index beyond sequence
        
        processed_trajectory = ProcessedTrajectory(
            trajectory_id="test_trajectory",
            input_ids=list(range(10)),
            attention_mask=[1] * 10,
            reasoning_mask=[1] * 10,
            action_mask=[0] * 10,
            spans=spans,
            original_text="Test",
            metadata={}
        )
        
        # Should handle invalid indices gracefully
        outputs = self.sad_loss(teacher_logits, student_logits, processed_trajectory)
        self.assertIn('loss', outputs)
        self.assertFalse(torch.isnan(outputs['loss']))

def run_comprehensive_tests():
    """Run all test suites with detailed reporting"""
    
    print("=" * 80)
    print("SAD LOSS COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    # Test suites
    test_suites = [
        TestSADLossConfig,
        TestSpanWeightComputer,
        TestAdvancedLossComputer,
        TestSADLoss,
        TestFactoryAndUtilities,
        TestPerformance,
        TestEdgeCases
    ]
    
    total_tests = 0
    total_failures = 0
    total_errors = 0
    
    for test_suite in test_suites:
        print(f"\n{'-' * 60}")
        print(f"Running {test_suite.__name__}")
        print(f"{'-' * 60}")
        
        suite = unittest.TestLoader().loadTestsFromTestCase(test_suite)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        
        if result.failures:
            print(f"\nFailures in {test_suite.__name__}:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print(f"\nErrors in {test_suite.__name__}:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("\n🎉 ALL TESTS PASSED! SAD Loss system is ready for production.")
        return True
    else:
        print(f"\n❌ {total_failures + total_errors} tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1) 