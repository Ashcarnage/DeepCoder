#!/usr/bin/env python3
"""
Test Script for SAD Training Pipeline
Validates Phase 2.3 implementation and integration
"""

import sys
import os
import unittest
import torch
import json
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_pipeline import (
    TrainingConfig, SADTrainer, TrajectoryDataset, 
    collate_fn, create_trainer
)
from trajectory_processing import (
    ProcessedTrajectory, TokenSpan, SpanType
)

class TestTrainingConfig(unittest.TestCase):
    """Test training configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        
        self.assertEqual(config.model_path, "/workspace/persistent/models/qwen3-30b-a3b")
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.learning_rate, 2e-4)
        self.assertTrue(config.use_mixed_precision)
        self.assertTrue(config.use_lora)
        self.assertEqual(config.sad_loss_config_type, "production")
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = TrainingConfig(
            batch_size=4,
            learning_rate=1e-4,
            num_epochs=5,
            use_wandb=False
        )
        
        self.assertEqual(config.batch_size, 4)
        self.assertEqual(config.learning_rate, 1e-4)
        self.assertEqual(config.num_epochs, 5)
        self.assertFalse(config.use_wandb)

class TestTrajectoryDataset(unittest.TestCase):
    """Test trajectory dataset for training"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock trajectory processor
        self.trajectory_processor = MagicMock()
        
        # Create sample processed trajectories
        self.sample_trajectories = self._create_sample_trajectories()
        
        # Save sample data
        self.data_file = os.path.join(self.temp_dir, "trajectories.jsonl")
        with open(self.data_file, 'w') as f:
            for traj in self.sample_trajectories:
                f.write(json.dumps(traj.to_dict()) + '\n')
    
    def _create_sample_trajectories(self):
        """Create sample processed trajectories"""
        trajectories = []
        
        for i in range(5):
            spans = [
                TokenSpan(SpanType.REASONING, 0, 10, f"thinking {i}"),
                TokenSpan(SpanType.ACTION, 10, 20, f"action {i}"),
                TokenSpan(SpanType.OBSERVATION, 20, 30, f"observation {i}")
            ]
            
            trajectory = ProcessedTrajectory(
                trajectory_id=f"test_trajectory_{i}",
                input_ids=list(range(30)),
                attention_mask=[1] * 30,
                reasoning_mask=[1] * 10 + [0] * 20,
                action_mask=[0] * 10 + [1] * 10 + [0] * 10,
                spans=spans,
                original_text=f"Test trajectory {i}",
                metadata={'test': True}
            )
            trajectories.append(trajectory)
        
        return trajectories
    
    def test_dataset_loading(self):
        """Test dataset loading from files"""
        dataset = TrajectoryDataset(
            data_dir=self.temp_dir,
            trajectory_processor=self.trajectory_processor,
            max_length=32
        )
        
        self.assertEqual(len(dataset), 5)
        
        # Test getting item
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        self.assertIn('reasoning_mask', item)
        self.assertIn('action_mask', item)
        self.assertIn('processed_trajectory', item)
        
        # Check tensor shapes
        self.assertEqual(len(item['input_ids']), 30)
        self.assertEqual(len(item['attention_mask']), 30)
        self.assertEqual(len(item['labels']), 30)
    
    def test_collate_function(self):
        """Test batch collation"""
        dataset = TrajectoryDataset(
            data_dir=self.temp_dir,
            trajectory_processor=self.trajectory_processor,
            max_length=32
        )
        
        # Get batch of items
        batch_items = [dataset[i] for i in range(3)]
        
        # Test collate function
        batch = collate_fn(batch_items)
        
        self.assertIn('input_ids', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('labels', batch)
        self.assertIn('reasoning_mask', batch)
        self.assertIn('action_mask', batch)
        self.assertIn('processed_trajectories', batch)
        
        # Check batch dimensions
        self.assertEqual(batch['input_ids'].shape[0], 3)  # batch size
        self.assertEqual(batch['input_ids'].shape[1], 30)  # sequence length
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestSADTrainer(unittest.TestCase):
    """Test SAD trainer functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create minimal config for testing
        self.config = TrainingConfig(
            model_path="/workspace/persistent/models/qwen3-30b-a3b",
            data_dir=self.temp_dir,
            output_dir=os.path.join(self.temp_dir, "output"),
            num_epochs=1,
            batch_size=1,
            max_steps=5,
            logging_steps=1,
            eval_steps=2,
            save_steps=3,
            use_wandb=False,
            dataloader_num_workers=0,
            eval_on_start=False
        )
        
        # Create sample data
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample training data"""
        # Create sample processed trajectories
        trajectories = []
        
        for i in range(3):
            spans = [
                TokenSpan(SpanType.REASONING, 0, 5, f"thinking {i}"),
                TokenSpan(SpanType.ACTION, 5, 10, f"action {i}")
            ]
            
            trajectory = ProcessedTrajectory(
                trajectory_id=f"test_trajectory_{i}",
                input_ids=list(range(10)),
                attention_mask=[1] * 10,
                reasoning_mask=[1] * 5 + [0] * 5,
                action_mask=[0] * 5 + [1] * 5,
                spans=spans,
                original_text=f"Test trajectory {i} with thinking and action",
                metadata={'test': True}
            )
            trajectories.append(trajectory)
        
        # Save to file
        data_file = os.path.join(self.temp_dir, "test_trajectories.jsonl")
        with open(data_file, 'w') as f:
            for traj in trajectories:
                f.write(json.dumps(traj.to_dict()) + '\n')
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = SADTrainer(self.config)
        
        self.assertEqual(trainer.config, self.config)
        self.assertEqual(trainer.global_step, 0)
        self.assertEqual(trainer.epoch, 0)
        self.assertEqual(trainer.best_loss, float('inf'))
        
        # Check device setup
        self.assertIsInstance(trainer.device, torch.device)
    
    @patch('training_pipeline.SGLangManager')
    @patch('training_pipeline.StudentModel')
    def test_component_setup(self, mock_student_model, mock_sglang_manager):
        """Test component setup with mocks"""
        # Mock SGLang manager
        mock_manager = MagicMock()
        mock_manager.is_server_running.return_value = True
        mock_manager.is_server_healthy.return_value = True
        mock_sglang_manager.return_value = mock_manager
        
        # Mock student model
        mock_model = MagicMock()
        mock_student_model.return_value = mock_model
        
        trainer = SADTrainer(self.config)
        
        # Test component setup
        trainer._setup_sglang_server()
        self.assertIsNotNone(trainer.sglang_manager)
        
        trainer._setup_model()
        self.assertIsNotNone(trainer.student_model)
        
        trainer._setup_sad_loss()
        self.assertIsNotNone(trainer.sad_loss)
        
        trainer._setup_trajectory_processor()
        self.assertIsNotNone(trainer.trajectory_processor)
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        trainer = SADTrainer(self.config)
        
        # Create dummy parameters
        dummy_model = torch.nn.Linear(10, 5)
        optimizer = trainer._create_optimizer(dummy_model.parameters())
        
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]['lr'], self.config.learning_rate)
        self.assertEqual(optimizer.param_groups[0]['weight_decay'], self.config.weight_decay)
    
    def test_scheduler_creation(self):
        """Test learning rate scheduler creation"""
        trainer = SADTrainer(self.config)
        
        # Create dummy optimizer
        dummy_model = torch.nn.Linear(10, 5)
        optimizer = trainer._create_optimizer(dummy_model.parameters())
        
        scheduler = trainer._create_scheduler(optimizer, num_training_steps=100)
        
        # Should create scheduler for linear type
        self.assertIsNotNone(scheduler)
    
    def test_checkpoint_saving(self):
        """Test checkpoint saving functionality"""
        trainer = SADTrainer(self.config)
        
        # Setup minimal components
        trainer._setup_sad_loss()
        dummy_model = torch.nn.Linear(10, 5)
        trainer.optimizer = trainer._create_optimizer(dummy_model.parameters())
        trainer.scheduler = trainer._create_scheduler(trainer.optimizer, 100)
        
        # Save checkpoint
        checkpoint_dir = os.path.join(self.temp_dir, "checkpoints")
        trainer.save_checkpoint(checkpoint_dir, epoch=0, step=1)
        
        # Check checkpoint file exists
        checkpoint_path = Path(checkpoint_dir) / "checkpoint-1.pt"
        self.assertTrue(checkpoint_path.exists())
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.assertIn('epoch', checkpoint)
        self.assertIn('global_step', checkpoint)
        self.assertIn('sad_loss_state_dict', checkpoint)
        self.assertIn('optimizer_state_dict', checkpoint)
        self.assertIn('config', checkpoint)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestTrainingIntegration(unittest.TestCase):
    """Test full training integration"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config for quick testing
        self.config = TrainingConfig(
            model_path="/workspace/persistent/models/qwen3-30b-a3b",
            data_dir=self.temp_dir,
            output_dir=os.path.join(self.temp_dir, "output"),
            num_epochs=1,
            batch_size=1,
            max_steps=2,  # Very short training for testing
            logging_steps=1,
            eval_steps=10,  # No evaluation during short test
            save_steps=10,  # No saving during short test
            use_wandb=False,
            dataloader_num_workers=0,
            eval_on_start=False,
            gradient_accumulation_steps=1
        )
        
        # Create sample data
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample training data"""
        trajectories = []
        
        for i in range(2):
            spans = [
                TokenSpan(SpanType.REASONING, 0, 3, f"thinking {i}"),
                TokenSpan(SpanType.ACTION, 3, 6, f"action {i}")
            ]
            
            trajectory = ProcessedTrajectory(
                trajectory_id=f"test_trajectory_{i}",
                input_ids=[1, 2, 3, 4, 5, 6],  # Small sequence
                attention_mask=[1, 1, 1, 1, 1, 1],
                reasoning_mask=[1, 1, 1, 0, 0, 0],
                action_mask=[0, 0, 0, 1, 1, 1],
                spans=spans,
                original_text=f"Test trajectory {i}",
                metadata={'test': True}
            )
            trajectories.append(trajectory)
        
        # Save to file
        data_file = os.path.join(self.temp_dir, "test_trajectories.jsonl")
        with open(data_file, 'w') as f:
            for traj in trajectories:
                f.write(json.dumps(traj.to_dict()) + '\n')
    
    @patch('training_pipeline.SGLangManager')
    @patch('training_pipeline.StudentModel')
    @patch('training_pipeline.wandb')
    def test_training_step_execution(self, mock_wandb, mock_student_model, mock_sglang_manager):
        """Test training step execution with mocks"""
        # Mock SGLang manager
        mock_manager = MagicMock()
        mock_manager.is_server_running.return_value = True
        mock_manager.is_server_healthy.return_value = True
        mock_sglang_manager.return_value = mock_manager
        
        # Mock student model response
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.logits = torch.randn(6, 100)  # 6 tokens, 100 vocab size
        mock_model.generate_response.return_value = mock_response
        mock_student_model.return_value = mock_model
        
        trainer = SADTrainer(self.config)
        
        # Setup components
        trainer._setup_sglang_server()
        trainer._setup_model()
        trainer._setup_sad_loss()
        trainer._setup_trajectory_processor()
        
        # Create dataset and dataloader
        from training_pipeline import TrajectoryDataset
        dataset = TrajectoryDataset(
            data_dir=self.temp_dir,
            trajectory_processor=trainer.trajectory_processor,
            max_length=10
        )
        
        # Get a batch
        batch_items = [dataset[0]]
        batch = collate_fn(batch_items)
        
        # Test training step
        metrics, loss = trainer.train_step(batch)
        
        # Verify metrics
        self.assertIsInstance(metrics, dict)
        self.assertIn('train_loss', metrics)
        self.assertIn('sad_loss', metrics)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_factory_function(self):
        """Test trainer factory function"""
        # Test with config dict
        trainer = create_trainer(
            batch_size=1,
            num_epochs=1,
            use_wandb=False
        )
        
        self.assertIsInstance(trainer, SADTrainer)
        self.assertEqual(trainer.config.batch_size, 1)
        self.assertEqual(trainer.config.num_epochs, 1)
        self.assertFalse(trainer.config.use_wandb)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)

class TestTrainingConfiguration(unittest.TestCase):
    """Test training configuration integration"""
    
    def test_config_yaml_integration(self):
        """Test configuration loading from YAML"""
        # Create temporary config file
        config_data = {
            'training': {
                'batch_size': 4,
                'learning_rate': 1e-4,
                'num_epochs': 2,
                'use_wandb': False
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            trainer = create_trainer(config_path=config_path)
            
            self.assertEqual(trainer.config.batch_size, 4)
            self.assertEqual(trainer.config.learning_rate, 1e-4)
            self.assertEqual(trainer.config.num_epochs, 2)
            self.assertFalse(trainer.config.use_wandb)
        finally:
            os.unlink(config_path)
    
    def test_sad_loss_config_types(self):
        """Test different SAD loss configuration types"""
        configs = ["production", "research", "fast"]
        
        for config_type in configs:
            trainer_config = TrainingConfig(
                sad_loss_config_type=config_type,
                use_wandb=False
            )
            
            trainer = SADTrainer(trainer_config)
            trainer._setup_sad_loss()
            
            self.assertIsNotNone(trainer.sad_loss)
            self.assertEqual(trainer.config.sad_loss_config_type, config_type)

def run_training_tests():
    """Run all training pipeline tests"""
    
    print("=" * 80)
    print("TRAINING PIPELINE TEST SUITE")
    print("=" * 80)
    
    test_suites = [
        TestTrainingConfig,
        TestTrajectoryDataset,
        TestSADTrainer,
        TestTrainingIntegration,
        TestTrainingConfiguration
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
        print("\n🎉 ALL TESTS PASSED! Training pipeline is ready for use.")
        return True
    else:
        print(f"\n❌ {total_failures + total_errors} tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_training_tests()
    sys.exit(0 if success else 1) 