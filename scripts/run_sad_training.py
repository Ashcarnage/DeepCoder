#!/usr/bin/env python3
"""
SAD Training Script for Phase 2.3
Run complete structured agent distillation training
"""

import sys
import os
import argparse
import logging
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_pipeline import create_trainer, TrainingConfig

def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/sad_training.log')
        ]
    )

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger = logging.getLogger(__name__)
    
    # Check if model exists
    model_path = "/workspace/persistent/models/qwen3-30b-a3b"
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        logger.error("Please download the model first:")
        logger.error("huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir /workspace/persistent/models/qwen3-30b-a3b")
        return False
    
    # Check if data directory exists
    data_dir = "data/processed_trajectories"
    if not Path(data_dir).exists() or not list(Path(data_dir).glob("*.jsonl")):
        logger.error(f"No processed trajectories found in {data_dir}")
        logger.error("Please run trajectory processing first:")
        logger.error("python scripts/process_trajectories.py")
        return False
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training will be slow on CPU.")
    else:
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return True

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Run SAD Training")
    
    # Configuration arguments
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="models/sad_trained",
                       help="Output directory for trained model")
    parser.add_argument("--data-dir", type=str, default="data/processed_trajectories",
                       help="Directory containing processed trajectories")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=2,
                       help="Training batch size")
    parser.add_argument("--num-epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=-1,
                       help="Maximum training steps (-1 for full training)")
    
    # SAD Loss arguments
    parser.add_argument("--sad-loss-type", type=str, default="production",
                       choices=["production", "research", "fast"],
                       help="SAD loss configuration type")
    
    # Logging arguments
    parser.add_argument("--logging-steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--eval-steps", type=int, default=100,
                       help="Evaluate every N steps")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    
    # Wandb arguments
    parser.add_argument("--use-wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="deepcoder-sad-training",
                       help="Wandb project name")
    parser.add_argument("--wandb-name", type=str, default=None,
                       help="Wandb run name")
    
    # Debug arguments
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Setup components but don't run training")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting SAD Training Pipeline")
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        return 1
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Create trainer configuration
    trainer_config = TrainingConfig(
        model_path="/workspace/persistent/models/qwen3-30b-a3b",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        sad_loss_config_type=args.sad_loss_type,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )
    
    logger.info("Training Configuration:")
    logger.info(f"  Model Path: {trainer_config.model_path}")
    logger.info(f"  Data Dir: {trainer_config.data_dir}")
    logger.info(f"  Output Dir: {trainer_config.output_dir}")
    logger.info(f"  Batch Size: {trainer_config.batch_size}")
    logger.info(f"  Learning Rate: {trainer_config.learning_rate}")
    logger.info(f"  Epochs: {trainer_config.num_epochs}")
    logger.info(f"  Max Steps: {trainer_config.max_steps}")
    logger.info(f"  SAD Loss Type: {trainer_config.sad_loss_config_type}")
    
    # Create trainer
    trainer = create_trainer(config_path=args.config, **trainer_config.__dict__)
    
    if args.dry_run:
        logger.info("Dry run mode - setting up components but not training")
        try:
            trainer._setup_sglang_server()
            trainer._setup_model()
            trainer._setup_sad_loss()
            trainer._setup_trajectory_processor()
            logger.info("✅ All components set up successfully!")
            logger.info("Dry run complete. Use --no-dry-run to start training.")
            return 0
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            return 1
        finally:
            trainer.cleanup()
    
    # Run training
    try:
        logger.info("🚀 Starting SAD training...")
        trainer.train()
        logger.info("✅ Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
        
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 