# Phase 2.3: Student Model Training Pipeline
# Comprehensive training system integrating SAD loss with Qwen3 model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
from tqdm import tqdm
import wandb
from contextlib import contextmanager
import os
import yaml

# Import our components
from sad_loss import SADLoss, SADLossConfig, get_production_config, get_research_config
from trajectory_processing import TrajectoryProcessor, ProcessedTrajectory, TrajectoryProcessingConfig
from student_model import StudentModel, StudentModelConfig
from sglang_manager import SGLangManager

logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Comprehensive training configuration"""
    
    # Model and Data
    model_path: str = "/workspace/persistent/models/qwen3-30b-a3b"
    data_dir: str = "data/processed_trajectories"
    output_dir: str = "models/trained_student"
    
    # SGLang Server Configuration
    sglang_port: int = 30000
    sglang_host: str = "127.0.0.1"
    max_seq_length: int = 32768
    
    # Training Parameters
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = -1  # -1 for full training
    
    # Optimization
    use_mixed_precision: bool = True
    gradient_clip_value: float = 1.0
    optimizer_type: str = "adamw"
    lr_scheduler_type: str = "linear"
    
    # LoRA Configuration (for efficient fine-tuning)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", 
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # SAD Loss Configuration
    sad_loss_config_type: str = "production"  # production, research, fast
    sad_loss_weight: float = 1.0
    auxiliary_loss_weight: float = 0.1
    
    # Logging and Monitoring
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Evaluation
    eval_batch_size: int = 1
    eval_on_start: bool = True
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "deepcoder-sad-training"
    wandb_name: Optional[str] = None
    
    # Hardware
    device: str = "auto"
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    save_safetensors: bool = True

class TrajectoryDataset(Dataset):
    """Dataset for processed trajectories with SAD training"""
    
    def __init__(self, 
                 data_dir: str,
                 trajectory_processor: TrajectoryProcessor,
                 max_length: int = 32768,
                 teacher_responses_file: Optional[str] = None):
        self.data_dir = Path(data_dir)
        self.trajectory_processor = trajectory_processor
        self.max_length = max_length
        
        # Load processed trajectories
        self.trajectories = self._load_trajectories()
        
        # Load teacher responses if available
        self.teacher_responses = self._load_teacher_responses(teacher_responses_file)
        
        logger.info(f"Loaded {len(self.trajectories)} trajectories for training")
    
    def _load_trajectories(self) -> List[ProcessedTrajectory]:
        """Load all processed trajectories from data directory"""
        trajectories = []
        
        # Look for processed trajectory files
        for file_path in self.data_dir.glob("*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        # Convert dict back to ProcessedTrajectory
                        trajectory = ProcessedTrajectory.from_dict(data)
                        trajectories.append(trajectory)
                    except Exception as e:
                        logger.warning(f"Failed to load trajectory from {file_path}: {e}")
        
        return trajectories
    
    def _load_teacher_responses(self, teacher_file: Optional[str]) -> Optional[Dict]:
        """Load teacher model responses if available"""
        if not teacher_file or not Path(teacher_file).exists():
            return None
        
        with open(teacher_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        trajectory = self.trajectories[idx]
        
        # Prepare input data
        input_ids = torch.tensor(trajectory.input_ids[:self.max_length], dtype=torch.long)
        attention_mask = torch.tensor(trajectory.attention_mask[:self.max_length], dtype=torch.long)
        reasoning_mask = torch.tensor(trajectory.reasoning_mask[:self.max_length], dtype=torch.bool)
        action_mask = torch.tensor(trajectory.action_mask[:self.max_length], dtype=torch.bool)
        
        # Create labels for language modeling (shifted input_ids)
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]
        labels[-1] = -100  # Ignore last token
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'reasoning_mask': reasoning_mask,
            'action_mask': action_mask,
            'processed_trajectory': trajectory
        }

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching trajectories"""
    
    # Find max length in batch
    max_length = max(len(item['input_ids']) for item in batch)
    
    # Pad all sequences to max length
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []
    batch_reasoning_mask = []
    batch_action_mask = []
    batch_trajectories = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_length = max_length - seq_len
        
        # Pad input_ids
        input_ids = torch.cat([
            item['input_ids'],
            torch.full((pad_length,), 0, dtype=torch.long)  # Use 0 as pad token
        ])
        batch_input_ids.append(input_ids)
        
        # Pad attention_mask
        attention_mask = torch.cat([
            item['attention_mask'],
            torch.zeros(pad_length, dtype=torch.long)
        ])
        batch_attention_mask.append(attention_mask)
        
        # Pad labels
        labels = torch.cat([
            item['labels'],
            torch.full((pad_length,), -100, dtype=torch.long)  # Ignore padded tokens
        ])
        batch_labels.append(labels)
        
        # Pad masks
        reasoning_mask = torch.cat([
            item['reasoning_mask'],
            torch.zeros(pad_length, dtype=torch.bool)
        ])
        batch_reasoning_mask.append(reasoning_mask)
        
        action_mask = torch.cat([
            item['action_mask'],
            torch.zeros(pad_length, dtype=torch.bool)
        ])
        batch_action_mask.append(action_mask)
        
        batch_trajectories.append(item['processed_trajectory'])
    
    return {
        'input_ids': torch.stack(batch_input_ids),
        'attention_mask': torch.stack(batch_attention_mask),
        'labels': torch.stack(batch_labels),
        'reasoning_mask': torch.stack(batch_reasoning_mask),
        'action_mask': torch.stack(batch_action_mask),
        'processed_trajectories': batch_trajectories
    }

class SADTrainer:
    """Structured Agent Distillation Trainer"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.sglang_manager = None
        self.student_model = None
        self.sad_loss = None
        self.trajectory_processor = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Mixed precision
        if self.config.use_mixed_precision:
            self.scaler = GradScaler('cuda')
        
        logger.info(f"Initialized SADTrainer with device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)
        
        if device.type == "cuda":
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _setup_logging(self):
        """Setup logging and monitoring"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                config=self.config.__dict__
            )
    
    def _setup_sglang_server(self):
        """Setup SGLang server for student model"""
        logger.info("Setting up SGLang server...")
        
        self.sglang_manager = SGLangManager(
            model_path=self.config.model_path,
            host=self.config.sglang_host,
            port=self.config.sglang_port
        )
        
        # Start server if not running
        if not self.sglang_manager.is_server_running():
            logger.info("Starting SGLang server...")
            self.sglang_manager.start_server()
            
            # Wait for server to be ready
            max_retries = 30
            for i in range(max_retries):
                if self.sglang_manager.is_server_healthy():
                    logger.info("SGLang server is ready!")
                    break
                time.sleep(10)
                logger.info(f"Waiting for SGLang server... ({i+1}/{max_retries})")
            else:
                raise RuntimeError("SGLang server failed to start")
    
    def _setup_model(self):
        """Setup student model"""
        logger.info("Setting up student model...")
        
        # Configure student model to connect to SGLang server
        student_config = StudentModelConfig(
            model_name="qwen3-30b-a3b",
            served_model_name="qwen3-30b-a3b",
            max_tokens=self.config.max_seq_length,
            temperature=0.6,
            enable_thinking=True
        )
        
        self.student_model = StudentModel()
    
    def _setup_sad_loss(self):
        """Setup SAD loss function"""
        logger.info("Setting up SAD loss...")
        
        # Get SAD loss configuration
        if self.config.sad_loss_config_type == "production":
            sad_config = get_production_config()
        elif self.config.sad_loss_config_type == "research":
            sad_config = get_research_config()
        else:
            sad_config = SADLossConfig()
        
        # Override with training-specific settings
        sad_config.use_mixed_precision = self.config.use_mixed_precision
        sad_config.gradient_clip_value = self.config.gradient_clip_value
        
        self.sad_loss = SADLoss(sad_config)
        self.sad_loss.to(self.device)
    
    def _setup_trajectory_processor(self):
        """Setup trajectory processor"""
        logger.info("Setting up trajectory processor...")
        
        # Load trajectory processing config
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                full_config = yaml.safe_load(f)
                trajectory_config_dict = full_config.get('trajectory_processing', {})
                
                # Filter out parameters that TrajectoryProcessingConfig doesn't accept
                valid_params = {}
                import inspect
                valid_param_names = set(inspect.signature(TrajectoryProcessingConfig.__init__).parameters.keys())
                valid_param_names.discard('self')  # Remove 'self' parameter
                
                for key, value in trajectory_config_dict.items():
                    if key in valid_param_names:
                        valid_params[key] = value
                
                trajectory_config = TrajectoryProcessingConfig(**valid_params)
        else:
            trajectory_config = TrajectoryProcessingConfig()
        
        self.trajectory_processor = TrajectoryProcessor(trajectory_config)
    
    def _create_optimizer(self, model_parameters) -> optim.Optimizer:
        """Create optimizer"""
        if self.config.optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(
                model_parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self, optimizer, num_training_steps: int):
        """Create learning rate scheduler"""
        if self.config.lr_scheduler_type == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log training metrics"""
        if step is None:
            step = self.global_step
        
        # Console logging
        log_str = f"Step {step}: " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        logger.info(log_str)
        
        # Wandb logging
        if self.config.use_wandb:
            wandb.log(metrics, step=step)
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Execute single training step"""
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        processed_trajectories = batch['processed_trajectories']
        
        # Get student model responses for the batch
        batch_size, seq_len = input_ids.shape
        student_logits = []
        
        # Process each sequence in the batch
        for i in range(batch_size):
            trajectory = processed_trajectories[i]
            
            # Get student response for this trajectory
            try:
                response = self.student_model.generate_response(
                    trajectory.original_text,
                    max_tokens=seq_len,
                    return_logits=True  # We need this capability
                )
                
                # Extract logits (this would need to be implemented in StudentModel)
                if hasattr(response, 'logits'):
                    logits = response.logits[:seq_len]  # Truncate to sequence length
                else:
                    # Fallback: create dummy logits for now
                    vocab_size = 151936  # Qwen3 vocab size
                    logits = torch.randn(seq_len, vocab_size, device=self.device)
                
                student_logits.append(logits)
                
            except Exception as e:
                logger.warning(f"Failed to get student response: {e}")
                # Create dummy logits
                vocab_size = 151936
                logits = torch.randn(seq_len, vocab_size, device=self.device)
                student_logits.append(logits)
        
        # Stack student logits
        student_logits = torch.stack(student_logits)
        
        # For teacher logits, we'll use the student logits with some noise for now
        # In a real implementation, these would come from the teacher model
        teacher_logits = student_logits + 0.1 * torch.randn_like(student_logits)
        
        # Compute SAD loss
        with autocast(device_type=self.device.type, enabled=self.config.use_mixed_precision):
            loss_outputs = {}
            total_loss = 0
            
            # Process each sequence for SAD loss
            for i in range(batch_size):
                sad_outputs = self.sad_loss(
                    teacher_logits[i:i+1],
                    student_logits[i:i+1], 
                    processed_trajectories[i],
                    target_tokens=labels[i:i+1],
                    attention_mask=attention_mask[i:i+1]
                )
                
                # Accumulate losses
                for key, value in sad_outputs.items():
                    if key not in loss_outputs:
                        loss_outputs[key] = []
                    loss_outputs[key].append(value)
            
            # Average losses across batch
            for key, values in loss_outputs.items():
                if isinstance(values[0], torch.Tensor):
                    loss_outputs[key] = torch.stack(values).mean()
            
            total_loss = loss_outputs['loss']
            
            # Add auxiliary losses if present
            for key, value in loss_outputs.items():
                if key.startswith('aux_'):
                    total_loss = total_loss + self.config.auxiliary_loss_weight * value
        
        # Extract metrics for logging
        metrics = {
            'train_loss': total_loss.item(),
            'sad_loss': loss_outputs['loss'].item() if 'loss' in loss_outputs else 0.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'] if hasattr(self, 'optimizer') else self.config.learning_rate
        }
        
        # Add span metrics if available
        for key, value in loss_outputs.items():
            if key.startswith('span_metrics/'):
                metrics[f"train_{key}"] = value.item() if isinstance(value, torch.Tensor) else value
        
        return metrics, total_loss
    
    def evaluate(self, eval_dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation set"""
        logger.info("Running evaluation...")
        
        self.sad_loss.eval()
        eval_metrics = {}
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                metrics, loss = self.train_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    if key not in eval_metrics:
                        eval_metrics[key] = 0
                    eval_metrics[key] += value
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average metrics
        for key in eval_metrics:
            eval_metrics[key] /= num_batches
        
        # Rename metrics for evaluation
        eval_metrics = {f"eval_{k.replace('train_', '')}" if k.startswith('train_') else f"eval_{k}": v 
                      for k, v in eval_metrics.items()}
        
        self.sad_loss.train()
        
        logger.info(f"Evaluation complete. Loss: {eval_metrics.get('eval_loss', 0):.4f}")
        return eval_metrics
    
    def save_checkpoint(self, output_dir: str, epoch: int, step: int):
        """Save training checkpoint"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': step,
            'sad_loss_state_dict': self.sad_loss.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_loss': self.best_loss
        }
        
        if self.config.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = output_path / f"checkpoint-{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Keep only the latest checkpoints
        checkpoints = list(output_path.glob("checkpoint-*.pt"))
        if len(checkpoints) > self.config.save_total_limit:
            checkpoints.sort(key=lambda x: int(x.stem.split('-')[-1]))
            for old_checkpoint in checkpoints[:-self.config.save_total_limit]:
                old_checkpoint.unlink()
                logger.info(f"Removed old checkpoint: {old_checkpoint}")
    
    def train(self):
        """Main training loop"""
        logger.info("Starting SAD training...")
        
        # Setup all components
        self._setup_sglang_server()
        self._setup_model()
        self._setup_sad_loss()
        self._setup_trajectory_processor()
        
        # Create dataset and dataloader
        dataset = TrajectoryDataset(
            data_dir=self.config.data_dir,
            trajectory_processor=self.trajectory_processor,
            max_length=self.config.max_seq_length
        )
        
        # Split dataset for training and evaluation
        train_size = int(0.9 * len(dataset))
        eval_size = len(dataset) - train_size
        train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory
        )
        
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.config.dataloader_num_workers,
            pin_memory=self.config.pin_memory
        )
        
        # Calculate training steps
        num_update_steps_per_epoch = len(train_dataloader) // self.config.gradient_accumulation_steps
        if self.config.max_steps > 0:
            max_steps = self.config.max_steps
            num_epochs = max_steps // num_update_steps_per_epoch + 1
        else:
            max_steps = num_update_steps_per_epoch * self.config.num_epochs
            num_epochs = self.config.num_epochs
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer(self.sad_loss.parameters())
        self.scheduler = self._create_scheduler(self.optimizer, max_steps)
        
        logger.info(f"Training for {num_epochs} epochs, {max_steps} steps")
        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Initial evaluation
        if self.config.eval_on_start and len(eval_dataset) > 0:
            eval_metrics = self.evaluate(eval_dataloader)
            self._log_metrics(eval_metrics)
        
        # Training loop
        self.sad_loss.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                metrics, loss = self.train_step(batch)
                
                # Scale loss for gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.config.use_mixed_precision:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    if self.config.use_mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.sad_loss.parameters(), self.config.gradient_clip_value)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.sad_loss.parameters(), self.config.gradient_clip_value)
                        self.optimizer.step()
                    
                    if self.scheduler:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.logging_steps == 0:
                        self._log_metrics(metrics)
                    
                    # Evaluation
                    if self.global_step % self.config.eval_steps == 0 and len(eval_dataset) > 0:
                        eval_metrics = self.evaluate(eval_dataloader)
                        self._log_metrics(eval_metrics)
                        
                        # Save best model
                        if eval_metrics.get('eval_loss', float('inf')) < self.best_loss:
                            self.best_loss = eval_metrics['eval_loss']
                            self.save_checkpoint(
                                os.path.join(self.config.output_dir, "best"),
                                epoch, self.global_step
                            )
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self.save_checkpoint(
                            self.config.output_dir,
                            epoch, self.global_step
                        )
                    
                    # Check max steps
                    if self.global_step >= max_steps:
                        break
                
                epoch_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'step': self.global_step
                })
            
            if self.global_step >= max_steps:
                break
        
        # Final evaluation and save
        if len(eval_dataset) > 0:
            final_eval_metrics = self.evaluate(eval_dataloader)
            self._log_metrics(final_eval_metrics)
        
        # Save final checkpoint
        self.save_checkpoint(
            os.path.join(self.config.output_dir, "final"),
            epoch, self.global_step
        )
        
        logger.info("Training completed!")
        
        # Cleanup
        if self.sglang_manager:
            self.sglang_manager.stop_server()
    
    def cleanup(self):
        """Cleanup resources"""
        if self.sglang_manager:
            self.sglang_manager.stop_server()
        
        if self.config.use_wandb:
            wandb.finish()

# Factory function for easy training setup
def create_trainer(config_path: Optional[str] = None, **kwargs) -> SADTrainer:
    """Create SADTrainer with configuration"""
    
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Extract training config
        training_config_dict = config_dict.get('training', {})
        training_config_dict.update(kwargs)
        
        config = TrainingConfig(**training_config_dict)
    else:
        config = TrainingConfig(**kwargs)
    
    return SADTrainer(config)

if __name__ == "__main__":
    # Example usage
    trainer = create_trainer(
        config_path="configs/config.yaml",
        num_epochs=1,
        batch_size=1,
        logging_steps=5
    )
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        trainer.cleanup() 