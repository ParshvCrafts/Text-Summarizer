"""
Model Trainer Component
=======================
Production-grade training pipeline using HuggingFace Seq2SeqTrainer with:
- Early stopping based on validation loss
- Learning rate scheduling (cosine/linear with warmup)
- Gradient clipping
- Automatic mixed precision (FP16)
- Checkpoint management
- Comprehensive logging
"""

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
)
from src.text_summarizer.logger import logger
from src.text_summarizer.entity import ModelTrainingConfig
import os
import json
import torch
import gc
from pathlib import Path
from datasets import load_from_disk
from datetime import datetime


class ModelTrainer:
    """
    Handles model training with HuggingFace Seq2SeqTrainer.
    
    Features:
    - Automatic mixed precision (FP16)
    - Early stopping based on validation loss
    - Cosine/linear learning rate scheduling with warmup
    - Gradient clipping
    - Best model checkpointing
    """
    
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup and log device information."""
        logger.info("=" * 50)
        logger.info("DEVICE SETUP")
        logger.info("=" * 50)
        
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device("cpu")
            logger.info("WARNING: CUDA not available! Using CPU")
            
        logger.info(f"Using device: {device}")
        logger.info("=" * 50)
        return device
        
    def print_memory_usage(self, stage=""):
        """Print current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(self.device) / 1024**3
            logger.info(f"{stage} GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        
    def _find_checkpoint(self):
        """Find the latest checkpoint for resumption."""
        checkpoint_dir = Path(self.config.root_dir)
        checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
            latest = checkpoints[-1]
            logger.info(f"Found checkpoint: {latest}")
            return str(latest)
        return None
    
    def _estimate_training_time(self, num_samples: int) -> dict:
        """Estimate training time based on configuration."""
        batch_size = self.config.per_device_train_batch_size
        grad_accum = self.config.gradient_accumulation_steps
        epochs = self.config.num_train_epochs
        
        steps_per_epoch = num_samples // (batch_size * grad_accum)
        total_steps = steps_per_epoch * epochs
        
        # Rough estimate: ~0.5-2 seconds per step depending on hardware
        seconds_per_step = 1.5 if torch.cuda.is_available() else 5.0
        estimated_seconds = total_steps * seconds_per_step
        
        return {
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "estimated_minutes": estimated_seconds / 60,
            "estimated_hours": estimated_seconds / 3600
        }

    def train(self):
        """
        Train the model using HuggingFace Seq2SeqTrainer.
        
        Features:
        - Checkpoint resumption (auto-detects existing checkpoints)
        - Data sampling (use fraction of data for faster training)
        - Gradient checkpointing (trade compute for memory)
        - Early stopping based on validation loss
        - Mixed precision training (FP16)
        """
        try:
            logger.info(f"Using device: {self.device}")
            self.print_memory_usage("Initial")

            # Load tokenizer
            logger.info(f"Loading tokenizer from: {self.config.model_ckpt}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")
            
            # Load model
            logger.info(f"Loading model from: {self.config.model_ckpt}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_ckpt,
                torch_dtype=torch.float16 if self.config.fp16 and torch.cuda.is_available() else torch.float32,
            )
            logger.info(f"Model loaded: {model.config.model_type}")
            logger.info(f"Model parameters: {model.num_parameters():,}")
            
            # Enable gradient checkpointing if configured
            if self.config.gradient_checkpointing:
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing ENABLED (saves memory)")
            
            self.print_memory_usage("After model loading")

            # Load dataset
            logger.info(f"Loading dataset from: {self.config.data_path}")
            dataset = load_from_disk(str(self.config.data_path))

            # Apply data sampling if configured
            if self.config.data_sample_fraction < 1.0:
                logger.info(f"Sampling {self.config.data_sample_fraction*100:.0f}% of training data")
                train_size = int(len(dataset["train"]) * self.config.data_sample_fraction)
                dataset["train"] = dataset["train"].shuffle(seed=self.config.seed).select(range(train_size))
                logger.info(f"Training on {train_size} samples (reduced from {len(dataset['train'])})")

            # Log dataset info
            logger.info("Dataset structure:")
            for split in dataset.keys():
                logger.info(f"  {split}: {len(dataset[split])} examples")
            
            # Estimate training time
            estimate = self._estimate_training_time(len(dataset["train"]))
            logger.info("=" * 50)
            logger.info("TRAINING TIME ESTIMATE")
            logger.info(f"  Steps per epoch: {estimate['steps_per_epoch']}")
            logger.info(f"  Total steps: {estimate['total_steps']}")
            logger.info(f"  Estimated time: {estimate['estimated_minutes']:.0f} min ({estimate['estimated_hours']:.1f} hrs)")
            if estimate['estimated_hours'] > 2:
                logger.warning("⚠️  Training may take >2 hours! Consider 'quick_test' or 'laptop_friendly' profile.")
            logger.info("=" * 50)

            # Data collator for seq2seq
            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                padding=True,
                pad_to_multiple_of=8 if torch.cuda.is_available() else None
            )

            # Check for checkpoint to resume from
            resume_from_checkpoint = self._find_checkpoint()
            if resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

            # Training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=str(self.config.root_dir),
                
                # Training parameters
                num_train_epochs=self.config.num_train_epochs,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_eval_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                
                # Learning rate
                learning_rate=self.config.learning_rate,
                warmup_ratio=self.config.warmup_ratio,
                lr_scheduler_type=self.config.lr_scheduler_type,
                weight_decay=self.config.weight_decay,
                max_grad_norm=self.config.max_grad_norm,
                
                # Evaluation & saving
                eval_strategy=self.config.eval_strategy,
                eval_steps=self.config.eval_steps,
                save_strategy=self.config.save_strategy,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                
                # Logging
                logging_steps=self.config.logging_steps,
                logging_first_step=self.config.logging_first_step,
                report_to=self.config.report_to,
                
                # Performance
                fp16=self.config.fp16 and torch.cuda.is_available(),
                dataloader_num_workers=self.config.dataloader_num_workers,
                dataloader_pin_memory=self.config.dataloader_pin_memory,
                
                # Misc
                prediction_loss_only=self.config.prediction_loss_only,
                remove_unused_columns=self.config.remove_unused_columns,
                seed=self.config.seed,
                
                # Generation config for evaluation
                predict_with_generate=False,  # Set True if computing ROUGE during training
            )

            # Early stopping callback
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=self.config.early_stopping_patience,
                early_stopping_threshold=self.config.early_stopping_threshold
            )

            # Initialize trainer
            trainer = Seq2SeqTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                tokenizer=tokenizer,
                data_collator=data_collator,
                callbacks=[early_stopping_callback],
            )

            # Log training configuration
            logger.info("=" * 50)
            logger.info("TRAINING CONFIGURATION")
            logger.info(f"  Model: {self.config.model_ckpt}")
            logger.info(f"  Epochs: {self.config.num_train_epochs}")
            logger.info(f"  Batch size: {self.config.per_device_train_batch_size}")
            logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
            logger.info(f"  Effective batch size: {self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}")
            logger.info(f"  Learning rate: {self.config.learning_rate}")
            logger.info(f"  LR scheduler: {self.config.lr_scheduler_type}")
            logger.info(f"  Warmup ratio: {self.config.warmup_ratio}")
            logger.info(f"  Early stopping patience: {self.config.early_stopping_patience}")
            logger.info(f"  Gradient checkpointing: {self.config.gradient_checkpointing}")
            logger.info(f"  Data sample fraction: {self.config.data_sample_fraction}")
            logger.info(f"  FP16: {self.config.fp16 and torch.cuda.is_available()}")
            logger.info("=" * 50)

            # Train (with checkpoint resumption if available)
            logger.info("Starting training...")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Save the best model
            best_model_dir = os.path.join(self.config.root_dir, "best_model")
            logger.info(f"Saving best model to: {best_model_dir}")
            trainer.save_model(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

            # Save training metrics
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            
            trainer.save_state()

            # Log final summary
            logger.info("=" * 50)
            logger.info("TRAINING COMPLETE")
            logger.info(f"  Total steps: {train_result.global_step}")
            logger.info(f"  Training loss: {metrics.get('train_loss', 'N/A'):.4f}")
            logger.info(f"  Training time: {metrics.get('train_runtime', 0):.2f}s")
            logger.info(f"  Best model saved to: {best_model_dir}")
            self.print_memory_usage("Training Complete")
            logger.info("=" * 50)

            # Clean up
            self.clear_gpu_memory()
            
            return train_result

        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self.clear_gpu_memory()
            raise