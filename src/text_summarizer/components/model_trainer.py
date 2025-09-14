from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
)
from src.text_summarizer.logger import logger
from src.text_summarizer.entity import ModelTrainingConfig
import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import gc
from datasets import load_from_disk

class ModelTrainer:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config
        self.setup_device()
        
    def setup_device(self):
        """Enhanced GPU setup with detailed diagnostics"""
        logger.info("=" * 50)
        logger.info("GPU DETECTION AND SETUP")
        logger.info("=" * 50)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        logger.info(f"CUDA Available: {cuda_available}")
        
        if cuda_available:
            # Get GPU info
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            logger.info(f"GPU Count: {gpu_count}")
            logger.info(f"Current GPU: {current_device}")
            logger.info(f"GPU Name: {gpu_name}")
            logger.info(f"GPU Memory: {gpu_memory:.1f} GB")
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            
            # Set device
            self.device = torch.device(f"cuda:{current_device}")
            
            # GPU optimizations for RTX 4060
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # Memory management
            torch.cuda.empty_cache()
            
            logger.info(f"Using device: {self.device}")
            logger.info("GPU optimizations enabled!")
        else:
            self.device = torch.device("cpu")
            logger.info("WARNING: CUDA not available! Using CPU")
            
        logger.info("=" * 50)
    
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
        
    def train(self):
        # Add this check at the beginning
        if os.environ.get('TRAINING_STARTED') == '1':
            logger.warning("Training already started - preventing restart")
            return
            
        os.environ['TRAINING_STARTED'] = '1'
    
        try:
            logger.info(f"Using device: {self.device}")
            self.print_memory_usage("Initial")

            # Load model and tokenizer
            logger.info(f"Loading model from: {self.config.model_ckpt}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            
            # Load model with optimizations
            model_kwargs = {
                "low_cpu_mem_usage": True,
                "use_safetensors": True,
                "torch_dtype": torch.float32
            }
            
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_ckpt,
                **model_kwargs
            )
            
            # Enable optimizations
            if torch.cuda.is_available() and hasattr(model, 'gradient_checkpointing_enable'):
                model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
            
            # Move to device
            model = model.to(self.device)
            self.print_memory_usage("After model loading")

            # Load dataset
            logger.info(f"Loading dataset from: {self.config.data_path}")
            dataset = load_from_disk(self.config.data_path)

            # Debug: Check dataset structure
            logger.info("Dataset structure:")
            for split in dataset.keys():
                logger.info(f"  {split}: {len(dataset[split])} examples")

            # Remove non-tensor columns
            columns_to_remove = ['id', 'dialogue', 'summary']
            for split in dataset.keys():
                existing_columns = set(dataset[split].column_names)
                columns_to_remove_in_split = [col for col in columns_to_remove if col in existing_columns]
                
                if columns_to_remove_in_split:
                    dataset[split] = dataset[split].remove_columns(columns_to_remove_in_split)
                    logger.info(f"Removed from {split}: {columns_to_remove_in_split}")

            logger.info(f"Remaining columns: {list(dataset['train'].column_names)}")

            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer, 
                model=model, 
                padding=True,
                pad_to_multiple_of=8 if torch.cuda.is_available() else None
            )

            # Create data loaders
            train_dataloader = DataLoader(
                dataset["train"], 
                batch_size=self.config.per_device_train_batch_size,
                shuffle=True,
                collate_fn=data_collator,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
                
            )

            val_dataloader = DataLoader(
                dataset["validation"], 
                batch_size=self.config.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=data_collator,
                num_workers=0,
                pin_memory=torch.cuda.is_available()
            )

            # Optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                eps=1e-6,
                fused=torch.cuda.is_available()
            )
            
            total_steps = len(train_dataloader) * self.config.num_train_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps
            )
            
            # Mixed precision scaler
            scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
            
            logger.info(f"Starting training for {self.config.num_train_epochs} epochs...")
            logger.info(f"Total training steps: {total_steps}")
            logger.info(f"Mixed precision training: {scaler is not None}")
            logger.info(f"Device: {self.device}")
            
            # Training metrics tracking
            training_history = []
            best_eval_loss = float('inf')
            global_step = 0
            
            # Training loop
            model.train()
            for epoch in range(self.config.num_train_epochs):
                logger.info(f"=== Epoch {epoch + 1}/{self.config.num_train_epochs} ===")
                
                epoch_loss = 0
                num_batches = 0
                
                progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
                
                for batch_idx, batch in enumerate(progress_bar):
                    # Move batch to device
                    batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Mixed precision forward pass
                    if scaler is not None:
                        with torch.amp.autocast('cuda'):
                            outputs = model(**batch)
                            loss = outputs.loss / self.config.gradient_accumulation_steps
                    else:
                        outputs = model(**batch)
                        loss = outputs.loss / self.config.gradient_accumulation_steps
                    
                    # Backward pass
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    
                    epoch_loss += loss.item() * self.config.gradient_accumulation_steps
                    num_batches += 1
                    
                    # Update weights
                    if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                        if scaler is not None:
                            # Safer gradient handling for FP16 models
                            try:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            except ValueError as e:
                                if "Attempting to unscale FP16 gradients" in str(e):
                                    # Fallback: skip unscale for FP16 gradients
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                    scaler.step(optimizer)
                                    scaler.update()
                                else:
                                    raise
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                        
                        scheduler.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                        # Clear GPU cache periodically
                        if global_step % 50 == 0:
                            self.clear_gpu_memory()
                        
                        # Logging
                        if global_step % self.config.logging_steps == 0:
                            current_lr = scheduler.get_last_lr()[0]
                            if torch.cuda.is_available():
                                gpu_mem = torch.cuda.memory_allocated(self.device) / 1024**3
                                logger.info(f"Step {global_step}/{total_steps} | Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {current_lr:.2e} | GPU Mem: {gpu_mem:.1f}GB")
                            else:
                                logger.info(f"Step {global_step}/{total_steps} | Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | LR: {current_lr:.2e}")
                        
                        # Evaluation
                        if global_step % self.config.eval_steps == 0:
                            eval_loss = self.evaluate(model, val_dataloader, scaler)
                            logger.info(f"Evaluation at step {global_step} | Eval Loss: {eval_loss:.4f}")
                            
                            # Save best model
                            if eval_loss < best_eval_loss:
                                best_eval_loss = eval_loss
                                logger.info(f"New best model! Eval loss: {eval_loss:.4f}")
                                self.save_checkpoint(model, tokenizer, global_step, eval_loss, is_best=True)
                            
                            # Record metrics
                            training_history.append({
                                'step': global_step,
                                'epoch': epoch + 1,
                                'train_loss': loss.item() * self.config.gradient_accumulation_steps,
                                'eval_loss': eval_loss,
                                'learning_rate': current_lr
                            })
                            
                            model.train()  # Back to training mode
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * self.config.gradient_accumulation_steps:.4f}',
                        'avg_loss': f'{epoch_loss/num_batches:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}',
                        'gpu_mem': f'{torch.cuda.memory_allocated(self.device) / 1024**3:.1f}GB' if torch.cuda.is_available() else 'CPU'
                    })
                
                avg_epoch_loss = epoch_loss / num_batches
                logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_epoch_loss:.4f}")
                self.print_memory_usage(f"End of Epoch {epoch + 1}")
                
                # End of epoch evaluation
                eval_loss = self.evaluate(model, val_dataloader, scaler)
                logger.info(f"End of Epoch {epoch + 1} | Eval Loss: {eval_loss:.4f}")
                
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    logger.info(f"New best model at end of epoch {epoch + 1}! Eval loss: {eval_loss:.4f}")
                    self.save_checkpoint(model, tokenizer, global_step, eval_loss, is_best=True)
            
            # Final save
            logger.info("Training completed! Saving final model...")
            self.save_final_model(model, tokenizer, training_history, best_eval_loss)
            
            logger.info("="*50)
            logger.info("TRAINING SUMMARY")
            logger.info(f"Total epochs: {self.config.num_train_epochs}")
            logger.info(f"Total steps: {global_step}")
            logger.info(f"Best evaluation loss: {best_eval_loss:.4f}")
            logger.info(f"Final learning rate: {scheduler.get_last_lr()[0]:.2e}")
            self.print_memory_usage("Training Complete")
            logger.info("="*50)
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            self.clear_gpu_memory()
            raise

    def evaluate(self, model, dataloader, scaler=None):
        """Evaluate the model"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        logger.info("Running evaluation...")
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                    
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def save_checkpoint(self, model, tokenizer, step, eval_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.config.root_dir, f"checkpoint-{step}")
        if is_best:
            checkpoint_dir = os.path.join(self.config.root_dir, "best_model")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save checkpoint info
        checkpoint_info = {
            'step': step,
            'eval_loss': eval_loss,
            'is_best': is_best
        }
        
        with open(os.path.join(checkpoint_dir, 'checkpoint_info.json'), 'w') as f:
            json.dump(checkpoint_info, f, indent=2)

    def save_final_model(self, model, tokenizer, training_history, best_eval_loss):
        """Save final model and training artifacts"""
        # Save final model
        model_dir = os.path.join(self.config.root_dir, "final_model")
        tokenizer_dir = os.path.join(self.config.root_dir, "tokenizer")
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        
        # Save training history
        with open(os.path.join(self.config.root_dir, "training_history.json"), "w") as f:
            json.dump(training_history, f, indent=2)
        
        # Save training summary
        summary = {
            "best_eval_loss": best_eval_loss,
            "total_epochs": self.config.num_train_epochs,
            "model_name": self.config.model_ckpt,
            "training_completed": True
        }
        
        with open(os.path.join(self.config.root_dir, "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Final model saved to: {model_dir}")
        logger.info(f"Tokenizer saved to: {tokenizer_dir}")
        logger.info(f"Training history saved to: {os.path.join(self.config.root_dir, 'training_history.json')}")