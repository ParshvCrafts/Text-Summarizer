from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str  # Model identifier string, not a path

@dataclass(frozen=True)
class ModelTrainingConfig:
    """Configuration for model training with all hyperparameters."""
    # Paths
    root_dir: Path
    data_path: Path
    model_ckpt: str  # HuggingFace model identifier
    
    # Core training
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    
    # Learning rate & scheduling
    learning_rate: float
    warmup_ratio: float
    warmup_steps: int
    lr_scheduler_type: str
    weight_decay: float
    max_grad_norm: float  # Gradient clipping
    
    # Evaluation & checkpointing
    eval_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    
    # Early stopping
    early_stopping_patience: int
    early_stopping_threshold: float
    
    # Logging
    logging_steps: int
    logging_first_step: bool
    report_to: str
    
    # Performance
    fp16: bool
    dataloader_num_workers: int
    dataloader_pin_memory: bool
    
    # Sequence lengths
    max_input_length: int
    max_target_length: int
    
    # Misc
    prediction_loss_only: bool
    remove_unused_columns: bool
    seed: int 
    
    # Resource optimization
    gradient_checkpointing: bool  # Trade compute for memory
    data_sample_fraction: float   # Fraction of data to use (0.1 = 10%) 
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path 
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    