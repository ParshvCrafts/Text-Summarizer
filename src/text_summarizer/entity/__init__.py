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
    tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int  
    weight_decay: float
    learning_rate: float  
    logging_steps: int
    eval_strategy: str
    eval_steps: int
    save_strategy: str 
    save_steps: int
    save_total_limit: int  
    gradient_accumulation_steps: int
    lr_scheduler_type: str  
    fp16: bool  
    load_best_model_at_end: bool  
    metric_for_best_model: str  
    greater_is_better: bool  
    prediction_loss_only: bool  
    remove_unused_columns: bool  
    report_to: str 
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path 
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    