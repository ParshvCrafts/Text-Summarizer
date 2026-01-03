from src.text_summarizer.constants import *
from src.text_summarizer.utils.common import read_yaml, create_directories
from src.text_summarizer.entity import (DataIngestionConfig, 
                                        DataValidationConfig,
                                        DataTransformationConfig,
                                        ModelTrainingConfig,
                                        ModelEvaluationConfig)
import os

class ConfigurationManager:
    def __init__(self,
                 config_filepath=CONFIG_FILE_PATH,
                 params_filepath=PARAMS_FILE_PATH):
        
        # Check if config files exist
        if not os.path.exists(config_filepath):
            raise FileNotFoundError(f"Configuration file not found: {config_filepath}")
        if not os.path.exists(params_filepath):
            raise FileNotFoundError(f"Parameters file not found: {params_filepath}")
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # Check if artifacts_root exists in config
        if not hasattr(self.config, 'artifacts_root'):
            raise KeyError("'artifacts_root' key not found in config.yaml. Please add: artifacts_root: artifacts")
        
        create_directories([self.config.artifacts_root])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Check if data_ingestion section exists
        if not hasattr(self.config, 'data_ingestion'):
            raise KeyError("'data_ingestion' section not found in config.yaml")
        
        config = self.config.data_ingestion
        
        # Check required keys
        required_keys = ['root_dir', 'source_URL', 'local_data_file', 'unzip_dir']
        for key in required_keys:
            if not hasattr(config, key):
                raise KeyError(f"'{key}' not found in data_ingestion section of config.yaml")
        
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=str(config.source_URL),
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )
        
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        
        create_directories([config.root_dir])
        
        data_validation_config = DataValidationConfig(
            root_dir= config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            ALL_REQUIRED_FILES= config.ALL_REQUIRED_FILES,
        )
        
        return data_validation_config
        
        
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])
        
        data_transformation_config = DataTransformationConfig(
            root_dir= config.root_dir,
            data_path= config.data_path,
            tokenizer_name= config.tokenizer_name
        )
        
        return data_transformation_config
    
    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainingConfig(
            # Paths
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=str(config.model_ckpt),
            
            # Core training
            num_train_epochs=int(params.num_train_epochs),
            per_device_train_batch_size=int(params.per_device_train_batch_size),
            per_device_eval_batch_size=int(params.per_device_eval_batch_size),
            gradient_accumulation_steps=int(params.gradient_accumulation_steps),
            
            # Learning rate & scheduling
            learning_rate=float(params.learning_rate),
            warmup_ratio=float(params.warmup_ratio),
            warmup_steps=int(params.warmup_steps),
            lr_scheduler_type=str(params.lr_scheduler_type),
            weight_decay=float(params.weight_decay),
            max_grad_norm=float(params.max_grad_norm),
            
            # Evaluation & checkpointing
            eval_strategy=str(params.eval_strategy),
            eval_steps=int(params.eval_steps),
            save_strategy=str(params.save_strategy),
            save_steps=int(params.save_steps),
            save_total_limit=int(params.save_total_limit),
            load_best_model_at_end=bool(params.load_best_model_at_end),
            metric_for_best_model=str(params.metric_for_best_model),
            greater_is_better=bool(params.greater_is_better),
            
            # Early stopping
            early_stopping_patience=int(params.early_stopping_patience),
            early_stopping_threshold=float(params.early_stopping_threshold),
            
            # Logging
            logging_steps=int(params.logging_steps),
            logging_first_step=bool(params.logging_first_step),
            report_to=str(params.report_to),
            
            # Performance
            fp16=bool(params.fp16),
            dataloader_num_workers=int(params.dataloader_num_workers),
            dataloader_pin_memory=bool(params.dataloader_pin_memory),
            
            # Sequence lengths
            max_input_length=int(params.max_input_length),
            max_target_length=int(params.max_target_length),
            
            # Misc
            prediction_loss_only=bool(params.prediction_loss_only),
            remove_unused_columns=bool(params.remove_unused_columns),
            seed=int(params.seed),
            
            # Resource optimization
            gradient_checkpointing=bool(getattr(params, 'gradient_checkpointing', True)),
            data_sample_fraction=float(getattr(params, 'data_sample_fraction', 1.0))
        )

        return model_trainer_config
    
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        
        create_directories([config.root_dir])
        
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.tokenizer_path),
            metric_file_name=Path(config.metric_file_name)
        )
        
        return model_evaluation_config