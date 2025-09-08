from src.text_summarizer.constants import *
from src.text_summarizer.utils.common import read_yaml, create_directories
from src.text_summarizer.entity import (DataIngestionConfig, DataValidationConfig)
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