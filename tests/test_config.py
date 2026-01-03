"""
Tests for configuration loading and validation.
"""
import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig,
    ModelEvaluationConfig
)


class TestConfigurationManager:
    """Test suite for ConfigurationManager."""
    
    def test_config_manager_initialization(self):
        """Test that ConfigurationManager initializes correctly."""
        config_manager = ConfigurationManager()
        assert config_manager.config is not None
        assert config_manager.params is not None
    
    def test_data_ingestion_config(self):
        """Test data ingestion configuration loading."""
        config_manager = ConfigurationManager()
        config = config_manager.get_data_ingestion_config()
        
        assert isinstance(config, DataIngestionConfig)
        assert config.root_dir is not None
        assert config.source_URL is not None
        assert "http" in config.source_URL
    
    def test_data_validation_config(self):
        """Test data validation configuration loading."""
        config_manager = ConfigurationManager()
        config = config_manager.get_data_validation_config()
        
        assert isinstance(config, DataValidationConfig)
        assert config.ALL_REQUIRED_FILES is not None
        assert len(config.ALL_REQUIRED_FILES) > 0
    
    def test_data_transformation_config(self):
        """Test data transformation configuration loading."""
        config_manager = ConfigurationManager()
        config = config_manager.get_data_transformation_config()
        
        assert isinstance(config, DataTransformationConfig)
        assert config.tokenizer_name is not None
    
    def test_model_training_config(self):
        """Test model training configuration loading."""
        config_manager = ConfigurationManager()
        config = config_manager.get_model_training_config()
        
        assert isinstance(config, ModelTrainingConfig)
        assert config.model_ckpt is not None
        assert config.num_train_epochs > 0
        assert config.learning_rate > 0
    
    def test_model_evaluation_config(self):
        """Test model evaluation configuration loading."""
        config_manager = ConfigurationManager()
        config = config_manager.get_model_evaluation_config()
        
        assert isinstance(config, ModelEvaluationConfig)
        assert config.model_path is not None
        assert config.tokenizer_path is not None


class TestTokenizerModelConsistency:
    """Test that tokenizer and model configurations are consistent."""
    
    def test_tokenizer_matches_model(self):
        """Verify tokenizer_name matches model_ckpt."""
        config_manager = ConfigurationManager()
        
        transform_config = config_manager.get_data_transformation_config()
        training_config = config_manager.get_model_training_config()
        
        # The tokenizer should match the model
        assert transform_config.tokenizer_name == training_config.model_ckpt, \
            f"Tokenizer mismatch: {transform_config.tokenizer_name} != {training_config.model_ckpt}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
