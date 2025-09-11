from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.logger import logger
from src.text_summarizer.components.model_trainer import ModelTrainer

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
    
        logger.info("Starting manual model training pipeline...")
        logger.info("This approach bypasses HuggingFace Trainer to avoid dependency issues")
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_training_config()
        model_trainer = ModelTrainer(model_trainer_config)
        model_trainer.train()
        logger.info("Manual model training pipeline completed successfully!")
            
        