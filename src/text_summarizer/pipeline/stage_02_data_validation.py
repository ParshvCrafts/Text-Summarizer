from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.components.data_validation import DataValidation
from src.text_summarizer.logger import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config = data_validation_config)
            data_validation.validate_all_files_exist()
            logger.info("Data validation completed successfully!")
        
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise e