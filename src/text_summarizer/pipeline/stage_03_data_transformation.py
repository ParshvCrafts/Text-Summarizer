from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.components.data_transformation import DataTransformation
from src.text_summarizer.logger import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass
    def main(self):
        try:
            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config = data_transformation_config)
            data_transformation.convert()
            logger.info("Data transformation completed successfully!")
        
        except Exception as e:
            logger.error(f"Data transformation failed: {str(e)}")
            raise e