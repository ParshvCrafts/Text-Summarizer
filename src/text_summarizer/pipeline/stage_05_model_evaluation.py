from src.text_summarizer.logger import logger
from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()
        logger.info("Model evaluation completed successfully!")