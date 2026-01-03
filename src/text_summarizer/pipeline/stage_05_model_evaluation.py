from src.text_summarizer.logger import logger
from src.text_summarizer.config.configuration import ConfigurationManager
from src.text_summarizer.components.model_evaluation import ModelEvaluation

class ModelEvaluationPipeline:
    def __init__(self):
        pass
    
    def main(self):
        try:
            config_manager = ConfigurationManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            results = model_evaluation.evaluate()
            logger.info(f"Model evaluation completed successfully! Results: {results}")
        except Exception as e:
            logger.error(f"Model evaluation pipeline failed: {str(e)}")
            raise