import multiprocessing
import os
from src.text_summarizer.logger import logger
logger.info("Starting the main application...")
from src.text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.text_summarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.text_summarizer.pipeline.stage_04_model_trainer import ModelTrainingPipeline
from src.text_summarizer.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

if __name__ == "__main__":
    
    multiprocessing.freeze_support() 
    
    STAGE_NAME= "Data Ingestion Stage"

    try:
        logger.info(f">>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<\n\n====================")
        
    except Exception as e:
        logger.exception(e)
        raise e



    STAGE_NAME= "Data Validation Stage"

    try:
        logger.info(f">>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logger.info(f">>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<\n\n====================")
        
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME= "Data Transformation Stage"


    try:
        logger.info(f">>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
        data_transformation = DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info(f">>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<\n\n====================")
        
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME= "Model Training Stage"


    try:
        logger.info(f">>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
        model_trainer = ModelTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<\n\n====================")
        
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Model Evaluation Stage"

    try:
        logger.info(f">>>>>>>>> {STAGE_NAME} started <<<<<<<<<<<<")
        model_evaluation = ModelEvaluationPipeline()
        model_evaluation.main()
        logger.info(f">>>>>>>>> {STAGE_NAME} completed <<<<<<<<<<<<\n\n====================")
        
    except Exception as e:
        logger.exception(e)
        raise e