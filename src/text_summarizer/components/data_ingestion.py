import urllib.request as request
import zipfile
from src.text_summarizer.logger import logger
from src.text_summarizer.utils.common import get_size
from pathlib import Path
from src.text_summarizer.entity import DataIngestionConfig
import os
from src.text_summarizer.config.configuration import ConfigurationManager

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=str(self.config.local_data_file)
            )
            logger.info(f"{filename} download with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")
            
    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        
        try:
            with zipfile.ZipFile(str(self.config.local_data_file), "r") as zip_ref:
                zip_ref.extractall(str(unzip_path))
            logger.info(f"Extracted zip file to: {unzip_path}")
        except zipfile.BadZipFile:
            logger.error(f"Bad zip file: {self.config.local_data_file}")
            raise
        except Exception as e:
            logger.error(f"Error extracting zip file: {str(e)}")
            raise

# Main execution
def run_data_ingestion():
    try:
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
        logger.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise e