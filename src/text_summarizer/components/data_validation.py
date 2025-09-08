import os
from src.text_summarizer.logger import logger
from src.text_summarizer.config.configuration import DataValidationConfig

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config
        
    def validate_all_files_exist(self) -> bool:
        try:
            data_dir = os.path.join("artifacts", "data_ingestion", "samsum_dataset")
            all_files = os.listdir(data_dir)
            
            # Check if ALL required files exist
            missing_files = []
            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    missing_files.append(required_file)
            
            validation_status = len(missing_files) == 0
            
            # Write detailed status
            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f'Validation status: {validation_status}\n')
                if missing_files:
                    f.write(f'Missing files: {", ".join(missing_files)}\n')
                else:
                    f.write('All required files are present\n')
            
            if validation_status:
                logger.info("✅ All required files are present")
            else:
                logger.error(f"❌ Missing files: {missing_files}")
                
            return validation_status
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            raise e