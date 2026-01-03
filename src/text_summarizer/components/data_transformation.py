from src.text_summarizer.logger import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
import os
from src.text_summarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        # Load tokenizer from config to ensure consistency with model
        logger.info(f"Loading tokenizer from: {self.config.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        logger.info(f"Tokenizer vocabulary size: {self.tokenizer.vocab_size}")
        
    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'], max_length=1024, truncation=True)
        
        # Use text_target parameter instead of deprecated as_target_tokenizer()
        target_encodings = self.tokenizer(example_batch['summary'], max_length=128, truncation=True)
            
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']}
        
    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, 'samsum_dataset'))