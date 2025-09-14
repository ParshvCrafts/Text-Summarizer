from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import evaluate
import torch
import pandas as pd
from tqdm import tqdm
from src.text_summarizer.logger import logger
from src.text_summarizer.entity import ModelEvaluationConfig
import os

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        """split the dataset into smaller batches that we can process simultaneously
        Yield successive batch-sized chunks from list_of_elements."""
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer,
                                batch_size=8, column_text="dialogue",
                                column_summary="summary"):
        """Calculate metrics on test dataset with proper error handling"""
        try:
            article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
            target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

            for article_batch, target_batch in tqdm(
                zip(article_batches, target_batches), total=len(article_batches), desc="Evaluating batches"):
                
                # Tokenize input
                inputs = tokenizer(
                    article_batch, 
                    max_length=1024,  
                    truncation=True,
                    padding="max_length", 
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate summaries
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            summaries = model.generate(
                                input_ids=inputs["input_ids"],
                                attention_mask=inputs["attention_mask"],
                                length_penalty=0.8, 
                                num_beams=4,  # Reduced for memory efficiency
                                max_length=128,
                                early_stopping=True
                            )
                    else:
                        summaries = model.generate(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            length_penalty=0.8, 
                            num_beams=4,
                            max_length=128,
                            early_stopping=True
                        )
                
                # Decode summaries
                decoded_summaries = [
                    tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                    for s in summaries
                ]

                metric.add_batch(predictions=decoded_summaries, references=target_batch)

            score = metric.compute()
            return score
            
        except Exception as e:
            logger.error(f"Error during metric calculation: {str(e)}")
            raise
    
    def evaluate(self):
        """Main evaluation method"""
        try:
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model from the same directory (best_model contains both)
            logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            
            logger.info(f"Loading model from: {self.config.model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            ).to(self.device)
            model.eval()
            
            # Load dataset
            logger.info(f"Loading dataset from: {self.config.data_path}")
            dataset_samsum_pt = load_from_disk(self.config.data_path)
            
            # Load rouge metric
            rouge_metric = evaluate.load('rouge')
            rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            
            # Calculate metrics
            logger.info("Starting evaluation on test dataset...")
            score = self.calculate_metric_on_test_ds(
                dataset_samsum_pt['test'], 
                rouge_metric, 
                model, 
                tokenizer, 
                batch_size=4,  # Reduced batch size for memory
                column_text='dialogue', 
                column_summary='summary'
            )
            
            # Process scores
            rouge_dict = {}
            for rn in rouge_names:
                if rn in score:
                    score_value = score[rn]
                    if hasattr(score_value, 'mid'):
                        rouge_dict[rn] = score_value.mid.fmeasure
                    elif isinstance(score_value, dict) and 'fmeasure' in score_value:
                        rouge_dict[rn] = score_value['fmeasure']
                    else:
                        rouge_dict[rn] = score_value
                else:
                    rouge_dict[rn] = 0.0
                    logger.warning(f"ROUGE metric {rn} not found in results")
            
            # Save results
            df = pd.DataFrame([rouge_dict], index=['pegasus-samsum'])
            os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
            df.to_csv(self.config.metric_file_name, index=False)
            
            logger.info(f"Evaluation results saved to: {self.config.metric_file_name}")
            logger.info(f"ROUGE Scores: {rouge_dict}")
            
            return rouge_dict
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise