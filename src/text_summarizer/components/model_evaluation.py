"""
Model Evaluation Component
==========================
Comprehensive evaluation pipeline with:
- ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum metrics
- Qualitative sample generation
- Length analysis
- Detailed metrics reporting
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_from_disk
import evaluate
import torch
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from src.text_summarizer.logger import logger
from src.text_summarizer.entity import ModelEvaluationConfig
import os
from datetime import datetime


class ModelEvaluation:
    """
    Comprehensive model evaluation with ROUGE metrics and qualitative analysis.
    """
    
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Evaluation will use device: {self.device}")
    
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
    
    def generate_summary(self, model, tokenizer, text, max_length=128, num_beams=4):
        """Generate a single summary for given text."""
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_beams=num_beams,
                length_penalty=0.8,
                early_stopping=True
            )
        
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    def generate_qualitative_samples(self, model, tokenizer, dataset, num_samples=20):
        """Generate sample summaries for qualitative analysis."""
        samples = []
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        logger.info(f"Generating {len(indices)} qualitative samples...")
        for idx in tqdm(indices, desc="Generating samples"):
            dialogue = dataset[int(idx)]['dialogue']
            reference = dataset[int(idx)]['summary']
            generated = self.generate_summary(model, tokenizer, dialogue)
            
            samples.append({
                'index': int(idx),
                'dialogue': dialogue,
                'reference_summary': reference,
                'generated_summary': generated,
                'reference_length': len(reference.split()),
                'generated_length': len(generated.split())
            })
        
        return samples
    
    def analyze_lengths(self, model, tokenizer, dataset, sample_size=100):
        """Analyze summary lengths compared to references."""
        indices = np.random.choice(len(dataset), min(sample_size, len(dataset)), replace=False)
        
        ref_lengths = []
        gen_lengths = []
        
        logger.info(f"Analyzing lengths on {len(indices)} samples...")
        for idx in tqdm(indices, desc="Length analysis"):
            dialogue = dataset[int(idx)]['dialogue']
            reference = dataset[int(idx)]['summary']
            generated = self.generate_summary(model, tokenizer, dialogue)
            
            ref_lengths.append(len(reference.split()))
            gen_lengths.append(len(generated.split()))
        
        analysis = {
            'reference': {
                'mean': float(np.mean(ref_lengths)),
                'std': float(np.std(ref_lengths)),
                'min': int(np.min(ref_lengths)),
                'max': int(np.max(ref_lengths))
            },
            'generated': {
                'mean': float(np.mean(gen_lengths)),
                'std': float(np.std(gen_lengths)),
                'min': int(np.min(gen_lengths)),
                'max': int(np.max(gen_lengths))
            },
            'length_ratio': float(np.mean(gen_lengths) / np.mean(ref_lengths)) if np.mean(ref_lengths) > 0 else 0
        }
        
        return analysis
    
    def evaluate(self):
        """
        Comprehensive evaluation with ROUGE metrics, qualitative samples, and length analysis.
        """
        try:
            logger.info("=" * 50)
            logger.info("MODEL EVALUATION")
            logger.info("=" * 50)
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer and model
            logger.info(f"Loading tokenizer from: {self.config.tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
            
            logger.info(f"Loading model from: {self.config.model_path}")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.config.model_path),
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)
            model.eval()
            
            # Load dataset
            logger.info(f"Loading dataset from: {self.config.data_path}")
            dataset = load_from_disk(str(self.config.data_path))
            test_dataset = dataset['test']
            
            # Ensure output directory exists
            output_dir = os.path.dirname(str(self.config.metric_file_name))
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Calculate ROUGE metrics
            logger.info("Calculating ROUGE metrics on test dataset...")
            rouge_metric = evaluate.load('rouge')
            rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
            
            score = self.calculate_metric_on_test_ds(
                test_dataset,
                rouge_metric,
                model,
                tokenizer,
                batch_size=4,
                column_text='dialogue',
                column_summary='summary'
            )
            
            # Process ROUGE scores
            rouge_dict = {}
            for rn in rouge_names:
                if rn in score:
                    score_value = score[rn]
                    if hasattr(score_value, 'mid'):
                        rouge_dict[rn] = float(score_value.mid.fmeasure)
                    elif isinstance(score_value, dict) and 'fmeasure' in score_value:
                        rouge_dict[rn] = float(score_value['fmeasure'])
                    else:
                        rouge_dict[rn] = float(score_value)
                else:
                    rouge_dict[rn] = 0.0
            
            # 2. Generate qualitative samples
            logger.info("Generating qualitative samples...")
            samples = self.generate_qualitative_samples(model, tokenizer, test_dataset, num_samples=25)
            
            # 3. Analyze lengths
            logger.info("Analyzing summary lengths...")
            length_analysis = self.analyze_lengths(model, tokenizer, test_dataset, sample_size=50)
            
            # 4. Compile comprehensive results
            results = {
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.config.model_path),
                'test_set_size': len(test_dataset),
                'rouge_scores': rouge_dict,
                'length_analysis': length_analysis
            }
            
            # Save ROUGE metrics CSV
            df = pd.DataFrame([rouge_dict])
            df.to_csv(str(self.config.metric_file_name), index=False)
            logger.info(f"ROUGE metrics saved to: {self.config.metric_file_name}")
            
            # Save comprehensive results JSON
            results_json_path = os.path.join(output_dir, "evaluation_results.json")
            with open(results_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Comprehensive results saved to: {results_json_path}")
            
            # Save qualitative samples
            samples_path = os.path.join(output_dir, "qualitative_samples.json")
            with open(samples_path, 'w') as f:
                json.dump(samples, f, indent=2)
            logger.info(f"Qualitative samples saved to: {samples_path}")
            
            # Log summary
            logger.info("=" * 50)
            logger.info("EVALUATION RESULTS")
            logger.info(f"  ROUGE-1: {rouge_dict.get('rouge1', 0):.4f}")
            logger.info(f"  ROUGE-2: {rouge_dict.get('rouge2', 0):.4f}")
            logger.info(f"  ROUGE-L: {rouge_dict.get('rougeL', 0):.4f}")
            logger.info(f"  ROUGE-Lsum: {rouge_dict.get('rougeLsum', 0):.4f}")
            logger.info(f"  Avg reference length: {length_analysis['reference']['mean']:.1f} words")
            logger.info(f"  Avg generated length: {length_analysis['generated']['mean']:.1f} words")
            logger.info(f"  Length ratio: {length_analysis['length_ratio']:.2f}")
            logger.info("=" * 50)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {str(e)}")
            raise