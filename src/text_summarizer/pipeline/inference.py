"""
Inference Pipeline
===================
Production-ready inference pipeline for text summarization.

Features:
- Model and tokenizer caching
- Batch prediction support
- Configurable generation parameters
- CLI interface
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from src.text_summarizer.logger import logger
from src.text_summarizer.config.configuration import ConfigurationManager
from pathlib import Path
from typing import List, Optional, Union
import argparse


class Summarizer:
    """
    Text summarization inference class.
    
    Usage:
        summarizer = Summarizer()
        summary = summarizer.summarize("Your dialogue text here...")
        
        # Or batch processing
        summaries = summarizer.summarize_batch(["text1", "text2", "text3"])
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    
    def __new__(cls, model_path: Optional[str] = None):
        """Singleton pattern to cache model across calls."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the summarizer.
        
        Args:
            model_path: Path to the trained model. If None, uses config default.
        """
        if self._model is not None:
            return  # Already initialized
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing Summarizer on device: {self.device}")
        
        # Get model path from config if not provided
        if model_path is None:
            config_manager = ConfigurationManager()
            eval_config = config_manager.get_model_evaluation_config()
            model_path = str(eval_config.model_path)
        
        self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model and tokenizer from path or HuggingFace Hub."""
        logger.info(f"Loading model from: {model_path}")
        
        # Check if local path exists, otherwise try HuggingFace Hub
        if not Path(model_path).exists():
            # Try to load from config's model_ckpt (HuggingFace Hub)
            try:
                config_manager = ConfigurationManager()
                training_config = config_manager.get_model_training_config()
                hub_model = training_config.model_ckpt
                logger.info(f"Local model not found. Loading from HuggingFace Hub: {hub_model}")
                model_path = hub_model
            except Exception as e:
                raise FileNotFoundError(
                    f"Model not found at {model_path} and could not load from HuggingFace Hub. "
                    f"Error: {e}. Please train the model first by running: python main.py"
                )
        
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)
        self._model.eval()
        
        logger.info(f"Model loaded successfully. Vocab size: {self._tokenizer.vocab_size}")
    
    @property
    def model(self):
        return self._model
    
    @property
    def tokenizer(self):
        return self._tokenizer
    
    def preprocess(self, text: str) -> str:
        """
        Preprocess input text for summarization.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Strip excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove excessive newlines but preserve dialogue structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def postprocess(self, summary: str) -> str:
        """
        Postprocess generated summary for quality.
        
        Args:
            summary: Raw generated summary
            
        Returns:
            Cleaned and polished summary
        """
        if not summary:
            return ""
        
        # Strip whitespace
        summary = summary.strip()
        
        # Capitalize first letter
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure proper ending punctuation
        if summary and summary[-1] not in '.!?':
            summary += '.'
        
        # Remove repeated phrases (simple deduplication)
        import re
        # Remove exact duplicate sentences
        sentences = re.split(r'(?<=[.!?])\s+', summary)
        seen = set()
        unique_sentences = []
        for s in sentences:
            s_lower = s.lower().strip()
            if s_lower and s_lower not in seen:
                seen.add(s_lower)
                unique_sentences.append(s)
        summary = ' '.join(unique_sentences)
        
        return summary
    
    def _get_length_params(self, target_length: str) -> dict:
        """
        Get generation parameters for target length.
        
        Args:
            target_length: 'short', 'medium', or 'long'
            
        Returns:
            Dictionary of generation parameters
        """
        length_configs = {
            'short': {'max_length': 50, 'min_length': 10, 'length_penalty': 0.5},
            'medium': {'max_length': 100, 'min_length': 30, 'length_penalty': 0.8},
            'long': {'max_length': 200, 'min_length': 50, 'length_penalty': 1.2},
        }
        return length_configs.get(target_length, length_configs['medium'])
    
    def summarize(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 0.8,
        early_stopping: bool = True,
        target_length: str = None,
        return_confidence: bool = False
    ):
        """
        Generate a summary for the given text.
        
        Args:
            text: Input dialogue/text to summarize
            max_length: Maximum length of generated summary
            min_length: Minimum length of generated summary
            num_beams: Number of beams for beam search
            length_penalty: Length penalty (< 1.0 = shorter, > 1.0 = longer)
            early_stopping: Stop when all beams finish
            target_length: Optional preset ('short', 'medium', 'long')
            return_confidence: If True, return (summary, confidence) tuple
            
        Returns:
            Generated summary string, or (summary, confidence) if return_confidence=True
        """
        if not text or not text.strip():
            return ("", 0.0) if return_confidence else ""
        
        # Preprocess input
        text = self.preprocess(text)
        
        # Apply length preset if specified
        if target_length:
            length_params = self._get_length_params(target_length)
            max_length = length_params['max_length']
            min_length = length_params['min_length']
            length_penalty = length_params['length_penalty']
        
        # Tokenize input
        inputs = self._tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary with scores for confidence
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                output_scores=return_confidence,
                return_dict_in_generate=return_confidence
            )
        
        # Extract summary IDs and confidence
        if return_confidence:
            summary_ids = outputs.sequences
            # Calculate confidence from sequence scores
            if hasattr(outputs, 'sequences_scores') and outputs.sequences_scores is not None:
                # Convert log probability to confidence (0-1 scale)
                log_prob = outputs.sequences_scores[0].item()
                # Normalize: typical log probs are -1 to -5, map to 0.5-1.0
                confidence = min(1.0, max(0.0, 1.0 + log_prob / 10))
            else:
                confidence = 0.85  # Default confidence if scores not available
        else:
            summary_ids = outputs
            confidence = None
        
        # Decode
        summary = self._tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Postprocess
        summary = self.postprocess(summary)
        
        if return_confidence:
            return summary, confidence
        return summary
    
    def summarize_batch(
        self,
        texts: List[str],
        max_length: int = 128,
        min_length: int = 30,
        num_beams: int = 4,
        batch_size: int = 8
    ) -> List[str]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Number of beams for beam search
            batch_size: Batch size for processing
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = self._tokenizer(
                batch,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                summary_ids = self._model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=0.8,
                    early_stopping=True
                )
            
            # Decode batch
            batch_summaries = self._tokenizer.batch_decode(
                summary_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            summaries.extend(batch_summaries)
        
        return summaries


def main():
    """CLI interface for text summarization."""
    parser = argparse.ArgumentParser(
        description="Text Summarizer - Generate summaries for dialogue text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Summarize text directly
  python -m src.text_summarizer.pipeline.inference --text "John: Hi! Sarah: Hello!"
  
  # Summarize from file
  python -m src.text_summarizer.pipeline.inference --file dialogue.txt
  
  # Interactive mode
  python -m src.text_summarizer.pipeline.inference --interactive
        """
    )
    
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to summarize"
    )
    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Path to file containing text to summarize"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=None,
        help="Path to trained model (default: artifacts/model_trainer/best_model)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum summary length (default: 128)"
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=4,
        help="Number of beams for beam search (default: 4)"
    )
    
    args = parser.parse_args()
    
    # Initialize summarizer
    print("Loading model...")
    summarizer = Summarizer(model_path=args.model_path)
    print("Model loaded successfully!\n")
    
    if args.interactive:
        # Interactive mode
        print("=" * 50)
        print("Interactive Summarization Mode")
        print("Enter dialogue text (type 'quit' to exit)")
        print("=" * 50)
        
        while True:
            print("\nEnter dialogue (press Enter twice to submit):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                if line.lower() == "quit":
                    print("Goodbye!")
                    return
                lines.append(line)
            
            if lines:
                text = "\n".join(lines)
                summary = summarizer.summarize(
                    text,
                    max_length=args.max_length,
                    num_beams=args.num_beams
                )
                print(f"\n📝 Summary: {summary}")
    
    elif args.file:
        # File mode
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        summary = summarizer.summarize(
            text,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        print(f"Summary: {summary}")
    
    elif args.text:
        # Direct text mode
        summary = summarizer.summarize(
            args.text,
            max_length=args.max_length,
            num_beams=args.num_beams
        )
        print(f"Summary: {summary}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
