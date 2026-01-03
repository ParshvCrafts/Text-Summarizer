"""
Benchmark Script - ROUGE Evaluation
====================================
Evaluates the model on SAMSum test set and generates comprehensive metrics.
Run with: python benchmark.py
"""

import sys
import os
import time
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import evaluate
from datasets import load_from_disk
from tqdm import tqdm
from src.text_summarizer.pipeline.inference import Summarizer


def run_benchmark(num_samples: int = None, save_examples: bool = True):
    """
    Run ROUGE benchmark on SAMSum test set.
    
    Args:
        num_samples: Number of samples to evaluate (None = all)
        save_examples: Whether to save example outputs
    """
    print("=" * 60)
    print("  TEXT SUMMARIZER - ROUGE BENCHMARK")
    print("=" * 60)
    
    # Load SAMSum test set from local artifacts
    print("\nLoading SAMSum test dataset...")
    data_path = Path("artifacts/data_ingestion/samsum_dataset")
    if not data_path.exists():
        print(f"Dataset not found at {data_path}")
        print("Please run the data ingestion pipeline first: python main.py")
        return None, None
    
    full_dataset = load_from_disk(str(data_path))
    dataset = full_dataset["test"]
    
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} samples")
    
    # Initialize summarizer
    print("\nLoading model...")
    start_load = time.time()
    summarizer = Summarizer()
    load_time = time.time() - start_load
    print(f"Model loaded in {load_time:.2f}s")
    
    # Load ROUGE metric
    rouge = evaluate.load("rouge")
    
    # Generate predictions
    print("\nGenerating summaries...")
    predictions = []
    references = []
    examples = []
    inference_times = []
    
    for i, item in enumerate(tqdm(dataset, desc="Summarizing")):
        dialogue = item["dialogue"]
        reference = item["summary"]
        
        start_time = time.time()
        prediction = summarizer.summarize(dialogue, max_length=128, num_beams=4)
        inference_time = time.time() - start_time
        
        predictions.append(prediction)
        references.append(reference)
        inference_times.append(inference_time)
        
        # Save some examples
        if save_examples and i < 25:
            examples.append({
                "id": i,
                "dialogue": dialogue,
                "reference": reference,
                "prediction": prediction,
                "inference_time": inference_time
            })
    
    # Compute ROUGE scores
    print("\nComputing ROUGE scores...")
    results = rouge.compute(predictions=predictions, references=references)
    
    # Print results
    print("\n" + "=" * 60)
    print("  ROUGE BENCHMARK RESULTS")
    print("=" * 60)
    print(f"\n  Model: philschmid/flan-t5-base-samsum")
    print(f"  Test samples: {len(dataset)}")
    print(f"  Model load time: {load_time:.2f}s")
    print(f"  Total inference time: {sum(inference_times):.2f}s")
    print(f"  Average per sample: {sum(inference_times)/len(inference_times):.2f}s")
    
    print("\n  ROUGE SCORES:")
    print("  " + "-" * 40)
    print(f"  ROUGE-1:    {results['rouge1']*100:.2f}")
    print(f"  ROUGE-2:    {results['rouge2']*100:.2f}")
    print(f"  ROUGE-L:    {results['rougeL']*100:.2f}")
    print(f"  ROUGE-Lsum: {results['rougeLsum']*100:.2f}")
    
    # Expected vs Actual comparison
    print("\n  COMPARISON TO REPORTED:")
    print("  " + "-" * 40)
    print(f"  Expected ROUGE-1: 47.24 (reported)")
    print(f"  Actual ROUGE-1:   {results['rouge1']*100:.2f}")
    diff = results['rouge1']*100 - 47.24
    print(f"  Difference:       {diff:+.2f}")
    
    # Save results
    output_dir = Path("artifacts/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics = {
        "model": "philschmid/flan-t5-base-samsum",
        "test_samples": len(dataset),
        "rouge1": results['rouge1'],
        "rouge2": results['rouge2'],
        "rougeL": results['rougeL'],
        "rougeLsum": results['rougeLsum'],
        "model_load_time": load_time,
        "total_inference_time": sum(inference_times),
        "avg_inference_time": sum(inference_times)/len(inference_times)
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved to: {output_dir / 'metrics.json'}")
    
    # Save examples
    if save_examples and examples:
        with open(output_dir / "examples.json", "w") as f:
            json.dump(examples, f, indent=2)
        print(f"  Examples saved to: {output_dir / 'examples.json'}")
        
        # Also save as markdown for portfolio
        with open(output_dir / "examples.md", "w", encoding="utf-8") as f:
            f.write("# Text Summarizer - Example Outputs\n\n")
            f.write(f"Model: `philschmid/flan-t5-base-samsum`\n\n")
            f.write(f"ROUGE-1: {results['rouge1']*100:.2f} | ")
            f.write(f"ROUGE-2: {results['rouge2']*100:.2f} | ")
            f.write(f"ROUGE-L: {results['rougeL']*100:.2f}\n\n")
            
            for ex in examples:
                f.write(f"## Example {ex['id'] + 1}\n\n")
                f.write("**Dialogue:**\n```\n")
                f.write(ex['dialogue'])
                f.write("\n```\n\n")
                f.write(f"**Reference Summary:** {ex['reference']}\n\n")
                f.write(f"**Model Summary:** {ex['prediction']}\n\n")
                f.write(f"*Inference time: {ex['inference_time']:.2f}s*\n\n")
                f.write("---\n\n")
        print(f"  Examples markdown saved to: {output_dir / 'examples.md'}")
    
    print("\n✅ Benchmark complete!")
    return results, examples


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ROUGE benchmark")
    parser.add_argument("--samples", "-n", type=int, default=100,
                       help="Number of samples to evaluate (default: 100, use -1 for all)")
    args = parser.parse_args()
    
    num_samples = None if args.samples == -1 else args.samples
    run_benchmark(num_samples=num_samples)
