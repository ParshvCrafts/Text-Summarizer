"""
Smoke Test Script
=================
Run this BEFORE any training to verify the pipeline works correctly.
Catches configuration errors, missing dependencies, and hardware issues.

Usage:
    python smoke_test.py

Expected time: < 2 minutes
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(test_name: str, passed: bool, message: str = ""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status}: {test_name}")
    if message and not passed:
        print(f"         → {message}")


def test_imports():
    """Test that all required packages can be imported."""
    print_header("TEST 1: Import Dependencies")
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("evaluate", "HuggingFace Evaluate"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("yaml", "PyYAML"),
        ("tqdm", "TQDM"),
    ]
    
    all_passed = True
    for module, name in required:
        try:
            __import__(module)
            print_result(f"Import {name}", True)
        except ImportError as e:
            print_result(f"Import {name}", False, str(e))
            all_passed = False
    
    return all_passed


def test_cuda():
    """Test CUDA availability and GPU info."""
    print_header("TEST 2: CUDA/GPU Check")
    
    import torch
    
    cuda_available = torch.cuda.is_available()
    print_result("CUDA Available", cuda_available, 
                 "Training will use CPU (slower)" if not cuda_available else "")
    
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"         GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Check if enough memory
        if gpu_memory < 4:
            print_result("GPU Memory", False, "Less than 4GB - may run out of memory")
        else:
            print_result("GPU Memory", True, f"{gpu_memory:.1f} GB available")
    
    return True  # Not a failure if no CUDA


def test_config_loading():
    """Test configuration files can be loaded."""
    print_header("TEST 3: Configuration Loading")
    
    try:
        from src.text_summarizer.config.configuration import ConfigurationManager
        config_manager = ConfigurationManager()
        print_result("Load ConfigurationManager", True)
        
        # Test each config
        config_manager.get_data_ingestion_config()
        print_result("Data Ingestion Config", True)
        
        config_manager.get_data_validation_config()
        print_result("Data Validation Config", True)
        
        config_manager.get_data_transformation_config()
        print_result("Data Transformation Config", True)
        
        training_config = config_manager.get_model_training_config()
        print_result("Model Training Config", True)
        print(f"         Model: {training_config.model_ckpt}")
        
        config_manager.get_model_evaluation_config()
        print_result("Model Evaluation Config", True)
        
        return True
        
    except Exception as e:
        print_result("Configuration Loading", False, str(e))
        return False


def test_tokenizer_model_consistency():
    """Verify tokenizer and model are consistent."""
    print_header("TEST 4: Tokenizer-Model Consistency")
    
    try:
        from src.text_summarizer.config.configuration import ConfigurationManager
        config_manager = ConfigurationManager()
        
        transform_config = config_manager.get_data_transformation_config()
        training_config = config_manager.get_model_training_config()
        
        tokenizer_name = transform_config.tokenizer_name
        model_name = training_config.model_ckpt
        
        if tokenizer_name == model_name:
            print_result("Tokenizer matches Model", True)
            print(f"         Both use: {model_name}")
            return True
        else:
            print_result("Tokenizer matches Model", False, 
                        f"Tokenizer: {tokenizer_name}, Model: {model_name}")
            return False
            
    except Exception as e:
        print_result("Consistency Check", False, str(e))
        return False


def test_model_loading():
    """Test that the model can be loaded (downloads if needed)."""
    print_header("TEST 5: Model Loading (may download ~1GB)")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.text_summarizer.config.configuration import ConfigurationManager
        
        config_manager = ConfigurationManager()
        training_config = config_manager.get_model_training_config()
        model_name = training_config.model_ckpt
        
        print(f"         Loading: {model_name}")
        start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_result("Load Tokenizer", True)
        print(f"         Vocab size: {tokenizer.vocab_size}")
        
        # Load model (this may take a while on first run)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elapsed = time.time() - start
        print_result("Load Model", True)
        print(f"         Time: {elapsed:.1f}s")
        print(f"         Parameters: {model.num_parameters():,}")
        
        return True
        
    except Exception as e:
        print_result("Model Loading", False, str(e))
        return False


def test_inference():
    """Test that inference works with a sample input."""
    print_header("TEST 6: Inference Test")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.text_summarizer.config.configuration import ConfigurationManager
        import torch
        
        config_manager = ConfigurationManager()
        training_config = config_manager.get_model_training_config()
        model_name = training_config.model_ckpt
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()
        
        # Sample dialogue
        sample = """
        John: Hey, are you coming to the party tonight?
        Sarah: I'm not sure, I have a lot of work.
        John: Come on, it'll be fun!
        Sarah: Okay, I'll try to come by 8.
        """
        
        inputs = tokenizer(sample, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64, num_beams=2)
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_result("Generate Summary", True)
        print(f"         Output: {summary[:100]}...")
        
        return True
        
    except Exception as e:
        print_result("Inference Test", False, str(e))
        return False


def test_training_step():
    """Test a single training step (forward + backward pass)."""
    print_header("TEST 7: Training Step (1 step only)")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.text_summarizer.config.configuration import ConfigurationManager
        import torch
        
        config_manager = ConfigurationManager()
        training_config = config_manager.get_model_training_config()
        model_name = training_config.model_ckpt
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.train()
        
        # Create dummy batch
        sample_input = "John: Hi! Sarah: Hello!"
        sample_target = "John and Sarah greet each other."
        
        inputs = tokenizer(sample_input, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
        labels = tokenizer(sample_target, return_tensors="pt", max_length=32, truncation=True, padding="max_length")
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs["labels"] = labels["input_ids"].to(device)
        
        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss
        print_result("Forward Pass", True)
        print(f"         Loss: {loss.item():.4f}")
        
        # Backward pass
        loss.backward()
        print_result("Backward Pass", True)
        
        # Check gradients
        has_gradients = any(p.grad is not None for p in model.parameters())
        print_result("Gradients Computed", has_gradients)
        
        return True
        
    except Exception as e:
        print_result("Training Step", False, str(e))
        return False


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    print_header("TEST 8: Checkpoint Save/Load")
    
    try:
        import tempfile
        import shutil
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        from src.text_summarizer.config.configuration import ConfigurationManager
        
        config_manager = ConfigurationManager()
        training_config = config_manager.get_model_training_config()
        model_name = training_config.model_ckpt
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Save to temp directory
        temp_dir = tempfile.mkdtemp()
        try:
            model.save_pretrained(temp_dir)
            tokenizer.save_pretrained(temp_dir)
            print_result("Save Checkpoint", True)
            
            # Load back
            loaded_model = AutoModelForSeq2SeqLM.from_pretrained(temp_dir)
            loaded_tokenizer = AutoTokenizer.from_pretrained(temp_dir)
            print_result("Load Checkpoint", True)
            
            # Verify
            assert loaded_model.num_parameters() == model.num_parameters()
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
            print_result("Verify Checkpoint", True)
            
        finally:
            shutil.rmtree(temp_dir)
        
        return True
        
    except Exception as e:
        print_result("Checkpoint Test", False, str(e))
        return False


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("  TEXT SUMMARIZER - SMOKE TEST")
    print("  Run this before training to catch issues early!")
    print("="*60)
    
    start_time = time.time()
    
    results = {
        "imports": test_imports(),
        "cuda": test_cuda(),
        "config": test_config_loading(),
        "consistency": test_tokenizer_model_consistency(),
        "model": test_model_loading(),
        "inference": test_inference(),
        "training": test_training_step(),
        "checkpoint": test_checkpoint_save_load(),
    }
    
    elapsed = time.time() - start_time
    
    # Summary
    print_header("SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    print(f"\n  Tests Passed: {passed}/{total}")
    print(f"  Time Elapsed: {elapsed:.1f}s")
    
    if all(results.values()):
        print("\n  ✅ ALL TESTS PASSED - Ready for training!")
        print("\n  Recommended next steps:")
        print("    1. For ZERO training (recommended):")
        print("       python -m src.text_summarizer.pipeline.inference --interactive")
        print("\n    2. For quick test training:")
        print("       Edit config.yaml: training_profile: quick_test")
        print("       python main.py")
        return 0
    else:
        print("\n  ❌ SOME TESTS FAILED - Fix issues before training!")
        failed = [k for k, v in results.items() if not v]
        print(f"  Failed tests: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
