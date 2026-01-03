# Troubleshooting Guide

Common issues and solutions for the Text Summarizer project.

## Training Issues

### "Training is too slow" (10+ hours)

**Solutions:**
1. **Use zero-training mode** (recommended):
   ```yaml
   # In config/config.yaml
   training_profile: zero_training
   ```
   Uses `philschmid/flan-t5-base-samsum` which is already fine-tuned. No training needed!

2. **Use quick_test profile**:
   ```yaml
   training_profile: quick_test
   ```
   Uses 10% of data, 1 epoch. ~10-15 minutes.

3. **Use laptop_friendly profile**:
   ```yaml
   training_profile: laptop_friendly
   ```
   Optimized for laptops. ~30-60 minutes.

### "Out of memory" / CUDA OOM

**Solutions:**
1. **Reduce batch size** in `config/params.yaml`:
   ```yaml
   per_device_train_batch_size: 1
   ```

2. **Enable gradient checkpointing**:
   ```yaml
   gradient_checkpointing: true
   ```

3. **Use a smaller model**:
   ```yaml
   # In config/config.yaml
   model_ckpt: google/flan-t5-small  # 308MB instead of 990MB
   ```

4. **Reduce sequence length**:
   ```yaml
   max_input_length: 256  # Instead of 512 or 1024
   ```

### "Laptop overheating"

**Solutions:**
1. Use `laptop_friendly` profile (smaller batch, fewer workers)
2. Add cooling pad or elevate laptop
3. Reduce `dataloader_num_workers` to 0
4. Consider using Google Colab instead

### "Training crashed - lost progress"

**Good news:** Checkpoints are saved automatically!

**To resume:**
```bash
# Just run training again - it auto-detects checkpoints
python main.py
```

The trainer will find the latest checkpoint and resume from there.

**To see available checkpoints:**
```bash
ls artifacts/model_trainer/checkpoint-*
```

### "Results getting worse" (3 epochs worse than 1)

This is **overfitting**. Solutions:

1. **Use fewer epochs**:
   ```yaml
   num_train_epochs: 1  # Not 3 or 5
   ```

2. **Lower learning rate**:
   ```yaml
   learning_rate: 1e-5  # Instead of 2e-5
   ```

3. **More aggressive early stopping**:
   ```yaml
   early_stopping_patience: 2  # Stop after 2 bad evaluations
   ```

4. **Use pre-trained model** (already optimized):
   ```yaml
   model_ckpt: philschmid/flan-t5-base-samsum
   ```

### "Google Colab ran out of memory"

**Solutions:**
1. Use Colab Pro for more RAM
2. Use `quick_test` profile with 10% data
3. Use smaller model (`google/flan-t5-small`)
4. Enable gradient checkpointing
5. Clear GPU cache before training:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

## Inference Issues

### "Model not found"

**Error:** `Model not found at artifacts/model_trainer/best_model`

**Solutions:**
1. **For zero-training mode**, the model downloads directly from HuggingFace:
   ```python
   from src.text_summarizer.pipeline.inference import Summarizer
   summarizer = Summarizer(model_path="philschmid/flan-t5-base-samsum")
   ```

2. **Run training first**:
   ```bash
   python main.py
   ```

### "Tokenizer mismatch error"

**Cause:** Tokenizer and model don't match.

**Solution:** Ensure both use the same model in `config/config.yaml`:
```yaml
data_transformation:
  tokenizer_name: philschmid/flan-t5-base-samsum

model_trainer:
  model_ckpt: philschmid/flan-t5-base-samsum  # Must match!
```

## Configuration Issues

### "KeyError: 'artifacts_root'"

**Solution:** Add to `config/config.yaml`:
```yaml
artifacts_root: artifacts
```

### "FileNotFoundError: config.yaml"

**Solution:** Run from project root directory:
```bash
cd Text-Summarizer
python main.py
```

## Quick Reference: Training Profiles

| Profile | Data | Epochs | Time | Use Case |
|---------|------|--------|------|----------|
| `zero_training` | 0% | 0 | 2 min | Just inference, no training |
| `quick_test` | 10% | 1 | 10-15 min | Test pipeline works |
| `laptop_friendly` | 50% | 1 | 30-60 min | Safe for laptops |
| `full_training` | 100% | 3 | 2-4 hrs | Cloud/powerful GPU |

## Quick Reference: Model Sizes

| Model | Size | ROUGE-1 | Training Needed |
|-------|------|---------|-----------------|
| `philschmid/flan-t5-base-samsum` | 990MB | 47.24 | No ⭐ |
| `google/flan-t5-small` | 308MB | ~35 | Yes |
| `sshleifer/distilbart-cnn-12-6` | 1.2GB | ~40 | Yes |
| `facebook/bart-large-cnn` | 1.6GB | ~42 | Yes |

## Still Having Issues?

1. Run the smoke test first:
   ```bash
   python smoke_test.py
   ```

2. Check the logs:
   ```bash
   cat logs/running_logs.log
   ```

3. Open an issue on GitHub with:
   - Error message
   - Your `config/config.yaml`
   - Your hardware specs
