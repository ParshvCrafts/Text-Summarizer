# Text Summarizer - Results & Benchmarks

## Project Overview

A production-grade dialogue summarization system using transformer models fine-tuned on the SAMSum dataset. This project demonstrates end-to-end ML pipeline development with a focus on practical deployment and resource optimization.

## Model Information

| Attribute | Value |
|-----------|-------|
| **Model** | `philschmid/flan-t5-base-samsum` |
| **Architecture** | FLAN-T5 Base (Encoder-Decoder) |
| **Parameters** | 247,577,856 (~248M) |
| **Model Size** | ~990 MB |
| **Pre-trained On** | SAMSum dialogue summarization dataset |
| **Training Required** | No (zero-training mode available) |

## Benchmark Results

### ROUGE Scores on SAMSum Test Set

| Metric | Score | Reported Score | Difference |
|--------|-------|----------------|------------|
| **ROUGE-1** | 43.53 | 47.24 | -3.71 |
| **ROUGE-2** | 20.01 | ~20 | ~0 |
| **ROUGE-L** | 34.78 | ~35 | ~0 |
| **ROUGE-Lsum** | 34.76 | ~35 | ~0 |

*Note: Slight variance from reported scores is expected due to different generation parameters (beam size, length penalty, etc.)*

### Performance Metrics

| Metric | Value |
|--------|-------|
| Model Load Time | ~1.4 seconds |
| Average Inference Time | ~3.7 seconds/sample (CPU) |
| Memory Usage | ~1.5 GB RAM |

## Training Profiles

| Profile | Data Used | Estimated Time | Best For |
|---------|-----------|----------------|----------|
| `zero_training` | 0% | 2 min (download) | **Recommended** - No training needed |
| `quick_test` | 10% | 10-15 min | Testing pipeline |
| `laptop_friendly` | 50% | 30-60 min | Resource-constrained |
| `full_training` | 100% | 2-4 hours | Cloud/powerful GPU |

## Example Outputs

### Example 1: Party Invitation
**Dialogue:**
```
John: Hey, are you coming to the party tonight?
Sarah: I'm not sure, I have a lot of work to do.
John: Come on, it'll be fun! Everyone's going to be there.
Sarah: Okay, I'll try to come by 8.
John: Great! See you then!
```

**Generated Summary:**
> Sarah is not sure if she will come to the party tonight, because she has a lot of work to do. She will try to come by 8.

---

### Example 2: Meeting Scheduling
**Dialogue:**
```
Alice: Hi Bob, can we schedule a meeting for tomorrow?
Bob: Sure, what time works for you?
Alice: How about 2 PM?
Bob: That works. Should I book the conference room?
Alice: Yes please. We need to discuss the Q4 budget.
Bob: Got it. I'll send the calendar invite.
```

**Generated Summary:**
> Alice and Bob will have a meeting tomorrow at 2 PM. They need to discuss the Q4 budget. Bob will send the calendar invite.

---

### Example 3: Technical Support
**Dialogue:**
```
Customer: My laptop won't turn on. I've tried everything.
Support: Have you tried holding the power button for 10 seconds?
Customer: Yes, nothing happens.
Support: Is the charging light on when you plug it in?
Customer: No, there's no light at all.
Support: It sounds like a power adapter issue. Try a different charger.
Customer: I don't have another one.
Support: I'll arrange for a replacement adapter to be sent to you.
```

**Generated Summary:**
> Customer's laptop won't turn on. He has a power adapter issue. He will get a replacement adapter from Support.

## Test Results

### Smoke Test: 8/8 Passing ✅

| Test | Status |
|------|--------|
| Import Dependencies | ✅ Pass |
| CUDA/GPU Check | ✅ Pass (CPU fallback) |
| Configuration Loading | ✅ Pass |
| Tokenizer-Model Consistency | ✅ Pass |
| Model Loading | ✅ Pass |
| Inference Test | ✅ Pass |
| Training Step | ✅ Pass |
| Checkpoint Save/Load | ✅ Pass |

### Pytest Suite: 8/10 Passing ✅

| Test | Status |
|------|--------|
| Config Manager Initialization | ✅ Pass |
| Data Ingestion Config | ✅ Pass |
| Data Validation Config | ✅ Pass |
| Data Transformation Config | ✅ Pass |
| Model Training Config | ✅ Pass |
| Model Evaluation Config | ✅ Pass |
| Tokenizer-Model Consistency | ✅ Pass |
| Summarizer Import | ✅ Pass |
| Summarizer Initialization | ⏭️ Skipped (needs model) |
| Summarize Text | ⏭️ Skipped (needs model) |

## Features Implemented

### Core Features
- ✅ 5-stage ML pipeline (Ingestion → Validation → Transformation → Training → Evaluation)
- ✅ Pre-trained model support (zero-training mode)
- ✅ Multiple training profiles for different hardware
- ✅ Checkpoint resumption (never lose progress)
- ✅ Early stopping to prevent overfitting
- ✅ Gradient checkpointing for memory optimization

### Inference & API
- ✅ CLI interface with interactive mode
- ✅ FastAPI REST API with batch support
- ✅ Single and batch summarization
- ✅ Configurable generation parameters

### Developer Experience
- ✅ Comprehensive smoke test script
- ✅ Pytest test suite
- ✅ Detailed documentation
- ✅ Troubleshooting guide

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run smoke test
python smoke_test.py

# 3. Run demo (zero-training mode)
python demo.py

# 4. Interactive CLI
python -m src.text_summarizer.pipeline.inference --interactive

# 5. Start API server
python app.py
```

## Known Limitations

1. **CPU Inference Speed**: ~3-4 seconds per summary on CPU. GPU recommended for production.
2. **Long Dialogues**: Very long dialogues (>1024 tokens) are truncated.
3. **Language**: Optimized for English dialogues only.
4. **Domain**: Best for casual/business conversations. May struggle with technical jargon.

## Future Improvements

1. **Model Quantization**: Reduce model size with INT8 quantization
2. **ONNX Export**: Faster inference with ONNX runtime
3. **Streaming**: Real-time summary generation
4. **Multi-language**: Support for other languages

## Files Structure

```
Text-Summarizer/
├── config/
│   ├── config.yaml          # Model and path configuration
│   └── params.yaml          # Training hyperparameters
├── src/text_summarizer/
│   ├── components/          # ML pipeline components
│   ├── pipeline/            # Pipeline orchestration
│   └── config/              # Configuration management
├── tests/                   # Pytest test suite
├── artifacts/               # Generated artifacts (gitignored)
├── app.py                   # FastAPI application
├── main.py                  # Training entry point
├── demo.py                  # Demo script
├── benchmark.py             # ROUGE benchmark script
├── smoke_test.py            # Pre-training validation
└── README.md                # Project documentation
```

## Author

Parshv Patel - [GitHub](https://github.com/ParshvCrafts)

## License

MIT License
