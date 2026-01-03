# Changelog

All notable changes to the Text Summarizer project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.7.0] - 2026-01-03

### Added
- **Confidence Scoring**: API now returns confidence score (0-1) with each summary
- **Length Control**: New `target_length` parameter ('short', 'medium', 'long')
- **Preprocessing**: Input normalization (whitespace, format handling)
- **Postprocessing**: Output polishing (capitalization, punctuation, deduplication)
- **Compression Ratio**: API response includes input/output length ratio
- **Edge Case Tests**: Tests for empty input, short input handling
- **ARCHITECTURE.md**: Comprehensive system architecture documentation

### Changed
- Updated inference tests to use HuggingFace Hub fallback (12/12 tests passing)
- Enhanced API response format with additional metrics
- Improved error handling in inference pipeline

### Fixed
- Skipped tests now run using HuggingFace Hub model fallback

## [1.6.0] - 2026-01-03

### Added
- **Smoke Test Script**: 8-test validation suite (`smoke_test.py`)
- **Demo Script**: Showcase with 8 example dialogues (`demo.py`)
- **Benchmark Script**: ROUGE evaluation on test set (`benchmark.py`)
- **API Test Script**: Automated API endpoint tests (`test_api.py`)
- **RESULTS.md**: Comprehensive benchmark results documentation
- **TROUBLESHOOTING.md**: Common issues and solutions guide

### Changed
- Inference pipeline now falls back to HuggingFace Hub when local model unavailable
- Fixed circular import in logger module
- Updated requirements.txt with evaluate and rouge-score packages

### Verified
- Smoke test: 8/8 passing
- Pytest: 8/10 passing (2 skipped)
- API tests: 6/6 passing
- ROUGE-1: 43.53 on 50 test samples

## [1.5.0] - 2026-01-02

### Added
- **Zero-Training Mode**: Use pre-trained `philschmid/flan-t5-base-samsum` without training
- **Training Profiles**: `zero_training`, `quick_test`, `laptop_friendly`, `full_training`
- **Checkpoint Resumption**: Auto-detect and resume from latest checkpoint
- **Training Time Estimation**: Estimate and warn before long training runs
- **Gradient Checkpointing**: Trade compute for memory savings
- **Data Sampling**: Use fraction of data for faster training
- **Early Stopping**: Aggressive early stopping to prevent overfitting

### Changed
- Default model changed to `philschmid/flan-t5-base-samsum` (pre-trained on SAMSum)
- Configuration system updated with profile support
- ModelTrainingConfig extended with new parameters

### Fixed
- Tokenizer-model mismatch issues
- Memory optimization for laptop training

## [1.0.0] - 2026-01-01

### Added
- **5-Stage ML Pipeline**:
  - Data Ingestion: Download SAMSum dataset
  - Data Validation: Verify dataset structure
  - Data Transformation: Tokenize dialogues
  - Model Trainer: Fine-tune with Seq2SeqTrainer
  - Model Evaluation: Compute ROUGE scores
- **FastAPI REST API**: `/summarize` and `/summarize/batch` endpoints
- **CLI Interface**: Interactive and file-based summarization
- **Configuration System**: YAML-based with dataclass entities
- **Pytest Suite**: Configuration and inference tests
- **README.md**: Project documentation

### Technical Details
- Base model: BART/FLAN-T5 variants
- Dataset: SAMSum (dialogue summarization)
- Framework: HuggingFace Transformers
- API: FastAPI with Pydantic validation

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.7.0 | 2026-01-03 | Confidence scoring, length control, 12/12 tests |
| 1.6.0 | 2026-01-03 | Smoke tests, benchmarks, documentation |
| 1.5.0 | 2026-01-02 | Zero-training mode, profiles, checkpointing |
| 1.0.0 | 2026-01-01 | Initial release with full pipeline |

## Roadmap

### Planned for v2.0.0
- [ ] Web interface with React frontend
- [ ] Model quantization (INT8) for faster inference
- [ ] ONNX export for production deployment
- [ ] Multi-language support
- [ ] Streaming summary generation
- [ ] Docker containerization
