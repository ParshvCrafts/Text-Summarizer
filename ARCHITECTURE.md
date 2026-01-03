# Text Summarizer - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TEXT SUMMARIZER                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   FastAPI    │    │     CLI      │    │   Python     │          │
│  │   REST API   │    │  Interface   │    │     API      │          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             │                                        │
│                    ┌────────▼────────┐                              │
│                    │   Summarizer    │                              │
│                    │   (Singleton)   │                              │
│                    └────────┬────────┘                              │
│                             │                                        │
│         ┌───────────────────┼───────────────────┐                   │
│         │                   │                   │                   │
│  ┌──────▼──────┐    ┌──────▼──────┐    ┌──────▼──────┐            │
│  │ Preprocess  │    │   Model     │    │ Postprocess │            │
│  │   Input     │    │  Generate   │    │   Output    │            │
│  └─────────────┘    └──────┬──────┘    └─────────────┘            │
│                            │                                        │
│                    ┌───────▼───────┐                               │
│                    │  HuggingFace  │                               │
│                    │ Transformers  │                               │
│                    │ (FLAN-T5)     │                               │
│                    └───────────────┘                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. ML Pipeline (Training)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Data     │───▶│    Data     │───▶│    Data     │
│  Ingestion  │    │ Validation  │    │ Transform   │
└─────────────┘    └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────┐    ┌──────▼──────┐
                   │   Model     │◀───│   Model     │
                   │ Evaluation  │    │   Trainer   │
                   └─────────────┘    └─────────────┘
```

| Stage | Component | Description |
|-------|-----------|-------------|
| 1 | `DataIngestion` | Downloads SAMSum dataset from GitHub |
| 2 | `DataValidation` | Verifies train/val/test splits exist |
| 3 | `DataTransformation` | Tokenizes dialogues using model tokenizer |
| 4 | `ModelTrainer` | Fine-tunes model with early stopping |
| 5 | `ModelEvaluation` | Computes ROUGE scores on test set |

### 2. Inference Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                      Summarizer Class                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input ──▶ Preprocess ──▶ Tokenize ──▶ Generate ──▶ Decode     │
│                                            │                     │
│                                     ┌──────▼──────┐             │
│                                     │  Confidence │             │
│                                     │   Scoring   │             │
│                                     └──────┬──────┘             │
│                                            │                     │
│  Output ◀── Postprocess ◀─────────────────┘                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Singleton Pattern**: Model loaded once, reused across requests
- **Preprocessing**: Whitespace normalization, format handling
- **Postprocessing**: Capitalization, punctuation, deduplication
- **Confidence Scoring**: Based on generation log probabilities
- **Length Control**: Short/medium/long presets

### 3. Configuration System

```
config/
├── config.yaml      # Paths, model names, profiles
└── params.yaml      # Training hyperparameters

        ┌─────────────────────┐
        │ ConfigurationManager │
        └──────────┬──────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│ Entity │   │  Entity  │   │  Entity  │
│ Config │   │  Config  │   │  Config  │
└────────┘   └──────────┘   └──────────┘
```

**Design Decisions:**
- YAML for human-readable configuration
- Dataclasses for type-safe config entities
- Profile system for different hardware constraints

### 4. API Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Endpoints:                                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │  GET /          │  │  GET /health    │  │  GET /docs      │ │
│  │  Root info      │  │  Health check   │  │  OpenAPI docs   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │  POST /summarize        │  │  POST /summarize/batch      │  │
│  │  Single summarization   │  │  Batch summarization        │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
│                                                                  │
│  Middleware:                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CORS (allow all origins for development)               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Training Flow

```
SAMSum Dataset (HuggingFace)
         │
         ▼
┌─────────────────┐
│  Download ZIP   │  ← Data Ingestion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Validate Splits │  ← Data Validation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tokenize      │  ← Data Transformation
│   (dialogue →   │
│    input_ids)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tune      │  ← Model Trainer
│  (Seq2SeqTrainer│
│   + callbacks)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ROUGE Eval     │  ← Model Evaluation
└────────┬────────┘
         │
         ▼
    Best Model
    (saved to disk)
```

### Inference Flow

```
User Input (dialogue text)
         │
         ▼
┌─────────────────┐
│   Preprocess    │  ← Normalize whitespace, format
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Tokenize     │  ← Convert to input_ids
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Generate     │  ← Beam search with length control
│  (FLAN-T5 model)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Decode      │  ← Convert tokens to text
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Postprocess   │  ← Capitalize, punctuate, dedupe
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Confidence    │  ← Calculate from log probs
└────────┬────────┘
         │
         ▼
    Summary + Confidence Score
```

## Key Design Decisions

### 1. Zero-Training Mode
**Decision**: Use pre-trained `philschmid/flan-t5-base-samsum` by default.

**Rationale**:
- Already fine-tuned on SAMSum with ROUGE-1 of 47.24
- Eliminates 2-4 hour training time
- Reduces resource requirements
- Immediate usability

### 2. Singleton Pattern for Model
**Decision**: Use singleton pattern for Summarizer class.

**Rationale**:
- Model loading takes ~2 seconds
- Model uses ~1.5GB memory
- Avoid reloading on every request
- Thread-safe for API usage

### 3. Profile-Based Configuration
**Decision**: Support multiple training profiles (zero_training, quick_test, laptop_friendly, full_training).

**Rationale**:
- Different users have different hardware
- Prevents crashes on resource-constrained machines
- Allows quick testing before long training runs

### 4. Checkpoint Resumption
**Decision**: Auto-detect and resume from checkpoints.

**Rationale**:
- Training can take hours
- Crashes/interruptions are common
- Prevents lost progress

### 5. Confidence Scoring
**Decision**: Return confidence score with each summary.

**Rationale**:
- Helps users assess summary quality
- Enables filtering low-confidence outputs
- Demonstrates ML engineering depth

## Directory Structure

```
Text-Summarizer/
├── config/                     # Configuration files
│   ├── config.yaml            # Paths and model settings
│   └── params.yaml            # Training hyperparameters
│
├── src/text_summarizer/       # Main package
│   ├── components/            # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   │
│   ├── pipeline/              # Pipeline orchestration
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── inference.py       # Production inference
│   │
│   ├── config/                # Configuration management
│   │   └── configuration.py
│   │
│   ├── entity/                # Data classes
│   │   └── __init__.py
│   │
│   ├── utils/                 # Utilities
│   │   └── common.py
│   │
│   └── logger/                # Logging
│       └── __init__.py
│
├── tests/                     # Test suite
│   ├── test_config.py
│   └── test_inference.py
│
├── artifacts/                 # Generated artifacts (gitignored)
│   ├── data_ingestion/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── model_evaluation/
│
├── app.py                     # FastAPI application
├── main.py                    # Training entry point
├── demo.py                    # Demo script
├── benchmark.py               # ROUGE benchmark
└── smoke_test.py              # Pre-training validation
```

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Model | FLAN-T5 Base | Seq2seq summarization |
| Framework | HuggingFace Transformers | Model loading, training |
| Training | Seq2SeqTrainer | Fine-tuning with callbacks |
| API | FastAPI | REST endpoints |
| Validation | Pydantic | Request/response schemas |
| Testing | Pytest | Unit and integration tests |
| Config | PyYAML + python-box | Configuration management |

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Model Size | 990 MB |
| Parameters | 248M |
| Load Time | ~2 seconds |
| Inference (CPU) | ~3-4 seconds |
| Inference (GPU) | ~0.5 seconds |
| Memory Usage | ~1.5 GB |
| ROUGE-1 Score | 43.53 |
