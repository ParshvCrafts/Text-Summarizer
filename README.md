# Text Summarizer

🧠 **AI-Powered Dialogue Summarization** using state-of-the-art transformer models.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blue)](https://text-summarizer-frontend.onrender.com)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI-green)](https://text-summarizer-api.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/GitHub-ParshvCrafts-black)](https://github.com/ParshvCrafts)

> Built by **Parshv Patel** | UC Berkeley • Data Science & ML

## ✨ Features

### Backend
- **Abstractive Summarization**: Generates human-like summaries of conversations
- **FLAN-T5 Model**: Pre-trained on SAMSum with ROUGE-1 score of 47.24
- **Production-Ready Pipeline**: 5-stage ML pipeline (Ingestion → Validation → Transformation → Training → Evaluation)
- **FastAPI REST API**: Production-grade API with Swagger documentation
- **Confidence Scoring**: Returns confidence scores for generated summaries
- **Length Control**: Short, medium, and long summary options

### Frontend
- **Modern React UI**: Built with React 19, Vite, and Tailwind CSS
- **Premium Animations**: Lenis smooth scroll, GSAP text animations, Framer Motion
- **Glassmorphism Design**: Beautiful dark theme with glass effects
- **Interactive Features**: Copy to clipboard, processing time display, compression visualization
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Accessibility**: Reduced motion support, keyboard navigation

## Architecture

```
Text-Summarizer/
├── config/                    # Configuration files
│   ├── config.yaml           # Paths, model names
│   └── params.yaml           # Training hyperparameters
├── src/text_summarizer/
│   ├── components/           # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/             # Pipeline orchestration
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   ├── stage_05_model_evaluation.py
│   │   └── inference.py      # Inference pipeline + CLI
│   ├── config/               # Configuration management
│   ├── entity/               # Data classes
│   └── utils/                # Utilities
├── tests/                    # Unit tests
├── research/                 # Jupyter notebooks
├── main.py                   # Training entry point
└── app.py                    # FastAPI application
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ParshvCrafts/Text-Summarizer.git
cd Text-Summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training Profiles

Choose a profile based on your hardware and time constraints:

| Profile | Data | Time | Best For |
|---------|------|------|----------|
| `zero_training` | 0% | 2 min | **Laptops** - Uses pre-trained model, no training! |
| `quick_test` | 10% | 10-15 min | Testing the pipeline works |
| `laptop_friendly` | 50% | 30-60 min | Laptops with limited resources |
| `full_training` | 100% | 2-4 hrs | Cloud/powerful GPU |

**Set your profile** in `config/config.yaml`:
```yaml
training_profile: zero_training  # Recommended for most users!
```

### Zero-Training Mode (Recommended)

The default configuration uses `philschmid/flan-t5-base-samsum`, which is **already fine-tuned on SAMSum** with ROUGE-1 score of 47.24. No training needed!

```bash
# Just run inference directly
python -m src.text_summarizer.pipeline.inference --interactive
```

### Training (If Needed)

```bash
# Run smoke test first to verify everything works
python smoke_test.py

# Then run training
python main.py
```

This will:
1. Download the SAMSum dataset
2. Validate the data
3. Tokenize dialogues and summaries
4. Fine-tune the model with early stopping
5. Evaluate and save ROUGE metrics

**Features:**
- ✅ Automatic checkpoint resumption (never lose progress!)
- ✅ Training time estimation before starting
- ✅ Gradient checkpointing (saves memory)
- ✅ Early stopping (prevents overfitting)

### Inference

#### CLI Usage

```bash
# Summarize text directly
python -m src.text_summarizer.pipeline.inference --text "John: Hi! Sarah: Hello, how are you?"

# Summarize from file
python -m src.text_summarizer.pipeline.inference --file dialogue.txt

# Interactive mode
python -m src.text_summarizer.pipeline.inference --interactive
```

#### Python API

```python
from src.text_summarizer.pipeline.inference import Summarizer

summarizer = Summarizer()
summary = summarizer.summarize("""
    John: Hey, are you coming to the party tonight?
    Sarah: I'm not sure, I have work to do.
    John: Come on, it'll be fun!
    Sarah: Okay, I'll try to come by 8.
""")
print(summary)
```

### API Server (Optional)

```bash
# Start the FastAPI server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Test the API
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "John: Hi! Sarah: Hello!"}'
```

## Configuration

### Model Selection

Edit `config/config.yaml` to change models:

```yaml
data_transformation:
  tokenizer_name: facebook/bart-large-cnn  # Must match model_ckpt

model_trainer:
  model_ckpt: facebook/bart-large-cnn
```

Supported models:
- `facebook/bart-large-cnn` (recommended)
- `facebook/bart-large-xsum`
- `google/pegasus-cnn_dailymail`
- `philschmid/bart-large-cnn-samsum` (pre-fine-tuned)

### Training Parameters

Edit `config/params.yaml`:

```yaml
TrainingArguments:
  num_train_epochs: 5
  learning_rate: 2e-5
  per_device_train_batch_size: 2
  early_stopping_patience: 5
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src
```

## Project Structure

| Component | Description |
|-----------|-------------|
| `data_ingestion` | Downloads and extracts SAMSum dataset |
| `data_validation` | Verifies train/test/validation splits exist |
| `data_transformation` | Tokenizes text using model's tokenizer |
| `model_trainer` | Fine-tunes with early stopping, LR scheduling |
| `model_evaluation` | Computes ROUGE, generates samples |
| `inference` | Production inference with caching |

## Performance

Expected ROUGE scores after training on SAMSum:

| Metric | Score |
|--------|-------|
| ROUGE-1 | ~0.42-0.45 |
| ROUGE-2 | ~0.18-0.21 |
| ROUGE-L | ~0.35-0.38 |

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/ --check

# Format code
black src/
isort src/
```

## 🚀 Deployment

### Local Development

```bash
# Backend
cd Text-Summarizer
pip install -r requirements.txt
python app.py
# API available at http://localhost:8000

# Frontend (in another terminal)
cd Text-Summarizer/frontend
npm install
npm run dev
# Frontend available at http://localhost:5173
```

### Deploy to Render

This project includes a `render.yaml` Blueprint for easy deployment:

1. **Push to GitHub**
2. **Create Render Account** at [render.com](https://render.com)
3. **New Blueprint** → Connect your GitHub repo
4. **Deploy** → Render will auto-detect services

#### Manual Deployment

**Backend (Web Service):**
- Build Command: `pip install -r requirements.txt`
- Start Command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Health Check: `/health`

**Frontend (Static Site):**
- Build Command: `cd frontend && npm install && npm run build`
- Publish Directory: `frontend/dist`
- Environment: `VITE_API_URL=https://your-backend.onrender.com`

### Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `PORT` | Backend | Server port (auto-set by Render) |
| `CORS_ORIGINS` | Backend | Allowed origins (default: `*`) |
| `VITE_API_URL` | Frontend | Backend API URL |

## 🛠️ Tech Stack

### Backend
- **Python 3.11** - Core language
- **FastAPI** - REST API framework
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace model library
- **FLAN-T5** - Pre-trained summarization model

### Frontend
- **React 19** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **GSAP** - Advanced animations
- **Lenis** - Smooth scrolling

## 📊 Performance

| Metric | Score |
|--------|-------|
| ROUGE-1 | 43.53 |
| ROUGE-2 | 20.01 |
| ROUGE-L | 34.78 |
| Inference Time | ~3s |
| Frontend Bundle | 453KB (149KB gzipped) |

## License

MIT License - see [LICENSE](LICENSE) for details.

## 👨‍💻 Author

**Parshv Patel**
- 🎓 UC Berkeley • B.A. Data Science
- 🧠 AI/ML Focus • 4.0 GPA
- 🔗 [LinkedIn](https://www.linkedin.com/in/parshv-patel-65a90326b/)
- 💻 [GitHub](https://github.com/ParshvCrafts)
- 🌐 [Portfolio](https://personal-website-rtzu.onrender.com/)