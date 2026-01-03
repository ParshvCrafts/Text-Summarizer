"""
Text Summarizer API
====================
FastAPI-based REST API for dialogue summarization.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
    
Endpoints:
    POST /summarize - Summarize dialogue text
    GET /health - Health check
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import uvicorn

# Environment configuration
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Lazy import to avoid loading model on import
_summarizer = None


def get_summarizer():
    """Lazy load the summarizer to avoid startup delay."""
    global _summarizer
    if _summarizer is None:
        from src.text_summarizer.pipeline.inference import Summarizer
        _summarizer = Summarizer()
    return _summarizer


# FastAPI app
app = FastAPI(
    title="Text Summarizer API",
    description="Generate summaries for dialogue/conversation text using fine-tuned transformer models.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware - allow configured origins or all
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class SummarizeRequest(BaseModel):
    """Request model for summarization."""
    text: str = Field(..., description="Dialogue text to summarize", min_length=1)
    max_length: Optional[int] = Field(128, description="Maximum summary length", ge=10, le=512)
    min_length: Optional[int] = Field(30, description="Minimum summary length", ge=1, le=100)
    num_beams: Optional[int] = Field(4, description="Number of beams for beam search", ge=1, le=10)
    target_length: Optional[str] = Field(None, description="Length preset: 'short', 'medium', or 'long'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "John: Hey, are you coming to the party tonight?\nSarah: I'm not sure, I have work.\nJohn: Come on, it'll be fun!\nSarah: Okay, I'll try to come by 8.",
                "max_length": 128,
                "num_beams": 4,
                "target_length": "medium"
            }
        }


class SummarizeResponse(BaseModel):
    """Response model for summarization."""
    summary: str = Field(..., description="Generated summary")
    confidence: float = Field(..., description="Confidence score (0-1)")
    input_length: int = Field(..., description="Number of characters in input")
    summary_length: int = Field(..., description="Number of characters in summary")
    compression_ratio: float = Field(..., description="Input/output length ratio")


class BatchSummarizeRequest(BaseModel):
    """Request model for batch summarization."""
    texts: List[str] = Field(..., description="List of dialogue texts to summarize")
    max_length: Optional[int] = Field(128, description="Maximum summary length")
    num_beams: Optional[int] = Field(4, description="Number of beams")


class BatchSummarizeResponse(BaseModel):
    """Response model for batch summarization."""
    summaries: List[str] = Field(..., description="List of generated summaries")
    count: int = Field(..., description="Number of summaries generated")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


# Endpoints
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text Summarizer API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    global _summarizer
    return HealthResponse(
        status="healthy",
        model_loaded=_summarizer is not None
    )


@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request: SummarizeRequest):
    """
    Generate a summary for the given dialogue text.
    
    - **text**: The dialogue/conversation text to summarize
    - **max_length**: Maximum length of the generated summary (default: 128)
    - **min_length**: Minimum length of the generated summary (default: 30)
    - **num_beams**: Number of beams for beam search (default: 4)
    - **target_length**: Length preset: 'short', 'medium', or 'long' (optional)
    """
    try:
        summarizer = get_summarizer()
        
        summary, confidence = summarizer.summarize(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            target_length=request.target_length,
            return_confidence=True
        )
        
        input_len = len(request.text)
        summary_len = len(summary)
        compression = input_len / summary_len if summary_len > 0 else 0
        
        return SummarizeResponse(
            summary=summary,
            confidence=round(confidence, 3),
            input_length=input_len,
            summary_length=summary_len,
            compression_ratio=round(compression, 2)
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first by running: python main.py"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


@app.post("/summarize/batch", response_model=BatchSummarizeResponse, tags=["Summarization"])
async def summarize_batch(request: BatchSummarizeRequest):
    """
    Generate summaries for multiple dialogue texts.
    
    - **texts**: List of dialogue texts to summarize
    - **max_length**: Maximum length of each summary (default: 128)
    - **num_beams**: Number of beams for beam search (default: 4)
    """
    try:
        summarizer = get_summarizer()
        
        summaries = summarizer.summarize_batch(
            texts=request.texts,
            max_length=request.max_length,
            num_beams=request.num_beams
        )
        
        return BatchSummarizeResponse(
            summaries=summaries,
            count=len(summaries)
        )
        
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Please train the model first."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch summarization failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )