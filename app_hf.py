"""
Text Summarizer API - Hugging Face Integration
===============================================
FastAPI backend using Hugging Face Inference API for free deployment.

Usage:
    uvicorn app_hf:app --reload --host 0.0.0.0 --port 8000
"""

import os
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import time

# Environment configuration
PORT = int(os.getenv("PORT", 8000))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Optional for higher rate limits

# Hugging Face API configuration (using new router endpoint)
HF_API_URL = "https://router.huggingface.co/hf-inference/models/philschmid/flan-t5-base-samsum"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

# FastAPI app
app = FastAPI(
    title="Text Summarizer API",
    description="Generate summaries using Hugging Face Inference API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS if CORS_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SummarizeRequest(BaseModel):
    text: str = Field(..., description="Dialogue text to summarize", min_length=1)
    target_length: Optional[str] = Field(None, description="Length preset: 'short', 'medium', or 'long'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "John: Hey, are you coming to the party tonight?\nSarah: I'm not sure, I have work.\nJohn: Come on, it'll be fun!\nSarah: Okay, I'll try to come by 8.",
                "target_length": "medium"
            }
        }

class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Generated summary")
    confidence: float = Field(..., description="Confidence score (simulated)")
    input_length: int = Field(..., description="Number of characters in input")
    summary_length: int = Field(..., description="Number of characters in summary")
    compression_ratio: float = Field(..., description="Input/output length ratio")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_available: bool

def generate_confidence_score(text: str, summary: str) -> float:
    """Generate a simulated confidence score based on summary quality."""
    # Simple heuristic based on length ratio and content
    ratio = len(summary) / len(text) if len(text) > 0 else 0
    
    # Ideal compression ratio for summaries is around 0.2-0.4
    if 0.2 <= ratio <= 0.4:
        base_confidence = 0.85
    elif 0.1 <= ratio <= 0.5:
        base_confidence = 0.75
    else:
        base_confidence = 0.65
    
    # Add some variation based on summary length
    length_factor = min(1.0, len(summary) / 50)  # Prefer summaries of at least 50 chars
    
    confidence = base_confidence * (0.7 + 0.3 * length_factor)
    return round(min(0.95, confidence), 3)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text Summarizer API - Hugging Face Edition",
        "docs": "/docs",
        "health": "/health",
        "model": "philschmid/flan-t5-base-samsum (via Hugging Face)"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and Hugging Face availability."""
    try:
        # Test Hugging Face API with a minimal request
        test_payload = {"inputs": "Test"}
        response = requests.post(HF_API_URL, json=test_payload, headers=HEADERS, timeout=5)
        api_available = response.status_code != 429  # Not rate limited
    except:
        api_available = False
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,  # Model is "loaded" on HF servers
        api_available=api_available
    )

@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request: SummarizeRequest):
    """
    Generate a summary using Hugging Face Inference API.
    
    - **text**: The dialogue/conversation text to summarize
    - **target_length**: Length preset: 'short', 'medium', or 'long' (optional)
    """
    start_time = time.time()
    
    try:
        # Call Hugging Face API - the model is fine-tuned for summarization
        # so we just pass the dialogue directly
        payload = {"inputs": request.text}
        
        response = requests.post(
            HF_API_URL, 
            json=payload, 
            headers=HEADERS,
            timeout=60  # 60 second timeout for cold starts
        )
        
        # Handle model loading (503 with estimated_time)
        if response.status_code == 503:
            result = response.json()
            estimated_time = result.get("estimated_time", 20)
            raise HTTPException(
                status_code=503,
                detail=f"Model is loading. Please try again in {int(estimated_time)} seconds."
            )
        elif response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again in a moment."
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Hugging Face API error: {response.text}"
            )
        
        # Extract summary from response
        # HF summarization models return [{"summary_text": "..."}]
        result = response.json()
        summary = ""
        
        if isinstance(result, list) and len(result) > 0:
            # Standard summarization response format
            summary = result[0].get("summary_text", "") or result[0].get("generated_text", "")
        elif isinstance(result, dict):
            summary = result.get("summary_text", "") or result.get("generated_text", "")
        
        summary = summary.strip()
        
        if not summary:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate summary. Please try again."
            )
        
        # Calculate metrics
        input_len = len(request.text)
        summary_len = len(summary)
        compression = input_len / summary_len if summary_len > 0 else 0
        confidence = generate_confidence_score(request.text, summary)
        processing_time = time.time() - start_time
        
        return SummarizeResponse(
            summary=summary,
            confidence=confidence,
            input_length=input_len,
            summary_length=summary_len,
            compression_ratio=round(compression, 2),
            processing_time=round(processing_time, 2)
        )
        
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=408,
            detail="Request timed out. The model might be loading. Please try again."
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "app_hf:app",
        host="0.0.0.0",
        port=PORT,
        reload=True
    )
