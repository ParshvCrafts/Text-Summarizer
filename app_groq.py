"""
Text Summarizer API - Groq Integration
=======================================
FastAPI backend using Groq's FREE API for ultra-fast inference.

Groq Free Tier Benefits:
- No credit card required
- Fast inference (LPU technology)
- Generous rate limits for personal projects
- Uses Llama 3.1 8B (excellent for summarization)

Usage:
    uvicorn app_groq:app --reload --host 0.0.0.0 --port 8000

Environment Variables:
    GROQ_API_KEY: Your Groq API key from console.groq.com
    PORT: Server port (default: 8000)
    CORS_ORIGINS: Allowed origins (default: *)
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq API configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"  # Fast, free, great for summarization

# FastAPI app
app = FastAPI(
    title="Text Summarizer API",
    description="Generate summaries using Groq's ultra-fast LLM inference (FREE)",
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
    confidence: float = Field(..., description="Confidence score")
    input_length: int = Field(..., description="Number of characters in input")
    summary_length: int = Field(..., description="Number of characters in summary")
    compression_ratio: float = Field(..., description="Input/output length ratio")
    processing_time: float = Field(..., description="Processing time in seconds")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_available: bool

def get_length_instruction(target_length: Optional[str]) -> str:
    """Get length instruction based on target."""
    if target_length == "short":
        return "Keep the summary very brief, 1-2 sentences maximum."
    elif target_length == "long":
        return "Provide a detailed summary covering all key points."
    else:  # medium or default
        return "Provide a concise summary in 2-3 sentences."

def calculate_confidence(summary: str, input_text: str) -> float:
    """Calculate confidence score based on summary quality heuristics."""
    if not summary or not input_text:
        return 0.5
    
    ratio = len(summary) / len(input_text)
    
    # Ideal compression ratio for summaries
    if 0.15 <= ratio <= 0.35:
        base_confidence = 0.90
    elif 0.1 <= ratio <= 0.5:
        base_confidence = 0.80
    else:
        base_confidence = 0.70
    
    # Adjust based on summary length (prefer 50-200 chars)
    if 50 <= len(summary) <= 200:
        length_bonus = 0.05
    else:
        length_bonus = 0
    
    return round(min(0.95, base_confidence + length_bonus), 3)

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text Summarizer API - Groq Edition (FREE & FAST)",
        "docs": "/docs",
        "health": "/health",
        "model": f"{MODEL} via Groq",
        "note": "Powered by Groq's free tier - ultra-fast inference!"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and Groq availability."""
    api_available = bool(GROQ_API_KEY)
    
    if api_available:
        try:
            # Quick test to verify API key works
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            test_payload = {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5
            }
            response = requests.post(GROQ_API_URL, json=test_payload, headers=headers, timeout=5)
            api_available = response.status_code == 200
        except:
            api_available = False
    
    return HealthResponse(
        status="healthy" if api_available else "degraded",
        model_loaded=True,
        api_available=api_available
    )

@app.post("/summarize", response_model=SummarizeResponse, tags=["Summarization"])
async def summarize(request: SummarizeRequest):
    """
    Generate a summary using Groq's ultra-fast LLM inference.
    
    - **text**: The dialogue/conversation text to summarize
    - **target_length**: Length preset: 'short', 'medium', or 'long' (optional)
    """
    if not GROQ_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="Groq API key not configured. Please set GROQ_API_KEY environment variable."
        )
    
    start_time = time.time()
    
    try:
        length_instruction = get_length_instruction(request.target_length)
        
        # Create the prompt for summarization
        system_prompt = """You are an expert dialogue summarizer. Your task is to create clear, accurate summaries of conversations.

Rules:
1. Capture the main points and outcomes of the conversation
2. Use third person (e.g., "John and Sarah discussed...")
3. Be concise but comprehensive
4. Do not add information not present in the dialogue
5. Output ONLY the summary, no explanations or prefixes"""

        user_prompt = f"""{length_instruction}

Summarize this dialogue:

{request.text}

Summary:"""

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.3,  # Lower temperature for more consistent summaries
        }
        
        response = requests.post(
            GROQ_API_URL,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please wait a moment and try again."
            )
        elif response.status_code == 401:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key. Please check your Groq API key."
            )
        elif response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Groq API error: {response.text}"
            )
        
        result = response.json()
        summary = result["choices"][0]["message"]["content"].strip()
        
        # Clean up the summary (remove any "Summary:" prefix if present)
        if summary.lower().startswith("summary:"):
            summary = summary[8:].strip()
        
        if not summary:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate summary. Please try again."
            )
        
        # Calculate metrics
        input_len = len(request.text)
        summary_len = len(summary)
        compression = input_len / summary_len if summary_len > 0 else 0
        confidence = calculate_confidence(summary, request.text)
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
            detail="Request timed out. Please try again."
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
        "app_groq:app",
        host="0.0.0.0",
        port=PORT,
        reload=True
    )
