# Deployment Guide - 100% FREE

This guide explains how to deploy the Text Summarizer for **completely free** using Groq's free API tier and Render's free hosting.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Frontend      │────▶│   Backend       │────▶│   Groq API      │
│   (Render/      │     │   (Render Free) │     │   (FREE)        │
│    Vercel)      │     │   512MB RAM     │     │   Llama 3.1 8B  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
     FREE                    FREE                    FREE
```

## Why This Works for Free

| Component | Service | Cost | Why It Works |
|-----------|---------|------|--------------|
| Frontend | Render Static / Vercel | FREE | Static files, no compute needed |
| Backend | Render Free Tier | FREE | Only 512MB RAM needed (no model loading!) |
| ML Model | Groq API | FREE | Model runs on Groq's infrastructure |

**Key Insight**: The heavy ML model (1.5GB+) runs on Groq's servers, not yours. Your backend just routes requests!

## Step 1: Get a Groq API Key (FREE)

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google/GitHub (no credit card required!)
3. Go to **API Keys** in the sidebar
4. Click **Create API Key**
5. Copy the key (starts with `gsk_...`)

**Groq Free Tier Limits:**
- 30 requests/minute
- 14,400 requests/day
- No credit card required!

## Step 2: Deploy to Render

### Option A: Blueprint (Recommended)

1. Push your code to GitHub
2. Go to [render.com](https://render.com) → **New** → **Blueprint**
3. Connect your GitHub repo
4. Render will detect `render.yaml` automatically
5. **IMPORTANT**: Before deploying, add your Groq API key:
   - Click on the backend service
   - Go to **Environment**
   - Add: `GROQ_API_KEY` = `your_groq_api_key_here`
6. Click **Deploy**

### Option B: Manual Deployment

#### Backend (Web Service)

1. Go to Render → **New** → **Web Service**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `text-summarizer-api`
   - **Runtime**: Python
   - **Build Command**: `pip install fastapi uvicorn requests pydantic`
   - **Start Command**: `uvicorn app_groq:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free
4. Add Environment Variable:
   - `GROQ_API_KEY` = `your_groq_api_key_here`
5. Deploy

#### Frontend (Static Site)

1. Go to Render → **New** → **Static Site**
2. Connect your GitHub repo
3. Configure:
   - **Name**: `text-summarizer-frontend`
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/dist`
4. Add Environment Variable:
   - `VITE_API_URL` = `https://text-summarizer-api.onrender.com`
5. Deploy

## Step 3: Verify Deployment

1. **Backend Health Check**:
   ```
   curl https://text-summarizer-api.onrender.com/health
   ```
   Should return: `{"status":"healthy","model_loaded":true,"api_available":true}`

2. **Test Summarization**:
   ```bash
   curl -X POST https://text-summarizer-api.onrender.com/summarize \
     -H "Content-Type: application/json" \
     -d '{"text": "John: Hi! Sarah: Hello, how are you?"}'
   ```

3. **Visit Frontend**:
   Open `https://text-summarizer-frontend.onrender.com`

## Environment Variables Reference

### Backend

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Your Groq API key from console.groq.com |
| `PORT` | No | Server port (auto-set by Render) |
| `CORS_ORIGINS` | No | Allowed origins (default: `*`) |

### Frontend

| Variable | Required | Description |
|----------|----------|-------------|
| `VITE_API_URL` | Yes | Backend API URL |

## Expected Behavior

### Cold Starts
- **Render Free Tier**: Services spin down after 15 minutes of inactivity
- **First request after spin-down**: 30-60 seconds (Render cold start)
- **Subsequent requests**: 0.5-2 seconds (Groq is FAST!)

### Rate Limits
- Groq Free Tier: ~30 requests/minute
- If rate limited, the API returns a helpful error message

## Troubleshooting

### "Groq API key not configured"
- Make sure `GROQ_API_KEY` is set in Render environment variables
- Redeploy after adding the variable

### "Rate limit exceeded"
- Wait 1 minute and try again
- Groq free tier has generous limits for personal projects

### CORS Errors
- Ensure `CORS_ORIGINS` includes your frontend URL
- Or set to `*` for development

### Slow First Request
- This is normal! Render free tier spins down after inactivity
- First request wakes up the service (30-60 seconds)
- Subsequent requests are fast

## Alternative: Deploy Frontend to Vercel

For faster frontend (better CDN):

1. Go to [vercel.com](https://vercel.com)
2. Import your GitHub repo
3. Set root directory to `frontend`
4. Add environment variable: `VITE_API_URL` = your backend URL
5. Deploy

## Cost Summary

| Service | Monthly Cost |
|---------|--------------|
| Render Backend (Free) | $0 |
| Render Frontend (Free) | $0 |
| Groq API (Free Tier) | $0 |
| **Total** | **$0** |

🎉 **Completely FREE deployment!**
