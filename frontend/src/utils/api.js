const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const summarizeText = async (text, targetLength = null, onStatusUpdate = null) => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 90000); // 90 second timeout for cold starts

  try {
    const body = { text };
    if (targetLength) {
      body.target_length = targetLength;
    }

    // Notify that request is starting
    if (onStatusUpdate) onStatusUpdate('Connecting to AI model...');

    const response = await fetch(`${API_BASE_URL}/summarize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      
      // Handle model loading (503)
      if (response.status === 503) {
        throw new Error(error.detail || 'Model is warming up. Please try again in a few seconds.');
      }
      // Handle rate limiting (429)
      if (response.status === 429) {
        throw new Error('Too many requests. Please wait a moment and try again.');
      }
      
      throw new Error(error.detail || `HTTP error! status: ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error('Request timed out. The model may be loading - please try again.');
    }
    throw error;
  }
};

export const checkHealth = async () => {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return await response.json();
  } catch (error) {
    return { status: 'error', model_loaded: false };
  }
};
