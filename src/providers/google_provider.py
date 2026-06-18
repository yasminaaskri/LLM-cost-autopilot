"""
Google provider — handles calls to gemini-2.5-flash
via the google-generativeai SDK.

Install: pip install google-generativeai
Env var: GOOGLE_API_KEY
"""

from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from src.models import ModelConfig, ProviderError

# ── FIXED: Correct path to .env in project root ──────────────────────────────
# google_provider.py is at: src/providers/google_provider.py
# Need to go up 3 levels to reach project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
ENV_PATH = PROJECT_ROOT / '.env'
load_dotenv(ENV_PATH)

# Debug info
print(f"📁 Looking for .env at: {ENV_PATH}")
print(f"📁 File exists: {ENV_PATH.exists()}")

if os.getenv("GOOGLE_API_KEY"):
    print("✅ GOOGLE_API_KEY loaded")
else:
    print("❌ GOOGLE_API_KEY NOT loaded - check .env file")
    print(f"   Tried to load from: {ENV_PATH}")

_client = None


def _get_client():
    """Singleton Google GenerativeAI client."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                model_id="google",
                message="GOOGLE_API_KEY environment variable not set. "
                        "Get your key at https://aistudio.google.com/apikey"
            )
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _client = genai
    return _client


def send(prompt: str,
         model_config: ModelConfig,
         max_tokens: int = 1024,
         temperature: float = 0.1) -> dict:
    """Send a prompt to Gemini via Google GenerativeAI SDK."""
    client = _get_client()

    try:
        model = client.GenerativeModel(model_config.model_id)
        response = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
        )
    except Exception as e:
        raise ProviderError(
            model_id=model_config.model_id,
            message=f"Google GenerativeAI API call failed: {e}",
            original_error=e,
        )

    text = response.text or ""
    input_tokens, output_tokens = _estimate_tokens(prompt, text)

    return {
        "text": text,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def _estimate_tokens(prompt: str, output: str) -> tuple[int, int]:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt)), len(enc.encode(output))
    except Exception:
        return len(prompt) // 4, len(output) // 4