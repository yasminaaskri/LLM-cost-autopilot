"""
Google provider — handles calls to gemini-2.5-flash
via the google-genai SDK.

Install: pip install google-genai
Env var: GOOGLE_API_KEY
"""

from __future__ import annotations
import os
from src.models import ModelConfig, ProviderError
from dotenv import load_dotenv

load_dotenv() 

if os.getenv("GOOGLE_API_KEY"):
    print("✅ GOOGLE_API_KEY loaded")
else:
    print("❌ GOOGLE_API_KEY NOT loaded - check .env file")


_client = None


def _get_client():
    """Singleton Google GenAI client. Initialised on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderError(
                model_id="google",
                message="GOOGLE_API_KEY environment variable not set. "
                        "Get your key at https://aistudio.google.com/apikey"
            )
        from google import genai
        _client = genai.Client(api_key=api_key)
    return _client


def send(prompt: str,
         model_config: ModelConfig,
         max_tokens: int = 1024,
         temperature: float = 0.1) -> dict:
    """
    Send a prompt to Gemini 2.5 Flash via the Google GenAI SDK.

    Args:
        prompt:       The user prompt string.
        model_config: ModelConfig from the registry.
        max_tokens:   Max output tokens.
        temperature:  Sampling temperature.

    Returns:
        dict with keys: text, input_tokens, output_tokens

    Raises:
        ProviderError on any API failure.
    """
    from google.genai import types

    client = _get_client()

    try:
        response = client.models.generate_content(
            model=model_config.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
    except Exception as e:
        raise ProviderError(
            model_id=model_config.model_id,
            message=f"Google GenAI API call failed: {e}",
            original_error=e,
        )

    text = response.text or ""

    # Extract token counts from usage_metadata
    meta          = response.usage_metadata
    input_tokens  = meta.prompt_token_count if meta else 0
    output_tokens = meta.candidates_token_count if meta else 0

    # Fallback: estimate via tiktoken if metadata missing
    if input_tokens == 0 or output_tokens == 0:
        input_tokens, output_tokens = _estimate_tokens(prompt, text)

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }


def _estimate_tokens(prompt: str, output: str) -> tuple[int, int]:
    """
    Rough token estimate when the API doesn't return usage metadata.
    Uses tiktoken with cl100k_base (good approximation for Gemini).
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(prompt)), len(enc.encode(output))
    except Exception:
        # Last resort: character-based estimate (4 chars ≈ 1 token)
        return len(prompt) // 4, len(output) // 4
"""
Google provider — handles calls to gemini-2.5-flash
via the google-genai SDK.

Install: pip install google-genai
Env var: GOOGLE_API_KEY
"""
