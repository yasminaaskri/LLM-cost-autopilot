"""
Groq provider — handles all calls to llama-3.3-70b-versatile
and llama-3.1-8b-instant via the Groq SDK.

Install: pip install groq
Env var: GROQ_API_KEY
"""

from __future__ import annotations
import os
import time
from groq import Groq
from src.models import ModelConfig, ProviderError
from dotenv import load_dotenv

load_dotenv()

if os.getenv("GROQ_API_KEY"):
    print("✅ GROQ_API_KEY loaded")
else:
    print("❌ GROQ_API_KEY NOT loaded - check .env file")
    
_client: Groq | None = None


def _get_client() -> Groq:
    """Singleton Groq client. Initialised on first call."""
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ProviderError(
                model_id="groq",
                message="GROQ_API_KEY environment variable not set. "
                        "Get your key at https://console.groq.com/keys"
            )
        _client = Groq(api_key=api_key)
    return _client


def send(prompt: str,
         model_config: ModelConfig,
         max_tokens: int = 1024,
         temperature: float = 0.1) -> dict:
    """
    Send a prompt to a Groq-hosted model.

    Args:
        prompt:       The user prompt string.
        model_config: ModelConfig from the registry.
        max_tokens:   Max output tokens.
        temperature:  Sampling temperature (low = more deterministic).

    Returns:
        dict with keys: text, input_tokens, output_tokens

    Raises:
        ProviderError on any API failure.
    """
    client = _get_client()

    try:
        response = client.chat.completions.create(
            model=model_config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:
        raise ProviderError(
            model_id=model_config.model_id,
            message=f"Groq API call failed: {e}",
            original_error=e,
        )

    text         = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    return {
        "text":          text,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
    }
