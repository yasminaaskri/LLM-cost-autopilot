"""
Feature extraction - turns text into numbers for the classifier.
"""

import re
import tiktoken
from typing import Dict, List

# Initialize tokenizer
try:
    enc = tiktoken.encoding_for_model("gpt-4")
except Exception:
    enc = tiktoken.get_encoding("cl100k_base")


def extract_features(prompt: str) -> Dict[str, float]:
    """
    Extract features from a single prompt.
    
    Returns 12 features that help determine complexity.
    """
    features = {
        "token_count": _count_tokens(prompt),
        "char_count": float(len(prompt)),
        "word_count": float(len(prompt.split())),
        "sentence_count": _count_sentences(prompt),
        "has_code_block": 1.0 if "```" in prompt else 0.0,
        "has_numbered_list": 1.0 if re.search(r'\d+\.\s', prompt) else 0.0,
        "num_constraints": _count_words(prompt, ["must", "should", "ensure", "only", "required"]),
        "reasoning_keywords": _count_words(prompt, ["analyze", "compare", "evaluate", "synthesize", "critique"]),
        "output_format_complexity": _output_complexity(prompt),
        "context_length": _context_length(prompt),
        "question_count": float(prompt.count("?")),
        "instruction_keywords": _count_words(prompt, ["extract", "summarize", "classify", "translate", "write", "create"]),
    }
    return features


def _count_tokens(text: str) -> float:
    """Count tokens using tiktoken."""
    try:
        return float(len(enc.encode(text)))
    except:
        return len(text) / 4.0


def _count_sentences(text: str) -> float:
    """Count sentences."""
    sentences = re.split(r'[.!?]+', text)
    return float(len([s for s in sentences if s.strip()]))


def _count_words(text: str, keywords: List[str]) -> float:
    """Count how many keywords appear."""
    text_lower = text.lower()
    return float(sum(1 for kw in keywords if kw in text_lower))


def _output_complexity(text: str) -> float:
    """Check output format complexity."""
    text_lower = text.lower()
    if "json" in text_lower or "structured" in text_lower:
        return 2.0
    elif "list" in text_lower or "bullet" in text_lower:
        return 1.0
    return 0.0


def _context_length(text: str) -> float:
    """Count tokens before the instruction word."""
    instruction_words = ["extract", "summarize", "classify", "translate", "write", "create", "analyze"]
    words = text.split()
    for i, word in enumerate(words):
        if word.lower() in instruction_words:
            return _count_tokens(" ".join(words[:i]))
    return 0.0


def get_feature_names() -> List[str]:
    """Return feature names in order."""
    return [
        "token_count", "char_count", "word_count", "sentence_count",
        "has_code_block", "has_numbered_list", "num_constraints",
        "reasoning_keywords", "output_format_complexity", "context_length",
        "question_count", "instruction_keywords"
    ]


def features_to_list(features: Dict[str, float]) -> List[float]:
    """Convert features dict to list for sklearn."""
    return [features.get(name, 0.0) for name in get_feature_names()]