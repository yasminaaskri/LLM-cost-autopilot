"""
Feature extraction — turns a prompt string into a numeric vector.

14 features (up from 12). The two additions fix the Tier 2/3 confusion:
  - complexity_keywords: verbs that signal complex output (write, design,
    develop, evaluate, propose, critique, compare) — strong Tier 3 signal
  - multi_step_signals: numbered steps or "first/then/finally" chains —
    strong Tier 2/3 signal absent from Tier 1

Root cause of the original misclassification:
  - reasoning_keywords fired 0 on most Tier 3 prompts because the list
    was too narrow ("analyze", "compare" etc.) while the actual Tier 3
    training data used "write", "design", "develop".
  - token_count dominated: short truncated test prompts (~16 tokens)
    looked identical to Tier 1 training examples.

Fix: broader keyword coverage + explicit complexity_keywords feature
that fires on "write a function", "design a system", etc.
"""

from __future__ import annotations
import re
from typing import Dict, List

# ── Tokeniser — graceful fallback when tiktoken BPE can't be downloaded ────────
try:
    import tiktoken as _tiktoken
    _enc = _tiktoken.encoding_for_model("gpt-4")
    _USE_TIKTOKEN = True
except Exception:
    _USE_TIKTOKEN = False
    _enc = None

# ── Keyword lists ──────────────────────────────────────────────────────────────
_CONSTRAINT_WORDS = [
    "must", "should", "ensure", "only", "required",
    "never", "always", "exactly", "strictly",
]

# Words that signal reasoning/analysis — Tier 2/3
_REASONING_WORDS = [
    "analyze", "analyse", "compare", "evaluate", "synthesize", "synthesise",
    "critique", "reason", "infer", "justify", "assess", "examine",
    "trade-off", "tradeoff", "implication", "consequence", "recommend",
    "strategy", "framework", "methodology", "impact", "competitive",
]

# Words that signal complex OUTPUT tasks — strong Tier 3 signal
# "write a function", "design a system", "develop a strategy"
_COMPLEXITY_WORDS = [
    "write a", "design a", "develop a", "build a", "implement a",
    "create a", "propose a", "architect", "implement",
    "production-ready", "end-to-end", "comprehensive",
]

# Simple instruction verbs — used by all tiers but dominated by Tier 1/2
_INSTRUCTION_WORDS = [
    "extract", "summarize", "summarise", "classify", "translate",
    "write", "create", "list", "identify", "reformat", "convert",
    "find", "count", "check", "is this", "what is",
]


def extract_features(prompt: str) -> Dict[str, float]:
    """
    Extract 14 numeric features from a single prompt string.
    All values are float for direct sklearn consumption.
    """
    return {
        "token_count":              _count_tokens(prompt),
        "char_count":               float(len(prompt)),
        "word_count":               float(len(prompt.split())),
        "sentence_count":           _count_sentences(prompt),
        "has_code_block":           1.0 if "```" in prompt else 0.0,
        "has_numbered_list":        1.0 if re.search(r"\d+[\.\)]\s", prompt) else 0.0,
        "num_constraints":          _count_keywords(prompt, _CONSTRAINT_WORDS),
        "reasoning_keywords":       _count_keywords(prompt, _REASONING_WORDS),
        "complexity_keywords":      _count_complexity(prompt),
        "output_format_complexity": _output_complexity(prompt),
        "context_length":           _context_length(prompt),
        "question_count":           float(prompt.count("?")),
        "instruction_keywords":     _count_keywords(prompt, _INSTRUCTION_WORDS),
        "multi_step_signals":       _multi_step(prompt),
    }


def get_feature_names() -> List[str]:
    """
    Stable ordered list of feature names.
    Must match extract_features() key order exactly.
    Saved in the pkl so inference always uses the same columns as training.
    """
    return [
        "token_count",
        "char_count",
        "word_count",
        "sentence_count",
        "has_code_block",
        "has_numbered_list",
        "num_constraints",
        "reasoning_keywords",
        "complexity_keywords",
        "output_format_complexity",
        "context_length",
        "question_count",
        "instruction_keywords",
        "multi_step_signals",
    ]


def features_to_list(features: Dict[str, float]) -> List[float]:
    """Convert features dict to ordered list for sklearn. Order = get_feature_names()."""
    return [features.get(name, 0.0) for name in get_feature_names()]


# ── Private helpers ────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> float:
    if _USE_TIKTOKEN and _enc is not None:
        try:
            return float(len(_enc.encode(text)))
        except Exception:
            pass
    return float(len(text) / 4)


def _count_sentences(text: str) -> float:
    parts = re.split(r"[.!?]+", text)
    return float(len([p for p in parts if p.strip()]))


def _count_keywords(text: str, keywords: List[str]) -> float:
    text_lower = text.lower()
    return float(sum(1 for kw in keywords if kw in text_lower))


def _count_complexity(text: str) -> float:
    """
    Score complexity-signalling phrases.
    "write a function", "design a system" etc. score 1.0 each.
    This is the key fix — these phrases dominate Tier 3 training data.
    """
    text_lower = text.lower()
    return float(sum(1 for phrase in _COMPLEXITY_WORDS if phrase in text_lower))


def _output_complexity(text: str) -> float:
    text_lower = text.lower()
    if any(w in text_lower for w in ["json", "structured", "csv", "yaml", "table"]):
        return 2.0
    if any(w in text_lower for w in ["list", "bullet", "enumerate", "numbered"]):
        return 1.0
    return 0.0


def _context_length(text: str) -> float:
    """
    Tokens AFTER the first instruction word.
    Longer context = more likely Tier 2/3.
    """
    words = text.split()
    for i, word in enumerate(words):
        if word.lower().rstrip(".,?:!") in _INSTRUCTION_WORDS:
            rest = " ".join(words[i + 1:])
            return _count_tokens(rest)
    return 0.0


def _multi_step(text: str) -> float:
    """
    Detect multi-step task signals:
      - Numbered list (1. 2. 3.) in the prompt body
      - Transition words indicating sequential steps
    """
    text_lower = text.lower()
    step_words = ["first,", "then,", "finally,", "additionally,",
                  "step 1", "step 2", "1)", "2)", "3)"]
    score = float(sum(1 for w in step_words if w in text_lower))
    # Also add 1.0 if numbered list pattern found
    if re.search(r"\d+[\.\)]\s", text):
        score += 1.0
    return min(score, 3.0)   # cap at 3 to avoid dominating
