"""
scripts/baseline_run.py

Sends 10 prompts (spanning all 3 future complexity tiers) to every model
in the registry. Logs all results to the DB and to data/baseline_results.csv.

Run with:  python scripts/baseline_run.py

Purpose:
  1. Validate all 3 provider connections work end-to-end.
  2. Collect real cost/latency/quality data per model.
  3. Reveal where Llama 3.1 8B fails vs Llama 3.3 70B and Gemini 2.5 Flash.
  4. Inform your tier definitions before building the classifier.

Output:
  - Prints a summary table to stdout
  - Saves data/baseline_results.csv
  - Inserts rows into baseline_runs DB table
"""

import sys
import os
import time
import csv
from datetime import datetime
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_registry, calculate_cost_if_highest_quality
from src.database import init_db, log_baseline_run

# ── 10 prompts spanning all 3 future tiers ────────────────────────────────────
# Labels are for your reference — the classifier doesn't exist yet.
PROMPTS = [
    # Tier 1 — Simple
    {
        "text": "Extract all email addresses from this text: "
                "Contact us at support@company.com or sales@firm.io for help.",
        "expected_tier": 1,
        "label": "extraction",
    },
    {
        "text": "Reformat this date from MM/DD/YYYY to DD-MM-YYYY: 06/17/2026",
        "expected_tier": 1,
        "label": "reformatting",
    },
    {
        "text": 'Is the following sentence grammatically correct? '
                'Reply with only "yes" or "no". '
                'Sentence: "She don\'t like coffee."',
        "expected_tier": 1,
        "label": "yes_no",
    },
    # Tier 2 — Moderate
    {
        "text": "Summarize the following paragraph in exactly 2 sentences: "
                "Artificial intelligence has transformed how companies operate, "
                "enabling automation of repetitive tasks, faster data analysis, "
                "and more personalised customer experiences. However, the rapid "
                "adoption of AI also raises concerns about job displacement, "
                "data privacy, and the concentration of power in the hands of "
                "a few large technology companies.",
        "expected_tier": 2,
        "label": "summarization",
    },
    {
        "text": "Classify this customer message into one of: "
                "complaint, question, compliment, or feature_request. "
                "Message: 'Your app keeps crashing whenever I try to export "
                "a PDF. This is really frustrating — I've lost work twice now.'",
        "expected_tier": 2,
        "label": "classification",
    },
    {
        "text": "Translate the following sentence to French: "
                "'The meeting has been rescheduled to Thursday at 3pm. "
                "Please update your calendars accordingly.'",
        "expected_tier": 2,
        "label": "translation",
    },
    {
        "text": "Extract all named entities (people, organisations, locations) "
                "from this text and list them by category: "
                "'Elon Musk announced that Tesla will open a new Gigafactory "
                "in Mexico City next year, in partnership with the Mexican "
                "government and local supplier Grupo Industrial Saltillo.'",
        "expected_tier": 2,
        "label": "ner",
    },
    # Tier 3 — Complex
    {
        "text": "Analyze the key trade-offs between using a monolithic "
                "architecture versus microservices for a startup building "
                "a B2B SaaS product. Consider: team size (5 engineers), "
                "expected scale (10k users in year 1), and the need to "
                "ship quickly. Provide a clear recommendation with reasoning.",
        "expected_tier": 3,
        "label": "multi_step_reasoning",
    },
    {
        "text": "Write a Python function called `parse_nested_config` that: "
                "1) Accepts a nested dict of arbitrary depth, "
                "2) Flattens it into a single-level dict using dot notation "
                "for keys (e.g. {'a': {'b': 1}} → {'a.b': 1}), "
                "3) Handles lists by indexing them (e.g. {'a': [1,2]} → "
                "{'a.0': 1, 'a.1': 2}). Include type hints and a docstring.",
        "expected_tier": 3,
        "label": "code_generation",
    },
    {
        "text": "What are the second-order economic consequences of widespread "
                "LLM adoption in knowledge work? Consider effects on: "
                "wage levels for different skill brackets, the market for "
                "junior professional roles, productivity growth distribution, "
                "and how this differs from previous automation waves. "
                "Give a nuanced, evidence-grounded analysis.",
        "expected_tier": 3,
        "label": "nuanced_analysis",
    },
]


def run_single(provider_module, model_config, prompt_text: str) -> dict:
    """
    Call one model with one prompt. Returns a result dict.
    Times the call and computes cost + baseline comparison.
    """
    start = time.monotonic()
    try:
        raw = provider_module.send(prompt_text, model_config)
        latency_ms = (time.monotonic() - start) * 1000

        cost_usd       = model_config.cost_for_tokens(
            raw["input_tokens"], raw["output_tokens"])
        cost_if_highest_quality  = calculate_cost_if_highest_quality(
            raw["input_tokens"], raw["output_tokens"])

        return {
            "success":       True,
            "output":        raw["text"],
            "input_tokens":  raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "latency_ms":    round(latency_ms, 1),
            "cost_usd":      cost_usd,
            "cost_if_highest_quality": cost_if_highest_quality,
            "savings_pct":   round((1 - cost_usd / cost_if_highest_quality) * 100, 1)
                             if cost_if_highest_quality > 0 else 0.0,
            "error":         None,
        }
    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        return {
            "success":       False,
            "output":        "",
            "input_tokens":  0,
            "output_tokens": 0,
            "latency_ms":    round(latency_ms, 1),
            "cost_usd":      0.0,
            "cost_if_highest_quality": 0.0,
            "savings_pct":   0.0,
            "error":         str(e),
        }


def main():
    print("=" * 70)
    print("LLM Cost Autopilot — Baseline Run")
    print(f"Started at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    # Init DB
    init_db()

    # Load registry
    registry = load_registry()
    model_keys = list(registry.keys())
    print(f"\n✓ Loaded {len(model_keys)} models: {', '.join(model_keys)}\n")

    # Import provider modules lazily (so missing API keys fail gracefully)
    providers = {}
    for key, config in registry.items():
        if config.provider == "groq":
            from src.providers import groq_provider as mod
            providers[key] = mod
        elif config.provider == "google":
            from src.providers import google_provider as mod
            providers[key] = mod

    run_timestamp = datetime.utcnow().isoformat()
    all_results   = []

    # ── Main loop ──────────────────────────────────────────────────────────────
    for p_idx, prompt_meta in enumerate(PROMPTS):
        prompt_text = prompt_meta["text"]
        print(f"\nPrompt {p_idx + 1}/{len(PROMPTS)} "
              f"[tier {prompt_meta['expected_tier']} · {prompt_meta['label']}]")
        print(f"  {prompt_text[:80]}{'...' if len(prompt_text) > 80 else ''}")

        for model_key in model_keys:
            config = registry[model_key]
            mod    = providers[model_key]

            print(f"  → {config.display_name} ...", end="", flush=True)
            result = run_single(mod, config, prompt_text)

            if result["success"]:
                print(f" {result['latency_ms']:.0f}ms "
                      f"${result['cost_usd']:.6f} "
                      f"(saved {result['savings_pct']:.1f}% vs GPT-4o)")

                # Log to DB
                log_baseline_run(
                    run_timestamp  = run_timestamp,
                    prompt_index   = p_idx,
                    prompt_preview = prompt_text,
                    model_key      = model_key,
                    output_preview = result["output"],
                    input_tokens   = result["input_tokens"],
                    output_tokens  = result["output_tokens"],
                    cost_usd       = result["cost_usd"],
                    latency_ms     = result["latency_ms"],
                )
            else:
                print(f" FAILED: {result['error']}")

            all_results.append({
                "prompt_index":    p_idx,
                "prompt_label":    prompt_meta["label"],
                "expected_tier":   prompt_meta["expected_tier"],
                "model_key":       model_key,
                "provider":        config.provider,
                "display_name":    config.display_name,
                "success":         result["success"],
                "input_tokens":    result["input_tokens"],
                "output_tokens":   result["output_tokens"],
                "latency_ms":      result["latency_ms"],
                "cost_usd":        result["cost_usd"],
                "cost_if_highest_quality": 0.0,
                "savings_pct":     result["savings_pct"],
                "output_preview":  result["output"][:150],
                "error":           result["error"],
            })

    # ── Save CSV ───────────────────────────────────────────────────────────────
    csv_path = ROOT / "data" / "baseline_results.csv"
    fieldnames = list(all_results[0].keys()) if all_results else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n✓ Results saved to {csv_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — Average cost and latency per model")
    print("=" * 70)
    print(f"{'Model':<30} {'Avg Cost':>12} {'Avg Latency':>14} {'Avg Savings%':>14}")
    print("-" * 70)

    for model_key in model_keys:
        model_results = [r for r in all_results
                         if r["model_key"] == model_key and r["success"]]
        if not model_results:
            print(f"{registry[model_key].display_name:<30} {'FAILED':>12}")
            continue
        avg_cost    = sum(r["cost_usd"] for r in model_results) / len(model_results)
        avg_latency = sum(r["latency_ms"] for r in model_results) / len(model_results)
        avg_savings = sum(r["savings_pct"] for r in model_results) / len(model_results)
        print(f"{registry[model_key].display_name:<30} "
              f"${avg_cost:>10.6f} "
              f"{avg_latency:>12.0f}ms "
              f"{avg_savings:>12.1f}%")

    print("-" * 70)
    total_cost  = sum(r["cost_usd"] for r in all_results if r["success"])
    total_base  = sum(r["cost_if_highest_quality"] for r in all_results if r["success"])
    if total_base > 0:
        overall_savings = (1 - total_cost / total_base) * 100
        print(f"\n✓ Total cost this run:     ${total_cost:.6f}")
        print(f"✓ Highest quality equivalent would: ${total_base:.6f}")
        print(f"✓ Overall savings:          {overall_savings:.1f}%")

    print("\n✓ Baseline run complete. Study the CSV before building the classifier.")
    print("  Key questions to answer:")
    print("  - Where does Llama 3.1 8B fail vs Llama 3.3 70B?")
    print("  - Which prompts does Gemini 2.5 Flash handle as well as 70B?")
    print("  - What patterns do Tier 3 failures share?\n")


if __name__ == "__main__":
    main()
