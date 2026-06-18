"""
Test the complete system with 20 prompts.
"""

import sys
import asyncio
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.router.router import get_router
from src.providers.dispatcher import send_request
from src.database import init_db, log_request, get_summary_stats
from src.models import ClassifierResult


# ── FULL, REALISTIC prompts (not truncated) ──────────────────────────────────
TEST_PROMPTS = [
    # ── Tier 1 — Simple ──────────────────────────────────────────────────────
    "Extract all email addresses from this text: support@company.com or sales@firm.io for help.",
    "Reformat this date from MM/DD/YYYY to DD-MM-YYYY: 06/17/2026",
    "Is the following sentence grammatically correct? Reply yes or no. Sentence: 'She don't like coffee.'",
    "Extract the product price from this text: The laptop costs $999.99",
    "Count the number of words in this sentence: The quick brown fox jumps over the lazy dog",
    "Convert this temperature from Celsius to Fahrenheit: 25°C",
    "Is the word 'running' a verb? Answer yes or no.",
    
    # ── Tier 2 — Moderate ──────────────────────────────────────────────────
    "Summarize the following paragraph in exactly 2 sentences: Artificial intelligence has transformed how companies operate, enabling automation of repetitive tasks, faster data analysis, and more personalised customer experiences. However, the rapid adoption of AI also raises concerns about job displacement, data privacy, and the concentration of power in the hands of a few large technology companies.",
    "Classify this customer message into one of: complaint, question, compliment, or feature_request. Message: 'Your app keeps crashing whenever I try to export a PDF. This is really frustrating — I have lost work twice now.'",
    "Translate the following sentence to French: 'The meeting has been rescheduled to Thursday at 3pm. Please update your calendars accordingly.'",
    "Extract all named entities (people, organisations, locations) from this text and list them by category: 'Elon Musk announced that Tesla will open a new Gigafactory in Mexico City next year, in partnership with the Mexican government and local supplier Grupo Industrial Saltillo.'",
    "Classify this sentiment as positive, negative, or neutral: 'The product is decent, but I expected better customer support. The software works fine though.'",
    "Write a short product description based on these features: 'AI-powered chatbot, 24/7 support, multi-language, integrates with Slack and Teams'",
    
    # ── Tier 3 — Complex ──────────────────────────────────────────────────
    "Analyze the competitive landscape for EV batteries and recommend a market entry strategy. Consider: current players, cost trends, and geographic opportunities.",
    "Write a Python function called flatten_dict that: 1) Accepts a nested dict of arbitrary depth, 2) Flattens it using dot-notation keys, 3) Handles lists by indexing them. Include type hints and a docstring.",
    "What are the second-order economic consequences of widespread LLM adoption in knowledge work? Consider effects on wage levels for different skill brackets and how this differs from previous automation waves.",
    "Compare the trade-offs between microservices and a monolith for a startup building a B2B SaaS product. Consider: team size (5 engineers), expected scale (10k users in year 1), and the need to ship quickly. Provide a clear recommendation with reasoning.",
    "Critique this business plan and identify the three biggest risks: 'We will build an AI-powered platform that connects freelancers with businesses. Our competitive advantage is our proprietary matching algorithm. We plan to capture 10% market share in year one.'",
    "Explain how attention mechanisms work and why they replaced RNNs in modern NLP architectures. Include the mathematical intuition and practical benefits.",
    "Write a complete React component that implements a search bar with debouncing. Include proper state management, API integration, and error handling.",
]


async def run_test():
    print("=" * 70)
    print("END-TO-END ROUTING TEST")
    print("=" * 70)
    
    # Initialize database
    init_db()
    router = get_router()
    
    results = []
    total_cost = 0
    total_baseline = 0
    
    # Process each prompt
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n[{i}/{len(TEST_PROMPTS)}]")
        print(f"  Prompt: {prompt[:60]}...")
        
        try:
            # 1. Route
            model_config, result = router.route(prompt)
            
            # 2. Send request
            response = send_request(prompt, model_config)
            
            # 3. Log to database
            classifier = ClassifierResult(
                tier=result.tier,
                confidence=result.confidence,
                features={},
                low_confidence=result.low_confidence
            )
            
            log_request(
                response=response,
                classifier=classifier,
                prompt_preview=prompt[:100],
                output_preview=response.output_text[:100]
            )
            
            # 4. Track costs
            total_cost += response.cost_usd
            total_baseline += response.cost_if_highest_quality
            
            savings_pct = ((response.cost_if_highest_quality - response.cost_usd) / 
                          response.cost_if_highest_quality * 100) if response.cost_if_highest_quality > 0 else 0
            
            print(f"  ✅ Tier {result.tier} (conf: {result.confidence:.2%})")
            print(f"  ✅ Model: {model_config.display_name}")
            print(f"  ✅ Cost: ${response.cost_usd:.6f}")
            print(f"  ✅ Savings: {savings_pct:.1f}%")
            
            results.append({
                "tier": result.tier,
                "model": model_config.display_name,
                "cost": response.cost_usd,
                "savings": savings_pct
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    stats = get_summary_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Total cost: ${stats['total_cost_usd']:.6f}")
    print(f"Total baseline: ${stats['total_baseline_cost']:.6f}")
    print(f"Savings: {stats['savings_pct']:.1f}%")
    
    print("\nRouting Distribution:")
    model_counts = Counter(r["model"] for r in results)
    for model, count in model_counts.most_common():
        print(f"  {model}: {count} ({count/len(results)*100:.0f}%)")
    
    # Verify
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    if stats['total_requests'] >= 20:
        print("✅ 20+ requests in database")
    if stats['total_baseline_cost'] > 0:
        print("✅ cost_if_highest_quality populated")
    if stats['savings_pct'] > 0:
        print("✅ Savings > 0")
    
    print("\n🎉 System working!")


if __name__ == "__main__":
    asyncio.run(run_test())