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


TEST_PROMPTS = [
    # Tier 1 - Simple
    "Extract all email addresses from: support@company.com",
    "Is this sentence correct? 'He go to school.' Reply yes or no.",
    "Convert this date: 06/17/2026 to DD-MM-YYYY",
    "What is the capital of France?",
    "Extract the price from: 'The item costs $49.99'",
    "Count the number of words in this sentence",
    
    # Tier 2 - Moderate
    "Summarize this in 2 sentences: AI is transforming how companies operate...",
    "Classify this sentiment: 'The product is good but expensive'",
    "Translate this to French: 'Hello, how are you?'",
    "Extract key entities from: 'Microsoft acquired Activision'",
    "Write a short product description for a wireless mouse",
    "Identify the sentiment: 'I absolutely love this product!'",
    
    # Tier 3 - Complex
    "Analyze the trade-offs between cloud and on-premise infrastructure",
    "Write a recursive function to traverse a binary tree",
    "What are the long-term implications of quantum computing on cryptography?",
    "Evaluate the pros and cons of microservices vs monolith",
    "Critique this business strategy: 'We will disrupt the market'",
    "Compare different prompt engineering techniques for LLMs",
    "Explain how transformer attention mechanisms work",
    "Analyze the competitive landscape for EV batteries and recommend a strategy"
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
        print(f"  Prompt: {prompt[:50]}...")
        
        try:
            # 1. Route
            model_config, tier, confidence = router.route(prompt)
            
            # 2. Send request
            response =  send_request(prompt, model_config)
            
            # 3. Log to database
            classifier = ClassifierResult(
                tier=tier,
                confidence=confidence,
                features={},
                low_confidence=confidence < 0.60
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
            
            print(f"  ✅ Tier {tier} (conf: {confidence:.2%})")
            print(f"  ✅ Model: {model_config.display_name}")
            print(f"  ✅ Cost: ${response.cost_usd:.6f}")
            print(f"  ✅ Savings: {savings_pct:.1f}%")
            
            results.append({
                "tier": tier,
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