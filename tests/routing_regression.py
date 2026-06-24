# tests/routing_regression.py
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.classifier.predict import classify_prompt

def main():
    with open(ROOT / "tests" / "golden_prompts.json") as f:
        golden = json.load(f)
    
    correct = sum(classify_prompt(g['prompt']).tier == g['expected_tier'] for g in golden)
    total = len(golden)
    pct = correct / total
    
    print(f"Golden Prompts Accuracy: {correct}/{total} ({pct:.0%})")
    assert pct >= 0.80, f"Accuracy {pct:.0%} below 80% gate"
    print("✅ Regression gate passed!")

if __name__ == "__main__":
    main()