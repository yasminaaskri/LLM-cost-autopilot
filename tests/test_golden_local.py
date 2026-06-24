import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.classifier.predict import classify_prompt

def main():
    golden_path = ROOT / "tests" / "golden_prompts.json"
    
    if not golden_path.exists():
        print("[ERROR] golden_prompts.json not found!")
        sys.exit(1)

    with open(golden_path) as f:
        golden = json.load(f)

    correct = 0
    total = len(golden)

    print(f"\n[INFO] Testing {total} golden prompts...\n")

    for i, test in enumerate(golden, 1):
        prompt = test["prompt"]
        expected = test["expected_tier"]
        result = classify_prompt(prompt)
        tier = result.tier
        confidence = result.confidence
        
        if tier == expected:
            status = "[PASS]"
            correct += 1
        else:
            status = "[FAIL]"
        
        print(f"{status} [{i}/{total}] Expected: T{expected}, Got: T{tier} (conf: {confidence:.2%})")
        print(f"   Prompt: {prompt[:60]}...")

    accuracy = correct / total
    print("\n" + "="*60)
    print(f"[RESULT] Golden Prompts Accuracy: {accuracy:.0%} ({correct}/{total})")
    print("="*60)

    if accuracy < 0.80:
        print(f"[ERROR] Accuracy {accuracy:.0%} is below 80% gate!")
        sys.exit(1)
    
    print(f"[SUCCESS] Accuracy {accuracy:.0%} meets or exceeds 80% gate!")

if __name__ == "__main__":
    main()