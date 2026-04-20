"""Run all bounce detection evaluation combinations and output comparison."""

import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.config import EvalConfig
from eval.runner import run_all_combinations
from eval.compare import build_comparison

def main():
    cfg = EvalConfig()

    print("Running all 4 evaluation combinations...")
    results = run_all_combinations(cfg)

    if results:
        comparison = build_comparison(results)

        os.makedirs("eval/results", exist_ok=True)
        with open("eval/results/comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nSaved: eval/results/comparison.json")

        print(f"\n{'Method':<30} {'Recall':>8} {'Prec':>8} {'F1':>8} {'Err_med':>8} {'Det':>6} {'FP':>6}")
        print("-" * 80)
        for r in comparison.get("summary_table", comparison.get("results", [])):
            print(f"{r['method']:<30} {r['recall']:>7.1%} {r['precision']:>7.1%} "
                  f"{r['f1']:>7.1%} {r.get('landing_error_median',0):>7.3f}m "
                  f"{r['detected_count']:>5} {r['false_positives']:>5}")
    else:
        print("No results generated.")


if __name__ == "__main__":
    main()
