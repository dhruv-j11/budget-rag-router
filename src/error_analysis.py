"""
error_analysis.py — Analyze misclassified queries from the neural router.

Shows:
  1. Which class pairs are most confused (confusion matrix analysis)
  2. Sample misclassified queries per confusion pair
  3. Domain-level error breakdown
  4. Query length and structural patterns in errors

Usage:
    python src/error_analysis.py --results results/neural_distilbert-base-uncased_combined.json
"""

import os
import json
import argparse
import numpy as np
from collections import Counter, defaultdict

LABEL_NAMES = {0: "single_hop", 1: "multi_hop", 2: "summary"}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/neural_distilbert-base-uncased_combined.json")
    parser.add_argument("--data_dir", default="data/")
    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    cm = np.array(results["confusion_matrix"])
    preds = results.get("predictions", None)
    labels = results.get("labels", None)

    print(f"\n{'='*60}")
    print(f"  Error Analysis: {results['model']} ({results['tag']})")
    print(f"  Test macro-F1: {results['test_macro_f1']:.4f}")
    print(f"{'='*60}")

    # 1. Confusion matrix analysis
    print(f"\n  Confusion Matrix:")
    print(f"  {'':>12} {'Pred SH':>10} {'Pred MH':>10} {'Pred Sum':>10}")
    for i, row_name in enumerate(["True SH", "True MH", "True Sum"]):
        print(f"  {row_name:>12} {cm[i][0]:>10} {cm[i][1]:>10} {cm[i][2]:>10}")

    # 2. Most confused pairs
    print(f"\n  Most confused class pairs:")
    pairs = []
    for i in range(3):
        for j in range(3):
            if i != j:
                pairs.append((cm[i][j], LABEL_NAMES[i], LABEL_NAMES[j]))
    pairs.sort(reverse=True)
    for count, true_label, pred_label in pairs[:4]:
        total_for_class = cm[list(LABEL_NAMES.keys())[list(LABEL_NAMES.values()).index(true_label)]].sum()
        pct = 100 * count / total_for_class if total_for_class > 0 else 0
        print(f"    {true_label} → {pred_label}: {count} errors ({pct:.1f}% of {true_label})")

    # 3. Load test data for per-example analysis
    test = load_jsonl(os.path.join(args.data_dir, "all_test.jsonl"))

    if preds is not None and labels is not None and len(preds) == len(test):
        # Find misclassified examples
        errors = []
        for i, ex in enumerate(test):
            if preds[i] != labels[i]:
                errors.append({
                    "question": ex["question"],
                    "true": LABEL_NAMES[labels[i]],
                    "pred": LABEL_NAMES[preds[i]],
                    "domain": ex["domain"],
                })

        print(f"\n  Total errors: {len(errors)} / {len(test)} ({100*len(errors)/len(test):.1f}%)")

        # 4. Domain breakdown
        print(f"\n  Errors by domain:")
        domain_errors = Counter(e["domain"] for e in errors)
        domain_totals = Counter(ex["domain"] for ex in test)
        for domain in sorted(domain_totals.keys()):
            n_err = domain_errors.get(domain, 0)
            n_total = domain_totals[domain]
            short = domain.replace("graphragBench_", "").replace("ultraDomain_", "")
            print(f"    {short:<12}: {n_err}/{n_total} ({100*n_err/n_total:.1f}%)")

        # 5. Sample errors per confusion pair
        print(f"\n  Sample misclassified queries:")
        error_by_pair = defaultdict(list)
        for e in errors:
            error_by_pair[(e["true"], e["pred"])].append(e)

        for (true_l, pred_l), examples in sorted(error_by_pair.items(), key=lambda x: -len(x[1])):
            print(f"\n    {true_l} → {pred_l} ({len(examples)} total):")
            for ex in examples[:3]:  # show 3 examples per pair
                q = ex["question"][:100] + ("..." if len(ex["question"]) > 100 else "")
                short_d = ex["domain"].replace("graphragBench_", "").replace("ultraDomain_", "")
                print(f"      [{short_d}] \"{q}\"")

        # 6. Save errors for manual review
        os.makedirs("results/", exist_ok=True)
        error_path = "results/error_examples.jsonl"
        with open(error_path, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
        print(f"\n  All errors saved → {error_path}")

    else:
        print("\n  Note: per-example predictions not found in results file.")
        print("  Re-run neural_router.py to generate them.")


if __name__ == "__main__":
    main()
