"""
data_loader.py — Load RAGRouter-Bench from HuggingFace and create train/val/test splits.

The dataset has 4 subsets (domains):
  - graphragBench_medical  (1,896 queries)
  - musique                (3,356 queries)
  - quality                (1,198 queries)
  - ultraDomain_legal      (1,277 queries)

Each query has a `type` field: single_hop | multi_hop | summary
These map to retrieval paradigm recommendations downstream.

Usage:
    python src/data_loader.py
    python src/data_loader.py --output_dir data/ --seed 42
"""

import os
import json
import argparse
from collections import Counter

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# ---------- Constants ----------
SUBSETS = [
    "graphragBench_medical",
    "musique",
    "quality",
    "ultraDomain_legal",
]
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# Relative token costs from the RAGRouter-Bench paper (Table 5).
# Mapping: query_type -> recommended paradigm -> relative token cost.
PARADIGM_COSTS = {
    "single_hop": {"paradigm": "NaiveRAG", "cost": 1.4},
    "multi_hop":  {"paradigm": "IterativeRAG", "cost": 3.5},
    "summary":    {"paradigm": "HybridRAG", "cost": 2.8},
}


def load_all_subsets():
    """Pull every subset from HuggingFace and return a unified list of dicts."""
    all_examples = []
    for subset in SUBSETS:
        print(f"Loading {subset}...")
        ds = load_dataset("Chaplain0908/RAGRouter", subset, split="train")
        for row in ds:
            all_examples.append({
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "type": row["type"],
                "label": LABEL_MAP[row["type"]],
                "domain": subset,
            })
    return all_examples


def print_distribution(examples, tag=""):
    """Print label and domain distributions."""
    print(f"\n{'='*60}")
    print(f"  Distribution: {tag}")
    print(f"  Total examples: {len(examples)}")
    print(f"{'='*60}")

    # Overall label distribution
    label_counts = Counter(ex["type"] for ex in examples)
    total = len(examples)
    print("\n  Label distribution (overall):")
    for label in ["single_hop", "multi_hop", "summary"]:
        count = label_counts.get(label, 0)
        pct = 100 * count / total
        bar = "█" * int(pct / 2)
        print(f"    {label:<12} {count:>5} ({pct:5.1f}%) {bar}")

    # Per-domain breakdown
    print("\n  Per-domain breakdown:")
    domains = sorted(set(ex["domain"] for ex in examples))
    for domain in domains:
        domain_examples = [ex for ex in examples if ex["domain"] == domain]
        domain_counts = Counter(ex["type"] for ex in domain_examples)
        domain_total = len(domain_examples)
        short_name = domain.replace("graphragBench_", "").replace("ultraDomain_", "")
        print(f"\n    {short_name} (n={domain_total}):")
        for label in ["single_hop", "multi_hop", "summary"]:
            count = domain_counts.get(label, 0)
            pct = 100 * count / domain_total
            print(f"      {label:<12} {count:>5} ({pct:5.1f}%)")


def create_splits(examples, test_size=0.15, val_size=0.15, seed=42):
    """
    Create stratified train/val/test splits.

    Stratify by (domain, label) to ensure each split has representative
    samples from every domain-label combination.
    """
    # Create a combined stratification key
    strat_keys = [f"{ex['domain']}_{ex['type']}" for ex in examples]

    # First split: separate out test set
    train_val, test, strat_tv, _ = train_test_split(
        examples, strat_keys,
        test_size=test_size,
        random_state=seed,
        stratify=strat_keys,
    )

    # Second split: separate train and val from the remaining
    strat_tv_keys = [f"{ex['domain']}_{ex['type']}" for ex in train_val]
    relative_val = val_size / (1 - test_size)  # adjust ratio
    train, val, _, _ = train_test_split(
        train_val, strat_tv_keys,
        test_size=relative_val,
        random_state=seed,
        stratify=strat_tv_keys,
    )

    return train, val, test


def save_split(examples, filepath):
    """Save a list of dicts as jsonl."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"  Saved {len(examples):>5} examples → {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/", help="Where to save splits")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load everything
    print("Downloading RAGRouter-Bench from HuggingFace...")
    all_examples = load_all_subsets()
    print_distribution(all_examples, tag="Full dataset")

    # 2. Create splits
    print("\nCreating stratified train/val/test splits (70/15/15)...")
    train, val, test = create_splits(all_examples, seed=args.seed)

    # 3. Print split distributions
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print_distribution(split, tag=name)

    # 4. Save combined splits
    save_split(train, os.path.join(args.output_dir, "all_train.jsonl"))
    save_split(val,   os.path.join(args.output_dir, "all_val.jsonl"))
    save_split(test,  os.path.join(args.output_dir, "all_test.jsonl"))

    # 5. Also save per-domain splits (for per-domain experiments)
    for domain in SUBSETS:
        d_train = [ex for ex in train if ex["domain"] == domain]
        d_val   = [ex for ex in val   if ex["domain"] == domain]
        d_test  = [ex for ex in test  if ex["domain"] == domain]
        short = domain.replace("graphragBench_", "").replace("ultraDomain_", "")
        save_split(d_train, os.path.join(args.output_dir, f"{short}_train.jsonl"))
        save_split(d_val,   os.path.join(args.output_dir, f"{short}_val.jsonl"))
        save_split(d_test,  os.path.join(args.output_dir, f"{short}_test.jsonl"))

    print("\n✓ Data pipeline complete.")
    print(f"  Combined: {len(train)} train / {len(val)} val / {len(test)} test")
    print(f"  Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
