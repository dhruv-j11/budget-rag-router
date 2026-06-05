"""
verify_data.py — Sanity checks on the processed data splits.

Checks:
  1. No ID leakage between train/val/test
  2. Label encoding is consistent
  3. All domains present in each split
  4. Tokenization produces valid tensors
  5. Class balance is roughly preserved across splits

Usage:
    python src/verify_data.py
"""

import os
import json
from collections import Counter
from transformers import AutoTokenizer

DATA_DIR = "data/"
SPLITS = ["all_train.jsonl", "all_val.jsonl", "all_test.jsonl"]
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def check_no_leakage(train, val, test):
    """Ensure no query IDs appear in multiple splits."""
    train_ids = {ex["id"] for ex in train}
    val_ids   = {ex["id"] for ex in val}
    test_ids  = {ex["id"] for ex in test}

    tv = train_ids & val_ids
    tt = train_ids & test_ids
    vt = val_ids & test_ids

    assert len(tv) == 0, f"Train-Val leak: {len(tv)} shared IDs"
    assert len(tt) == 0, f"Train-Test leak: {len(tt)} shared IDs"
    assert len(vt) == 0, f"Val-Test leak: {len(vt)} shared IDs"
    print("  ✓ No ID leakage between splits")


def check_labels(examples, split_name):
    """Ensure all labels are valid and encoding is correct."""
    for ex in examples:
        assert ex["type"] in LABEL_MAP, f"Unknown type: {ex['type']} in {split_name}"
        assert ex["label"] == LABEL_MAP[ex["type"]], (
            f"Label mismatch: {ex['type']}→{ex['label']} in {split_name}"
        )
    print(f"  ✓ All labels valid in {split_name}")


def check_domains(train, val, test):
    """Ensure all 4 domains are present in every split."""
    for name, split in [("train", train), ("val", val), ("test", test)]:
        domains = set(ex["domain"] for ex in split)
        assert len(domains) == 4, f"{name} missing domains: have {domains}"
    print("  ✓ All 4 domains present in every split")


def check_balance(train, val, test):
    """Check that class proportions are roughly preserved (within 3%)."""
    def get_proportions(examples):
        counts = Counter(ex["type"] for ex in examples)
        total = len(examples)
        return {k: v / total for k, v in counts.items()}

    train_p = get_proportions(train)
    for name, split in [("val", val), ("test", test)]:
        split_p = get_proportions(split)
        for label in LABEL_MAP:
            diff = abs(train_p.get(label, 0) - split_p.get(label, 0))
            assert diff < 0.03, (
                f"Class imbalance: {label} differs by {diff:.3f} in {name}"
            )
    print("  ✓ Class proportions preserved across splits (within 3%)")


def check_tokenization(examples):
    """Verify that a tokenizer can process the queries."""
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    sample = examples[:50]
    texts = [ex["question"] for ex in sample]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128,
                        return_tensors="pt")
    assert encoded["input_ids"].shape[0] == len(sample)
    assert encoded["input_ids"].shape[1] <= 128
    print(f"  ✓ Tokenization works (sample shape: {encoded['input_ids'].shape})")


def main():
    print("Running data verification checks...\n")

    # Load splits
    train = load_jsonl(os.path.join(DATA_DIR, "all_train.jsonl"))
    val   = load_jsonl(os.path.join(DATA_DIR, "all_val.jsonl"))
    test  = load_jsonl(os.path.join(DATA_DIR, "all_test.jsonl"))

    print(f"  Loaded: {len(train)} train / {len(val)} val / {len(test)} test\n")

    check_no_leakage(train, val, test)
    check_labels(train, "train")
    check_labels(val, "val")
    check_labels(test, "test")
    check_domains(train, val, test)
    check_balance(train, val, test)
    check_tokenization(train)

    print("\n✓ All verification checks passed.")


if __name__ == "__main__":
    main()
