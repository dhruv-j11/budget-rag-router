"""
domain_transfer.py — Leave-one-domain-out (LODO) transfer evaluation.

For each of the 4 domains, train on the other 3 and test on the held-out one.
This answers: "Can a routing classifier generalize to unseen domains?"

Nobody has tested this on RAGRouter-Bench. The Bansal paper explicitly lists
out-of-distribution generalization as an untested limitation.

Usage:
    python src/domain_transfer.py --epochs 5
"""

import os
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, classification_report

# ---------- Constants ----------
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}
NUM_LABELS = 3

DOMAINS = {
    "medical": "graphragBench_medical",
    "musique": "musique",
    "quality": "quality",
    "legal": "ultraDomain_legal",
}


# ---------- Data ----------

class QueryDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex["question"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


# ---------- Model ----------

class QueryRouter(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = self.dropout(outputs.last_hidden_state[:, 0, :])
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# ---------- Training ----------

def train_and_eval(train_examples, test_examples, model_name, epochs, lr,
                   batch_size, device, tag=""):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = DataLoader(QueryDataset(train_examples, tokenizer),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(QueryDataset(test_examples, tokenizer),
                             batch_size=batch_size)

    model = QueryRouter(model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(0.1 * total_steps), total_steps)

    # Train
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += out["loss"].item()

    # Eval
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {
        "macro_f1": float(macro_f1),
        "predictions": [int(p) for p in all_preds],
        "labels": [int(l) for l in all_labels],
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Load ALL data
    all_train = load_jsonl(os.path.join(args.data_dir, "all_train.jsonl"))
    all_val = load_jsonl(os.path.join(args.data_dir, "all_val.jsonl"))
    all_test = load_jsonl(os.path.join(args.data_dir, "all_test.jsonl"))
    all_data = all_train + all_val + all_test

    # Also load in-domain combined results for comparison
    in_domain_path = os.path.join(args.output_dir, "neural_distilbert-base-uncased_combined.json")
    in_domain_results = {}
    if os.path.exists(in_domain_path):
        with open(in_domain_path) as f:
            in_domain_results = json.load(f).get("domain_results", {})

    # Leave-one-domain-out evaluation
    lodo_results = {}

    for held_out_short, held_out_full in DOMAINS.items():
        print(f"\n{'='*60}")
        print(f"  LODO: Held-out domain = {held_out_short}")
        print(f"{'='*60}")

        # Split: train on other 3 domains, test on held-out domain
        train_ex = [ex for ex in all_data if ex["domain"] != held_out_full]
        test_ex = [ex for ex in all_data if ex["domain"] == held_out_full]

        print(f"  Train: {len(train_ex)} examples (3 domains)")
        print(f"  Test:  {len(test_ex)} examples ({held_out_short})")

        # Check label distribution in test set
        from collections import Counter
        label_dist = Counter(ex["type"] for ex in test_ex)
        for label, count in sorted(label_dist.items()):
            print(f"    {label}: {count} ({100*count/len(test_ex):.1f}%)")

        t0 = time.time()
        result = train_and_eval(
            train_ex, test_ex, args.model, args.epochs,
            args.lr, args.batch_size, device, tag=held_out_short,
        )
        elapsed = time.time() - t0

        # Compare to in-domain
        domain_name_map = {
            "medical": "Medical", "musique": "MuSiQue",
            "quality": "QuALITY", "legal": "Legal",
        }
        short_to_display = domain_name_map[held_out_short]
        in_domain_f1 = in_domain_results.get(short_to_display, 0)
        transfer_gap = result["macro_f1"] - in_domain_f1

        result["held_out"] = held_out_short
        result["in_domain_f1"] = in_domain_f1
        result["transfer_gap"] = transfer_gap
        result["train_size"] = len(train_ex)
        result["test_size"] = len(test_ex)
        result["time_s"] = elapsed

        lodo_results[held_out_short] = result

        print(f"\n  LODO F1:      {result['macro_f1']:.4f}")
        print(f"  In-domain F1: {in_domain_f1:.4f}")
        print(f"  Transfer gap: {transfer_gap:+.4f}")
        print(f"  Time: {elapsed:.1f}s")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  LEAVE-ONE-DOMAIN-OUT: Transfer Summary")
    print(f"{'='*70}")
    print(f"  {'Held-out':<12} | {'LODO F1':>8} | {'In-domain F1':>12} | {'Gap':>8} | {'Transfer?':>10}")
    print(f"  {'-'*60}")

    for short, r in lodo_results.items():
        display = domain_name_map[short]
        gap = r["transfer_gap"]
        # If gap > -0.05, transfer is reasonable
        transfer_ok = "YES" if gap > -0.05 else "PARTIAL" if gap > -0.15 else "NO"
        print(f"  {display:<12} | {r['macro_f1']:>8.4f} | {r['in_domain_f1']:>12.4f} | {gap:>+8.4f} | {transfer_ok:>10}")

    avg_lodo = np.mean([r["macro_f1"] for r in lodo_results.values()])
    avg_in_domain = np.mean([r["in_domain_f1"] for r in lodo_results.values()])
    print(f"\n  Average LODO F1:      {avg_lodo:.4f}")
    print(f"  Average in-domain F1: {avg_in_domain:.4f}")
    print(f"  Average transfer gap: {avg_lodo - avg_in_domain:+.4f}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "lodo_transfer.json")
    with open(out_path, "w") as f:
        json.dump(lodo_results, f, indent=2, default=float)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
