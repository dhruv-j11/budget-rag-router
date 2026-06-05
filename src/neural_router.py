"""
neural_router.py — Fine-tune transformer models for query-type routing.

This is the core contribution: training neural classifiers on RAGRouter-Bench,
which has not been done before (Bansal & Agarwal 2026 only used classical ML).

Models:
  - distilbert-base-uncased (66M params, 6 layers)
  - huawei-noah/TinyBERT_General_4L_312D (14.5M params, 4 layers)

Task: 3-class classification (single_hop / multi_hop / summary)

Usage:
    # Combined training (all domains)
    python src/neural_router.py --model distilbert-base-uncased --epochs 5 --lr 2e-5

    # Per-domain training
    python src/neural_router.py --model distilbert-base-uncased --per_domain

    # TinyBERT
    python src/neural_router.py --model huawei-noah/TinyBERT_General_4L_312D --epochs 5 --lr 3e-5
"""

import os
import json
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------- Constants ----------
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = 3


# ---------- Dataset ----------

class QueryDataset(Dataset):
    """PyTorch dataset for query classification."""

    def __init__(self, examples, tokenizer, max_length=128):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex["question"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }


# ---------- Model ----------

class QueryRouter(nn.Module):
    """
    Transformer encoder + classification head.

    Architecture:
        input_ids → transformer → [CLS] pooling → dropout → linear → 3 logits

    The classification head is intentionally simple — the hypothesis is that
    a pre-trained encoder already captures enough query semantics that a single
    linear layer suffices for routing.
    """

    def __init__(self, model_name, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ---------- Training ----------

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                n_batches += 1

    avg_loss = total_loss / n_batches if n_batches > 0 else 0
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
    }


# ---------- Main training loop ----------

def train_model(
    model_name: str,
    train_examples: list,
    val_examples: list,
    test_examples: list,
    epochs: int = 5,
    lr: float = 2e-5,
    batch_size: int = 32,
    max_length: int = 128,
    output_dir: str = "checkpoints/",
    tag: str = "combined",
):
    """
    Full training pipeline for a neural router.

    Returns:
        dict with final test metrics + training history
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    # Tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ds = QueryDataset(train_examples, tokenizer, max_length)
    val_ds   = QueryDataset(val_examples, tokenizer, max_length)
    test_ds  = QueryDataset(test_examples, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size)
    test_loader  = DataLoader(test_ds, batch_size=batch_size)

    # Model
    model = QueryRouter(model_name).to(device)
    total_params, trainable_params = model.count_params()
    print(f"  Model: {model_name}")
    print(f"  Parameters: {total_params:,} total / {trainable_params:,} trainable")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Training loop
    history = []
    best_val_f1 = 0
    best_epoch = 0

    print(f"\n  Training for {epochs} epochs ({total_steps} steps, {warmup_steps} warmup)...")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val F1':>7} | {'Time':>6}")
    print(f"  {'-'*50}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_results = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        print(f"  {epoch:>5} | {train_loss:>10.4f} | {val_results['loss']:>8.4f} | "
              f"{val_results['macro_f1']:>7.4f} | {elapsed:>5.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_results["loss"],
            "val_macro_f1": val_results["macro_f1"],
        })

        # Save best model
        if val_results["macro_f1"] > best_val_f1:
            best_val_f1 = val_results["macro_f1"]
            best_epoch = epoch
            save_path = os.path.join(output_dir, f"{tag}_best.pt")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"\n  Best val F1: {best_val_f1:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(
        os.path.join(output_dir, f"{tag}_best.pt"),
        map_location=device,
        weights_only=True,
    ))
    test_results = evaluate(model, test_loader, device)

    print(f"\n  Test macro-F1: {test_results['macro_f1']:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        test_results["labels"], test_results["predictions"],
        target_names=["single_hop", "multi_hop", "summary"],
    ))

    # Per-domain evaluation on test set
    test_domains = [ex["domain"] for ex in test_examples]
    domain_results = {}
    domain_map = {
        "graphragBench_medical": "Medical",
        "musique": "MuSiQue",
        "quality": "QuALITY",
        "ultraDomain_legal": "Legal",
    }
    for domain, short_name in domain_map.items():
        mask = [d == domain for d in test_domains]
        yt = [l for l, m in zip(test_results["labels"], mask) if m]
        yp = [p for p, m in zip(test_results["predictions"], mask) if m]
        if yt:
            domain_results[short_name] = float(f1_score(yt, yp, average="macro"))
            print(f"    {short_name:<10}: {domain_results[short_name]:.4f}")

    return {
        "model": model_name,
        "tag": tag,
        "test_macro_f1": test_results["macro_f1"],
        "domain_results": domain_results,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "confusion_matrix": confusion_matrix(
            test_results["labels"], test_results["predictions"]
        ).tolist(),
    }


# ---------- CLI ----------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--per_domain", action="store_true",
                        help="Train separate model per domain instead of combined")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    args = parser.parse_args()

    # Clean model name for filenames
    model_short = args.model.split("/")[-1]

    if not args.per_domain:
        # --- Combined training ---
        print(f"\n{'='*60}")
        print(f"  Neural Router: {args.model} (combined)")
        print(f"{'='*60}")

        train = load_jsonl(os.path.join(args.data_dir, "all_train.jsonl"))
        val   = load_jsonl(os.path.join(args.data_dir, "all_val.jsonl"))
        test  = load_jsonl(os.path.join(args.data_dir, "all_test.jsonl"))

        results = train_model(
            model_name=args.model,
            train_examples=train,
            val_examples=val,
            test_examples=test,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            max_length=args.max_length,
            output_dir=args.checkpoint_dir,
            tag=f"{model_short}_combined",
        )

        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"neural_{model_short}_combined.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved → {out_path}")

    else:
        # --- Per-domain training ---
        domains = {
            "medical":  ("medical", "graphragBench_medical"),
            "musique":  ("musique", "musique"),
            "quality":  ("quality", "quality"),
            "legal":    ("legal", "ultraDomain_legal"),
        }

        all_domain_results = {}
        for short_name, (file_prefix, full_name) in domains.items():
            print(f"\n{'='*60}")
            print(f"  Neural Router: {args.model} (domain={short_name})")
            print(f"{'='*60}")

            train = load_jsonl(os.path.join(args.data_dir, f"{file_prefix}_train.jsonl"))
            val   = load_jsonl(os.path.join(args.data_dir, f"{file_prefix}_val.jsonl"))
            test  = load_jsonl(os.path.join(args.data_dir, f"{file_prefix}_test.jsonl"))

            results = train_model(
                model_name=args.model,
                train_examples=train,
                val_examples=val,
                test_examples=test,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                max_length=args.max_length,
                output_dir=args.checkpoint_dir,
                tag=f"{model_short}_{short_name}",
            )
            all_domain_results[short_name] = results

        # Save
        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir, f"neural_{model_short}_per_domain.json")
        with open(out_path, "w") as f:
            json.dump(all_domain_results, f, indent=2, default=str)
        print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
