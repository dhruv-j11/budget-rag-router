"""
calibration.py — Confidence calibration analysis with temperature scaling.

Measures:
  1. Expected Calibration Error (ECE) before and after temperature scaling
  2. Reliability diagram (predicted confidence vs actual accuracy)
  3. Per-class calibration breakdown

Temperature scaling: learn a single scalar T on the validation set that
divides logits before softmax. T>1 = less confident, T<1 = more confident.

Usage:
    python src/calibration.py
"""

import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score

# ---------- Constants ----------
NUM_LABELS = 3
LABEL_NAMES = {0: "single_hop", 1: "multi_hop", 2: "summary"}


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


# ---------- Calibration Metrics ----------

def compute_ece(confidences, predictions, labels, n_bins=10):
    """
    Expected Calibration Error.

    Bins predictions by confidence, computes |accuracy - confidence| per bin,
    weighted by bin size.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if i == n_bins - 1:  # include 1.0 in last bin
            mask = (confidences >= lo) & (confidences <= hi)

        bin_size = mask.sum()
        if bin_size == 0:
            bin_data.append({"lo": lo, "hi": hi, "count": 0,
                             "accuracy": 0, "confidence": 0, "gap": 0})
            continue

        bin_acc = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += gap * (bin_size / len(confidences))

        bin_data.append({
            "lo": float(lo), "hi": float(hi), "count": int(bin_size),
            "accuracy": float(bin_acc), "confidence": float(bin_conf),
            "gap": float(gap),
        })

    return float(ece), bin_data


def learn_temperature(logits, labels, lr=0.01, max_iter=200):
    """
    Learn optimal temperature T on validation set.

    T is a single scalar. We minimize NLL loss with logits/T.
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    def closure():
        optimizer.zero_grad()
        # Clamp temperature to avoid division by zero or negative
        t = temperature.clamp(min=0.1)
        loss = nll(logits_tensor / t, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.clamp(min=0.1).item())


# ---------- Collect Logits ----------

def collect_logits(model, dataloader, device):
    """Run model on data and collect raw logits + labels."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            all_logits.append(out["logits"].cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())

    return np.concatenate(all_logits, axis=0), np.array(all_labels)


# ---------- Plot ----------

def plot_reliability(bin_data_before, bin_data_after, ece_before, ece_after,
                     temperature, output_dir="figures/"):
    """Plot reliability diagrams before and after temperature scaling."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
    })

    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, bins, ece, title in [
        (ax1, bin_data_before, ece_before, f"Before (T=1.0, ECE={ece_before:.4f})"),
        (ax2, bin_data_after, ece_after, f"After (T={temperature:.2f}, ECE={ece_after:.4f})"),
    ]:
        confs = [b["confidence"] for b in bins if b["count"] > 0]
        accs = [b["accuracy"] for b in bins if b["count"] > 0]
        counts = [b["count"] for b in bins if b["count"] > 0]
        midpoints = [(b["lo"] + b["hi"]) / 2 for b in bins if b["count"] > 0]

        # Bar chart
        width = 0.08
        ax.bar(midpoints, accs, width=width, color="#5B8DEF", alpha=0.8,
               label="Accuracy", edgecolor="white")
        ax.bar(midpoints, [c - a if c > a else 0 for c, a in zip(confs, accs)],
               width=width, bottom=accs, color="#FF6B6B", alpha=0.5,
               label="Gap", edgecolor="white")

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect")

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "reliability_diagram.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"  Reliability diagram saved → {path}")
    plt.close()


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--checkpoint", default="checkpoints/distilbert-base-uncased_combined_best.pt")
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Calibration Analysis")
    print(f"{'='*60}")

    # Load data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    val_ex = load_jsonl(os.path.join(args.data_dir, "all_val.jsonl"))
    test_ex = load_jsonl(os.path.join(args.data_dir, "all_test.jsonl"))
    val_loader = DataLoader(QueryDataset(val_ex, tokenizer), batch_size=args.batch_size)
    test_loader = DataLoader(QueryDataset(test_ex, tokenizer), batch_size=args.batch_size)

    # Load model
    model = QueryRouter(args.model).to(device)
    if os.path.exists(args.checkpoint):
        print(f"  Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
    else:
        print(f"  No checkpoint found at {args.checkpoint}")
        print(f"  Training a fresh model for calibration analysis...")
        # Quick train
        train_ex = load_jsonl(os.path.join(args.data_dir, "all_train.jsonl"))
        train_loader = DataLoader(QueryDataset(train_ex, tokenizer),
                                  batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        model.train()
        for epoch in range(5):
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                out = model(**batch)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            print(f"    Epoch {epoch+1}/5 done")

    # Collect logits
    print(f"\n  Collecting logits...")
    val_logits, val_labels = collect_logits(model, val_loader, device)
    test_logits, test_labels = collect_logits(model, test_loader, device)

    # --- Before temperature scaling ---
    test_probs = torch.softmax(torch.tensor(test_logits), dim=-1).numpy()
    test_confs = test_probs.max(axis=1)
    test_preds = test_probs.argmax(axis=1)

    ece_before, bins_before = compute_ece(test_confs, test_preds, test_labels)
    f1_before = f1_score(test_labels, test_preds, average="macro")

    print(f"\n  BEFORE temperature scaling:")
    print(f"    Macro-F1: {f1_before:.4f}")
    print(f"    ECE: {ece_before:.4f}")
    print(f"    Mean confidence: {test_confs.mean():.4f}")
    print(f"    Mean accuracy: {(test_preds == test_labels).mean():.4f}")

    # --- Learn temperature on validation set ---
    print(f"\n  Learning temperature on validation set...")
    temperature = learn_temperature(val_logits, val_labels)
    print(f"    Optimal T = {temperature:.4f}")

    # --- After temperature scaling ---
    scaled_logits = test_logits / temperature
    scaled_probs = torch.softmax(torch.tensor(scaled_logits), dim=-1).numpy()
    scaled_confs = scaled_probs.max(axis=1)
    scaled_preds = scaled_probs.argmax(axis=1)

    ece_after, bins_after = compute_ece(scaled_confs, scaled_preds, test_labels)
    f1_after = f1_score(test_labels, scaled_preds, average="macro")

    print(f"\n  AFTER temperature scaling (T={temperature:.4f}):")
    print(f"    Macro-F1: {f1_after:.4f} (unchanged — T doesn't change predictions)")
    print(f"    ECE: {ece_after:.4f} (was {ece_before:.4f})")
    print(f"    Mean confidence: {scaled_confs.mean():.4f} (was {test_confs.mean():.4f})")
    print(f"    ECE reduction: {(1 - ece_after/ece_before)*100:.1f}%")

    # Per-class calibration
    print(f"\n  Per-class confidence (after scaling):")
    for cls_id, cls_name in LABEL_NAMES.items():
        mask = test_preds == cls_id
        if mask.sum() > 0:
            cls_conf = scaled_confs[mask].mean()
            cls_acc = (test_labels[mask] == cls_id).mean()
            print(f"    {cls_name:<12}: conf={cls_conf:.4f}, acc={cls_acc:.4f}, gap={abs(cls_conf-cls_acc):.4f}")

    # Plot
    plot_reliability(bins_before, bins_after, ece_before, ece_after, temperature)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "temperature": temperature,
        "before": {
            "ece": ece_before, "f1": f1_before,
            "mean_confidence": float(test_confs.mean()),
            "bins": bins_before,
        },
        "after": {
            "ece": ece_after, "f1": f1_after,
            "mean_confidence": float(scaled_confs.mean()),
            "bins": bins_after,
        },
        "ece_reduction_pct": float((1 - ece_after/ece_before)*100) if ece_before > 0 else 0,
    }
    out_path = os.path.join(args.output_dir, "calibration.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
