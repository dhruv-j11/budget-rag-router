"""
cost_aware.py — Cost-aware neural routing with Pareto frontier optimization.

Core idea:
  Loss = L_classification + λ * L_cost

  L_classification = standard cross-entropy
  L_cost = expected token cost = Σ softmax(logit_i) * paradigm_cost_i

  By sweeping λ from 0 to 1.0, we trace a Pareto frontier of cost vs quality.

This is the main novel contribution of the paper.

Usage:
    # Single lambda value
    python src/cost_aware.py --lambda_cost 0.1

    # Full Pareto sweep (7 lambda values)
    python src/cost_aware.py --sweep

    # Quick test with fewer epochs
    python src/cost_aware.py --sweep --epochs 3
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------- Constants ----------
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}
NUM_LABELS = 3

# Token costs from RAGRouter-Bench (Table 5), relative to LLM-Only = 1.0
# single_hop → NaiveRAG (1.4x), multi_hop → IterativeRAG (3.5x), summary → HybridRAG (2.8x)
PARADIGM_COSTS = torch.tensor([1.4, 3.5, 2.8], dtype=torch.float32)

# Cost of always routing to the most expensive paradigm (IterativeRAG)
ALWAYS_EXPENSIVE_COST = 3.5


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

class CostAwareRouter(nn.Module):
    """
    DistilBERT + classification head with cost-aware loss.

    The model architecture is identical to the Phase 3 neural router.
    The only difference is the loss function.
    """

    def __init__(self, model_name="distilbert-base-uncased", num_labels=NUM_LABELS,
                 dropout=0.1, lambda_cost=0.0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.lambda_cost = lambda_cost

        # Register cost vector as buffer (moves to device with model)
        self.register_buffer("costs", PARADIGM_COSTS.clone())

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        loss = None
        cost_loss = None
        ce_loss = None

        if labels is not None:
            # Standard cross-entropy
            ce_loss = nn.CrossEntropyLoss()(logits, labels)

            # Expected cost: dot product of softmax probs with cost vector
            probs = torch.softmax(logits, dim=-1)  # (batch, 3)
            expected_cost = torch.sum(probs * self.costs, dim=-1)  # (batch,)
            cost_loss = expected_cost.mean()

            # Combined loss
            loss = ce_loss + self.lambda_cost * cost_loss

        return {
            "loss": loss,
            "logits": logits,
            "ce_loss": ce_loss,
            "cost_loss": cost_loss,
        }


# ---------- Training ----------

def train_one_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss, total_ce, total_cost = 0, 0, 0
    n = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)

        outputs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += outputs["loss"].item()
        total_ce += outputs["ce_loss"].item()
        total_cost += outputs["cost_loss"].item()
        n += 1

    return {
        "loss": total_loss / n,
        "ce_loss": total_ce / n,
        "cost_loss": total_cost / n,
    }


def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss, total_ce, total_cost = 0, 0, 0
    n = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            probs = torch.softmax(outputs["logits"], dim=-1)
            preds = torch.argmax(outputs["logits"], dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if outputs["loss"] is not None:
                total_loss += outputs["loss"].item()
                total_ce += outputs["ce_loss"].item()
                total_cost += outputs["cost_loss"].item()
                n += 1

    all_probs = np.array(all_probs)
    costs_np = PARADIGM_COSTS.numpy()

    # Compute actual routing cost: sum of predicted paradigm costs
    predicted_costs = [costs_np[p] for p in all_preds]
    total_predicted_cost = sum(predicted_costs)

    # Cost of always using IterativeRAG (most expensive)
    always_expensive = ALWAYS_EXPENSIVE_COST * len(all_preds)

    # Token savings
    savings_pct = (always_expensive - total_predicted_cost) / always_expensive * 100

    # Oracle savings (perfect routing)
    oracle_costs = [costs_np[l] for l in all_labels]
    oracle_total = sum(oracle_costs)
    oracle_savings = (always_expensive - oracle_total) / always_expensive * 100

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return {
        "loss": total_loss / n if n > 0 else 0,
        "ce_loss": total_ce / n if n > 0 else 0,
        "cost_loss": total_cost / n if n > 0 else 0,
        "macro_f1": macro_f1,
        "predictions": all_preds,
        "labels": all_labels,
        "avg_predicted_cost": total_predicted_cost / len(all_preds),
        "token_savings_pct": savings_pct,
        "oracle_savings_pct": oracle_savings,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


# ---------- Single Training Run ----------

def train_single(
    lambda_cost, model_name="distilbert-base-uncased",
    epochs=5, lr=2e-5, batch_size=32, max_length=128,
    data_dir="data/", checkpoint_dir="checkpoints/",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_ex = load_jsonl(os.path.join(data_dir, "all_train.jsonl"))
    val_ex = load_jsonl(os.path.join(data_dir, "all_val.jsonl"))
    test_ex = load_jsonl(os.path.join(data_dir, "all_test.jsonl"))

    train_loader = DataLoader(QueryDataset(train_ex, tokenizer, max_length),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(QueryDataset(val_ex, tokenizer, max_length),
                            batch_size=batch_size)
    test_loader = DataLoader(QueryDataset(test_ex, tokenizer, max_length),
                             batch_size=batch_size)

    # Model
    model = CostAwareRouter(model_name=model_name, lambda_cost=lambda_cost).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # Train
    best_val_f1 = 0
    history = []

    print(f"\n  {'Epoch':>5} | {'Loss':>8} | {'CE':>8} | {'Cost':>8} | {'Val F1':>7} | {'Savings':>8}")
    print(f"  {'-'*55}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_results = evaluate(model, val_loader, device)

        print(f"  {epoch:>5} | {train_metrics['loss']:>8.4f} | {train_metrics['ce_loss']:>8.4f} | "
              f"{train_metrics['cost_loss']:>8.4f} | {val_results['macro_f1']:>7.4f} | "
              f"{val_results['token_savings_pct']:>7.1f}%")

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_ce": train_metrics["ce_loss"],
            "train_cost": train_metrics["cost_loss"],
            "val_f1": val_results["macro_f1"],
            "val_savings": val_results["token_savings_pct"],
        })

        if val_results["macro_f1"] > best_val_f1:
            best_val_f1 = val_results["macro_f1"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f"cost_aware_lambda_{lambda_cost}.pt"))

    # Load best and evaluate on test
    model.load_state_dict(torch.load(
        os.path.join(checkpoint_dir, f"cost_aware_lambda_{lambda_cost}.pt"),
        map_location=device, weights_only=True,
    ))
    test_results = evaluate(model, test_loader, device)

    # Per-domain
    test_domains = [ex["domain"] for ex in test_ex]
    domain_map = {
        "graphragBench_medical": "Medical", "musique": "MuSiQue",
        "quality": "QuALITY", "ultraDomain_legal": "Legal",
    }
    domain_results = {}
    for domain, short in domain_map.items():
        mask = [d == domain for d in test_domains]
        yt = [l for l, m in zip(test_results["labels"], mask) if m]
        yp = [p for p, m in zip(test_results["predictions"], mask) if m]
        if yt:
            domain_results[short] = float(f1_score(yt, yp, average="macro"))

    return {
        "lambda_cost": lambda_cost,
        "test_macro_f1": test_results["macro_f1"],
        "test_savings_pct": test_results["token_savings_pct"],
        "oracle_savings_pct": test_results["oracle_savings_pct"],
        "avg_predicted_cost": test_results["avg_predicted_cost"],
        "domain_results": domain_results,
        "confusion_matrix": test_results["confusion_matrix"],
        "history": history,
    }


# ---------- Pareto Sweep ----------

def run_pareto_sweep(lambdas, **kwargs):
    """Train one model per lambda value and collect Pareto frontier points."""
    all_results = []

    for lam in lambdas:
        print(f"\n{'='*60}")
        print(f"  Cost-Aware Training: λ = {lam}")
        print(f"{'='*60}")

        result = train_single(lambda_cost=lam, **kwargs)

        print(f"\n  Test F1: {result['test_macro_f1']:.4f} | "
              f"Savings: {result['test_savings_pct']:.1f}% | "
              f"Avg cost: {result['avg_predicted_cost']:.2f}")
        for d, f1 in result["domain_results"].items():
            print(f"    {d:<10}: {f1:.4f}")

        all_results.append(result)

    return all_results


def print_pareto_table(results):
    """Print the Pareto frontier as a table."""
    print(f"\n{'='*80}")
    print(f"  PARETO FRONTIER: Cost vs Quality Tradeoff")
    print(f"{'='*80}")
    print(f"  {'λ':>6} | {'F1':>7} | {'Savings':>8} | {'Avg Cost':>8} | "
          f"{'MuSiQue':>8} | {'QuALITY':>8} | {'Legal':>8} | {'Medical':>8}")
    print(f"  {'-'*85}")

    for r in results:
        dr = r["domain_results"]
        print(f"  {r['lambda_cost']:>6.3f} | {r['test_macro_f1']:>7.4f} | "
              f"{r['test_savings_pct']:>7.1f}% | {r['avg_predicted_cost']:>8.2f} | "
              f"{dr.get('MuSiQue', 0):>8.4f} | {dr.get('QuALITY', 0):>8.4f} | "
              f"{dr.get('Legal', 0):>8.4f} | {dr.get('Medical', 0):>8.4f}")

    print(f"\n  Oracle savings (perfect routing): {results[0]['oracle_savings_pct']:.1f}%")
    print(f"  Always-expensive cost: {ALWAYS_EXPENSIVE_COST}")


def save_results(results, output_dir="results/"):
    """Save all sweep results."""
    os.makedirs(output_dir, exist_ok=True)

    # Full results
    out_path = os.path.join(output_dir, "cost_aware_sweep.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Full results saved → {out_path}")

    # Pareto points for plotting
    pareto_points = [{
        "lambda": r["lambda_cost"],
        "f1": r["test_macro_f1"],
        "savings": r["test_savings_pct"],
        "avg_cost": r["avg_predicted_cost"],
    } for r in results]
    pareto_path = os.path.join(output_dir, "pareto_points.json")
    with open(pareto_path, "w") as f:
        json.dump(pareto_points, f, indent=2, default=float)
    print(f"  Pareto points saved → {pareto_path}")


# ---------- Pareto Frontier Plot ----------

def plot_pareto(results, output_dir="figures/"):
    """Generate the Pareto frontier figure — the centerpiece of the paper."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "axes.labelsize": 12, "axes.titlesize": 13,
    })

    os.makedirs(output_dir, exist_ok=True)

    lambdas = [r["lambda_cost"] for r in results]
    f1s = [r["test_macro_f1"] for r in results]
    savings = [r["test_savings_pct"] for r in results]
    avg_costs = [r["avg_predicted_cost"] for r in results]

    # --- Figure 1: F1 vs Token Savings ---
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(savings, f1s, "o-", color="#534AB7", linewidth=2, markersize=8, zorder=3)

    # Label each point with its lambda
    for i, lam in enumerate(lambdas):
        offset = (5, 5) if i % 2 == 0 else (5, -12)
        ax.annotate(f"λ={lam}", (savings[i], f1s[i]),
                    textcoords="offset points", xytext=offset,
                    fontsize=9, color="#534AB7")

    # Reference lines
    oracle_sav = results[0]["oracle_savings_pct"]
    ax.axvline(x=oracle_sav, color="#888", linestyle="--", alpha=0.5, label="Oracle savings")
    ax.axhline(y=f1s[0], color="#1D9E75", linestyle="--", alpha=0.5, label=f"λ=0 baseline (F1={f1s[0]:.3f})")

    ax.set_xlabel("Token savings vs always-IterativeRAG (%)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Pareto frontier: accuracy vs cost savings")
    ax.legend(loc="lower left", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_frontier.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"  Pareto figure saved → {path}")
    plt.close()

    # --- Figure 2: F1 vs Average Cost (direct) ---
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(avg_costs, f1s, "s-", color="#D85A30", linewidth=2, markersize=8, zorder=3)

    for i, lam in enumerate(lambdas):
        offset = (5, 5) if i % 2 == 0 else (5, -12)
        ax.annotate(f"λ={lam}", (avg_costs[i], f1s[i]),
                    textcoords="offset points", xytext=offset,
                    fontsize=9, color="#D85A30")

    ax.set_xlabel("Average predicted paradigm cost (× LLM-Only)")
    ax.set_ylabel("Test macro-F1")
    ax.set_title("Cost-quality tradeoff: direct cost view")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "cost_quality_tradeoff.pdf")
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
    print(f"  Cost-quality figure saved → {path}")
    plt.close()


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambda_cost", type=float, default=0.1)
    parser.add_argument("--sweep", action="store_true",
                        help="Run full Pareto sweep over lambda values")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--checkpoint_dir", default="checkpoints/")
    args = parser.parse_args()

    common_kwargs = dict(
        model_name=args.model, epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    if args.sweep:
        lambdas = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

        print(f"\n{'#'*60}")
        print(f"  PARETO SWEEP: {len(lambdas)} lambda values")
        print(f"  λ ∈ {lambdas}")
        print(f"  Epochs: {args.epochs} per run")
        print(f"  Estimated time: ~{len(lambdas) * 6} min on T4 GPU")
        print(f"{'#'*60}")

        results = run_pareto_sweep(lambdas, **common_kwargs)
        print_pareto_table(results)
        save_results(results, args.output_dir)
        plot_pareto(results, output_dir="figures/")

    else:
        print(f"\n{'='*60}")
        print(f"  Cost-Aware Training: λ = {args.lambda_cost}")
        print(f"{'='*60}")

        result = train_single(lambda_cost=args.lambda_cost, **common_kwargs)

        print(f"\n  Test F1: {result['test_macro_f1']:.4f}")
        print(f"  Token savings: {result['test_savings_pct']:.1f}%")
        print(f"  Avg cost: {result['avg_predicted_cost']:.2f}")
        for d, f1 in result["domain_results"].items():
            print(f"    {d:<10}: {f1:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        out_path = os.path.join(args.output_dir,
                                f"cost_aware_lambda_{args.lambda_cost}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=float)
        print(f"\n  Results saved → {out_path}")


if __name__ == "__main__":
    main()
