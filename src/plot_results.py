"""
plot_results.py — Generate paper-quality figures from results.

Generates:
  1. Bar chart: macro-F1 across domains (best classical vs neural)
  2. Confusion matrix heatmaps
  3. Training curves
  4. Model size vs accuracy scatter

Usage:
    python src/plot_results.py
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

DOMAINS = ["MuSiQue", "QuALITY", "Legal", "Medical"]
LABEL_NAMES = ["single_hop", "multi_hop", "summary"]
FIG_DIR = "figures/"


def load_results(results_dir="results/"):
    results = {}
    for f in Path(results_dir).glob("*.json"):
        with open(f) as fh:
            results[f.stem] = json.load(fh)
    return results


def plot_domain_comparison(results):
    """Bar chart comparing best classical vs neural across domains."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Find best classical per domain
    best_classical = {d: 0 for d in DOMAINS}
    best_classical["Overall"] = 0
    best_classical_name = ""

    for key in results:
        if not key.startswith("baselines_"):
            continue
        for clf_name, metrics in results[key].items():
            overall = metrics.get("overall", 0)
            if overall > best_classical["Overall"]:
                best_classical["Overall"] = overall
                feature = key.replace("baselines_", "").upper()
                best_classical_name = f"{clf_name} ({feature})"
                for d in DOMAINS:
                    best_classical[d] = metrics.get(d, 0)

    # Find neural results
    neural_results = {d: 0 for d in DOMAINS}
    neural_results["Overall"] = 0
    neural_name = ""
    for key in results:
        if key.startswith("neural_") and "combined" in key:
            data = results[key]
            neural_results["Overall"] = data.get("test_macro_f1", 0)
            neural_name = data["model"].split("/")[-1]
            for d in DOMAINS:
                neural_results[d] = data.get("domain_results", {}).get(d, 0)

    if not neural_name:
        print("  No neural results found, skipping domain comparison plot.")
        return

    # Plot
    x = np.arange(len(DOMAINS) + 1)
    width = 0.35
    labels = DOMAINS + ["Overall"]

    classical_vals = [best_classical[d] for d in labels]
    neural_vals = [neural_results[d] for d in labels]

    bars1 = ax.bar(x - width/2, classical_vals, width, label=f"Best Classical ({best_classical_name})",
                   color="#5B8DEF", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, neural_vals, width, label=f"Neural ({neural_name})",
                   color="#FF6B6B", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.set_title("Query Routing: Classical vs Neural Classifiers")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "domain_comparison.pdf")
    plt.savefig(path, bbox_inches="tight")
    plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close()


def plot_confusion_matrix(results):
    """Plot confusion matrix heatmap from neural results."""
    for key in results:
        if not (key.startswith("neural_") and "combined" in key):
            continue
        data = results[key]
        cm = np.array(data.get("confusion_matrix", []))
        if cm.size == 0:
            continue

        fig, ax = plt.subplots(figsize=(5, 4))

        # Normalize by row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        # Add text annotations
        for i in range(3):
            for j in range(3):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                        ha="center", va="center", color=color, fontsize=10)

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        short_names = ["Factual", "Reasoning", "Summary"]
        ax.set_xticklabels(short_names)
        ax.set_yticklabels(short_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        model_short = data["model"].split("/")[-1]
        ax.set_title(f"Confusion Matrix: {model_short}")

        plt.colorbar(im, ax=ax, label="Proportion")
        plt.tight_layout()

        path = os.path.join(FIG_DIR, f"confusion_{model_short}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close()


def plot_training_curves(results):
    """Plot loss and F1 training curves."""
    for key in results:
        if not (key.startswith("neural_") and "combined" in key):
            continue
        data = results[key]
        history = data.get("history", [])
        if not history:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        epochs = [h["epoch"] for h in history]

        # Loss
        ax1.plot(epochs, [h["train_loss"] for h in history], "o-", color="#5B8DEF", label="Train")
        ax1.plot(epochs, [h["val_loss"] for h in history], "o-", color="#FF6B6B", label="Val")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # F1
        ax2.plot(epochs, [h["val_macro_f1"] for h in history], "o-", color="#2ECC71", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Macro-F1")
        ax2.set_title("Validation Macro-F1")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        model_short = data["model"].split("/")[-1]
        fig.suptitle(f"Training Curves: {model_short}", y=1.02)
        plt.tight_layout()

        path = os.path.join(FIG_DIR, f"training_curves_{model_short}.pdf")
        plt.savefig(path, bbox_inches="tight")
        plt.savefig(path.replace(".pdf", ".png"), bbox_inches="tight")
        print(f"  Saved → {path}")
        plt.close()


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_results()

    if not results:
        print("No results found in results/. Run baselines and neural training first.")
        return

    print("\nGenerating figures...")
    plot_domain_comparison(results)
    plot_confusion_matrix(results)
    plot_training_curves(results)
    print("\n✓ All figures generated.")


if __name__ == "__main__":
    main()
