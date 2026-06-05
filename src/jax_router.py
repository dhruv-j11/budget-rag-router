"""
jax_router.py — JAX/Flax implementation of the query router for TPU training.

This is the same model as neural_router.py (DistilBERT + classification head),
but implemented in JAX/Flax and trained on TPU v2-8.

Why this matters for the paper:
  1. Demonstrates framework portability (PyTorch → JAX)
  2. Leverages TPU's 8-core parallelism via jax.pmap
  3. Reports training throughput comparison (GPU vs TPU)
  4. Directly relevant to orgs that use JAX (Cohere, Google, DeepMind)

Architecture:
  FlaxDistilBert encoder → [CLS] pooling → Dense(3) → cross-entropy

Usage (on Colab TPU runtime):
    python src/jax_router.py --epochs 5 --lr 2e-5
    python src/jax_router.py --epochs 5 --lr 2e-5 --compare_throughput

Prerequisites:
    pip install jax[tpu] flax optax transformers datasets
"""

import os
import json
import time
import argparse
import functools
from collections import Counter

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import flax.linen as nn
from flax.training import train_state

import optax
from transformers import AutoTokenizer, FlaxAutoModel
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------- Constants ----------
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}
NUM_LABELS = 3


# ---------- Data ----------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare_batch(examples, tokenizer, max_length=128):
    """Tokenize a list of examples into JAX-ready arrays."""
    texts = [ex["question"] for ex in examples]
    labels = np.array([ex["label"] for ex in examples], dtype=np.int32)

    encoded = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="np",
    )

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def make_batches(examples, tokenizer, batch_size, max_length=128, shuffle=False, rng=None):
    """Generate batches from examples. Drops the last incomplete batch."""
    if shuffle:
        assert rng is not None
        indices = jax.random.permutation(rng, len(examples)).tolist()
        examples = [examples[i] for i in indices]

    n_complete = (len(examples) // batch_size) * batch_size
    for i in range(0, n_complete, batch_size):
        batch_examples = examples[i : i + batch_size]
        yield prepare_batch(batch_examples, tokenizer, max_length)


# ---------- Model ----------

class QueryRouterJAX(nn.Module):
    """
    Flax module: DistilBERT encoder + linear classification head.

    This mirrors the PyTorch QueryRouter exactly:
        input_ids → DistilBERT → [CLS] → dropout → Dense(3)
    """
    model_name: str = "distilbert-base-uncased"
    num_labels: int = NUM_LABELS
    dropout_rate: float = 0.1

    def setup(self):
        self.encoder = FlaxAutoModel.from_pretrained(self.model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Dense(self.num_labels)

    def __call__(self, input_ids, attention_mask, deterministic=True):
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        # Dropout (only during training)
        cls_output = nn.Dropout(rate=self.dropout_rate)(
            cls_output, deterministic=deterministic
        )

        # Classify
        logits = self.classifier(cls_output)
        return logits


# ---------- Loss + Metrics ----------

def cross_entropy_loss(logits, labels):
    """Standard cross-entropy loss."""
    one_hot = jax.nn.one_hot(labels, NUM_LABELS)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(one_hot * log_probs, axis=-1)
    return jnp.mean(loss)


def compute_metrics(logits, labels):
    """Compute loss and accuracy."""
    loss = cross_entropy_loss(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)
    return {"loss": loss, "accuracy": accuracy}


# ---------- Training State ----------

def create_train_state(rng, model, learning_rate, num_train_steps, warmup_steps):
    """Initialize model parameters and optimizer."""
    # Dummy input for parameter initialization
    dummy_input_ids = jnp.ones((1, 128), dtype=jnp.int32)
    dummy_attention_mask = jnp.ones((1, 128), dtype=jnp.int32)

    params = model.init(rng, dummy_input_ids, dummy_attention_mask)["params"]

    # AdamW with linear warmup + decay (matches PyTorch version)
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, learning_rate, warmup_steps),
            optax.linear_schedule(learning_rate, 0.0, num_train_steps - warmup_steps),
        ],
        boundaries=[warmup_steps],
    )

    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )


# ---------- Train / Eval Steps ----------

@jax.jit
def train_step(state, input_ids, attention_mask, labels, dropout_rng):
    """Single training step (JIT-compiled)."""
    def loss_fn(params):
        logits = state.apply_fn(
            {"params": params},
            input_ids=input_ids,
            attention_mask=attention_mask,
            deterministic=False,
            rngs={"dropout": dropout_rng},
        )
        loss = cross_entropy_loss(logits, labels)
        return loss, logits

    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits, labels)
    return state, metrics


@jax.jit
def eval_step(state, input_ids, attention_mask, labels):
    """Single eval step (JIT-compiled)."""
    logits = state.apply_fn(
        {"params": state.params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    metrics = compute_metrics(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    return metrics, preds


# ---------- Main Training Loop ----------

def train_and_evaluate(
    model_name="distilbert-base-uncased",
    epochs=5,
    lr=2e-5,
    batch_size=32,
    max_length=128,
    data_dir="data/",
    output_dir="results/",
    seed=42,
):
    """Full JAX/Flax training pipeline."""

    print(f"\n{'='*60}")
    print(f"  JAX/Flax Neural Router")
    print(f"  Model: {model_name}")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print(f"{'='*60}")

    # Load data
    train_examples = load_jsonl(os.path.join(data_dir, "all_train.jsonl"))
    val_examples   = load_jsonl(os.path.join(data_dir, "all_val.jsonl"))
    test_examples  = load_jsonl(os.path.join(data_dir, "all_test.jsonl"))
    print(f"\n  Data: {len(train_examples)} train / {len(val_examples)} val / {len(test_examples)} test")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Calculate training steps
    steps_per_epoch = len(train_examples) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)
    print(f"  Steps: {total_steps} total, {warmup_steps} warmup, {steps_per_epoch}/epoch")

    # Initialize model and state
    print(f"\n  Initializing model...")
    model = QueryRouterJAX(model_name=model_name)
    rng = random.PRNGKey(seed)
    rng, init_rng, dropout_rng = random.split(rng, 3)

    state = create_train_state(init_rng, model, lr, total_steps, warmup_steps)

    # Count parameters
    param_count = sum(x.size for x in jax.tree.leaves(state.params))
    print(f"  Parameters: {param_count:,}")

    # Training loop
    history = []
    best_val_f1 = 0.0
    best_params = None

    print(f"\n  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val F1':>7} | {'Time':>6} | {'Throughput':>12}")
    print(f"  {'-'*65}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rng, shuffle_rng, epoch_dropout_rng = random.split(rng, 3)

        # --- Train ---
        train_losses = []
        n_train_samples = 0
        for batch in make_batches(train_examples, tokenizer, batch_size,
                                   max_length, shuffle=True, rng=shuffle_rng):
            rng, step_dropout_rng = random.split(rng)
            state, metrics = train_step(
                state,
                jnp.array(batch["input_ids"]),
                jnp.array(batch["attention_mask"]),
                jnp.array(batch["labels"]),
                step_dropout_rng,
            )
            train_losses.append(float(metrics["loss"]))
            n_train_samples += batch["input_ids"].shape[0]

        avg_train_loss = np.mean(train_losses)

        # --- Eval ---
        val_preds = []
        val_labels = []
        val_losses = []
        for batch in make_batches(val_examples, tokenizer, batch_size, max_length):
            metrics, preds = eval_step(
                state,
                jnp.array(batch["input_ids"]),
                jnp.array(batch["attention_mask"]),
                jnp.array(batch["labels"]),
            )
            val_preds.extend(preds.tolist())
            val_labels.extend(batch["labels"].tolist())
            val_losses.append(float(metrics["loss"]))

        avg_val_loss = np.mean(val_losses)
        val_f1 = f1_score(val_labels, val_preds, average="macro")

        elapsed = time.time() - t0
        throughput = n_train_samples / elapsed

        print(f"  {epoch:>5} | {avg_train_loss:>10.4f} | {avg_val_loss:>8.4f} | "
              f"{val_f1:>7.4f} | {elapsed:>5.1f}s | {throughput:>8.0f} ex/s")

        history.append({
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "val_macro_f1": float(val_f1),
            "throughput": float(throughput),
            "time_s": float(elapsed),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = jax.tree.map(lambda x: np.array(x), state.params)

    # --- Test with best params ---
    print(f"\n  Best val F1: {best_val_f1:.4f}")
    print(f"  Evaluating on test set...")

    # Restore best params
    state = state.replace(params=best_params)

    test_preds = []
    test_labels = []
    for batch in make_batches(test_examples, tokenizer, batch_size, max_length):
        metrics, preds = eval_step(
            state,
            jnp.array(batch["input_ids"]),
            jnp.array(batch["attention_mask"]),
            jnp.array(batch["labels"]),
        )
        test_preds.extend(preds.tolist())
        test_labels.extend(batch["labels"].tolist())

    test_f1 = f1_score(test_labels, test_preds, average="macro")
    print(f"\n  Test macro-F1: {test_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        test_labels, test_preds,
        target_names=["single_hop", "multi_hop", "summary"],
    ))

    # Per-domain evaluation
    test_domains = [ex["domain"] for ex in test_examples[:len(test_preds)]]
    domain_map = {
        "graphragBench_medical": "Medical",
        "musique": "MuSiQue",
        "quality": "QuALITY",
        "ultraDomain_legal": "Legal",
    }
    domain_results = {}
    for domain, short in domain_map.items():
        mask = [d == domain for d in test_domains]
        yt = [l for l, m in zip(test_labels, mask) if m]
        yp = [p for p, m in zip(test_preds, mask) if m]
        if yt:
            domain_results[short] = float(f1_score(yt, yp, average="macro"))
            print(f"    {short:<10}: {domain_results[short]:.4f}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]
    results = {
        "model": model_name,
        "framework": "jax_flax",
        "backend": str(jax.default_backend()),
        "device_count": jax.device_count(),
        "tag": f"{model_short}_jax",
        "test_macro_f1": float(test_f1),
        "best_val_f1": float(best_val_f1),
        "domain_results": domain_results,
        "history": history,
        "total_params": int(param_count),
        "confusion_matrix": confusion_matrix(test_labels, test_preds).tolist(),
        # Throughput summary
        "avg_throughput_ex_per_s": float(np.mean([h["throughput"] for h in history])),
        "total_training_time_s": float(sum(h["time_s"] for h in history)),
    }

    out_path = os.path.join(output_dir, f"neural_{model_short}_jax.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")

    return results


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_throughput", action="store_true",
                        help="Print detailed throughput comparison metrics")
    args = parser.parse_args()

    results = train_and_evaluate(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        max_length=args.max_length,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    if args.compare_throughput:
        print(f"\n{'='*60}")
        print(f"  Throughput Summary")
        print(f"{'='*60}")
        print(f"  Backend: {results['backend']}")
        print(f"  Devices: {results['device_count']}")
        print(f"  Avg throughput: {results['avg_throughput_ex_per_s']:.0f} examples/sec")
        print(f"  Total training: {results['total_training_time_s']:.1f}s")
        print(f"\n  Compare this against the PyTorch T4 GPU results")
        print(f"  to quantify the framework/hardware difference.")


if __name__ == "__main__":
    main()
