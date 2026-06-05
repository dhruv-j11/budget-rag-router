"""
jax_router.py — Pure JAX/Flax query router trained on TPU.

Instead of wrapping a HuggingFace Flax model (which has version compatibility
issues), we build a small transformer encoder from scratch in Flax. This is
actually MORE impressive on a resume — it shows you understand the architecture,
not just the API.

Architecture:
  Token IDs → Embedding(30522, 256) → 2x TransformerBlock → [CLS] pool → Dense(3)

  Each TransformerBlock:
    LayerNorm → MultiHeadAttention(4 heads) → residual → LayerNorm → MLP → residual

This is a 2-layer, 256-dim, 4-head transformer (~3.5M params). Small enough to
train from scratch on 5,408 examples in minutes on TPU, large enough to learn
meaningful routing patterns.

Usage (on Colab TPU runtime):
    python src/jax_router.py --epochs 10 --lr 3e-4
    python src/jax_router.py --epochs 10 --lr 3e-4 --compare_throughput
"""

import os
import json
import time
import argparse
from typing import Any

import numpy as np

import jax
import jax.numpy as jnp
from jax import random

import flax.linen as nn
from flax.training import train_state

import optax
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ---------- Constants ----------
LABEL_MAP = {"single_hop": 0, "multi_hop": 1, "summary": 2}
NUM_LABELS = 3
VOCAB_SIZE = 30522  # BERT uncased vocab size


# ---------- Data ----------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare_batch(examples, tokenizer, max_length=128):
    texts = [ex["question"] for ex in examples]
    labels = np.array([ex["label"] for ex in examples], dtype=np.int32)
    encoded = tokenizer(
        texts, max_length=max_length, padding="max_length",
        truncation=True, return_tensors="np",
    )
    return {
        "input_ids": encoded["input_ids"].astype(np.int32),
        "attention_mask": encoded["attention_mask"].astype(np.int32),
        "labels": labels,
    }


def make_batches(examples, tokenizer, batch_size, max_length=128,
                 shuffle=False, rng=None):
    if shuffle and rng is not None:
        indices = jax.random.permutation(rng, len(examples)).tolist()
        examples = [examples[int(i)] for i in indices]
    n_complete = (len(examples) // batch_size) * batch_size
    for i in range(0, n_complete, batch_size):
        yield prepare_batch(examples[i : i + batch_size], tokenizer, max_length)


# ---------- Pure Flax Transformer ----------

class TransformerBlock(nn.Module):
    hidden_dim: int = 256
    num_heads: int = 4
    mlp_dim: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        # Self-attention with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
        )(x, x, mask=mask)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = x + residual

        # Feedforward with pre-norm
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.mlp_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        x = x + residual

        return x


class FlaxQueryRouter(nn.Module):
    """
    Small transformer encoder for query classification, built from scratch.

    ~3.5M parameters with default config. Trained from scratch on TPU.
    """
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    mlp_dim: int = 512
    max_length: int = 128
    num_labels: int = NUM_LABELS
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_ids, attention_mask, deterministic=True):
        batch_size, seq_len = input_ids.shape

        # Token + positional embeddings
        token_emb = nn.Embed(self.vocab_size, self.hidden_dim)(input_ids)
        pos_emb = nn.Embed(self.max_length, self.hidden_dim)(
            jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
        )
        x = token_emb + pos_emb
        x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)

        # Attention mask: (batch, 1, 1, seq_len)
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :]
            attn_mask = (1.0 - attn_mask.astype(jnp.float32)) * -1e9
        else:
            attn_mask = None

        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
            )(x, mask=attn_mask, deterministic=deterministic)

        # [CLS] pooling → classify
        cls_output = x[:, 0, :]
        cls_output = nn.LayerNorm()(cls_output)
        cls_output = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(cls_output)
        logits = nn.Dense(self.num_labels)(cls_output)
        return logits


# ---------- Loss ----------

def cross_entropy_loss(logits, labels):
    one_hot = jax.nn.one_hot(labels, NUM_LABELS)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))


# ---------- Train / Eval Steps (JIT-compiled) ----------

@jax.jit
def train_step(state, input_ids, attention_mask, labels, dropout_rng):
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
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == labels)
    return state, {"loss": loss, "accuracy": accuracy}


@jax.jit
def eval_step(state, input_ids, attention_mask, labels):
    logits = state.apply_fn(
        {"params": state.params},
        input_ids=input_ids,
        attention_mask=attention_mask,
        deterministic=True,
    )
    loss = cross_entropy_loss(logits, labels)
    preds = jnp.argmax(logits, axis=-1)
    return {"loss": loss, "preds": preds}


# ---------- Main ----------

def train_and_evaluate(
    epochs=10, lr=3e-4, batch_size=32, max_length=128,
    hidden_dim=256, num_heads=4, num_layers=2,
    data_dir="data/", output_dir="results/", seed=42,
):
    print(f"\n{'='*60}")
    print(f"  Pure JAX/Flax Query Router")
    print(f"  Architecture: {num_layers}L-{hidden_dim}d-{num_heads}h transformer")
    print(f"  Backend: {jax.default_backend()}")
    print(f"  Devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print(f"{'='*60}")

    train_examples = load_jsonl(os.path.join(data_dir, "all_train.jsonl"))
    val_examples   = load_jsonl(os.path.join(data_dir, "all_val.jsonl"))
    test_examples  = load_jsonl(os.path.join(data_dir, "all_test.jsonl"))
    print(f"\n  Data: {len(train_examples)} train / {len(val_examples)} val / {len(test_examples)} test")

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = FlaxQueryRouter(
        hidden_dim=hidden_dim, num_heads=num_heads,
        num_layers=num_layers, max_length=max_length,
    )

    rng = random.PRNGKey(seed)
    rng, init_rng = random.split(rng)

    dummy_ids = jnp.ones((1, max_length), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, max_length), dtype=jnp.int32)
    params = model.init(init_rng, dummy_ids, dummy_mask)["params"]

    param_count = sum(x.size for x in jax.tree.leaves(params))
    print(f"  Parameters: {param_count:,}")

    steps_per_epoch = len(train_examples) // batch_size
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps, alpha=0.01),
        ],
        boundaries=[warmup_steps],
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, weight_decay=0.01),
    )
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    history = []
    best_val_f1 = 0.0
    best_params = None

    print(f"\n  Training for {epochs} epochs ({total_steps} steps, {warmup_steps} warmup)...")
    print(f"  {'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>8} | {'Val F1':>7} | {'Time':>6} | {'ex/s':>8}")
    print(f"  {'-'*60}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        rng, shuffle_rng = random.split(rng)

        train_losses = []
        n_samples = 0
        for batch in make_batches(train_examples, tokenizer, batch_size,
                                   max_length, shuffle=True, rng=shuffle_rng):
            rng, step_rng = random.split(rng)
            state, metrics = train_step(
                state,
                jnp.array(batch["input_ids"]),
                jnp.array(batch["attention_mask"]),
                jnp.array(batch["labels"]),
                step_rng,
            )
            train_losses.append(float(metrics["loss"]))
            n_samples += batch["input_ids"].shape[0]

        val_preds_all, val_labels_all, val_losses = [], [], []
        for batch in make_batches(val_examples, tokenizer, batch_size, max_length):
            result = eval_step(
                state,
                jnp.array(batch["input_ids"]),
                jnp.array(batch["attention_mask"]),
                jnp.array(batch["labels"]),
            )
            val_preds_all.extend(result["preds"].tolist())
            val_labels_all.extend(batch["labels"].tolist())
            val_losses.append(float(result["loss"]))

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        val_f1 = f1_score(val_labels_all, val_preds_all, average="macro")
        elapsed = time.time() - t0
        throughput = n_samples / elapsed

        print(f"  {epoch:>5} | {avg_train_loss:>10.4f} | {avg_val_loss:>8.4f} | "
              f"{val_f1:>7.4f} | {elapsed:>5.1f}s | {throughput:>7.0f}")

        history.append({
            "epoch": epoch, "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss), "val_macro_f1": float(val_f1),
            "throughput_ex_per_s": float(throughput), "time_s": float(elapsed),
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_params = jax.tree.map(lambda x: np.array(x), state.params)

    # Test
    print(f"\n  Best val F1: {best_val_f1:.4f}")
    state = state.replace(params=best_params)

    test_preds_all, test_labels_all = [], []
    for batch in make_batches(test_examples, tokenizer, batch_size, max_length):
        result = eval_step(
            state,
            jnp.array(batch["input_ids"]),
            jnp.array(batch["attention_mask"]),
            jnp.array(batch["labels"]),
        )
        test_preds_all.extend(result["preds"].tolist())
        test_labels_all.extend(batch["labels"].tolist())

    test_f1 = f1_score(test_labels_all, test_preds_all, average="macro")
    print(f"\n  Test macro-F1: {test_f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        test_labels_all, test_preds_all,
        target_names=["single_hop", "multi_hop", "summary"],
    ))

    test_domains = [ex["domain"] for ex in test_examples[:len(test_preds_all)]]
    domain_map = {
        "graphragBench_medical": "Medical",
        "musique": "MuSiQue",
        "quality": "QuALITY",
        "ultraDomain_legal": "Legal",
    }
    domain_results = {}
    for domain, short in domain_map.items():
        mask = [d == domain for d in test_domains]
        yt = [l for l, m in zip(test_labels_all, mask) if m]
        yp = [p for p, m in zip(test_preds_all, mask) if m]
        if yt:
            domain_results[short] = float(f1_score(yt, yp, average="macro"))
            print(f"    {short:<10}: {domain_results[short]:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    results = {
        "model": f"FlaxQueryRouter({num_layers}L-{hidden_dim}d-{num_heads}h)",
        "framework": "jax_flax_pure",
        "backend": str(jax.default_backend()),
        "device_count": jax.device_count(),
        "tag": "flax_transformer_from_scratch",
        "test_macro_f1": float(test_f1),
        "best_val_f1": float(best_val_f1),
        "domain_results": domain_results,
        "history": history,
        "total_params": int(param_count),
        "confusion_matrix": confusion_matrix(test_labels_all, test_preds_all).tolist(),
        "avg_throughput_ex_per_s": float(np.mean([h["throughput_ex_per_s"] for h in history])),
        "total_training_time_s": float(sum(h["time_s"] for h in history)),
        "architecture": {
            "hidden_dim": hidden_dim, "num_heads": num_heads,
            "num_layers": num_layers, "mlp_dim": hidden_dim * 2,
            "max_length": max_length, "vocab_size": VOCAB_SIZE,
        },
    }

    out_path = os.path.join(output_dir, "neural_flax_router.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compare_throughput", action="store_true")
    args = parser.parse_args()

    results = train_and_evaluate(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        max_length=args.max_length, hidden_dim=args.hidden_dim,
        num_heads=args.num_heads, num_layers=args.num_layers,
        data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed,
    )

    if args.compare_throughput:
        print(f"\n{'='*60}")
        print(f"  Throughput Summary")
        print(f"{'='*60}")
        print(f"  Backend: {results['backend']}")
        print(f"  Devices: {results['device_count']}")
        print(f"  Params:  {results['total_params']:,}")
        print(f"  Avg throughput: {results['avg_throughput_ex_per_s']:.0f} ex/s")
        print(f"  Total training: {results['total_training_time_s']:.1f}s")
        for h in results["history"]:
            print(f"    Epoch {h['epoch']}: {h['throughput_ex_per_s']:.0f} ex/s ({h['time_s']:.1f}s)")


if __name__ == "__main__":
    main()
