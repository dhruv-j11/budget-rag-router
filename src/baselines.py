"""
baselines.py — Classical ML baselines for query-type routing.

Reproduces the Bansal & Agarwal (April 2026) approach:
  - 5 classifiers: SVM, Logistic Regression, Random Forest, Gradient Boosting, k-NN
  - 3 feature regimes: TF-IDF, MiniLM embeddings, hand-crafted structural features

Usage:
    python src/baselines.py --feature tfidf
    python src/baselines.py --feature minilm
    python src/baselines.py --feature handcrafted
    python src/baselines.py --feature all
"""

import os
import json
import re
import argparse
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# ---------- Data loading ----------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_splits(data_dir="data/"):
    train = load_jsonl(os.path.join(data_dir, "all_train.jsonl"))
    val   = load_jsonl(os.path.join(data_dir, "all_val.jsonl"))
    test  = load_jsonl(os.path.join(data_dir, "all_test.jsonl"))
    return train, val, test


# ---------- Feature extraction ----------

QUESTION_WORDS = {"who", "what", "where", "when", "why", "how", "which", "whom"}
COMPARISON_WORDS = {"compare", "contrast", "difference", "versus", "vs", "between", "similarities"}
AGGREGATION_WORDS = {"summarize", "overview", "describe", "explain", "discuss", "outline", "review", "list"}
MULTI_HOP_SIGNALS = {"and", "both", "also", "relationship", "connection", "affect", "influence", "lead"}


def extract_handcrafted(question: str) -> np.ndarray:
    """
    Extract structural features from a query string.

    Features (14-dim):
      0: character length
      1: word count
      2: avg word length
      3: question mark count
      4-6: starts with comparison/aggregation/multi-hop word (binary)
      7: number of question words present
      8: number of comparison words present
      9: number of aggregation words present
      10: number of multi-hop signal words present
      11: contains "and" or "or" (binary)
      12: number of named entities (rough: capitalized words not at sentence start)
      13: ends with question mark (binary)
    """
    words = question.lower().split()
    word_set = set(words)

    char_len = len(question)
    word_count = len(words)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    qmark_count = question.count("?")
    starts_comparison = int(words[0] in COMPARISON_WORDS) if words else 0
    starts_aggregation = int(words[0] in AGGREGATION_WORDS) if words else 0
    starts_multihop = int(words[0] in MULTI_HOP_SIGNALS) if words else 0
    n_question_words = len(word_set & QUESTION_WORDS)
    n_comparison = len(word_set & COMPARISON_WORDS)
    n_aggregation = len(word_set & AGGREGATION_WORDS)
    n_multihop = len(word_set & MULTI_HOP_SIGNALS)
    has_conjunction = int("and" in word_set or "or" in word_set)
    # Rough entity count: capitalized words not at sentence start
    raw_words = question.split()
    n_entities = sum(1 for i, w in enumerate(raw_words) if i > 0 and w[0].isupper()) if len(raw_words) > 1 else 0
    ends_qmark = int(question.strip().endswith("?"))

    return np.array([
        char_len, word_count, avg_word_len, qmark_count,
        starts_comparison, starts_aggregation, starts_multihop,
        n_question_words, n_comparison, n_aggregation, n_multihop,
        has_conjunction, n_entities, ends_qmark,
    ], dtype=np.float32)


def get_features(examples, method="tfidf", tfidf_vec=None, st_model=None):
    """
    Extract features for a list of examples.

    Returns:
        X: feature matrix (n_samples, n_features)
        y: label array (n_samples,)
        tfidf_vec: fitted TfidfVectorizer (only for tfidf, on train)
    """
    questions = [ex["question"] for ex in examples]
    y = np.array([ex["label"] for ex in examples])

    if method == "tfidf":
        if tfidf_vec is None:
            tfidf_vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
            X = tfidf_vec.fit_transform(questions)
        else:
            X = tfidf_vec.transform(questions)
        return X, y, tfidf_vec

    elif method == "minilm":
        from sentence_transformers import SentenceTransformer
        if st_model is None:
            st_model = SentenceTransformer("all-MiniLM-L6-v2")
        X = st_model.encode(questions, show_progress_bar=True, batch_size=128)
        return X, y, st_model

    elif method == "handcrafted":
        X = np.stack([extract_handcrafted(q) for q in questions])
        scaler = StandardScaler()
        if tfidf_vec is None:  # reuse param slot for scaler
            X = scaler.fit_transform(X)
            return X, y, scaler
        else:
            X = tfidf_vec.transform(X)  # tfidf_vec is actually the scaler here
            return X, y, tfidf_vec

    else:
        raise ValueError(f"Unknown feature method: {method}")


# ---------- Classifiers ----------

def get_classifiers():
    return {
        "SVM": SVC(kernel="linear", C=1.0, random_state=42),
        "LogReg": LogisticRegression(max_iter=1000, random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "GBM": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "kNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }


# ---------- Evaluation ----------

DOMAIN_NAMES = {
    "graphragBench_medical": "Medical",
    "musique": "MuSiQue",
    "quality": "QuALITY",
    "ultraDomain_legal": "Legal",
}
LABEL_NAMES = {0: "single_hop", 1: "multi_hop", 2: "summary"}


def evaluate_per_domain(y_true, y_pred, domains):
    """Compute macro-F1 overall and per domain."""
    results = {}
    results["overall"] = float(f1_score(y_true, y_pred, average="macro"))

    unique_domains = sorted(set(domains))
    for domain in unique_domains:
        mask = [d == domain for d in domains]
        yt = [y for y, m in zip(y_true, mask) if m]
        yp = [y for y, m in zip(y_pred, mask) if m]
        short = DOMAIN_NAMES.get(domain, domain)
        results[short] = float(f1_score(yt, yp, average="macro"))

    return results


# ---------- Main ----------

def run_baselines(feature_method, data_dir="data/", output_dir="results/"):
    """Train all classifiers with the given feature method and evaluate."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Running baselines: features={feature_method}")
    print(f"{'='*60}")

    # Load data
    train, val, test = load_splits(data_dir)

    # Extract features
    print(f"\nExtracting {feature_method} features...")
    t0 = time.time()
    X_train, y_train, fitted = get_features(train, method=feature_method)
    X_val, y_val, _ = get_features(val, method=feature_method, tfidf_vec=fitted)
    X_test, y_test, _ = get_features(test, method=feature_method, tfidf_vec=fitted)
    print(f"  Feature extraction: {time.time() - t0:.1f}s")
    print(f"  Train shape: {X_train.shape if hasattr(X_train, 'shape') else 'sparse'}")

    test_domains = [ex["domain"] for ex in test]

    # Train and evaluate each classifier
    all_results = {}
    classifiers = get_classifiers()

    for clf_name, clf in classifiers.items():
        print(f"\n  Training {clf_name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0

        # Predict on test
        y_pred = clf.predict(X_test)

        # Evaluate
        results = evaluate_per_domain(y_test.tolist(), y_pred.tolist(), test_domains)
        results["train_time_s"] = round(train_time, 2)
        all_results[clf_name] = results

        print(f"    Time: {train_time:.1f}s | Overall macro-F1: {results['overall']:.4f}")
        for domain_name in ["MuSiQue", "QuALITY", "Legal", "Medical"]:
            if domain_name in results:
                print(f"    {domain_name:<10}: {results[domain_name]:.4f}")

    # Save results
    output_path = os.path.join(output_dir, f"baselines_{feature_method}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved → {output_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", default="all",
                        choices=["tfidf", "minilm", "handcrafted", "all"])
    parser.add_argument("--data_dir", default="data/")
    parser.add_argument("--output_dir", default="results/")
    args = parser.parse_args()

    if args.feature == "all":
        for method in ["tfidf", "handcrafted", "minilm"]:
            run_baselines(method, args.data_dir, args.output_dir)
    else:
        run_baselines(args.feature, args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
