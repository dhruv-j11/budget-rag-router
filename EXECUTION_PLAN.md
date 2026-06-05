# Budget-Constrained Neural Routing for Adaptive RAG
## Hours 0–12 Execution Plan

---

## PHASE 1: Environment + Data (Hours 0–2)

### Step 1.1 — Clone and set up environment (~15 min)

```bash
cd ~/budget-rag-router
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1.2 — Download and inspect the dataset (~15 min)

Run: `python src/data_loader.py`

This will:
- Pull all 4 subsets from HuggingFace (`Chaplain0908/RAGRouter`)
- Print label distributions per domain
- Create train/val/test splits (70/15/15) stratified by label
- Save splits to `data/` as jsonl files

**What to understand:** Look at the printout. Notice the class imbalance.
MuSiQue is 77% multi_hop. Medical is 58% single_hop. Legal is more balanced.
This is why macro-F1 matters — accuracy would be misleading.

### Step 1.3 — Sanity check: read 20 examples (~15 min)

Open `data/all_train.jsonl` and actually read queries from each class.
You need to develop intuition for what makes a query "single_hop" vs "multi_hop" vs "summary".

Examples:
- single_hop: "What is the most common type of skin cancer?" → one fact, one lookup
- multi_hop: "Who is the spouse of the Green performer?" → need to find who performed Green, then find their spouse
- summary: "Summarize the treatment options for BCC" → need to aggregate across multiple passages

**This is not optional.** If you can't explain the label taxonomy to someone, you don't understand your own paper.

### Step 1.4 — Verify data pipeline (~15 min)

Run: `python src/verify_data.py`

Checks that splits don't leak, label encoding is correct, tokenization works.

---

## PHASE 2: Classical Baselines (Hours 2–5)

These are the numbers you need to BEAT. The Bansal paper (April 2026) reported:
- TF-IDF + SVM: ~0.95 macro-F1 on legal, ~0.80 on medical
- MiniLM + LR: similar range

You reproduce them to (a) validate your pipeline and (b) have baselines in your paper.

### Step 2.1 — Run TF-IDF baselines (~30 min)

Run: `python src/baselines.py --feature tfidf`

This trains SVM, Logistic Regression, Random Forest, and Gradient Boosting
on TF-IDF features for each domain AND the combined dataset.
Results saved to `results/baselines_tfidf.json`.

### Step 2.2 — Run MiniLM embedding baselines (~30 min)

Run: `python src/baselines.py --feature minilm`

Same classifiers but using `all-MiniLM-L6-v2` sentence embeddings as features.
This is the stronger baseline — MiniLM captures semantic meaning, not just word frequency.

### Step 2.3 — Run hand-crafted feature baselines (~30 min)

Run: `python src/baselines.py --feature handcrafted`

Features: query length, word count, question word (who/what/how/why),
presence of comparison words, number of entities, etc.

### Step 2.4 — Compile baseline results table (~30 min)

Run: `python src/compile_results.py --phase baselines`

This generates a LaTeX table + a summary printout.
**Checkpoint:** Your numbers should be within ~2% of the Bansal paper.
If they're wildly different, something is wrong with your data split or preprocessing.

### Step 2.5 — Understand what you just did (~30 min)

Read through the results. Ask yourself:
- Which domain is easiest to route? (Legal — formulaic language)
- Which is hardest? (Medical — single corpus, less surface variation)
- Which classifier won? (Probably SVM or LR — they're strong for text)
- Which features won? (Probably MiniLM — it captures semantics)

Write 2-3 bullet points of observations. These become your Related Work comparison.

---

## PHASE 3: Neural Router Training (Hours 5–10)

This is your main contribution. Nobody has done this on RAGRouter-Bench.

### Step 3.1 — Fine-tune DistilBERT on combined data (~1.5 hours)

Run: `python src/neural_router.py --model distilbert-base-uncased --epochs 5 --lr 2e-5`

What's happening:
- DistilBERT (66M params, 6 layers) gets a classification head (Linear → 3 classes)
- Input: tokenized query (max 128 tokens)
- Output: softmax over [single_hop, multi_hop, summary]
- Loss: cross-entropy
- Optimizer: AdamW with linear warmup (10% of steps) then linear decay
- Batch size: 32
- Training on combined data from all 4 domains

**Training takes ~15-20 min on a GPU, ~1-2 hours on CPU.**
If you have a GPU (even a free Colab T4), use it.

### Step 3.2 — Fine-tune per-domain models (~1 hour)

Run: `python src/neural_router.py --model distilbert-base-uncased --per_domain`

Trains 4 separate models, one per domain.
Compare per-domain vs combined to see if domain-specific routing helps.

### Step 3.3 — Try TinyBERT as a smaller alternative (~30 min)

Run: `python src/neural_router.py --model huawei-noah/TinyBERT_General_4L_312D --epochs 5 --lr 3e-5`

TinyBERT is 14.5M params — 4.5x smaller than DistilBERT.
If it matches DistilBERT's accuracy, that's a great finding (smaller router = cheaper routing).

### Step 3.4 — Evaluate all neural models (~30 min)

Run: `python src/evaluate.py --phase neural`

This computes:
- Macro-F1 per domain and overall
- Confusion matrices (which misroutes happen most?)
- Per-class precision/recall
- Comparison table against classical baselines

### Step 3.5 — Error analysis (~1 hour)

Run: `python src/error_analysis.py`

This shows you:
- The queries your model gets WRONG
- Which class pairs get confused most (multi_hop ↔ single_hop is typical)
- Domain-level breakdown of errors

**Actually read the misclassified queries.** Can YOU tell the difference?
If the query is ambiguous even to a human, that's worth noting in the paper.

---

## PHASE 4: Checkpoint + Documentation (Hours 10–12)

### Step 4.1 — Generate comparison table (~30 min)

Run: `python src/compile_results.py --phase all`

This produces the main results table for your paper:
| Model | Features | MuSiQue | QuALITY | Legal | Medical | Overall | Token Savings |

### Step 4.2 — Generate figures (~30 min)

Run: `python src/plot_results.py`

Generates:
- Bar chart: macro-F1 across domains (classical vs neural)
- Confusion matrix heatmaps per domain
- Model size vs accuracy scatter

### Step 4.3 — Write notes for the paper (~1 hour)

Open `notes/findings.md` and write down:
1. Did neural routers beat classical? By how much?
2. Which domain was hardest? Why?
3. Does model size matter (DistilBERT vs TinyBERT)?
4. What types of queries get misrouted?

These become your Experiments section.

---

## Key checkpoints

After hour 2: You can load and inspect the data. You understand the label taxonomy.
After hour 5: You have baseline numbers that match the literature.
After hour 10: You have neural router results that (hopefully) beat baselines.
After hour 12: You have a complete results table and figures ready for the paper.

Then we move to Phase 5 (hours 12-20): the cost-aware objective — your MAIN contribution.
