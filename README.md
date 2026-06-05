# Budget-Constrained Neural Routing for Adaptive RAG

> First neural routing baselines on RAGRouter-Bench with cost-aware training.

## Paper

**Budget-Constrained Neural Routing for Adaptive RAG Systems**

We demonstrate that lightweight neural classifiers (DistilBERT, TinyBERT) outperform
classical baselines on the RAGRouter-Bench query routing task, and introduce a
cost-penalty training objective that enables tunable cost-quality tradeoffs via
Pareto frontier optimization.

## Quick Start

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Phase 1: Data
python src/data_loader.py
python src/verify_data.py

# Phase 2: Baselines
python src/baselines.py --feature all

# Phase 3: Neural routers
python src/neural_router.py --model distilbert-base-uncased --epochs 5
python src/neural_router.py --model huawei-noah/TinyBERT_General_4L_312D --epochs 5

# Phase 4: Analysis
python src/compile_results.py --phase all
python src/plot_results.py
python src/error_analysis.py
```

## Project Structure

```
budget-rag-router/
├── src/
│   ├── data_loader.py        # Download + split RAGRouter-Bench
│   ├── verify_data.py        # Sanity checks on data pipeline
│   ├── baselines.py          # Classical ML baselines (SVM, LR, RF, GBM, kNN)
│   ├── neural_router.py      # DistilBERT/TinyBERT fine-tuning
│   ├── cost_aware.py         # Cost-penalty training objective (Phase 5)
│   ├── error_analysis.py     # Misclassification analysis
│   ├── compile_results.py    # Results tables (terminal + LaTeX)
│   └── plot_results.py       # Paper-quality figures
├── data/                     # Processed train/val/test splits
├── results/                  # JSON result files + LaTeX tables
├── figures/                  # Generated plots
├── checkpoints/              # Saved model weights
├── EXECUTION_PLAN.md         # Hour-by-hour build plan
└── requirements.txt
```

## Dataset

[RAGRouter-Bench](https://huggingface.co/datasets/Chaplain0908/RAGRouter) (Wang et al., 2026):
- 7,727 queries across 4 domains
- 3 query types: single_hop, multi_hop, summary
- 5 RAG paradigms with known token costs

## Citation

```bibtex
@article{jain2026budget,
  title={Budget-Constrained Neural Routing for Adaptive RAG Systems},
  author={Jain, Dhruv},
  year={2026}
}
```
