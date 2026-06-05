"""
compile_results.py — Compile all results into comparison tables.

Generates:
  - Terminal-friendly comparison table
  - LaTeX table for the paper
  - Summary JSON with all numbers

Usage:
    python src/compile_results.py --phase baselines
    python src/compile_results.py --phase neural
    python src/compile_results.py --phase all
"""

import os
import json
import argparse
from pathlib import Path

DOMAINS = ["MuSiQue", "QuALITY", "Legal", "Medical"]

# Token costs from RAGRouter-Bench (relative to LLM-Only = 1.0x)
PARADIGM_COSTS = {
    "single_hop": 1.4,   # NaiveRAG
    "multi_hop":  3.5,   # IterativeRAG
    "summary":    2.8,   # HybridRAG
}
ALWAYS_EXPENSIVE_COST = 3.5  # IterativeRAG for everything


def load_results(results_dir="results/"):
    """Load all result files from the results directory."""
    results = {}
    for f in Path(results_dir).glob("*.json"):
        with open(f) as fh:
            results[f.stem] = json.load(fh)
    return results


def format_baseline_table(results):
    """Format classical baseline results as a table."""
    rows = []
    for feature_type in ["tfidf", "handcrafted", "minilm"]:
        key = f"baselines_{feature_type}"
        if key not in results:
            continue
        for clf_name, metrics in results[key].items():
            row = {
                "Method": f"{clf_name}",
                "Features": feature_type.upper() if feature_type != "minilm" else "MiniLM",
            }
            for domain in DOMAINS:
                row[domain] = metrics.get(domain, 0)
            row["Overall"] = metrics.get("overall", 0)
            rows.append(row)
    return rows


def format_neural_table(results):
    """Format neural router results as a table."""
    rows = []
    for key, data in results.items():
        if not key.startswith("neural_"):
            continue
        if "domain_results" in data:
            # Single combined result
            model_short = data["model"].split("/")[-1]
            params_m = data.get("total_params", 0) / 1e6
            row = {
                "Method": f"{model_short}",
                "Features": f"Neural ({params_m:.0f}M)",
            }
            for domain in DOMAINS:
                row[domain] = data["domain_results"].get(domain, 0)
            row["Overall"] = data.get("test_macro_f1", 0)
            rows.append(row)
    return rows


def print_table(rows, title=""):
    """Print a formatted comparison table."""
    if not rows:
        print(f"  No results found for: {title}")
        return

    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")

    # Header
    header = f"  {'Method':<15} {'Features':<12}"
    for domain in DOMAINS:
        header += f" {domain:>8}"
    header += f" {'Overall':>8}"
    print(header)
    print(f"  {'-'*85}")

    # Find best per column
    best = {}
    for domain in DOMAINS + ["Overall"]:
        vals = [r.get(domain, 0) for r in rows if isinstance(r.get(domain, 0), (int, float))]
        best[domain] = max(vals) if vals else 0

    # Rows
    for row in rows:
        line = f"  {row['Method']:<15} {row['Features']:<12}"
        for domain in DOMAINS:
            val = row.get(domain, 0)
            marker = "*" if abs(val - best[domain]) < 0.001 and val > 0 else " "
            line += f" {val:>7.4f}{marker}"
        overall = row.get("Overall", 0)
        marker = "*" if abs(overall - best["Overall"]) < 0.001 and overall > 0 else " "
        line += f" {overall:>7.4f}{marker}"
        print(line)

    print(f"\n  * = best in column")


def generate_latex_table(all_rows, title="Results"):
    """Generate a LaTeX table for the paper."""
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + title + r"}")
    lines.append(r"\begin{tabular}{ll" + "c" * (len(DOMAINS) + 1) + "}")
    lines.append(r"\toprule")

    header = r"Method & Features"
    for d in DOMAINS:
        header += f" & {d}"
    header += r" & Overall \\"
    lines.append(header)
    lines.append(r"\midrule")

    for row in all_rows:
        line = f"{row['Method']} & {row['Features']}"
        for d in DOMAINS:
            val = row.get(d, 0)
            line += f" & {val:.3f}"
        line += f" & {row.get('Overall', 0):.3f}"
        line += r" \\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["baselines", "neural", "all"])
    parser.add_argument("--results_dir", default="results/")
    args = parser.parse_args()

    results = load_results(args.results_dir)

    if args.phase in ["baselines", "all"]:
        baseline_rows = format_baseline_table(results)
        print_table(baseline_rows, "Classical Baselines (Bansal & Agarwal reproduction)")

    if args.phase in ["neural", "all"]:
        neural_rows = format_neural_table(results)
        print_table(neural_rows, "Neural Routers (our contribution)")

    if args.phase == "all":
        all_rows = format_baseline_table(results) + format_neural_table(results)
        print_table(all_rows, "Full Comparison: Classical vs Neural Routers")

        # Generate LaTeX
        latex = generate_latex_table(all_rows, "Macro-F1 across domains for classical and neural routing classifiers.")
        latex_path = os.path.join(args.results_dir, "main_table.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\n  LaTeX table saved → {latex_path}")


if __name__ == "__main__":
    main()
