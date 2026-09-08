"""
End-to-End RAG Pipeline Orchestrator.
Runs the entire RAG system against a golden set to measure overall system accuracy.
Tests the synergy of chunking, retrieval, and generation.

Usage:
`python scripts/run_rag_eval.py`
"""

import sys
import os
import json
import asyncio

# Ensure src/ and root are in PYTHONPATH
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

from config.logging_config import setup_logger
from eval.golden import load_golden_set, run_evaluation, aggregate_metrics
from eval.judge import LLMJudge
from pipeline.inference import InferencePipeline

logger = setup_logger(__name__)


def generate_report(metrics: dict, thresholds: dict, report_path: str, results: list = None):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    def evaluate_metric(
        metric_val: float, thresh_dict: dict, reverse: bool = False
    ) -> str:
        # if reverse, lower is better (e.g. latency, cost)
        acceptable = thresh_dict.get("acceptable", 0)
        good = thresh_dict.get("good", 0)

        if reverse:
            if metric_val <= good:
                return "Good"
            elif metric_val <= acceptable:
                return "Acceptable"
            else:
                return "Poor"
        else:
            if metric_val >= good:
                return "Good"
            elif metric_val >= acceptable:
                return "Acceptable"
            else:
                return "Poor"

    # Extract numerical values from metrics for comparison
    tsr = float(metrics.get("Task Success Rate", "0").strip("%")) / 100.0
    grd = float(metrics.get("Groundedness", "0").strip("%")) / 100.0
    rhr = float(metrics.get("Retrieval Hit Rate", "0").strip("%")) / 100.0
    cost = float(metrics.get("Cost / Query", "0").strip("$"))
    latency = float(metrics.get("Latency (p95)", "0").strip("s"))

    evals = {
        "Task Success Rate": evaluate_metric(
            tsr, thresholds.get("task_success_rate", {}), False
        ),
        "Groundedness": evaluate_metric(grd, thresholds.get("groundedness", {}), False),
        "Retrieval Hit Rate": evaluate_metric(
            rhr, thresholds.get("retrieval_rate", {}), False
        ),
        "Cost / Query": evaluate_metric(
            cost, thresholds.get("cost_per_query", {}), True
        ),
        "Latency (p95)": evaluate_metric(
            latency, thresholds.get("latency_p95", {}), True
        ),
    }

    with open(report_path, "w") as f:
        f.write("# RAG Evaluation Report\n\n")
        f.write("## Final Metrics vs Thresholds\n\n")
        f.write("| Metric | Acceptable Threshold | Good Threshold | Value | Threshold Evaluation |\n")
        f.write("|---|---|---|---|---|\n")
        
        # Helper to get threshold string safely
        def get_thresh_str(metric_key, thresh_type):
            mapping = {
                "Task Success Rate": "task_success_rate",
                "Groundedness": "groundedness",
                "Retrieval Hit Rate": "retrieval_rate",
                "Cost / Query": "cost_per_query",
                "Latency (p95)": "latency_p95"
            }
            if metric_key in mapping:
                t = thresholds.get(mapping[metric_key], {})
                val = t.get(thresh_type, "N/A")
                return str(val)
            return "N/A"

        for key, val in metrics.items():
            eval_status = evals.get(key, "N/A")
            acc_str = get_thresh_str(key, "acceptable")
            good_str = get_thresh_str(key, "good")
            f.write(f"| {key} | {acc_str} | {good_str} | {val} | {eval_status} |\n")

        if results:
            f.write("\n## Metrics by Path Level\n\n")
            f.write("| Path Level | Count | Task Success Rate | Groundedness | Retrieval Hit Rate |\n")
            f.write("|---|---|---|---|---|\n")
            
            levels = {}
            for r in results:
                lvl = r.get("level", "unknown")
                if lvl not in levels:
                    levels[lvl] = {"count": 0, "ts": 0.0, "gr": 0.0, "rh": 0.0}
                levels[lvl]["count"] += 1
                levels[lvl]["ts"] += r.get("task_success", 0.0)
                levels[lvl]["gr"] += r.get("groundedness", 0.0)
                levels[lvl]["rh"] += r.get("retrieval_hit", 0.0)
                
            for lvl, data in levels.items():
                c = data["count"]
                ts_pct = (data["ts"] / c) if c > 0 else 0
                gr_pct = (data["gr"] / c) if c > 0 else 0
                rh_pct = (data["rh"] / c) if c > 0 else 0
                f.write(f"| {lvl} | {c} | {ts_pct:.2%} | {gr_pct:.2%} | {rh_pct:.2%} |\n")

        f.write("\n## Threshold Definitions\n")
        f.write("```json\n")
        f.write(json.dumps(thresholds, indent=2))
        f.write("\n```\n")

    print(f"\nReport successfully generated at: {report_path}")


async def main():
    print("==========================================")
    print("Starting Comprehensive RAG Evaluation...")
    print("==========================================\n")

    golden_path = os.path.join(root_dir, "data", "golden_set.jsonl")
    thresholds_path = os.path.join(root_dir, "data", "evals", "thresholds.json")
    report_path = os.path.join(root_dir, "docs", "evals", "report.md")

    # Load dataset
    golden_dataset = load_golden_set(golden_path)
    if not golden_dataset:
        print("Error: Could not load golden dataset.")
        sys.exit(1)

    print(f"Successfully loaded {len(golden_dataset)} items from {golden_path}\n")

    # Load thresholds
    try:
        with open(thresholds_path, "r") as f:
            thresholds = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load thresholds: {e}")
        thresholds = {}

    # Initialize components
    pipeline = InferencePipeline()
    judge = LLMJudge()

    # Run evaluation
    results = await run_evaluation(golden_dataset, pipeline, judge)

    # Save all results into a single json file in data/evals
    evals_dir = os.path.join(root_dir, "data", "evals")
    os.makedirs(evals_dir, exist_ok=True)
    file_path = os.path.join(evals_dir, "all_results.json")
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    # Aggregate and print metrics
    metrics = aggregate_metrics(results)

    print("\n==========================================")
    print("Evaluation Complete. Final Metrics:")
    print("==========================================")
    for key, value in metrics.items():
        print(f"{key.ljust(25)}: {value}")
    print("==========================================")

    # Generate report
    generate_report(metrics, thresholds, report_path, results)


if __name__ == "__main__":
    asyncio.run(main())
