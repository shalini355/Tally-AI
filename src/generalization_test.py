"""Anti-cherry-picking generalization test runner.

Generates a second, freshly seeded 50+ record synthetic dataset, runs the full
reconciliation pipeline against it, and compares accuracy/throughput/calibration
metrics against the primary run to prove generalization.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from .evaluate_accuracy import evaluate_report, print_evaluation_summary
    from .generate_data import generate_dataset
    from .reconcile import reconcile
except ImportError:
    from evaluate_accuracy import evaluate_report, print_evaluation_summary
    from generate_data import generate_dataset
    from reconcile import reconcile


ROOT_DIR = Path(__file__).resolve().parents[1]
PRIMARY_DIR = ROOT_DIR / "data"
GENERALIZATION_DIR = PRIMARY_DIR / "dataset_2"


def run_generalization_test(
    seed: int = 20260823,
    provider: str = "groq",
    model: str | None = None,
    output_dir: Path = GENERALIZATION_DIR,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate Dataset 2, run reconciliation, evaluate, and compare against Dataset 1."""
    print("=======================================================")
    print("     RUNNING GENERALIZATION TEST ON DATASET 2")
    print("=======================================================")

    # 1. Generate Dataset 2
    erp_csv, bank_csv, gt_json = generate_dataset(
        seed=seed,
        output_dir=output_dir,
        erp_count=60,
        matched_count=52,
    )

    report_csv = output_dir / "reconciled_report.csv"
    exceptions_csv = output_dir / "exceptions_list.csv"

    # 2. Run Reconciliation on Dataset 2
    start_time = time.perf_counter()
    reconcile(
        erp_csv,
        bank_csv,
        report_csv,
        exceptions_csv,
        provider=provider,
        model=model,
    )
    elapsed_time = time.perf_counter() - start_time

    # 3. Evaluate Dataset 2
    ds2_summary = evaluate_report(report_csv, exceptions_csv, gt_json, elapsed_time)
    print_evaluation_summary(ds2_summary, dataset_name="Generalization (Dataset 2)")

    # 4. Load or Evaluate Dataset 1 for comparison
    ds1_report = PRIMARY_DIR / "reconciled_report.csv"
    ds1_exceptions = PRIMARY_DIR / "exceptions_list.csv"
    ds1_gt = PRIMARY_DIR / ".ground_truth_mappings.json"

    if ds1_report.exists() and ds1_gt.exists():
        ds1_summary = evaluate_report(ds1_report, ds1_exceptions, ds1_gt, 0.0)
    else:
        # Generate and run dataset 1 if it doesn't exist
        g1_erp, g1_bank, g1_gt = generate_dataset(seed=20260822, output_dir=PRIMARY_DIR)
        t1_start = time.perf_counter()
        reconcile(g1_erp, g1_bank, ds1_report, ds1_exceptions, provider=provider, model=model)
        ds1_summary = evaluate_report(ds1_report, ds1_exceptions, g1_gt, time.perf_counter() - t1_start)

    # 5. Print Comparison Table
    print_comparative_table(ds1_summary, ds2_summary)

    # 6. Save comparative artifact
    comparison_artifact = PRIMARY_DIR / "generalization_comparison.json"
    comparison_artifact.write_text(
        json.dumps(
            {
                "dataset_1_primary": ds1_summary,
                "dataset_2_generalization": ds2_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved comparative metrics to {comparison_artifact}")

    return ds1_summary, ds2_summary


def print_comparative_table(ds1: dict[str, Any], ds2: dict[str, Any]) -> None:
    print("\n=========================================================================================")
    print("             ANTI-CHERRY-PICKING GENERALIZATION COMPARISON SUMMARY")
    print("=========================================================================================")
    header = f"{'Metric':<35} | {'Primary Dataset 1':<22} | {'Generalization Dataset 2':<22}"
    print(header)
    print("-" * len(header))

    # Overall Metrics
    m1_ov = ds1.get("overall", {})
    m2_ov = ds2.get("overall", {})
    print(f"{'Overall Match Accuracy':<35} | {m1_ov.get('accuracy', 0.0):<21.2f}% | {m2_ov.get('accuracy', 0.0):<21.2f}%")
    print(f"{'Overall Precision':<35} | {m1_ov.get('precision', 0.0):<21.2f}% | {m2_ov.get('precision', 0.0):<21.2f}%")
    print(f"{'Overall Recall':<35} | {m1_ov.get('recall', 0.0):<21.2f}% | {m2_ov.get('recall', 0.0):<21.2f}%")
    print(f"{'Overall F1 Score':<35} | {m1_ov.get('f1', 0.0):<22.2f} | {m2_ov.get('f1', 0.0):<22.2f}")

    print("-" * len(header))
    # Category F1 Scores
    for cat in ["exact", "fuzzy_name_match", "rounding_discrepancy", "true_exception"]:
        f1_1 = ds1.get("category_metrics", {}).get(cat, {}).get("f1", 0.0)
        f1_2 = ds2.get("category_metrics", {}).get(cat, {}).get("f1", 0.0)
        print(f"{f'F1 Score ({cat})':<35} | {f1_1:<22.2f} | {f2_2 if (f2_2 := f1_2) else 0.0:<22.2f}")

    print("-" * len(header))
    # Throughput
    tb1 = ds1.get("throughput_breakdown", {})
    tb2 = ds2.get("throughput_breakdown", {})
    print(f"{'Deterministic Pass Resolved %':<35} | {tb1.get('deterministic_match_percent', 0.0):<21.1f}% | {tb2.get('deterministic_match_percent', 0.0):<21.1f}%")
    print(f"{'LLM Matcher Pass Resolved %':<35} | {tb1.get('llm_match_percent', 0.0):<21.1f}% | {tb2.get('llm_match_percent', 0.0):<21.1f}%")
    print(f"{'Total Pipeline Time (s)':<35} | {tb1.get('total_time_seconds', 0.0):<22.2f} | {tb2.get('total_time_seconds', 0.0):<22.2f}")

    print("-" * len(header))
    # Calibration High-Conf Bucket
    cal1 = ds1.get("confidence_calibration", {}).get("0.8-1.0", {}).get("accuracy", 0.0)
    cal2 = ds2.get("confidence_calibration", {}).get("0.8-1.0", {}).get("accuracy", 0.0)
    print(f"{'High Conf (0.8-1.0) Bucket Accuracy':<35} | {cal1:<21.2f}% | {cal2:<21.2f}%")
    print("=========================================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260823, help="Seed for Dataset 2")
    parser.add_argument("--provider", choices=("gemini", "groq"), default="groq")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, default=GENERALIZATION_DIR)
    args = parser.parse_args()

    run_generalization_test(
        seed=args.seed,
        provider=args.provider,
        model=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
