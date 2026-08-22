"""Comprehensive evaluation script reporting per-category precision/recall/F1, throughput breakdown, categorized exceptions, and confidence calibration."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .reconcile import reconcile
except ImportError:
    from reconcile import reconcile


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ERP_PATH = ROOT_DIR / "data" / "erp_ledger.csv"
DEFAULT_BANK_PATH = ROOT_DIR / "data" / "bank_statement.csv"
DEFAULT_REPORT_PATH = ROOT_DIR / "data" / "reconciled_report.csv"
DEFAULT_EXCEPTIONS_PATH = ROOT_DIR / "data" / "exceptions_list.csv"


def _pair_set(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(str(row["erp_id"]), str(row["bank_ref"])) for row in rows}


def _find_ground_truth(data_dir: Path, ground_truth_override: Path | None = None) -> Path:
    if ground_truth_override is not None and ground_truth_override.exists():
        return ground_truth_override
    candidates = (
        data_dir / ".ground_truth_mappings.json",
        data_dir / "ground_truth_mappings.json",
        ROOT_DIR / "data" / ".ground_truth_mappings.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find .ground_truth_mappings.json in {data_dir}")


def _calc_prf1(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 100.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def evaluate_report(
    report_path: str | Path,
    exceptions_path: str | Path,
    ground_truth_path: str | Path,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Calculate per-category PRF1, throughput breakdown, exceptions, and confidence calibration."""
    report_path = Path(report_path)
    exceptions_path = Path(exceptions_path)
    report_dir = report_path.parent

    # Load matched report and exceptions
    report = pd.read_csv(report_path) if report_path.exists() else pd.DataFrame()
    exceptions = pd.read_csv(exceptions_path) if exceptions_path.exists() else pd.DataFrame()

    # Load Ground Truth
    with Path(ground_truth_path).open(encoding="utf-8") as gt_file:
        gt_data = json.load(gt_file)

    gt_matches = gt_data.get("matches", [])
    gt_exceptions_erp = set(gt_data.get("exceptions", {}).get("erp_only", []))
    gt_exceptions_bank = set(gt_data.get("exceptions", {}).get("bank_only", []))

    # Categorize ground truth pairs
    gt_by_cat: dict[str, set[tuple[str, str]]] = {
        "exact": set(),
        "fuzzy_name_match": set(),
        "rounding_discrepancy": set(),
    }
    all_gt_pairs: set[tuple[str, str]] = set()

    for m in gt_matches:
        pair = (str(m["erp_id"]), str(m["bank_ref"]))
        all_gt_pairs.add(pair)
        raw_cat = str(m.get("match_type", "exact"))
        if raw_cat in {"exact"}:
            gt_by_cat["exact"].add(pair)
        elif raw_cat in {"fuzzy_name", "fuzzy_name_match"}:
            gt_by_cat["fuzzy_name_match"].add(pair)
        elif raw_cat in {"rounding_difference", "rounding_discrepancy"}:
            gt_by_cat["rounding_discrepancy"].add(pair)

    # Categorize predicted report pairs
    pred_by_cat: dict[str, set[tuple[str, str]]] = {
        "exact": set(),
        "fuzzy_name_match": set(),
        "rounding_discrepancy": set(),
    }
    all_pred_pairs: set[tuple[str, str]] = set()

    if not report.empty:
        for _, row in report.iterrows():
            pair = (str(row["erp_id"]), str(row["bank_ref"]))
            all_pred_pairs.add(pair)
            m_type = str(row.get("match_type", "exact"))
            # Exact-category performance belongs to the hard-coded first pass;
            # an LLM decision must not be counted as deterministic exact.
            if m_type == "exact" and str(row.get("source", "deterministic")) != "deterministic":
                continue
            if m_type in pred_by_cat:
                pred_by_cat[m_type].add(pair)
            elif m_type == "exact":
                pred_by_cat["exact"].add(pair)
            elif m_type in {"fuzzy_name", "fuzzy_name_match"}:
                pred_by_cat["fuzzy_name_match"].add(pair)
            elif m_type in {"rounding_difference", "rounding_discrepancy"}:
                pred_by_cat["rounding_discrepancy"].add(pair)

    # 1. Per-category Precision, Recall, F1
    category_metrics: dict[str, dict[str, float]] = {}

    for cat_key in ["exact", "fuzzy_name_match", "rounding_discrepancy"]:
        gt_pairs = gt_by_cat[cat_key]
        pred_pairs = pred_by_cat[cat_key]
        tp = len(pred_pairs & gt_pairs)
        fp = len(pred_pairs - gt_pairs)
        fn = len(gt_pairs - pred_pairs)
        category_metrics[cat_key] = _calc_prf1(tp, fp, fn)

    # True Exception category evaluation
    gt_all_exceptions = gt_exceptions_erp | gt_exceptions_bank
    pred_all_exceptions = set(exceptions["record_id"].astype(str)) if not exceptions.empty else set()

    tp_exc = len(pred_all_exceptions & gt_all_exceptions)
    fp_exc = len(pred_all_exceptions - gt_all_exceptions)
    fn_exc = len(gt_all_exceptions - pred_all_exceptions)
    category_metrics["true_exception"] = _calc_prf1(tp_exc, fp_exc, fn_exc)

    # Overall Match metrics
    overall_tp = len(all_pred_pairs & all_gt_pairs)
    overall_fp = len(all_pred_pairs - all_gt_pairs)
    overall_fn = len(all_gt_pairs - all_pred_pairs)
    overall_metrics = _calc_prf1(overall_tp, overall_fp, overall_fn)
    overall_metrics["accuracy"] = round(overall_tp / len(all_gt_pairs) * 100, 2) if all_gt_pairs else 0.0

    # 2. Throughput Breakdown
    metrics_file = report_dir / "reconciliation_metrics.json"
    if metrics_file.exists():
        with metrics_file.open(encoding="utf-8") as mf:
            throughput_metrics = json.load(mf)
    else:
        throughput_metrics = {
            "deterministic_matches_count": len(pred_by_cat["exact"]),
            "deterministic_match_percent": round(len(pred_by_cat["exact"]) / max(len(all_pred_pairs), 1) * 100, 2),
            "deterministic_time_seconds": 0.0,
            "llm_matches_count": len(all_pred_pairs) - len(pred_by_cat["exact"]),
            "llm_match_percent": round((len(all_pred_pairs) - len(pred_by_cat["exact"])) / max(len(all_pred_pairs), 1) * 100, 2),
            "llm_time_seconds": elapsed_seconds,
            "llm_calls_made": 0,
            "total_time_seconds": elapsed_seconds,
        }

    # 3. Categorized Honest Exceptions
    categorized_exceptions = {}
    if not exceptions.empty and "category" in exceptions.columns:
        categorized_exceptions = exceptions["category"].value_counts().to_dict()

    # 4. Confidence Calibration Analysis
    eval_log_file = report_dir / "confidence_calibration_log.json"
    calibration_buckets: dict[str, dict[str, Any]] = {
        "0.0-0.5": {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0},
        "0.5-0.8": {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0},
        "0.8-1.0": {"total": 0, "correct": 0, "incorrect": 0, "accuracy": 0.0},
    }

    if eval_log_file.exists():
        with eval_log_file.open(encoding="utf-8") as ef:
            eval_log = json.load(ef)

        for log_entry in eval_log:
            pair = (str(log_entry["erp_id"]), str(log_entry["bank_ref"]))
            is_true = pair in all_gt_pairs
            pred_match = log_entry["is_match"]
            is_correct = (pred_match == is_true)
            conf = float(log_entry["confidence_score"])

            if conf < 0.5:
                bucket_key = "0.0-0.5"
            elif conf < 0.8:
                bucket_key = "0.5-0.8"
            else:
                bucket_key = "0.8-1.0"

            calibration_buckets[bucket_key]["total"] += 1
            if is_correct:
                calibration_buckets[bucket_key]["correct"] += 1
            else:
                calibration_buckets[bucket_key]["incorrect"] += 1

        for b in calibration_buckets.values():
            if b["total"] > 0:
                b["accuracy"] = round(b["correct"] / b["total"] * 100, 2)

    # Check Audit Trail completeness
    audit_trail_ok = False
    if not report.empty and "reasoning" in report.columns:
        audit_trail_ok = bool(report["reasoning"].map(
            lambda v: isinstance(v, str) and bool(v.strip())
        ).all())


    summary = {
        "overall": overall_metrics,
        "category_metrics": category_metrics,
        "throughput_breakdown": throughput_metrics,
        "categorized_exceptions": categorized_exceptions,
        "confidence_calibration": calibration_buckets,
        "audit_trail_ok": audit_trail_ok,
    }

    # Save summary artifact
    summary_path = report_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def print_evaluation_summary(summary: dict[str, Any], dataset_name: str = "Dataset") -> None:
    print(f"\n=======================================================")
    print(f"       RECONCILIATION EVALUATION SUMMARY ({dataset_name.upper()})")
    print(f"=======================================================")

    overall = summary["overall"]
    print(f"\n--- OVERALL METRICS ---")
    print(f"Match Accuracy  : {overall.get('accuracy', 0.0):.2f}%")
    print(f"Match Precision : {overall['precision']:.2f}%")
    print(f"Match Recall    : {overall['recall']:.2f}%")
    print(f"Match F1 Score  : {overall['f1']:.2f}")

    print(f"\n--- CATEGORY-SPECIFIC PERFORMANCE (Precision / Recall / F1) ---")
    header = f"{'Match Category':<25} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}"
    print(header)
    print("-" * len(header))
    for cat, m in summary["category_metrics"].items():
        print(f"{cat:<25} | {m['precision']:<9.2f}% | {m['recall']:<9.2f}% | {m['f1']:<9.2f}")

    tb = summary["throughput_breakdown"]
    print(f"\n--- THROUGHPUT STAGE BREAKDOWN ---")
    print(f"Deterministic Pass : {tb.get('deterministic_matches_count', 0)} resolved ({tb.get('deterministic_match_percent', 0):.1f}%) in {tb.get('deterministic_time_seconds', 0):.4f}s")
    print(f"LLM Matcher Pass   : {tb.get('llm_matches_count', 0)} resolved ({tb.get('llm_match_percent', 0):.1f}%) in {tb.get('llm_time_seconds', 0):.2f}s ({tb.get('llm_calls_made', 0)} calls)")
    print(f"Total Pipeline Time: {tb.get('total_time_seconds', 0):.2f} seconds")

    print(f"\n--- CATEGORIZED HONEST EXCEPTIONS ---")
    cats = summary["categorized_exceptions"]
    if cats:
        for k, v in cats.items():
            print(f"  - {k:<30}: {v} records")
    else:
        print("  None")

    print(f"\n--- CONFIDENCE CALIBRATION ---")
    cal_header = f"{'Confidence Bucket':<20} | {'Evaluations':<12} | {'Correct':<8} | {'Accuracy':<10}"
    print(cal_header)
    print("-" * len(cal_header))
    for b_name, b_data in summary["confidence_calibration"].items():
        print(f"{b_name:<20} | {b_data['total']:<12} | {b_data['correct']:<8} | {b_data['accuracy']:<9.2f}%")

    print(f"\nAudit Trail: {'PASS (all rows have reasoning)' if summary['audit_trail_ok'] else 'FAIL'}")
    print(f"=======================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erp", type=Path, default=DEFAULT_ERP_PATH)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK_PATH)
    parser.add_argument("--ground-truth", type=Path)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--exceptions", type=Path, default=DEFAULT_EXCEPTIONS_PATH)
    parser.add_argument(
        "--provider",
        choices=("gemini", "groq"),
        default="groq",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Evaluate the existing report without running reconciliation.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    if not args.skip_reconcile:
        reconcile(
            args.erp,
            args.bank,
            args.report,
            args.exceptions,
            provider=args.provider,
            model=args.model,
        )
    elapsed_seconds = time.perf_counter() - started_at

    data_dir = args.report.parent
    ground_truth_path = _find_ground_truth(data_dir, args.ground_truth)

    summary = evaluate_report(args.report, args.exceptions, ground_truth_path, elapsed_seconds)
    print_evaluation_summary(summary, dataset_name=data_dir.name)


if __name__ == "__main__":
    main()
