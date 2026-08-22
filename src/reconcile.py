"""Run deterministic and LLM-assisted reconciliation for both input datasets with timing and exception breakdown."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

try:
    from .deterministic_filter import DeterministicMatcher
    from .llm_matcher import MatchDecision, MatchType, evaluate_potential_match
except ImportError:
    from deterministic_filter import DeterministicMatcher
    from llm_matcher import MatchDecision, MatchType, evaluate_potential_match


DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MAX_CANDIDATES = 2


def _tokens(value: Any) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9]+", str(value))
        if len(token) > 2 and token not in {"pvt", "ltd", "settlement", "payout"}
    }


def _candidate_score(erp_row: Mapping[str, Any], bank_row: Mapping[str, Any]) -> float:
    """Rank likely candidates cheaply before spending an LLM request."""
    erp_name = _tokens(erp_row.get("merchant_name", ""))
    bank_name = _tokens(bank_row.get("description", ""))
    name_overlap = len(erp_name & bank_name) / max(len(erp_name), 1)
    expected_net = (
        float(erp_row.get("amount", 0))
        + float(erp_row.get("tax_line_item", 0))
        - float(bank_row.get("fee_deducted", 0))
    )
    amount_delta = abs(expected_net - float(bank_row.get("net_amount", 0)))
    amount_score = 1 / (1 + amount_delta)
    try:
        date_delta = abs(
            pd.Timestamp(erp_row["date"]) - pd.Timestamp(bank_row["settlement_date"])
        ).days
    except (KeyError, TypeError, ValueError):
        date_delta = 30
    date_score = 1 / (1 + date_delta)
    return (name_overlap * 0.55) + (amount_score * 0.30) + (date_score * 0.15)


def _matched_output_row(
    erp_row: Mapping[str, Any],
    bank_row: Mapping[str, Any],
    match_type: str,
    confidence_score: float,
    reasoning: str,
    source: str,
) -> dict[str, Any]:
    return {
        "erp_id": erp_row["erp_id"],
        "bank_ref": bank_row["bank_ref"],
        "date": erp_row.get("date", ""),
        "settlement_date": bank_row.get("settlement_date", ""),
        "merchant_name": erp_row.get("merchant_name", ""),
        "description": bank_row.get("description", ""),
        "amount": erp_row.get("amount", ""),
        "tax_line_item": erp_row.get("tax_line_item", ""),
        "net_amount": bank_row.get("net_amount", ""),
        "fee_deducted": bank_row.get("fee_deducted", ""),
        "match_type": match_type,
        "confidence_score": confidence_score,
        "reasoning": reasoning,
        "source": source,
    }


def _rank_candidates(
    erp_row: Mapping[str, Any], bank_rows: pd.DataFrame, limit: int
) -> list[dict[str, Any]]:
    candidates: list[tuple[float, dict[str, Any]]] = []
    erp_name = _tokens(erp_row.get("merchant_name", ""))

    for _, bank_row in bank_rows.iterrows():
        b_dict = bank_row.to_dict()
        b_name = _tokens(b_dict.get("description", ""))
        name_overlap = len(erp_name & b_name)
        expected_net = (
            float(erp_row.get("amount", 0))
            + float(erp_row.get("tax_line_item", 0))
            - float(b_dict.get("fee_deducted", 0))
        )
        amount_delta = abs(expected_net - float(b_dict.get("net_amount", 0)))

        # Plausibility filter: Require token overlap or close net amount
        if name_overlap > 0 or amount_delta <= 50.0:
            score = float(_candidate_score(erp_row, b_dict))
            candidates.append((score, b_dict))

    candidates.sort(key=lambda item: item[0], reverse=True)
    return [bank_row for _, bank_row in candidates[:limit]]




def reconcile(
    erp_path: str | Path,
    bank_path: str | Path,
    report_path: str | Path,
    exceptions_path: str | Path,
    *,
    provider: str = "gemini",
    model: str | None = None,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    max_candidates_per_erp: int = DEFAULT_MAX_CANDIDATES,
    evaluator: Callable[..., MatchDecision] = evaluate_potential_match,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconcile inputs and write matched and exception CSV reports with timing breakdown."""
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    if max_candidates_per_erp < 1:
        raise ValueError("max_candidates_per_erp must be at least 1")

    pipeline_start = time.perf_counter()

    # Stage 1: Deterministic matching pass
    det_start = time.perf_counter()
    matcher = DeterministicMatcher(erp_path, bank_path)
    deterministic_pairs = matcher.run_first_pass()
    det_time = time.perf_counter() - det_start

    unresolved_erp, unresolved_bank = matcher.unresolved_pool()

    matched_rows: list[dict[str, Any]] = []
    matched_erp_ids: set[str] = set()
    matched_bank_refs: set[str] = set()

    for pair in deterministic_pairs:
        erp_row = matcher.erp.iloc[pair["erp_row"]].to_dict()
        bank_row = matcher.bank.iloc[pair["bank_row"]].to_dict()
        matched_erp_ids.add(str(pair["erp"]))
        matched_bank_refs.add(str(pair["bank"]))
        matched_rows.append(
            _matched_output_row(
                erp_row,
                bank_row,
                MatchType.EXACT.value,
                1.0,
                "ERP ID embedded in bank description and amount matched exactly.",
                "deterministic",
            )
        )

    deterministic_count = len(matched_rows)

    # Stage 2: LLM fuzzy & semantic matching pass
    llm_start = time.perf_counter()
    llm_calls_made = 0
    llm_evaluations_log: list[dict[str, Any]] = []

    # Map candidate evaluation history per unresolved erp_id
    erp_eval_history: dict[str, list[dict[str, Any]]] = {}

    for _, erp_series in unresolved_erp.iterrows():
        erp_row = erp_series.to_dict()
        erp_id = str(erp_row["erp_id"])
        if erp_id in matched_erp_ids:
            continue

        erp_eval_history[erp_id] = []
        available_bank = unresolved_bank[
            ~unresolved_bank["bank_ref"].astype(str).isin(matched_bank_refs)
        ]
        candidates = _rank_candidates(erp_row, available_bank, max_candidates_per_erp)

        for bank_row in candidates:
            bank_ref = str(bank_row["bank_ref"])
            if bank_ref in matched_bank_refs:
                continue

            llm_calls_made += 1
            decision = evaluator(
                erp_row,
                bank_row,
                provider=provider,
                model=model,
            )

            eval_record = {
                "erp_id": erp_id,
                "bank_ref": bank_ref,
                "is_match": decision.is_match,
                "confidence_score": decision.confidence_score,
                "match_type": decision.match_type.value,
                "reasoning": decision.reasoning,
                "merchant_erp": erp_row.get("merchant_name", ""),
                "description_bank": bank_row.get("description", ""),
                "currency": erp_row.get("currency", ""),
            }
            llm_evaluations_log.append(eval_record)
            erp_eval_history[erp_id].append(eval_record)

            if (
                decision.is_match
                and decision.match_type is not MatchType.NO_MATCH
                and decision.confidence_score >= confidence_threshold
            ):
                matched_erp_ids.add(erp_id)
                matched_bank_refs.add(bank_ref)
                matched_rows.append(
                    _matched_output_row(
                        erp_row,
                        bank_row,
                        decision.match_type.value,
                        decision.confidence_score,
                        decision.reasoning,
                        "llm",
                    )
                )
                break

    llm_time = time.perf_counter() - llm_start
    llm_count = len(matched_rows) - deterministic_count
    total_pipeline_time = time.perf_counter() - pipeline_start

    # Stage 3: Categorized honest exceptions tagging
    exceptions: list[dict[str, Any]] = []

    for _, erp_row in matcher.erp.iterrows():
        erp_id = str(erp_row["erp_id"])
        if erp_id not in matched_erp_ids:
            eval_history = erp_eval_history.get(erp_id, [])
            currency = str(erp_row.get("currency", "")).upper()

            # Determine category
            if currency not in {"INR", ""}:
                category = "amount_currency_mismatch"
                reason = f"Currency is {currency} (foreign settlement/currency mismatch)."
            elif not eval_history:
                category = "no_counterpart_found"
                reason = "No candidate bank settlement found in candidate search pool."
            else:
                high_conf_candidates = [
                    e for e in eval_history if e["confidence_score"] >= 0.5
                ]
                if len(high_conf_candidates) > 1:
                    category = "ambiguous_multiple_candidates"
                    reason = f"Multiple candidate bank records had ambiguous confidence scores (count={len(high_conf_candidates)})."
                else:
                    top_conf = max(e["confidence_score"] for e in eval_history)
                    if top_conf < confidence_threshold:
                        category = "below_confidence_threshold"
                        reason = f"Highest LLM confidence score ({top_conf:.2f}) was below threshold ({confidence_threshold})."
                    else:
                        category = "no_counterpart_found"
                        reason = "No matching bank settlement record satisfied business constraints."

            exceptions.append(
                {
                    "dataset": "erp",
                    "record_id": erp_id,
                    "related_record_id": eval_history[0]["bank_ref"] if eval_history else "",
                    "confidence_score": max((e["confidence_score"] for e in eval_history), default=0.0),
                    "match_type": MatchType.NO_MATCH.value,
                    "category": category,
                    "reason": reason,
                    "details": erp_row.to_json(),
                }
            )

    for _, bank_row in matcher.bank.iterrows():
        bank_ref = str(bank_row["bank_ref"])
        if bank_ref not in matched_bank_refs:
            # Check if this bank record was evaluated by any ERP row
            bank_evals = [e for e in llm_evaluations_log if e["bank_ref"] == bank_ref]
            if not bank_evals:
                category = "no_counterpart_found"
                reason = "No candidate ERP transaction found matching bank settlement details."
            else:
                top_conf = max(e["confidence_score"] for e in bank_evals)
                if top_conf < confidence_threshold:
                    category = "below_confidence_threshold"
                    reason = f"Evaluated candidate confidence ({top_conf:.2f}) was below threshold ({confidence_threshold})."
                else:
                    category = "ambiguous_multiple_candidates"
                    reason = "Bank record had ambiguous candidate matches in ERP ledger."

            exceptions.append(
                {
                    "dataset": "bank",
                    "record_id": bank_ref,
                    "related_record_id": bank_evals[0]["erp_id"] if bank_evals else "",
                    "confidence_score": max((e["confidence_score"] for e in bank_evals), default=0.0),
                    "match_type": MatchType.NO_MATCH.value,
                    "category": category,
                    "reason": reason,
                    "details": bank_row.to_json(),
                }
            )

    matched_report = pd.DataFrame(
        matched_rows,
        columns=[
            "erp_id",
            "bank_ref",
            "date",
            "settlement_date",
            "merchant_name",
            "description",
            "amount",
            "tax_line_item",
            "net_amount",
            "fee_deducted",
            "match_type",
            "confidence_score",
            "reasoning",
            "source",
        ],
    )
    exceptions_report = pd.DataFrame(
        exceptions,
        columns=[
            "dataset",
            "record_id",
            "related_record_id",
            "confidence_score",
            "match_type",
            "category",
            "reason",
            "details",
        ],
    )

    report_path = Path(report_path)
    exceptions_path = Path(exceptions_path)
    report_dir = report_path.parent
    report_dir.mkdir(parents=True, exist_ok=True)

    matched_report.to_csv(report_path, index=False)
    exceptions_report.to_csv(exceptions_path, index=False)

    # Save metrics JSON & calibration log
    total_matched = len(matched_rows)
    total_processed = len(matcher.erp)

    det_pct = (deterministic_count / total_matched * 100) if total_matched else 0.0
    llm_pct = (llm_count / total_matched * 100) if total_matched else 0.0

    exception_category_counts = exceptions_report["category"].value_counts().to_dict()

    metrics = {
        "total_erp_records": total_processed,
        "total_bank_records": len(matcher.bank),
        "total_matched_pairs": total_matched,
        "deterministic_matches_count": deterministic_count,
        "deterministic_match_percent": round(det_pct, 2),
        "deterministic_time_seconds": round(det_time, 4),
        "llm_matches_count": llm_count,
        "llm_match_percent": round(llm_pct, 2),
        "llm_time_seconds": round(llm_time, 4),
        "llm_calls_made": llm_calls_made,
        "total_time_seconds": round(total_pipeline_time, 4),
        "unresolved_exceptions_count": len(exceptions),
        "categorized_exceptions": exception_category_counts,
    }

    metrics_file = report_dir / "reconciliation_metrics.json"
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    eval_log_file = report_dir / "confidence_calibration_log.json"
    eval_log_file.write_text(json.dumps(llm_evaluations_log, indent=2), encoding="utf-8")

    return matched_report, exceptions_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erp", default="data/erp_ledger.csv")
    parser.add_argument("--bank", default="data/bank_statement.csv")
    parser.add_argument("--report", default="data/reconciled_report.csv")
    parser.add_argument("--exceptions", default="data/exceptions_list.csv")
    parser.add_argument(
        "--provider",
        choices=("gemini", "groq"),
        default="gemini",
    )
    parser.add_argument("--model")
    parser.add_argument("--max-candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    args = parser.parse_args()

    matched, exceptions = reconcile(
        args.erp,
        args.bank,
        args.report,
        args.exceptions,
        provider=args.provider,
        model=args.model,
        max_candidates_per_erp=args.max_candidates,
    )
    print(f"Matched pairs: {len(matched)}")
    print(f"Exceptions: {len(exceptions)}")


if __name__ == "__main__":
    main()
