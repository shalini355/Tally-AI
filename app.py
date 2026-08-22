"""Streamlit dashboard for Tally AI reconciliation with full audit metrics."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

try:
    from src.reconcile import reconcile
except ImportError:
    from reconcile import reconcile


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATASET2_DIR = DATA_DIR / "dataset_2"

st.set_page_config(
    page_title="Tally AI - Reconciliation Dashboard",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetric"] {
        border-left: 4px solid #1f7a8c;
        background-color: var(--secondary-background-color);
        padding: 0.8rem 1rem;
        border-radius: 6px;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-color);
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: var(--text-color);
        margin-bottom: 1.5rem;
    }
    .section-card {
        background-color: var(--secondary-background-color);
        padding: 1.2rem;
        border-radius: 8px;
        border: 1px solid var(--secondary-background-color);
        margin-bottom: 1rem;
    }
    .exception-caption { color: #ef4444; font-weight: 700; margin: 0.5rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _load_dataset_files(data_dir: Path) -> dict[str, Any]:
    matched_path = data_dir / "reconciled_report.csv"
    exceptions_path = data_dir / "exceptions_list.csv"
    metrics_path = data_dir / "reconciliation_metrics.json"
    summary_path = data_dir / "evaluation_summary.json"
    gt_path = data_dir / ".ground_truth_mappings.json"

    matched = pd.read_csv(matched_path) if matched_path.exists() else pd.DataFrame()
    exceptions = pd.read_csv(exceptions_path) if exceptions_path.exists() else pd.DataFrame()

    metrics = {}
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            metrics = json.load(f)

    summary = {}
    if summary_path.exists():
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)

    total_records = len(matched)
    if gt_path.exists():
        with gt_path.open(encoding="utf-8") as f:
            truth = json.load(f)
        total_records = int(truth.get("record_counts", {}).get("erp", len(matched)))
    elif (data_dir / "erp_ledger.csv").exists():
        total_records = len(pd.read_csv(data_dir / "erp_ledger.csv"))

    return {
        "matched": matched,
        "exceptions": exceptions,
        "metrics": metrics,
        "summary": summary,
        "total_records": total_records,
    }


def _run_uploaded_reconciliation(erp_upload, bank_upload, provider: str, model: str | None) -> None:
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_path = Path(temporary_dir)
        erp_path = temporary_path / "erp_ledger.csv"
        bank_path = temporary_path / "bank_statement.csv"
        report_path = temporary_path / "reconciled_report.csv"
        exceptions_path = temporary_path / "exceptions_list.csv"
        erp_path.write_bytes(erp_upload.getvalue())
        bank_path.write_bytes(bank_upload.getvalue())

        started_at = time.perf_counter()
        with st.spinner("Running deterministic & LLM reconciliation pipeline..."):
            reconcile(
                erp_path,
                bank_path,
                report_path,
                exceptions_path,
                provider=provider,
                model=model,
            )
        throughput = time.perf_counter() - started_at

        data = _load_dataset_files(temporary_path)
        st.session_state["uploaded_matched"] = data["matched"]
        st.session_state["uploaded_exceptions"] = data["exceptions"]
        st.session_state["uploaded_metrics"] = data["metrics"]
        st.session_state["uploaded_summary"] = data["summary"]
        st.session_state["uploaded_throughput"] = throughput
        st.session_state["uploaded_total"] = len(pd.read_csv(erp_path))


# Main Header
st.markdown('<div class="main-header">Tally AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Multi-Source Verification • Deterministic & LLM Pipeline • Honest Exception Audit</div>',
    unsafe_allow_html=True,
)

# Sidebar setup
with st.sidebar:
    st.header("⚙️ Data Source")
    dataset_choice = st.radio(
        "Select Dataset View",
        options=["Primary Dataset 1", "Generalization Dataset 2", "Live Upload / Test"],
        index=0,
    )

    st.divider()
    st.subheader("🚀 Live Reconciliation Run")
    erp_upload = st.file_uploader("Upload ERP Ledger (CSV)", type="csv", key="erp_upload")
    bank_upload = st.file_uploader("Upload Bank Statement (CSV)", type="csv", key="bank_upload")
    provider = st.selectbox("LLM Provider", ["groq", "gemini"], index=0)
    model = st.text_input("Model Override", placeholder="Default provider model")

    run_live = st.button(
        "Run Reconciliation Pipeline",
        type="primary",
        disabled=erp_upload is None or bank_upload is None,
        use_container_width=True,
    )

    if run_live and erp_upload is not None and bank_upload is not None:
        try:
            _run_uploaded_reconciliation(erp_upload, bank_upload, provider, model or None)
            st.success("Reconciliation complete!")
            st.session_state["view_mode"] = "uploaded"
        except Exception as error:
            st.error(f"Reconciliation failed: {error}")

# Determine active dataset to display
if dataset_choice == "Live Upload / Test" and "uploaded_matched" in st.session_state:
    matched_df = st.session_state["uploaded_matched"]
    exceptions_df = st.session_state["uploaded_exceptions"]
    metrics_data = st.session_state.get("uploaded_metrics", {})
    summary_data = st.session_state.get("uploaded_summary", {})
    total_records = st.session_state.get("uploaded_total", len(matched_df))
    active_dataset_label = "Live Upload"
elif dataset_choice == "Generalization Dataset 2":
    ds_data = _load_dataset_files(DATASET2_DIR)
    matched_df = ds_data["matched"]
    exceptions_df = ds_data["exceptions"]
    metrics_data = ds_data["metrics"]
    summary_data = ds_data["summary"]
    total_records = ds_data["total_records"]
    active_dataset_label = "Generalization Dataset 2 (Seed 20260823)"
else:
    ds_data = _load_dataset_files(DATA_DIR)
    matched_df = ds_data["matched"]
    exceptions_df = ds_data["exceptions"]
    metrics_data = ds_data["metrics"]
    summary_data = ds_data["summary"]
    total_records = ds_data["total_records"]
    active_dataset_label = "Primary Dataset 1 (Seed 20260822)"

# Top Level Key Metrics
match_count = len(matched_df)
exception_count = len(exceptions_df)
overall_match_rate = (match_count / max(total_records, 1)) * 100
total_throughput_time = metrics_data.get("total_time_seconds", 0.0)

metric_cols = st.columns(4)
metric_cols[0].metric("Records Processed", total_records, help="Total ERP records evaluated")
metric_cols[1].metric("Overall Match Rate", f"{overall_match_rate:.1f}%", help="Percentage of records successfully reconciled")
metric_cols[2].metric("Throughput Speed", f"{total_throughput_time:.2f} s", help="Total pipeline execution time")
metric_cols[3].metric("Unresolved Exceptions", exception_count, help="Flagged records requiring human audit")

st.divider()

# Section 1: Category-Specific Precision / Recall / F1 & Throughput Breakdown
col_perf, col_through = st.columns([1.1, 0.9])

with col_perf:
    st.subheader("🎯 Measured Accuracy & Category PRF1")
    cat_metrics = summary_data.get("category_metrics", {})
    if cat_metrics:
        prf1_rows = []
        for cat_name, values in cat_metrics.items():
            prf1_rows.append(
                {
                    "Category": cat_name,
                    "Precision (%)": values.get("precision", 0.0),
                    "Recall (%)": values.get("recall", 0.0),
                    "F1 Score": values.get("f1", 0.0),
                    "True Positives": values.get("tp", 0),
                    "False Positives": values.get("fp", 0),
                }
            )
        prf1_df = pd.DataFrame(prf1_rows)
        st.dataframe(prf1_df, use_container_width=True, hide_index=True)
    else:
        st.info("Run evaluation script to view category-level Precision, Recall, and F1 scores.")

with col_through:
    st.subheader("⚡ Throughput Stage Breakdown")
    det_count = metrics_data.get("deterministic_matches_count", 0)
    det_pct = metrics_data.get("deterministic_match_percent", 0.0)
    det_time = metrics_data.get("deterministic_time_seconds", 0.0)

    llm_count = metrics_data.get("llm_matches_count", 0)
    llm_pct = metrics_data.get("llm_match_percent", 0.0)
    llm_time = metrics_data.get("llm_time_seconds", 0.0)
    llm_calls = metrics_data.get("llm_calls_made", 0)

    t_col1, t_col2 = st.columns(2)
    t_col1.metric("Deterministic Fast Pass", f"{det_count} ({det_pct:.1f}%)", f"{det_time:.4f} s")
    t_col2.metric("LLM Matcher Pass", f"{llm_count} ({llm_pct:.1f}%)", f"{llm_time:.2f} s ({llm_calls} calls)")

    # Chart for stage throughput
    throughput_df = pd.DataFrame(
        [
            {"Stage": "Deterministic Pass", "Resolved Matches": det_count, "Time (s)": det_time},
            {"Stage": "LLM Matcher Pass", "Resolved Matches": llm_count, "Time (s)": llm_time},
        ]
    )
    chart = (
        alt.Chart(throughput_df)
        .mark_bar()
        .encode(
            x=alt.X("Stage:N", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Resolved Matches:Q"),
            color=alt.Color("Stage:N", scale=alt.Scale(range=["#0284c7", "#0d9488"])),
            tooltip=["Stage", "Resolved Matches", "Time (s)"],
        )
        .properties(height=180)
    )
    st.altair_chart(chart, use_container_width=True)

st.divider()

# Section 2: Categorized Honest Exceptions & Confidence Calibration
col_exc, col_cal = st.columns([1, 1])

with col_exc:
    st.subheader("⚠️ Categorized Honest Exceptions")
    st.caption("Every unresolved record is tagged with a precise audit reason")
    cat_exceptions = metrics_data.get("categorized_exceptions", {})
    if cat_exceptions:
        exc_df = pd.DataFrame(
            [{"Exception Reason Category": k, "Record Count": v} for k, v in cat_exceptions.items()]
        )
        exc_chart = (
            alt.Chart(exc_df)
            .mark_bar(color="#e11d48")
            .encode(
                x=alt.X("Exception Reason Category:N", axis=alt.Axis(labelAngle=-20)),
                y=alt.Y("Record Count:Q"),
                tooltip=["Exception Reason Category", "Record Count"],
            )
            .properties(height=220)
        )
        st.altair_chart(exc_chart, use_container_width=True)
    else:
        st.info("No exception categorization data available.")

with col_cal:
    st.subheader("📊 LLM Confidence Calibration")
    st.caption("Verification accuracy by confidence bucket to justify 0.80 threshold")
    calibration_data = summary_data.get("confidence_calibration", {})
    if calibration_data:
        cal_rows = []
        for bucket, b_data in calibration_data.items():
            cal_rows.append(
                {
                    "Confidence Bucket": bucket,
                    "Evaluations": b_data.get("total", 0),
                    "Correct": b_data.get("correct", 0),
                    "Incorrect": b_data.get("incorrect", 0),
                    "Accuracy (%)": b_data.get("accuracy", 0.0),
                }
            )
        cal_df = pd.DataFrame(cal_rows)
        cal_chart = (
            alt.Chart(cal_df)
            .mark_bar()
            .encode(
                x=alt.X("Confidence Bucket:N", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Accuracy (%):Q", scale=alt.Scale(domain=[0, 100])),
                color=alt.value("#22c55e"),
                tooltip=["Confidence Bucket", "Evaluations", "Correct", "Accuracy (%)"],
            )
            .properties(height=220)
        )
        st.altair_chart(cal_chart, use_container_width=True)
    else:
        st.info("Run evaluation script to generate confidence calibration buckets.")

st.divider()

# Section 3: Data Tables
st.subheader("📋 Reconciled Transactions Log")
if matched_df.empty:
    st.info("No reconciled transactions available.")
else:
    st.dataframe(
        matched_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence_score": st.column_config.ProgressColumn(
                "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
            )
        },
    )

st.subheader("🚨 Unresolved Exception List (Accountant Review Required)")
if exceptions_df.empty:
    st.success("No unresolved exceptions! All records successfully matched.")
else:
    st.dataframe(
        exceptions_df,
        use_container_width=True,
        hide_index=True,
    )
