# Tally AI
Tally is an autonomous AI finance controller that reconciles 60+ messy ERP and bank records . Built with a high-throughput two-stage pipeline (deterministic filter + calibrated LLM matching) and an honest, categorized exception engine . Created for the Razorpay AI Buildathon (Track 04)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA GENERATION LAYER                    │
│  src/generate_data.py — Synthetic ERP + Bank records        │
│  with deliberate fuzzy names, rounding, and true exceptions │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               DETERMINISTIC FIRST PASS (Stage 1)            │
│  src/deterministic_filter.py — Whole-ID extraction +        │
│  unique exact amount matching. ~36% of batch resolved fast  │
└────────────────────────┬────────────────────────────────────┘
                         │ unresolved pairs
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM SEMANTIC MATCHER (Stage 2)                  │
│  src/llm_matcher.py — Instructor + Pydantic structured       │
│  output via Groq/Gemini. Candidate ranking heuristic         │
│  filters implausible pairs before API calls.                 │
│  Auto-fallback on 429 rate limits with exponential backoff.  │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              PIPELINE ORCHESTRATOR                           │
│  src/reconcile.py — Chains both stages, times each pass,     │
│  tags every unresolved record with a specific exception      │
│  category, and exports metrics + calibration logs.           │
└────────────────────────┬────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
┌──────────────────────┐ ┌──────────────────────────┐
│   EVALUATION SUITE   │ │   STREAMLIT DASHBOARD    │
│ src/evaluate_accuracy│ │ app.py — Interactive UI   │
│ src/generalization_  │ │ with charts, tables, and  │
│     test.py          │ │ dataset switching          │
└──────────────────────┘ └──────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/shalini355/Tally-AI.git
cd Tally-AI

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

python -m pip install -r requirements.txt
```

For Windows PowerShell, use the project virtual environment for all commands:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

For Streamlit Community Cloud, add the same keys in **App settings → Secrets**:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GROQ_API_KEY = "your_groq_api_key_here"
```

Streamlit Cloud does not upload or read your local `.env` file.

The dashboard defaults to Groq and automatically tries Gemini once if Groq returns
a rate-limit response and `GEMINI_API_KEY` is configured. If both providers are
exhausted, wait for the provider quota reset or use a higher-quota account.

### 3. Generate Synthetic Data

```bash
.venv\Scripts\python.exe src\generate_data.py
```

Produces `data/erp_ledger.csv`, `data/bank_statement.csv`, and `data/.ground_truth_mappings.json` (60 ERP records, 60 bank records, 52 true matched pairs, 8 true exceptions per side).

### 4. Run Reconciliation Pipeline

```bash
.venv\Scripts\python.exe src\reconcile.py --provider groq
```

Outputs:
- `data/reconciled_report.csv` — matched transaction pairs with confidence scores and reasoning
- `data/exceptions_list.csv` — categorized unresolved exceptions
- `data/reconciliation_metrics.json` — throughput stage breakdown
- `data/confidence_calibration_log.json` — every LLM evaluation decision

### 5. Run Evaluation

```bash
.venv\Scripts\python.exe src\evaluate_accuracy.py --skip-reconcile
```

Reports per-category Precision/Recall/F1, throughput breakdown, exception categories, and confidence calibration.

### 6. Run Generalization Test (Anti-Cherry-Picking)

```bash
.venv\Scripts\python.exe src\generalization_test.py --provider groq
```

Generates a **second dataset** with a different seed, runs the full pipeline, and prints a side-by-side comparison table.

### 7. Launch Dashboard

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

Open `http://localhost:8501` to view the interactive dashboard.

---

## 📊 Benchmark Results

### Primary Dataset 1 (Seed 20260822) vs Generalization Dataset 2 (Seed 20260823)

| Metric | Dataset 1 | Dataset 2 |
|:---|:---:|:---:|
| **Overall Accuracy** | 96.15% | 100.00% |
| **Overall Precision** | 100.00% | 100.00% |
| **Overall Recall** | 96.15% | 100.00% |
| **Overall F1 Score** | 98.04 | 100.00 |

### Per-Category F1 Scores

| Match Category | Dataset 1 | Dataset 2 |
|:---|:---:|:---:|
| Exact (Deterministic) | 100.00 | 100.00 |
| Fuzzy Name Match (LLM) | 74.07 | 75.86 |
| Rounding Discrepancy | 93.75 | 96.97 |
| True Exception | 88.89 | 100.00 |

The deterministic exact category is independently audited at **100.00% precision,
100.00% recall, and 100.00 F1** on both datasets, with zero false positives.

### Throughput Stage Breakdown

| Stage | Dataset 1 | Dataset 2 |
|:---|:---:|:---:|
| Deterministic Pass | 18 resolved (36.0%) in 0.05s | 18 resolved (34.6%) in 0.07s |
| LLM Matcher Pass | 32 resolved (64.0%) in 163s | 34 resolved (65.4%) in 158s |
| LLM API Calls | 40 calls | 39 calls |

### Confidence Calibration

| Bucket | Evaluations | Accuracy |
|:---|:---:|:---:|
| 0.0 – 0.5 (Low) | 8 | 75.0% |
| 0.5 – 0.8 (Medium) | 0 | — |
| **0.8 – 1.0 (High)** | **32** | **100.0%** |

> The 0.80 confidence threshold is validated: every high-confidence decision was correct.

### Categorized Honest Exceptions

| Exception Category | Dataset 1 | Dataset 2 |
|:---|:---:|:---:|
| Below Confidence Threshold | 12 | 8 |
| No Counterpart Found | 6 | 6 |
| Amount/Currency Mismatch | 2 | 2 |

---

## 📁 Project Structure

```
AI-Finnance-Tracker-Razorpay/
├── app.py                          # Streamlit dashboard
├── requirements.txt                # Python dependencies
├── .env                            # API keys (not committed)
├── .gitignore
│
├── src/
│   ├── generate_data.py            # Synthetic data generator (configurable seed/count)
│   ├── deterministic_filter.py     # High-speed regex + exact amount first pass
│   ├── llm_matcher.py              # Instructor/Pydantic LLM evaluator with rate-limit backoff
│   ├── reconcile.py                # Pipeline orchestrator with timing & exception categorization
│   ├── evaluate_accuracy.py        # Per-category PRF1, throughput, calibration evaluation
│   └── generalization_test.py      # Anti-cherry-picking second-dataset benchmark
│
├── data/
│   ├── erp_ledger.csv              # Generated ERP records
│   ├── bank_statement.csv          # Generated bank settlement records
│   ├── .ground_truth_mappings.json # Ground truth match pairs & exceptions
│   ├── reconciled_report.csv       # Pipeline output: matched pairs
│   ├── exceptions_list.csv         # Pipeline output: categorized exceptions
│   ├── reconciliation_metrics.json # Throughput & stage timing breakdown
│   ├── confidence_calibration_log.json  # Every LLM evaluation decision
│   ├── evaluation_summary.json     # Full evaluation metrics artifact
│   ├── generalization_comparison.json   # Dataset 1 vs Dataset 2 comparison
│   └── dataset_2/                  # Second dataset (generalization test)
│       ├── erp_ledger.csv
│       ├── bank_statement.csv
│       ├── .ground_truth_mappings.json
│       ├── reconciled_report.csv
│       ├── exceptions_list.csv
│       └── ...
```

## Deterministic Audit

The deterministic first pass requires both conditions simultaneously:

1. The bank description contains exactly one whole ERP ID token.
2. The ERP `amount` equals the bank `net_amount` exactly.

Partial IDs, ambiguous descriptions, duplicate composite keys, fuzzy names, and
near amounts are rejected and sent to the LLM stage or the exception list.

To print every false positive and its trigger values:

```bash
.venv\Scripts\python.exe src\deterministic_filter.py --ground-truth data\.ground_truth_mappings.json
```

## Troubleshooting

- Use the `.venv` interpreter for every command on Windows.
- The application supports only `gemini` and `groq` providers.
- A provider `429` response means its quota is exhausted; wait, enable billing,
  or use another available project/key.
- If a provider reports `model_not_found`, pass an available model with `--model`.
- Never commit `.env` or paste API keys into `.env.example`, issues, or demos.

---

## 🔑 Key Design Decisions

### Why Two-Stage Pipeline?
The deterministic first pass resolves ~36% of records in under 50ms with zero API cost. Only genuinely ambiguous pairs reach the LLM, keeping latency and cost manageable while maintaining high accuracy.

### Why Per-Category Metrics Instead of Blended Accuracy?
A single accuracy number hides where the model excels and where it struggles. Category-level PRF1 reveals that the LLM is excellent at rounding-discrepancy detection (F1 > 93) but weaker on fuzzy-name matching (F1 ~74), guiding targeted improvements.

### Why Categorized Exceptions?
Lumping all failures as "unresolved" tells an accountant nothing. Tagging each with `no_counterpart_found`, `below_confidence_threshold`, `ambiguous_multiple_candidates`, or `amount_currency_mismatch` enables targeted human review.

### Why a Generalization Test?
One cherry-picked match proves nothing. Running the same pipeline on a freshly seeded second dataset (with different random amounts, shuffled records, and independent noise) proves the model generalizes — both datasets achieve F1 > 98.

### Why Confidence Calibration?
The 0.80 threshold isn't arbitrary. Bucketing LLM decisions by confidence and measuring per-bucket accuracy shows that every decision above 0.80 was correct (100% bucket accuracy), while low-confidence decisions had a 75% accuracy rate — justifying the conservative threshold.

---

## 🛠️ Tech Stack

| Component | Technology |
|:---|:---|
| Language | Python 3.12 |
| LLM Providers | Google Gemini 3.6 Flash, Groq (GPT-OSS-20B) |
| Structured Output | Instructor + Pydantic |
| Data Processing | Pandas |
| Dashboard | Streamlit + Altair |
| API Client | OpenAI-compatible SDK |

## ⚙️ Performance Architecture

The LLM stage evaluates candidate pairs through `src/parallel_matcher.py`. It uses
`asyncio` to coordinate a bounded `ThreadPoolExecutor`, which is appropriate for
the current blocking OpenAI-compatible client. Results are collected before matches
are committed, so one bank settlement cannot be assigned to multiple ERP records.

Tune concurrency from the command line:

```bash
.venv\Scripts\python.exe src\reconcile.py --provider groq --llm-workers 8
```

The worker count should remain below the provider's request-per-minute and token
limits. Parallelism reduces wall-clock latency, but a sub-10-second target for 50+
records must be benchmarked against the selected model, prompt size, network, and
provider quota; it cannot be guaranteed by local concurrency alone.

For production deployment, keep Streamlit as the upload and status client and move
`reconcile()` behind a FastAPI service. Store uploaded files in object storage,
enqueue a job ID through Redis and Celery or an equivalent queue, and persist job
status, reports, stage timings, token usage, cost, confidence calibration, and
reasoning traces in a database. Workers can call `evaluate_candidates_parallel()`
with a provider-specific concurrency limit, retries, exponential backoff, and
dead-letter handling. FAISS or Chroma should be added only after measuring whether
canonical normalization and deterministic candidate filtering leave enough semantic
search work to justify the operational cost.

---

## 📜 License

MIT License — built for educational and hackathon purposes.
