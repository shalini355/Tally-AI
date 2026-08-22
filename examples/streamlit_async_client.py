"""Non-blocking Streamlit client for the FastAPI job API."""

import requests
import streamlit as st

API_URL = st.secrets.get("TALLY_API_URL", "http://localhost:8000")

st.title("Tally AI")
erp_file = st.file_uploader("ERP ledger", type=["csv"], key="erp")
bank_file = st.file_uploader("Bank statement", type=["csv"], key="bank")

if st.button("Start reconciliation", disabled=not (erp_file and bank_file)):
    keys = {}
    for filename, content in (("erp_ledger.csv", erp_file.getvalue()), ("bank_statement.csv", bank_file.getvalue())):
        presign = requests.post(f"{API_URL}/v1/uploads/presign", params={"filename": filename}, timeout=10)
        presign.raise_for_status()
        upload = presign.json()
        uploaded = requests.put(upload["upload_url"], data=content, headers={"Content-Type": "text/csv"}, timeout=30)
        uploaded.raise_for_status()
        keys[filename] = upload["key"]
    response = requests.post(f"{API_URL}/v1/reconciliation-jobs/from-keys", json={"erp_key": keys["erp_ledger.csv"], "bank_key": keys["bank_statement.csv"]}, timeout=10)
    response.raise_for_status()
    st.session_state["job_id"] = response.json()["task_id"]
    st.rerun()

job_id = st.session_state.get("job_id")
if job_id:
    st.caption(f"Task: {job_id}")
    status_box = st.empty()
    progress = st.progress(0)

    @st.fragment(run_every="3s")
    def poll_job() -> None:
        response = requests.get(f"{API_URL}/v1/reconciliation-jobs/{job_id}", timeout=10)
        response.raise_for_status()
        job = response.json()
        state = job["status"]
        progress_value = {"pending": 10, "processing": 60, "success": 100, "failed": 100}[state]
        progress.progress(progress_value)
        status_box.info(f"Status: {state}") if state in {"pending", "processing"} else status_box.success(f"Status: {state}") if state == "success" else status_box.error(job.get("error") or "Job failed")
        if state == "success":
            st.json(job["result"])

    poll_job()
