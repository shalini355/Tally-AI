import hashlib
import random
import tempfile
import time
from pathlib import Path
from uuid import UUID

import redis
from celery import Task
from sqlalchemy import update

from src.reconcile import reconcile

from .celery_app import celery_app
from .config import get_settings
from .db import SessionLocal
from .models import FailedJob, JobStatus, ReconciliationJob, TransactionAudit
from .storage import ObjectStorage


def _is_retryable_error(error: Exception) -> bool:
    text = str(error).lower()
    return isinstance(error, (ConnectionError, TimeoutError)) or "429" in text or "rate_limit" in text or "quota" in text


class ReconciliationTask(Task):
    autoretry_for = (ConnectionError, TimeoutError)
    max_retries = get_settings().celery_task_max_retries
    retry_backoff = True
    retry_backoff_max = get_settings().celery_task_backoff_max
    retry_jitter = True

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = UUID(str(kwargs["job_id"]))
        payload = {"job_id": str(job_id), "input_key": kwargs["input_key"], "bank_key": kwargs.get("bank_key", "")}
        with SessionLocal.begin() as session:
            session.execute(
                update(ReconciliationJob)
                .where(ReconciliationJob.id == job_id)
                .values(status=JobStatus.FAILED, error_message=str(exc))
            )
            session.add(
                FailedJob(
                    job_id=job_id,
                    payload=payload,
                    error_message=str(exc),
                    retry_count=self.request.retries,
                )
            )


@celery_app.task(bind=True, base=ReconciliationTask, name="tally_ai.reconcile")
def reconcile_job(self, *, job_id: str, input_key: str, bank_key: str) -> dict:
    settings = get_settings()
    lock_key = f"tally-ai:idempotency:{job_id}"
    lock = redis.Redis.from_url(settings.redis_url).lock(
        lock_key, timeout=settings.idempotency_lock_seconds, blocking=False
    )
    if not lock.acquire():
        return {"job_id": job_id, "status": "already_processing"}

    try:
        with SessionLocal.begin() as session:
            job = session.get(ReconciliationJob, UUID(job_id))
            if job is None:
                raise ValueError(f"Unknown job: {job_id}")
            if job.status == JobStatus.SUCCESS:
                return {"job_id": job_id, "status": JobStatus.SUCCESS.value, "result": job.result}
            job.status = JobStatus.PROCESSING

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            erp_path = root / "erp_ledger.csv"
            bank_path = root / "bank_statement.csv"
            report_path = root / "reconciled_report.csv"
            exceptions_path = root / "exceptions_list.csv"
            storage = ObjectStorage()
            storage.download(input_key, erp_path)
            storage.download(bank_key, bank_path)

            started = time.perf_counter()
            matched, exceptions = reconcile(
                erp_path, bank_path, report_path, exceptions_path, llm_workers=8
            )
            duration_ms = int((time.perf_counter() - started) * 1000)
            output_prefix = input_key.rsplit("/", 1)[0] + "/"
            result_key = output_prefix + "reconciled_report.csv"
            exception_key = output_prefix + "exceptions_list.csv"
            result_uri = storage.upload(report_path, result_key)
            exception_uri = storage.upload(exceptions_path, exception_key)

        result = {
            "matched_count": len(matched),
            "exception_count": len(exceptions),
            "report_uri": result_uri,
            "exceptions_uri": exception_uri,
            "duration_ms": duration_ms,
        }
        with SessionLocal.begin() as session:
            for row in matched.to_dict("records"):
                transaction_hash = hashlib.sha256(
                    f"{row['erp_id']}:{row['bank_ref']}".encode("utf-8")
                ).hexdigest()
                session.add(
                    TransactionAudit(
                        job_id=UUID(job_id),
                        transaction_hash=transaction_hash,
                        stage="reconciliation",
                        decision={
                            "match_type": row.get("match_type"),
                            "confidence_score": row.get("confidence_score"),
                            "reasoning": row.get("reasoning"),
                        },
                        duration_ms=duration_ms,
                    )
                )
            session.execute(
                update(ReconciliationJob)
                .where(ReconciliationJob.id == UUID(job_id))
                .values(status=JobStatus.SUCCESS, result=result)
            )
        return {"job_id": job_id, "status": JobStatus.SUCCESS.value, "result": result}
    except Exception as exc:
        if _is_retryable_error(exc) and self.request.retries < self.max_retries:
            base_delay = min(2 ** (self.request.retries + 1), settings.celery_task_backoff_max)
            raise self.retry(exc=exc, countdown=base_delay + random.uniform(0, base_delay))
        raise
    finally:
        try:
            lock.release()
        except redis.exceptions.LockError:
            pass
