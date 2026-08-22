from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID, uuid4

import redis
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from .config import get_settings
from .db import Base, engine, get_db
from .models import JobStatus, ReconciliationJob
from .storage import ObjectStorage
from .tasks import reconcile_job

app = FastAPI(title="Tally AI API", version="1.0.0")


class JobKeys(BaseModel):
    erp_key: str
    bank_key: str
    user_id: str = "anonymous"


@app.on_event("startup")
def create_tables() -> None:
    """Create local-development tables; production must use migrations."""
    Base.metadata.create_all(bind=engine)


@app.get("/health")
def health() -> dict[str, str]:
    """Report API health without hiding dependency failures."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        redis.Redis.from_url(get_settings().redis_url).ping()
    except Exception as error:
        raise HTTPException(status_code=503, detail="A required dependency is unavailable") from error
    return {"status": "ok"}


@app.post("/v1/uploads/presign")
def presign_upload(filename: str, user_id: str = "anonymous") -> dict[str, str]:
    if filename not in {"erp_ledger.csv", "bank_statement.csv"}:
        raise HTTPException(status_code=400, detail="Unsupported filename")
    key = f"uploads/{user_id}/{uuid4()}/{filename}"
    return {"key": key, "upload_url": ObjectStorage().presigned_upload(key, "text/csv")}


@app.post("/v1/reconciliation-jobs/from-keys", status_code=status.HTTP_202_ACCEPTED)
def create_job_from_keys(payload: JobKeys, db: Session = Depends(get_db)) -> dict[str, str]:
    if not payload.erp_key.endswith("/erp_ledger.csv") or not payload.bank_key.endswith("/bank_statement.csv"):
        raise HTTPException(status_code=400, detail="Invalid input object keys")
    job_id = uuid4()
    job = ReconciliationJob(
        id=job_id,
        user_id=payload.user_id,
        input_key=payload.erp_key,
        bank_key=payload.bank_key,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    task = reconcile_job.apply_async(kwargs={"job_id": str(job_id), "input_key": payload.erp_key, "bank_key": payload.bank_key})
    job.celery_task_id = task.id
    db.commit()
    return {"task_id": str(job_id), "job_id": str(job_id), "status": JobStatus.PENDING.value}


@app.post("/v1/reconciliation-jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    erp_file: UploadFile = File(...),
    bank_file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict[str, str]:
    settings = get_settings()
    if erp_file.filename != "erp_ledger.csv" or bank_file.filename != "bank_statement.csv":
        raise HTTPException(status_code=400, detail="Expected erp_ledger.csv and bank_statement.csv")

    job_id = uuid4()
    prefix = f"jobs/{job_id}/"
    storage = ObjectStorage()
    with TemporaryDirectory() as directory:
        erp_path = Path(directory) / erp_file.filename
        bank_path = Path(directory) / bank_file.filename
        for upload, destination in ((erp_file, erp_path), (bank_file, bank_path)):
            size = 0
            with destination.open("wb") as output:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(status_code=413, detail="File exceeds upload size limit")
                    output.write(chunk)
        storage.upload(erp_path, prefix + erp_file.filename)
        storage.upload(bank_path, prefix + bank_file.filename)

    input_key = prefix + erp_file.filename
    job = ReconciliationJob(id=job_id, input_key=input_key, bank_key=prefix + bank_file.filename, status=JobStatus.PENDING)
    db.add(job)
    db.commit()
    task = reconcile_job.apply_async(kwargs={"job_id": str(job_id), "input_key": input_key, "bank_key": prefix + bank_file.filename})
    job.celery_task_id = task.id
    db.commit()
    return {"job_id": str(job_id), "status": JobStatus.PENDING.value}


@app.get("/v1/reconciliation-jobs/{job_id}")
def get_job(job_id: UUID, db: Session = Depends(get_db)) -> dict:
    job = db.get(ReconciliationJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "result": job.result,
        "error": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
