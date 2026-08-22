from pathlib import Path

import boto3

from .config import get_settings


class ObjectStorage:
    def __init__(self) -> None:
        settings = get_settings()
        self.bucket = settings.s3_bucket
        self.client = boto3.client("s3", region_name=settings.aws_region)

    def upload(self, local_path: Path, key: str) -> str:
        self.client.upload_file(
            str(local_path),
            self.bucket,
            key,
            ExtraArgs={"ServerSideEncryption": "AES256"},
        )
        return f"s3://{self.bucket}/{key}"

    def download(self, key: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(local_path))

    def presigned_upload(self, key: str, content_type: str, expires: int = 900) -> str:
        return self.client.generate_presigned_url(
            "put_object",
            Params={"Bucket": self.bucket, "Key": key, "ContentType": content_type},
            ExpiresIn=expires,
        )
