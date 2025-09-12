from typing import List, Optional, Tuple
import os
import logging
import pathlib

import boto3
from botocore.config import Config
from botocore import UNSIGNED

logger = logging.getLogger(__name__)


# ---------- S3 helpers ----------

def _parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """
    s3://bucket/prefix  -> ('bucket', 'prefix')
    s3://bucket         -> ('bucket', '')
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    rest = s3_uri[5:]
    if "/" in rest:
        bucket, prefix = rest.split("/", 1)
    else:
        bucket, prefix = rest, ""
    return bucket, prefix.rstrip("/")


def _boto3_s3_client_from_args(s3_sync_args: Optional[List[str]] = None):
    """
    기존 코드의 s3_sync_args=["--no-sign-request"] 의미를 존중해
    boto3에서 anonymous/unsigned 접근을 선택적으로 사용합니다.
    """
    s3_sync_args = s3_sync_args or []
    unsigned = "--no-sign-request" in s3_sync_args
    if unsigned:
        return boto3.client("s3", config=Config(signature_version=UNSIGNED))
    return boto3.client("s3")


def _ensure_parent_dir(path: str):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


# ---------- Public APIs (기존 시그니처 유지) ----------

def get_hash_from_bucket(
    bucket_uri: str, s3_sync_args: Optional[List[str]] = None
) -> str:
    """
    과거: awsv2 s3 cp s3://.../refs/main .
    현재: boto3 get_object로 'refs/main' 내용(커밋 해시)을 로컬 파일 ./main 으로 저장.
    """
    bucket, prefix = _parse_s3_uri(bucket_uri)
    key = f"{prefix}/refs/main" if prefix else "refs/main"

    s3 = _boto3_s3_client_from_args(s3_sync_args)
    resp = s3.get_object(Bucket=bucket, Key=key)
    body = resp["Body"].read().decode("utf-8").strip()

    with open("main", "w") as f:
        f.write(body)

    return body


def get_checkpoint_and_refs_dir(
    model_id: str,
    bucket_uri: str,
    s3_sync_args: Optional[List[str]] = None,
    mkdir: bool = False,
) -> Tuple[str, str]:
    """
    TRANSFORMERS_CACHE 아래에 HF 캐시 레이아웃을 유지합니다.
    refs/main 의 해시를 읽어 snapshots/<hash> 경로를 만들고(옵션),
    해당 경로를 반환합니다.
    """
    from transformers.utils.hub import TRANSFORMERS_CACHE

    f_hash = get_hash_from_bucket(bucket_uri, s3_sync_args)

    path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
    refs_dir = os.path.join(path, "refs")
    checkpoint_dir = os.path.join(path, "snapshots", f_hash)

    if mkdir:
        os.makedirs(refs_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir, refs_dir


def get_download_path(model_id: str):
    from transformers.utils.hub import TRANSFORMERS_CACHE
    path = os.path.join(TRANSFORMERS_CACHE, f"models--{model_id.replace('/', '--')}")
    return path


def download_model(
    model_id: str,
    bucket_uri: str,
    s3_sync_args: Optional[List[str]] = None,
    tokenizer_only: bool = False,
) -> None:
    """
    S3에서 모델 디렉터리 전체(또는 토크나이저 관련 파일만)를
    TRANSFORMERS_CACHE/models--<model_id> 로 동기화합니다.
    """
    path = get_download_path(model_id)
    os.makedirs(path, exist_ok=True)

    bucket, prefix = _parse_s3_uri(bucket_uri)
    s3 = _boto3_s3_client_from_args(s3_sync_args)

    paginator = s3.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix + "/"

    for page in paginator.paginate(**kwargs):
        contents = page.get("Contents") or []
        for obj in contents:
            key = obj["Key"]
            # 디렉터리 placeholder 무시
            if key.endswith("/"):
                continue

            # prefix 이후 상대 경로 계산
            rel = key[len(prefix) + 1 :] if prefix else key
            if tokenizer_only and ("token" not in rel.lower()):
                continue

            dest = os.path.join(path, rel)
            _ensure_parent_dir(dest)
            s3.download_file(bucket, key, dest)

    logger.info("Model downloaded to %s", path)


def get_mirror_link(model_id: str) -> str:
    """
    (기본값) 미러 S3 경로를 구성합니다.
    필요 시 ray_finetune_llm_deepspeed.py에서 --model-s3-uri로 override 가능.
    """
    return f"s3://llama-2-weights/models--{model_id.replace('/', '--')}"