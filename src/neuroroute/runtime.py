"""Execution guards and small provenance utilities."""
import hashlib
import json
import os
from pathlib import Path
import socket


def require_compute(gpu=False):
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Run experiments through Slurm on a compute node.")
    if gpu:
        import jax
        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("A GPU allocation and functioning JAX GPU backend are required.")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

