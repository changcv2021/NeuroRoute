#!/usr/bin/env python3
"""Check the explicit code-only export without importing scientific packages."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    "data", "artifacts", "outputs", "runs", "results", "results_summary", "reports",
    "logs", "checkpoints", "training_states", "videos", "figures", "paper_csv",
    "implementation_audit",
}
ALLOWED_SUFFIXES = {".py", ".json", ".toml", ".md", ".txt", ".sh", ".yml", ".in", ".lock"}
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
    re.compile(r"AKIA[A-Z0-9]{16}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def main():
    manifest = json.loads((ROOT / "manifests/source.json").read_text())
    expected = set(manifest["published_files"])
    errors = []
    counts = {"python": 0, "json": 0, "toml": 0}
    byte_count = 0
    git = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=False)
    if git.returncode == 0 and git.stdout:
        tracked = set(git.stdout.decode().rstrip("\0").split("\0"))
        for name in sorted(tracked - expected):
            errors.append(f"Unapproved tracked file: {name}")
        for name in sorted(expected - tracked):
            errors.append(f"Expected file not tracked: {name}")
    for name in sorted(expected):
        relative = Path(name)
        path = ROOT / relative
        if (relative.is_absolute() or ".." in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(ROOT)):
            errors.append(f"Invalid export path: {name}")
            continue
        if FORBIDDEN_PARTS.intersection(relative.parts):
            errors.append(f"Output directory in export: {name}")
        if name != ".gitignore" and relative.suffix not in ALLOWED_SUFFIXES:
            errors.append(f"Unapproved file type: {name}")
        if not path.is_file():
            errors.append(f"Missing file: {name}")
            continue
        raw = path.read_bytes()
        byte_count += len(raw)
        if len(raw) > 1_000_000:
            errors.append(f"Unexpected large file: {name}")
        expected_hash = manifest["source_sha256"].get(name)
        if expected_hash and hashlib.sha256(raw).hexdigest() != expected_hash:
            errors.append(f"Source checksum changed: {name}")
        try:
            content = raw.decode("utf-8-sig")
            if any(pattern.search(content) for pattern in SECRET_PATTERNS):
                errors.append(f"Possible credential detected in: {name}")
            if relative.suffix == ".py":
                ast.parse(content, filename=name)
                counts["python"] += 1
            elif relative.suffix == ".json":
                json.loads(content)
                counts["json"] += 1
            elif relative.suffix == ".toml":
                tomllib.loads(content)
                counts["toml"] += 1
        except (UnicodeError, ValueError, SyntaxError) as error:
            errors.append(f"Parse failure: {name}: {error}")
    for name in manifest["source_sha256"]:
        if name not in expected:
            errors.append(f"Source absent from export list: {name}")
    for error in errors:
        print(error, file=sys.stderr)
    print(json.dumps(dict(status="FAILED" if errors else "PASSED", files=len(expected),
                         checked_sources=len(manifest["source_sha256"]), bytes=byte_count,
                         parsed=counts, errors=len(errors),
                         scope="Static source-only checks; no GPU/runtime tests"), indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
