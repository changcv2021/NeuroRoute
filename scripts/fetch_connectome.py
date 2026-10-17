#!/usr/bin/env python3
"""Fetch only pinned analysis inputs; run bulk transfers inside an allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess


def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("manifests/connectome_manifest.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/connectome_banc_v888.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    wanted = {config[k] for k in ("metadata_file", "edges_file", "cluster_file")}
    wanted |= {"supplemental_data_1.txt", "influence_README.md"}
    files = [f for f in manifest["files"] if f["filename"] in wanted]
    if {f["filename"] for f in files} != wanted:
        raise ValueError("Manifest does not contain all exact required filenames")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for item in files:
        target = args.data_dir / item["filename"]
        algorithm = item["checksum"]["type"].lower()
        if item["filesize"] > 1000000 and (not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login")):
            raise RuntimeError("Bulk data transfer and hashing require a compute allocation")
        if not target.exists():
            partial = target.with_suffix(target.suffix + ".partial")
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "30", "--max-time", "900", "--retry", "3",
                "--max-filesize", str(item["filesize"]), item["url"], "--output", str(partial)], check=True)
            if partial.stat().st_size != item["filesize"] or digest(partial, algorithm) != item["checksum"]["value"]:
                raise ValueError(f"Download checksum/size mismatch: {partial}")
            partial.replace(target)
        if target.stat().st_size != item["filesize"] or digest(target, algorithm) != item["checksum"]["value"]:
            raise ValueError(f"Existing file checksum/size mismatch: {target}")
        item.update(status="verified", path=str(target.resolve()), sha256=digest(target, "sha256"))
        print(f"verified {item['filename']} {item['filesize']} bytes", flush=True)
    manifest["acquisition_job_id"] = os.environ.get("SLURM_JOB_ID")
    manifest["prior_status"] = "pending_extraction_and_sensitivity_audit"
    (args.data_dir / "acquisition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
