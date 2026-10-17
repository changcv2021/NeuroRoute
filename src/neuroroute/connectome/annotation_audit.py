"""Summarize real annotation labels without presuming boolean encodings."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import socket

from .extract import clean, select_neurons, sha256


def main():
    import pyarrow.feather as feather
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/connectome_banc_v888.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Full annotation processing requires a compute allocation")
    cfg = json.loads(args.config.read_text())
    columns = [cfg[k] for k in ("id_column", "type_column", "role_column", "sensory_body_column", "motor_body_column")]
    quality = ["proofread", "roughly_proofread", "status"]
    rows = feather.read_table(args.data_dir / cfg["metadata_file"], columns=columns + quality).to_pylist()
    with (args.data_dir / cfg["cluster_file"]).open() as stream:
        clusters = list(csv.DictReader(stream))
    selected, coverage = select_neurons(rows, clusters, cfg)
    selected_ids = {r["root_id"] for r in selected}
    counts, tag_counts, intersections = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    for row in rows:
        role = clean(row[cfg["role_column"]])
        scope = "selected" if str(row[cfg["id_column"]]) in selected_ids else "excluded"
        if role not in {"sensory", "motor", "ascending", "descending", "sensory_ascending", "sensory_descending"}:
            continue
        for field in quality:
            counts[f"{scope}:{role}:{field}"][clean(row[field])] += 1
        combo = tuple(clean(row[field]) for field in quality)
        intersections[f"{scope}:{role}"][combo] += 1
        for field in (cfg["sensory_body_column"], cfg["motor_body_column"]):
            value = clean(row[field])
            if "," in value:
                tokens = sorted({t.strip() for t in value.split(",")})
                matched_legs = sorted(set(tokens) & set(cfg["body_groups"]))
                tag_counts[f"{scope}:{role}:{field}"][(value, tuple(matched_legs))] += 1
    output = {"status": "observed_values_not_inferred_semantics", "metadata_sha256": sha256(args.data_dir / cfg["metadata_file"]),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "row_count": len(rows),
        "field_values_by_scope_role": {k: dict(v) for k, v in sorted(counts.items())},
        "joint_quality_values": {k: [{"proofread": t[0], "roughly_proofread": t[1], "status": t[2], "count": n}
                                    for t, n in sorted(v.items())] for k, v in sorted(intersections.items())},
        "multi_body_tags": {k: [{"raw_value": t[0], "leg_tokens": list(t[1]), "count": n}
                              for t, n in sorted(v.items())] for k, v in sorted(tag_counts.items())},
        "base_selection_coverage": coverage}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": len(rows), "selected": len(selected)}))


if __name__ == "__main__":
    main()
