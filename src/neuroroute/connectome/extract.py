"""Read pinned BANC inputs and aggregate small comparable directional matrices."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import socket

from .normalize import aggregate_type_balanced


def clean(value):
    if value is None:
        return "unknown"
    result = str(value).strip()
    return "unknown" if result in {"", "NA", "nan", "None", "unknown"} else result


def body_group(value, groups, policy="exact"):
    value = clean(value)
    if policy == "exact":
        return value
    if policy != "unique_leg_token":
        raise ValueError(f"Unknown body-tag policy: {policy}")
    matches = set(part.strip() for part in value.split(",")) & set(groups)
    if len(matches) == 1:
        return next(iter(matches))
    return "ambiguous_leg_annotation" if len(matches) > 1 else value


def select_neurons(records, clusters, config):
    """Use source BANC IDs only; cluster CSV has its own verified ID field."""
    present = {role: {clean(r["super_cluster"]) for r in clusters if r["super_class"] == role}
               for role in ("ascending", "descending")}
    shared = sorted((present["ascending"] & present["descending"]) - {"unknown"})
    cluster_lookup = {str(r["id"]): r for r in clusters}
    if len(cluster_lookup) != len(clusters):
        raise ValueError("Duplicate IDs in published cluster source")
    selection, excluded, seen = [], Counter(), set()
    annotation_counts = Counter()
    for record in records:
        root = str(record[config["id_column"]])
        if root in seen:
            raise ValueError(f"Duplicate BANC root ID {root}")
        seen.add(root)
        role = clean(record[config["role_column"]])
        cell_type = clean(record[config["type_column"]])
        annotation_counts[f"role_total:{role}"] += 1
        if cell_type == "unknown":
            annotation_counts[f"unknown_type:{role}"] += 1
        group, direction, side = "unknown", None, None
        if role in config["sensory_roles"]:
            group = body_group(record[config["sensory_body_column"]], config["body_groups"],
                               config.get("annotation_policy", {}).get("body_tag_policy", "exact"))
            direction, side = "upward", "row"
        elif role == "motor":
            group = body_group(record[config["motor_body_column"]], config["body_groups"],
                               config.get("annotation_policy", {}).get("body_tag_policy", "exact"))
            direction, side = "downward", "column"
        elif role in {"ascending", "descending"}:
            cluster = cluster_lookup.get(root)
            if cluster is not None and cluster["super_class"] != role:
                raise ValueError(f"Role disagreement between pinned annotation files for {root}")
            group = clean(cluster["super_cluster"]) if cluster else "unknown"
            direction, side = ("upward", "column") if role == "ascending" else ("downward", "row")
        else:
            excluded["out_of_scope_role"] += 1
            continue
        predicates = config.get("annotation_policy", {}).get("quality_any", [])
        if predicates and not any(clean(record[p["field"]]) in p["accepted_values"] for p in predicates):
            excluded[f"quality_filter:{role}"] += 1
            continue
        if group == "unknown" or cell_type == "unknown":
            excluded[f"unknown_annotation:{role}"] += 1
            continue
        wanted = shared if role in {"ascending", "descending"} else config["body_groups"]
        if group not in wanted:
            excluded[f"out_of_scope_group:{role}:{group}"] += 1
            continue
        selection.append({"root_id": root, "role": role, "cell_type": cell_type,
            "group": group, "direction": direction, "matrix_side": side,
            "annotation_source": config["metadata_file"],
            "group_source": config["cluster_file"] if role in {"ascending", "descending"} else config["metadata_file"]})
    group_counts = Counter((r["direction"], r["matrix_side"], r["group"]) for r in selection)
    group_types = {}
    for r in selection:
        key = (r["direction"], r["matrix_side"], r["group"])
        group_types.setdefault(key, set()).add(r["cell_type"])
    return selection, {"total_metadata_rows": len(seen), "selected_rows": len(selection),
        "annotation_counts": dict(annotation_counts),
        "group_coverage": [{"direction": k[0], "matrix_side": k[1], "group": k[2],
                            "cells": n, "types": len(group_types[k])}
                           for k, n in sorted(group_counts.items())],
        "selected_by_role": dict(Counter(r["role"] for r in selection)),
        "excluded": dict(excluded), "shared_clusters": shared,
        "ascending_only_clusters": sorted(present["ascending"] - present["descending"]),
        "descending_only_clusters": sorted(present["descending"] - present["ascending"])}


def edge_rows(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        required = ["pre", "post", "count", "post_count"]
        if not set(required).issubset(reader.schema.names):
            raise ValueError(f"Unexpected edge schema: {reader.schema.names}")
        for i in range(reader.num_record_batches):
            data = reader.get_batch(i).select(required).to_pydict()
            yield from zip(*(data[key] for key in required))


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    import numpy as np
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.feather as feather
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/connectome_banc_v888.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=int)
    parser.add_argument("--normalization", choices=["type_pair_mean_input_fraction", "cell_pair_mean_input_fraction"])
    parser.add_argument("--annotation-policy", type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().startswith("login"):
        raise RuntimeError("Full annotation/edge processing requires a compute allocation")
    config = json.loads(args.config.read_text())
    if args.annotation_policy:
        config["annotation_policy"] = json.loads(args.annotation_policy.read_text())
    manifest = json.loads((args.data_dir / "acquisition_manifest.json").read_text())
    provenance = {"config_sha256": sha256(args.config), "dataset": "BANC", "materialization": 888,
        "dataverse_version": manifest["dataverse_version"], "synapse_version": config["synapse_version"],
        "input_sha256": {}, "slurm_job_id": os.environ["SLURM_JOB_ID"]}
    provenance["source_sha256"] = {name: sha256(Path(__file__).parent / name)
                                   for name in ("extract.py", "normalize.py", "derive_prior.py", "make_controls.py")}
    if args.annotation_policy:
        provenance["annotation_policy"] = config["annotation_policy"]
        provenance["annotation_policy_sha256"] = sha256(args.annotation_policy)
    for key in ("metadata_file", "edges_file", "cluster_file"):
        name = config[key]
        item = next(f for f in manifest["files"] if f["filename"] == name)
        actual = sha256(args.data_dir / name)
        if item["status"] != "verified" or actual != item["sha256"]:
            raise ValueError(f"Unverified or changed raw file: {name}")
        provenance["input_sha256"][name] = actual
    columns = [config[k] for k in ("id_column", "type_column", "role_column", "sensory_body_column", "motor_body_column")]
    columns += sorted({p["field"] for p in config.get("annotation_policy", {}).get("quality_any", [])} - set(columns))
    with pa.memory_map(str(args.data_dir / config["metadata_file"]), "r") as source:
        metadata_schema = ipc.open_file(source).schema
    if not set(columns).issubset(metadata_schema.names):
        raise ValueError(f"Required fields absent; actual metadata schema: {metadata_schema}")
    meta = feather.read_table(args.data_dir / config["metadata_file"], columns=columns).to_pylist()
    with (args.data_dir / config["cluster_file"]).open() as stream:
        clusters = list(csv.DictReader(stream))
    selected, coverage = select_neurons(meta, clusters, config)
    coverage["metadata_schema"] = {field.name: str(field.type) for field in metadata_schema}
    coverage["annotation_coverage_sensitivity"] = {
        "status": "pending_actual_annotation_field_review",
        "candidate_fields": [name for name in metadata_schema.names if any(
            word in name.lower() for word in ("proof", "review", "status", "quality"))],
        "note": "Field names are reported, never guessed as validated inclusion rules."}
    args.output.mkdir(parents=True, exist_ok=True)
    fields = ["root_id", "role", "cell_type", "group", "direction", "matrix_side", "annotation_source", "group_source"]
    for name in ("selected_neurons.csv", "group_assignment.csv"):
        with (args.output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fields); writer.writeheader(); writer.writerows(selected)
    axes = {"upward": [config["body_groups"], coverage["shared_clusters"]],
            "downward": [coverage["shared_clusters"], config["body_groups"]]}
    threshold = args.threshold if args.threshold is not None else config["minimum_pair_synapses"]
    if threshold < 1:
        raise ValueError("Synapse threshold must be positive")
    normalization = args.normalization or config["normalization"]
    provenance.update(threshold=threshold, normalization=normalization)
    for direction in axes:
        neurons = {r["root_id"]: (r["group"], r["cell_type"], r["matrix_side"])
                   for r in selected if r["direction"] == direction}
        matrix = aggregate_type_balanced(edge_rows(args.data_dir / config["edges_file"]),
            neurons, *axes[direction], threshold=threshold, normalization=normalization)
        np.savez_compressed(args.output / f"{direction}_matrix.npz", matrix=matrix)
    coverage.update(axes=axes, provenance=provenance, status="candidate_requires_sensitivity_audit")
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "selected": len(selected), "axes": axes}))


if __name__ == "__main__":
    main()
