"""Machine-check the complete fixed preprocessing audit without selecting a prior."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def summarize(base, threshold_root, annotation_root, require_source_hashes=False):
    expected = [("base", base)]
    expected += [(f"threshold_{t}_{n}", threshold_root / f"threshold_{t}_{n}")
                 for t in [1, 5, 10] for n in ["type_pair_mean_input_fraction", "cell_pair_mean_input_fraction"]]
    expected += [(f"annotation_{i}", annotation_root / f"variant_{i}") for i in range(6)]
    variants, errors, warnings = [], [], []
    source_inputs, axes, protocol_hash, source_code = None, None, None, None
    for label, path in expected:
        try:
            p = json.loads((path / "prior.json").read_text())
            c = json.loads((path / "coverage.json").read_text())
            pi = p["pi_bio"]
            if len(pi) != 2 or any(not math.isfinite(x) or x <= 0 for x in pi) or not math.isclose(sum(pi), 1):
                raise ValueError("invalid prior")
            if p["direction_order"] != ["upward", "downward"]:
                raise ValueError("wrong direction order")
            if p["provenance"] != c["provenance"]:
                raise ValueError("prior and coverage provenance differ")
            provenance = p["provenance"]
            if source_inputs is None:
                source_inputs, axes, protocol_hash = provenance["input_sha256"], c["axes"], provenance["config_sha256"]
                source_code = provenance.get("source_sha256")
            if provenance["input_sha256"] != source_inputs or c["axes"] != axes or provenance["config_sha256"] != protocol_hash:
                raise ValueError("input snapshot, axes, or base protocol differ")
            if c["axes"]["upward"] != list(reversed(c["axes"]["downward"])):
                raise ValueError("unmatched anatomical/cluster axes")
            if not c["group_coverage"] or any(g["cells"] <= 0 or g["types"] <= 0 for g in c["group_coverage"]):
                raise ValueError("unobserved group cannot be interpreted as zero")
            observed = {(g["direction"], g["matrix_side"], g["group"]) for g in c["group_coverage"]}
            expected_groups = {(direction, side, group) for direction, pair in c["axes"].items()
                               for side, groups in zip(("row", "column"), pair) for group in groups}
            if observed != expected_groups:
                raise ValueError("group coverage does not match all declared axes")
            if source_code and provenance.get("source_sha256") and provenance["source_sha256"] != source_code:
                raise ValueError("source implementation differs across variants")
            if not provenance.get("source_sha256"):
                (errors if require_source_hashes else warnings).append(f"{label}: historical run lacks source hashes; rerun for full reproducibility")
            variants.append({"variant": label, "path": str(path), "pi_bio": pi,
                "upward_minus_downward": pi[0]-pi[1], "effective_rank": p["effective_rank"],
                "selected_by_role": c["selected_by_role"], "excluded": c["excluded"],
                "annotation_policy": provenance.get("annotation_policy"),
                "prior_sha256": hashlib.sha256((path / "prior.json").read_bytes()).hexdigest()})
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{label}: {exc}")
    upward = [v["pi_bio"][0] for v in variants]
    signs = {0 if math.isclose(x, 0.5, abs_tol=1e-12) else (1 if x > 0.5 else -1) for x in upward}
    stable = len(signs) == 1
    status = "technical_pass_candidate_mapping" if not errors and stable else "insufficient_preprocessing_evidence"
    return {"status": status, "expected_variants": len(expected), "completed_variants": len(variants),
        "errors": errors, "warnings": warnings, "all_direction_relations_agree": stable,
        "upward_fraction_range": [min(upward),max(upward)] if upward else None,
        "frozen_candidate": variants[0]["pi_bio"] if variants and variants[0]["variant"] == "base" else None,
        "selection_rule": "retain original base; never choose largest asymmetry or robot outcome",
        "scope": "three leg classes and 16 shared published AN/DN groups; excludes 89 AN-only proprioceptive cells",
        "biological_hypothesis_status": "untested_in_robots",
        "limitations": ["effective rank is not communication capacity", "quality status notes can conflict with explicit proofreading fields",
            "unmatched groups and missing annotations are excluded and reported, never zero-imputed",
            "three body groups impose rank<=3; result depends on this declared engineering mapping"],
        "variants": variants}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-source-hashes", action="store_true")
    args = parser.parse_args()
    result = summarize(args.root / "base", args.root / "sensitivity", args.root / "annotation_sensitivity", args.require_source_hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k:result[k] for k in ["status","completed_variants","errors","warnings","upward_fraction_range"]}))
    if result["status"] != "technical_pass_candidate_mapping":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
