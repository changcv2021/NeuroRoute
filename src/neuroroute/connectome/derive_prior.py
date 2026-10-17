"""Effective rank is an engineering structural statistic, not biological capacity."""
import argparse
import json
from pathlib import Path


def effective_rank(matrix):
    import numpy as np
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Expected a finite, nonnegative nonempty matrix")
    singular = np.linalg.svd(a, compute_uv=False)
    if singular.sum() <= 0:
        raise ValueError("Zero matrix has no normalized singular spectrum")
    p = singular[singular > 0] / singular.sum()
    return float(np.exp(-(p * np.log(p)).sum())), p.tolist()


def derive(upward, downward, upward_axes, downward_axes):
    if list(upward_axes[0]) != list(downward_axes[1]) or list(upward_axes[1]) != list(downward_axes[0]):
        raise ValueError("U/D need the same anatomical and shared cluster axes")
    if upward.shape != downward.T.shape:
        raise ValueError("U/D dimensions are not comparable")
    ur, us = effective_rank(upward)
    dr, ds = effective_rank(downward)
    return {"status": "candidate_requires_sensitivity_audit", "direction_order": ["upward", "downward"],
            "pi_bio": [ur / (ur + dr), dr / (ur + dr)],
            "effective_rank": {"upward": ur, "downward": dr},
            "normalized_singular_spectrum": {"upward": us, "downward": ds},
            "interpretation": "engineering structural prior; not measured communication capacity"}


def main():
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()
    coverage = json.loads((args.artifact_dir / "coverage.json").read_text())
    u = np.load(args.artifact_dir / "upward_matrix.npz", allow_pickle=False)
    d = np.load(args.artifact_dir / "downward_matrix.npz", allow_pickle=False)
    prior = derive(u["matrix"], d["matrix"], coverage["axes"]["upward"], coverage["axes"]["downward"])
    prior["provenance"] = coverage["provenance"]
    (args.artifact_dir / "prior.json").write_text(json.dumps(prior, indent=2) + "\n")


if __name__ == "__main__":
    main()
