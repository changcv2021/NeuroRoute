"""Counterfactual allocation priors; swapping is not graph rewiring."""
import argparse
import json
import math
from pathlib import Path


def make_controls(prior):
    p = prior["pi_bio"]
    if len(p) != 2 or any(not math.isfinite(v) or v <= 0 for v in p) or not math.isclose(sum(p), 1.0):
        raise ValueError("A real normalized two-direction prior is required")
    return {"B_ASYM": {"lambda_bio": 0, "pi": None},
            "B_SYM": {"pi": [0.5, 0.5]},
            "M_BIO": {"pi": p}, "M_REVERSE": {"pi": list(reversed(p))},
            "status": prior["status"], "source_provenance": prior.get("provenance")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(make_controls(json.loads(args.prior.read_text())), indent=2) + "\n")


if __name__ == "__main__":
    main()
