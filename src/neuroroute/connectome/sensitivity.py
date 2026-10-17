"""One fixed variant per Slurm array element; independent tasks need no dependencies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/connectome_banc_v888.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True, choices=range(6))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    variants = [(t, n) for t in config["sensitivity_pair_thresholds"]
                for n in config["sensitivity_normalizations"]]
    if len(variants) != 6:
        raise ValueError("The frozen sensitivity protocol requires exactly six variants")
    threshold, normalization = variants[args.index]
    output = args.output_root / f"threshold_{threshold}_{normalization}"
    subprocess.run([sys.executable, "-m", "neuroroute.connectome.extract", "--config", str(args.config),
        "--data-dir", str(args.data_dir), "--output", str(output), "--threshold", str(threshold),
        "--normalization", normalization], check=True)
    subprocess.run([sys.executable, "-m", "neuroroute.connectome.derive_prior", str(output)], check=True)
    subprocess.run([sys.executable, "-m", "neuroroute.connectome.make_controls",
        str(output / "prior.json"), str(output / "controls.json")], check=True)
    print(json.dumps({"index": args.index, "output": str(output), "threshold": threshold,
                      "normalization": normalization}))


if __name__ == "__main__":
    main()
