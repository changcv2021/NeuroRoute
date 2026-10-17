"""Run one predeclared annotation-coverage variant, independent of other variants."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, type=int, choices=range(6))
    parser.add_argument("--config", type=Path, default=Path("configs/connectome_banc_v888.json"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    policy = args.config.parent / f"connectome_annotation_{args.index}.json"
    output = args.output_root / f"variant_{args.index}"
    subprocess.run([sys.executable, "-m", "neuroroute.connectome.extract", "--config", str(args.config),
        "--data-dir", str(args.data_dir), "--output", str(output), "--annotation-policy", str(policy)], check=True)
    subprocess.run([sys.executable, "-m", "neuroroute.connectome.derive_prior", str(output)], check=True)
    subprocess.run([sys.executable, "-m", "neuroroute.connectome.make_controls", str(output / "prior.json"),
                    str(output / "controls.json")], check=True)


if __name__ == "__main__":
    main()
