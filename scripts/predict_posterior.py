"""Predict from public posterior arrays using raw-feature CSV in recorded order."""

import argparse
import json
from pathlib import Path

import numpy as np

from sampling_lab.artifacts import sha256, write_json
from sampling_lab.posterior import predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="Name under results/reference/sampling")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if Path(args.protocol).name != args.protocol:
        raise ValueError("Protocol must be a single published name")
    folder = Path("results/reference/sampling") / args.protocol
    record = json.loads((folder / "run.json").read_text())
    metrics = json.loads((folder / "metrics.json").read_text())
    source = Path(record["numeric_source"])
    if record["status"] != "complete" or sha256(source) != record["numeric_sha256"]:
        raise ValueError("Incomplete or changed posterior evidence")
    with np.load(source, allow_pickle=False) as data:
        features = data["feature_names"].tolist()
    header = args.input.open().readline().strip().split(",")
    if header != features:
        raise ValueError("CSV headers must match the published feature_names exactly")
    raw = np.loadtxt(args.input, delimiter=",", skiprows=1, ndmin=2)
    output = predict(source, raw, metrics["target"])
    output.update(
        protocol=args.protocol,
        source_sha256=sha256(source),
        input_sha256=sha256(args.input),
        diagnostics=metrics["diagnostics"],
    )
    write_json(args.output, output)
    print("Predicted", len(raw), "observations")


if __name__ == "__main__":
    main()
