"""Normalized densities from a portable owned energy model with refined quadrature."""

import argparse
from pathlib import Path

import numpy as np
import torch

from sampling_lab.artifacts import sha256, write_json
from sampling_lab.ebm import load, log_partition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    x = np.loadtxt(args.input, delimiter=",", ndmin=2)
    if x.shape[1] != 2 or not np.isfinite(x).all():
        raise ValueError("Two finite coordinates per CSV row are required")
    model = load(args.model)
    with torch.no_grad():
        partition = float(log_partition(model, 401, 18))
        score = (-model(torch.as_tensor(x, dtype=torch.float32)) - partition).numpy()
    write_json(
        args.output,
        dict(
            log_density=score,
            log_partition=partition,
            model_sha256=sha256(args.model),
            input_sha256=sha256(args.input),
            scope="Finite quadrature on ±18 with 401 points per axis; check published refinement and tail bounds. This is an owned 2D model, not an exact high-dimensional normalization oracle.",
        ),
    )
    print("Scored", len(x), "points")


if __name__ == "__main__":
    main()
