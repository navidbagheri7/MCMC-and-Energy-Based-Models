"""Run owned sampler or EBM protocols; quick runs always use private smoke outputs."""

import argparse
import json
from pathlib import Path

from .ebm import train
from .experiments import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=["sampling", "ebm"])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--filter", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = args.config or Path(__file__).parent / "default_configs" / f"{args.family}.json"
    items = json.loads(config.read_text())["experiments"]
    output = (
        args.output or Path("results/smoke" if args.quick else "results/reference") / args.family
    )
    if args.quick and output.is_relative_to("results/reference"):
        raise ValueError("Quick fits cannot overwrite reference evidence")
    if (
        args.output
        and not args.quick
        and output != Path("results/reference") / args.family
        and output.is_relative_to("results/reference")
    ):
        raise ValueError("Use a separate checkout for alternative reference protocols")
    source_root = (
        (output / "source_data") if args.quick or args.output else Path("assets/source_data")
    )
    for item in items:
        if args.filter and args.filter not in json.dumps(item, sort_keys=True):
            continue
        config = dict(item)
        if args.quick:
            if args.family == "sampling":
                config.update(draws=60, warmup=100)
            else:
                config.update(steps=2)
        (evaluate if args.family == "sampling" else train)(config, output, source_root)


if __name__ == "__main__":
    main()
