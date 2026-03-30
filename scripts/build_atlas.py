"""Build oracle or population atlas maps from segmentation labels."""

import argparse
from pathlib import Path

from anatcoder.data.atlas import AtlasBuilder


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for atlas construction."""
    parser = argparse.ArgumentParser(description="Build anatomy atlas probability maps.")
    parser.add_argument("--mode", choices=["oracle", "population"], required=True)
    parser.add_argument("--seg_path")
    parser.add_argument("--seg_dir")
    parser.add_argument("--reference_path")
    parser.add_argument("--n_cases", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    """Run atlas-building entrypoint."""
    args = parse_args()

    if args.mode == "oracle":
        if not args.seg_path:
            raise ValueError("--seg_path is required for oracle mode")
        atlas = AtlasBuilder.from_oracle(args.seg_path)
    else:
        if not args.seg_dir or not args.reference_path:
            raise ValueError("--seg_dir and --reference_path are required for population mode")
        seg_paths = sorted(str(p) for p in Path(args.seg_dir).glob("**/*.nii.gz"))
        atlas = AtlasBuilder.from_population(seg_paths, args.reference_path, n_cases=args.n_cases)

    _ = atlas
    raise NotImplementedError("TODO: save atlas tensor to numpy file")


if __name__ == "__main__":
    main()
