"""Batch preprocessing script for raw CT datasets."""

import argparse

from anatcoder.data.preprocess import batch_preprocess


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch preprocessing."""
    parser = argparse.ArgumentParser(description="Batch preprocess raw CT cases.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--crop_size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    """Run batch preprocessing entrypoint."""
    args = parse_args()
    batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    main()
