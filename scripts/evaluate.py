"""Standalone evaluation script for trained checkpoints."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate reconstruction checkpoints.")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--metrics_config", default="configs/eval/default.yaml")
    return parser.parse_args()


def main() -> None:
    """Run evaluation entrypoint."""
    _ = parse_args()
    raise NotImplementedError("TODO: implement standalone evaluator")


if __name__ == "__main__":
    main()
