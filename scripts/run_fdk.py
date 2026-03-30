"""Run the classical FDK baseline reconstruction pipeline."""

import argparse


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for FDK baseline execution."""
    parser = argparse.ArgumentParser(description="Run FDK baseline on sparse projections.")
    parser.add_argument("--proj_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--geo_config", required=True)
    return parser.parse_args()


def main() -> None:
    """Run FDK reconstruction entrypoint."""
    _ = parse_args()
    raise NotImplementedError("TODO: implement FDK baseline script")


if __name__ == "__main__":
    main()
