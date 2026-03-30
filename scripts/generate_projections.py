"""Generate sparse-view projection datasets using TIGRE wrappers."""

import argparse
from pathlib import Path

from anatcoder.data.projection import generate_sparse_projections
from anatcoder.utils.io import load_yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for projection generation."""
    parser = argparse.ArgumentParser(description="Generate sparse CBCT projections.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_views", nargs="+", type=int, required=True)
    parser.add_argument("--geo_config", required=True)
    return parser.parse_args()


def main() -> None:
    """Run sparse projection generation for all cases."""
    args = parse_args()
    geo_cfg = load_yaml(args.geo_config)

    processed_dir = Path(args.data_dir)
    for case_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        volume_path = case_dir / "volume.npy"
        if not volume_path.exists():
            continue
        generate_sparse_projections(
            volume_path=volume_path,
            n_views_list=args.n_views,
            output_dir=Path(args.output_dir) / case_dir.name,
            geo_params=geo_cfg.get("geo", geo_cfg),
        )


if __name__ == "__main__":
    main()
