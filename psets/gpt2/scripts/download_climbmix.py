import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ClimbMix downloader."""
    parser = argparse.ArgumentParser(
        description="Download a deterministic cluster-covered ClimbMix sample."
    )
    parser.add_argument("--dataset", default="gvlassis/ClimbMix")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/raw/climbmix")
    parser.add_argument("--max-docs-per-cluster", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.max_docs_per_cluster <= 0:
        raise ValueError("--max-docs-per-cluster must be positive")
    return args


def main() -> None:
    """Download a deterministic cluster-covered sample of ClimbMix text rows.

    The script must write one JSONL file per cluster under the output directory.
    Each row must contain at least `text`, `cluster_id`, and `source_row_index`.
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id in range(1, 21):
        dataset = load_dataset(
            args.dataset,
            f"cluster_id={cluster_id}",
            split=args.split,
        )
        rng = np.random.default_rng(args.seed + cluster_id)
        candidate_count = min(len(dataset), args.max_docs_per_cluster * 10)
        candidate_indices = rng.choice(
            len(dataset),
            size=candidate_count,
            replace=False,
        )

        output_path = output_dir / f"cluster_{cluster_id:02d}.jsonl"
        rows_written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for source_row_index in candidate_indices:
                row = dataset[int(source_row_index)]
                text = row.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                record = {
                    "text": text,
                    "cluster_id": cluster_id,
                    "source_row_index": int(source_row_index),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1
                if rows_written >= args.max_docs_per_cluster:
                    break

        if rows_written == 0:
            raise RuntimeError(f"no non-empty rows found for cluster {cluster_id}")


if __name__ == "__main__":
    main()
