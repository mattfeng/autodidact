import argparse
import json
from pathlib import Path
import sys

from datasets import load_dataset


def log(message: str) -> None:
    """Print a progress message immediately."""
    print(message, file=sys.stderr, flush=True)


def add_source_row_index(row: dict, index: int) -> dict:
    """Attach the row's position in the streamed cluster before shuffling."""
    row["source_row_index"] = int(index)
    return row


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ClimbMix downloader."""
    parser = argparse.ArgumentParser(
        description="Download a deterministic cluster-covered ClimbMix sample."
    )
    parser.add_argument("--dataset", default="gvlassis/ClimbMix")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="data/raw/climbmix")
    parser.add_argument("--max-docs-per-cluster", type=int, default=2000)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.max_docs_per_cluster <= 0:
        raise ValueError("--max-docs-per-cluster must be positive")
    if args.shuffle_buffer_size <= 0:
        raise ValueError("--shuffle-buffer-size must be positive")
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    return args


def main() -> None:
    """Download a deterministic cluster-covered sample of ClimbMix text rows.

    The script must write one JSONL file per cluster under the output directory.
    Each row must contain at least `text`, `cluster_id`, and `source_row_index`.
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(
        "Streaming ClimbMix rows "
        f"dataset={args.dataset!r} split={args.split!r} "
        f"max_docs_per_cluster={args.max_docs_per_cluster} "
        f"shuffle_buffer_size={args.shuffle_buffer_size} seed={args.seed}"
    )

    for cluster_id in range(1, 21):
        log(f"[cluster {cluster_id:02d}/20] loading streaming dataset")
        dataset = load_dataset(
            args.dataset,
            f"cluster_id={cluster_id}",
            split=args.split,
            streaming=True,
        )
        dataset = dataset.map(add_source_row_index, with_indices=True)
        dataset = dataset.shuffle(
            seed=args.seed + cluster_id,
            buffer_size=args.shuffle_buffer_size,
        )

        output_path = output_dir / f"cluster_{cluster_id:02d}.jsonl"
        rows_written = 0
        rows_seen = 0
        rows_skipped = 0
        log(
            f"[cluster {cluster_id:02d}/20] writing up to "
            f"{args.max_docs_per_cluster} rows to {output_path}"
        )
        with output_path.open("w", encoding="utf-8") as handle:
            for row in dataset:
                rows_seen += 1
                text = row.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    rows_skipped += 1
                    continue
                record = {
                    "text": text,
                    "cluster_id": cluster_id,
                    "source_row_index": int(row["source_row_index"]),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                rows_written += 1
                if rows_written % args.progress_every == 0:
                    log(
                        f"[cluster {cluster_id:02d}/20] wrote "
                        f"{rows_written}/{args.max_docs_per_cluster} rows "
                        f"(seen={rows_seen}, skipped_empty={rows_skipped})"
                    )
                if rows_written >= args.max_docs_per_cluster:
                    break

        if rows_written == 0:
            raise RuntimeError(f"no non-empty rows found for cluster {cluster_id}")
        log(
            f"[cluster {cluster_id:02d}/20] done: wrote {rows_written} rows "
            f"(seen={rows_seen}, skipped_empty={rows_skipped})"
        )

    log("Finished streaming ClimbMix sample.")


if __name__ == "__main__":
    main()
