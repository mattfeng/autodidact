import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tiktoken


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for ClimbMix token-array construction."""
    parser = argparse.ArgumentParser(
        description="Create token-balanced GPT-2 arrays from ClimbMix JSONL files."
    )
    parser.add_argument("--input-dir", default="data/raw/climbmix")
    parser.add_argument("--output-dir", default="data/processed/climbmix_gpt2_1m")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--dataset", default="gvlassis/ClimbMix")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tokens-per-cluster", type=int, default=50000)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--val-fraction", type=float, default=0.01)
    parser.add_argument("--test-fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.tokens_per_cluster <= 0:
        raise ValueError("--tokens-per-cluster must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if not 0.0 < args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1")
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    if args.val_fraction + args.test_fraction >= 1.0:
        raise ValueError("validation plus test fraction must be less than 1")
    return args


def sha256_file(path: Path) -> str:
    """Return the SHA256 hex digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    """Read one JSONL file into a list of dictionaries."""
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    """Create token-balanced GPT-2 token arrays from downloaded ClimbMix JSONL.

    The script must sample approximately the same number of GPT-2 tokens from
    every cluster, concatenate them in a deterministic shuffled cluster order,
    split by token position, save NumPy arrays, and write a JSON manifest.
    """
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tiktoken.get_encoding("gpt2")
    cluster_chunks = []
    cluster_stats = {}

    for cluster_id in range(1, 21):
        input_path = input_dir / f"cluster_{cluster_id:02d}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"missing {input_path}")

        rows = read_jsonl(input_path)
        token_parts = []
        selected_rows = 0
        total_tokens = 0

        for row in rows:
            text = row.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue
            token_ids = tokenizer.encode(text)
            token_ids.append(tokenizer.eot_token)
            tokens = np.asarray(token_ids, dtype=np.int32)
            token_parts.append(tokens)
            selected_rows += 1
            total_tokens += int(tokens.shape[0])
            if total_tokens >= args.tokens_per_cluster:
                break

        if total_tokens < args.tokens_per_cluster:
            raise RuntimeError(
                f"cluster {cluster_id} has only {total_tokens} tokens; "
                f"need {args.tokens_per_cluster}"
            )

        cluster_tokens = np.concatenate(token_parts).astype(np.int32)
        cluster_tokens = cluster_tokens[: args.tokens_per_cluster]
        cluster_chunks.append((cluster_id, cluster_tokens))
        cluster_stats[str(cluster_id)] = {
            "rows": selected_rows,
            "tokens": int(cluster_tokens.shape[0]),
        }

    rng = np.random.default_rng(args.seed)
    order = np.arange(len(cluster_chunks))
    rng.shuffle(order)
    shuffled_chunks = [cluster_chunks[int(index)][1] for index in order]
    all_tokens = np.concatenate(shuffled_chunks).astype(np.int32)

    total = int(all_tokens.shape[0])
    test_len = int(total * args.test_fraction)
    val_len = int(total * args.val_fraction)
    train_len = total - val_len - test_len
    min_len = args.block_size + 1
    if min(train_len, val_len, test_len) < min_len:
        raise RuntimeError(
            "train, validation, and test splits must each contain at least "
            f"{min_len} tokens; increase --tokens-per-cluster or reduce fractions"
        )

    train_tokens = all_tokens[:train_len]
    val_tokens = all_tokens[train_len : train_len + val_len]
    test_tokens = all_tokens[train_len + val_len :]

    paths = {
        "train_tokens": output_dir / "train_tokens.npy",
        "val_tokens": output_dir / "val_tokens.npy",
        "test_tokens": output_dir / "test_tokens.npy",
    }
    np.save(paths["train_tokens"], train_tokens)
    np.save(paths["val_tokens"], val_tokens)
    np.save(paths["test_tokens"], test_tokens)

    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "seed": args.seed,
        "tokenizer": "gpt2",
        "target_tokens_per_cluster": args.tokens_per_cluster,
        "block_size": args.block_size,
        "cluster_order": [int(cluster_chunks[int(index)][0]) for index in order],
        "clusters": cluster_stats,
        "splits": {
            "train": int(train_tokens.shape[0]),
            "val": int(val_tokens.shape[0]),
            "test": int(test_tokens.shape[0]),
        },
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
    }
    manifest_path = manifest_dir / "climbmix_gpt2_1m_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
