import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_FILES = [
    "model.safetensors",
    "config.json",
    "vocab.json",
    "merges.txt",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for downloading GPT-2 checkpoint files."""
    parser = argparse.ArgumentParser(
        description="Download OpenAI GPT-2-small files from HuggingFace."
    )
    parser.add_argument("--repo-id", default="openai-community/gpt2")
    parser.add_argument("--output-dir", default="checkpoints/openai-gpt2")
    return parser.parse_args()


def main() -> None:
    """Download OpenAI GPT-2-small tokenizer and safetensors files from HuggingFace.

    The script may use `huggingface_hub.snapshot_download`, but must not import
    or use `transformers`.
    """
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=output_dir,
        allow_patterns=MODEL_FILES,
    )

    missing = [name for name in MODEL_FILES if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing downloaded files: {missing}")

    manifest = {
        "repo_id": args.repo_id,
        "files": MODEL_FILES,
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
