# Problem Set: Implement GPT-2 in JAX and Equinox

## Table of Contents

- [Overview](#overview)
- [Learning Goals](#learning-goals)
- [Environment](#environment)
- [Directory Structure](#directory-structure)
- [Dataset](#dataset)
- [Problem 1: Configuration, Tokenization, and Dataset Construction](#problem-1-configuration-tokenization-and-dataset-construction)
- [Problem 2: Grain Data Loading for Language Modeling](#problem-2-grain-data-loading-for-language-modeling)
- [Problem 3: GPT-2 Modules in Equinox](#problem-3-gpt-2-modules-in-equinox)
- [Problem 4: Loss, Optimization, and Training](#problem-4-loss-optimization-and-training)
- [Problem 5: Load OpenAI GPT-2 Weights Without Transformers](#problem-5-load-openai-gpt-2-weights-without-transformers)
- [Problem 6: Integration and Training Behavior](#problem-6-integration-and-training-behavior)
- [End-to-End Checks](#end-to-end-checks)

## Overview

You will build a decoder-only Transformer language model with the GPT-2 architecture using JAX, Equinox, and Optax. You will train it with next-token prediction on a deterministic, token-balanced subset of ClimbMix loaded through Grain. You may use `datasets` and `huggingface_hub` for dataset access and file downloads. You must implement GPT-2 byte-pair encoding yourself from `vocab.json` and `merges.txt` in `src/tokenizer.py`; student-authored source must not use `transformers`, `tiktoken`, or `tokenizers`. The instructor-provided dataset preparation script may use `tiktoken` for faster offline preprocessing.

The final system must:

- construct a valid subset of ClimbMix from text rows with `cluster_id` metadata;
- tokenize text with the GPT-2 byte-pair encoder;
- load batches with Grain as integer token arrays;
- implement GPT-2 modules in Equinox with correct masking, residual streams, layer normalization, and tied output embeddings;
- train with cross-entropy on shifted targets;
- generate text autoregressively; and
- load OpenAI GPT-2 124M pretrained weights from downloaded HuggingFace `safetensors` files without importing `transformers`.

Correctness means that every tensor shape matches the GPT-2-small contract, training loss decreases on a small subset, causal masking prevents attention to future tokens, and your Equinox model produces logits close to the reference pretrained checkpoint after weight loading.

## Learning Goals

- You will implement GPT-2 byte-level pre-tokenization, byte-to-Unicode mapping, BPE merges, token ID lookup, and decoding from downloaded `vocab.json` and `merges.txt`.
- You will construct a deterministic ClimbMix subset that preserves cluster coverage instead of accidentally taking an ordered or single-cluster prefix.
- You will use Grain to shuffle, transform, batch, and iterate token examples for JAX training.
- You will implement multi-head masked self-attention, MLP blocks, residual connections, and final language-model logits in Equinox.
- You will implement numerically stable cross-entropy and verify it on fixed logits.
- You will train a GPT-2 model with Optax and verify that parameters update under JIT-compiled JAX steps.
- You will load OpenAI GPT-2 weights from `safetensors` into your own Equinox PyTree and check numerical agreement.

## Environment

Use Python 3.11 or 3.12. The commands below install CPU-compatible packages. A GPU is optional; all correctness checks must pass on CPU, though full training will be slower.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "jax[cpu]>=0.5.0" "equinox>=0.11.10" "optax>=0.2.4" "grain[parquet]>=0.2.16" "numpy>=1.26" "regex>=2024.5.15" "tiktoken>=0.7.0" "datasets>=3.0.0" "huggingface_hub>=0.25.0" "safetensors>=0.4.5" "pyarrow>=17.0.0" "pytest>=8.0.0" "tqdm>=4.66.0"
```

Use one random seed throughout the assignment unless a task states otherwise:

```bash
export PYTHONHASHSEED=0
export GPT2_PSET_SEED=1234
```

The setup commands above are the only commands in this section that you should expect to run before creating the project files. Commands that invoke `src.*` modules, `scripts/*.py`, or `pytest` are correctness checks to run after you have created the provided files and implemented the files named in the relevant problem.

After you create the provided tests and implement the source files, run the full test suite with:

```bash
pytest -q
```

After you implement `src/train.py`, `src/config.py`, `src/data.py`, `src/model.py`, and `src/loss.py`, run a tiny end-to-end training check with:

```bash
python -m src.train --config configs/tiny.yaml
```

After you implement `src/check_pretrained.py`, `src/load_openai.py`, `src/model.py`, and `src/tokenizer.py`, and after checkpoint files exist under `checkpoints/openai-gpt2`, run the pretrained weight-loading check with:

```bash
python -m src.check_pretrained --checkpoint-dir checkpoints/openai-gpt2
```

Do not install or import `transformers` or `tokenizers`. Do not import `tiktoken` in `src/`; it is installed only for the provided dataset preparation script. The provided tests include a guard for these boundaries.

## Directory Structure

Create this directory structure:

```text
gpt2_jax_equinox/
  README.md
  configs/
    tiny.yaml
    gpt2_small.yaml
  data/
    raw/
    processed/
    manifests/
  checkpoints/
    openai-gpt2/
  scripts/
    download_climbmix.py
    build_climbmix_subset.py
    download_openai_gpt2.py
  src/
    config.py
    tokenizer.py
    data.py
    model.py
    loss.py
    train.py
    generate.py
    load_openai.py
    check_pretrained.py
  tests/
    test_no_transformers.py
    test_tokenizer.py
    test_data.py
    test_attention.py
    test_loss.py
    test_model_shapes.py
    test_weight_loading.py
    test_integration.py
```

You will edit every file under `src/`. The files under `configs/`, `scripts/`, and `tests/` are instructor-provided support files: create them exactly as shown and use them unchanged. The `data/raw/`, `data/processed/`, `data/manifests/`, and `checkpoints/openai-gpt2/` contents are generated by commands in this problem set.

## Dataset

ClimbMix is a language-model pretraining dataset introduced with CLIMB, a clustering-based data mixture method. The official NVIDIA dataset is large. For this assignment, use the HuggingFace-hosted text-form mirror `gvlassis/ClimbMix`, which exposes data by `cluster_id` and avoids requiring you to load a terabyte-scale ordered corpus. HuggingFace reports the original ClimbMix as a text-generation dataset derived from the CLIMB paper, and the text-form mirror documents that the original release is inconvenient for small users because cluster ordering and missing precise ratios can make naive subsampling invalid.

Your subset must be a properly constructed subset, meaning:

- it must contain examples from all 20 ClimbMix clusters, with `cluster_id` values `1` through `20`;
- it must be sampled with a fixed seed;
- it must not be formed by taking the first rows of the original dataset;
- it must target a fixed number of tokenized GPT-2 tokens per cluster, not only a fixed number of documents;
- it must store a manifest containing dataset name, split, seed, tokenizer name, target tokens per cluster, selected row counts, selected token counts, and SHA256 hashes of output files.

For development, create a small subset with `50_000` GPT-2 tokens per cluster, about `1_000_000` total tokens. For larger training, increase this to `2_000_000` tokens per cluster.

After you create the provided `scripts/download_climbmix.py`, download raw text rows with:

```bash
python scripts/download_climbmix.py \
  --dataset gvlassis/ClimbMix \
  --split train \
  --output-dir data/raw/climbmix \
  --max-docs-per-cluster 2000 \
  --seed 1234
```

After you create the provided `scripts/download_openai_gpt2.py`, download the GPT-2 vocabulary, merge, and checkpoint files with:

```bash
python scripts/download_openai_gpt2.py
```

After you create the provided `scripts/build_climbmix_subset.py`, build token shards with:

```bash
python scripts/build_climbmix_subset.py \
  --input-dir data/raw/climbmix \
  --output-dir data/processed/climbmix_gpt2_1m \
  --manifest-dir data/manifests \
  --tokens-per-cluster 50000 \
  --block-size 1024 \
  --val-fraction 0.01 \
  --test-fraction 0.01 \
  --seed 1234
```

The processed files must be NumPy arrays saved as:

- `data/processed/climbmix_gpt2_1m/train_tokens.npy`
- `data/processed/climbmix_gpt2_1m/val_tokens.npy`
- `data/processed/climbmix_gpt2_1m/test_tokens.npy`

Each array has dtype `uint16` or `int32` and shape `(num_tokens,)`. GPT-2 token IDs must be integers in `[0, 50256]`, where `50256` is the end-of-text token.

Training examples are contiguous language-model blocks:

- input `x`: shape `(block_size,)`, dtype `int32`;
- target `y`: shape `(block_size,)`, dtype `int32`;
- relation: `y[t] == tokens[start + t + 1]` and `x[t] == tokens[start + t]`.

Use train/validation/test splits by token position after deterministic cluster-balanced construction and before making blocks. Do not split individual examples after blocks are made, because adjacent shifted blocks can leak nearly identical context between splits.

## Problem 1: Configuration, Tokenization, and Dataset Construction

### Context

Language models train on token IDs, not raw strings. GPT-2 uses byte-pair encoding with a vocabulary of 50,257 tokens. For next-token prediction, a long sequence of token IDs is divided into fixed context windows. The input window is all tokens except the next shifted token, and the target at each position is the next token.

A naive ClimbMix subset can be wrong even when it has the right file format. If the source is ordered by cluster or topic, taking the first rows can silently train on a narrow slice. Your subset must explicitly cover all clusters and record how it was created.

### Tasks

Required files to edit: `src/config.py` and `src/tokenizer.py`.

Instructor-provided files to create unchanged: `configs/tiny.yaml`, `configs/gpt2_small.yaml`, `scripts/download_climbmix.py`, `scripts/build_climbmix_subset.py`, `scripts/download_openai_gpt2.py`, `tests/test_tokenizer.py`, and `tests/test_no_transformers.py`.

#### 1.a Define typed configuration objects

Create `src/config.py` and include:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class GPT2Config:
    """Configuration for a GPT-2 style decoder-only Transformer.

    Attributes:
        vocab_size: Number of token IDs, including GPT-2 end-of-text.
        block_size: Maximum sequence length used by positional embeddings.
        n_layer: Number of Transformer blocks.
        n_head: Number of attention heads.
        n_embd: Residual stream width.
        dropout: Dropout probability used during training. You may set this to
            0.0 for deterministic tests.
        layer_norm_eps: Epsilon used by layer normalization.
    """
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    layer_norm_eps: float


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for optimization and data iteration.

    Attributes:
        data_dir: Directory containing processed token arrays.
        batch_size: Number of sequences per update.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW decoupled weight decay.
        max_steps: Number of optimizer updates.
        eval_interval: Number of steps between validation loss estimates.
        eval_batches: Number of validation batches per estimate.
        seed: Integer seed used for model initialization, data order, and sampling.
    """
    data_dir: str
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    eval_interval: int
    eval_batches: int
    seed: int


def load_yaml_config(path: str) -> tuple[GPT2Config, TrainConfig]:
    """Load a YAML config file and return model and training configs.

    Args:
        path: Path to a YAML file with `model` and `train` sections.

    Returns:
        A pair `(model_config, train_config)`.
    """
    pass
```

Create `configs/tiny.yaml` from the provided file at [`configs/tiny.yaml`](configs/tiny.yaml). It defines a model small enough for CPU tests.

Create `configs/gpt2_small.yaml` from the provided file at [`configs/gpt2_small.yaml`](configs/gpt2_small.yaml). It defines the GPT-2 124M configuration.

Pseudocode:

1. Read the YAML file as a dictionary.
2. Extract the `model` section and pass its values to `GPT2Config`.
3. Extract the `train` section and pass its values to `TrainConfig`.
4. Return both dataclass instances.

Correctness checks:

- `load_yaml_config("configs/tiny.yaml")[0].n_layer == 2`.
- `load_yaml_config("configs/gpt2_small.yaml")[0].n_embd == 768`.
- `n_embd % n_head == 0` must be true for every config.

#### 1.b Inspect GPT-2 byte-level BPE tokenization

Create `src/tokenizer.py` and include:

```python
import json
from pathlib import Path

import numpy as np
import regex as re


GPT2_PRETOKEN_PATTERN = (
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)


def bytes_to_unicode() -> dict[int, str]:
    """Return the reversible GPT-2 byte-to-Unicode map.

    GPT-2 maps raw UTF-8 bytes to Unicode characters before applying BPE so
    every byte sequence can be represented without using unknown tokens.
    """
    pass


def gpt2_pretokenize(text: str) -> list[str]:
    """Split text with OpenAI GPT-2's regex pre-tokenization pattern.

    The pattern keeps common English contractions as separate pieces, groups
    letters and numbers separately, and includes a leading space with most
    word pieces after the first word.
    """
    pass


def byte_encode_piece(piece: str, byte_encoder: dict[int, str]) -> tuple[str, ...]:
    """Encode one pre-tokenized piece as GPT-2 byte-level Unicode symbols.

    Args:
        piece: One string returned by `gpt2_pretokenize`.
        byte_encoder: Mapping returned by `bytes_to_unicode`.

    Returns:
        Tuple of Unicode symbols, one for each UTF-8 byte in `piece`.
    """
    pass


def bpe_merge_once(symbols: tuple[str, ...], ranks: dict[tuple[str, str], int]) -> tuple[str, ...]:
    """Apply one GPT-2 BPE merge step to a tuple of symbols.

    Args:
        symbols: Current byte-level or merged symbols.
        ranks: Mapping from adjacent symbol pairs to lower-is-better merge rank.

    Returns:
        Updated symbols after merging every non-overlapping occurrence of the
        best-ranked adjacent pair. If no adjacent pair is ranked, return the
        input symbols unchanged.
    """
    pass


def bpe_encode_piece(piece: str, ranks: dict[tuple[str, str], int]) -> tuple[str, ...]:
    """Encode one pre-tokenized piece into final BPE token strings.

    This helper is also used by `GPT2Tokenizer.encode`; it must not delegate to
    `transformers`, `tiktoken`, or `tokenizers`.
    """
    pass


def load_merges(path: str) -> dict[tuple[str, str], int]:
    """Load GPT-2 merge ranks from `merges.txt`.

    Args:
        path: Path to the downloaded GPT-2 merges file.

    Returns:
        Mapping from symbol pairs to merge rank, where lower rank merges first.
    """
    pass


class GPT2Tokenizer:
    """GPT-2 tokenizer implemented from vocabulary and merge files.

    The tokenizer must use GPT-2 byte-pair encoding, have vocabulary size
    50257, and expose the end-of-text token ID 50256.
    """

    def __init__(
        self,
        vocab_path: str = "checkpoints/openai-gpt2/vocab.json",
        merges_path: str = "checkpoints/openai-gpt2/merges.txt",
    ) -> None:
        """Create the GPT-2 tokenizer from downloaded vocab and merge files."""
        pass

    @property
    def vocab_size(self) -> int:
        """Return the number of GPT-2 token IDs."""
        pass

    @property
    def eot_token(self) -> int:
        """Return the GPT-2 end-of-text token ID."""
        pass

    def encode(self, text: str, add_eot: bool = False) -> np.ndarray:
        """Encode text as a one-dimensional integer NumPy array.

        Args:
            text: Input Unicode string.
            add_eot: Whether to append the GPT-2 end-of-text token.

        Returns:
            Token IDs with shape `(num_tokens,)` and dtype `int32`.
        """
        pass

    def decode(self, token_ids: np.ndarray) -> str:
        """Decode a one-dimensional array of GPT-2 token IDs into text.

        Args:
            token_ids: Array with shape `(num_tokens,)`.

        Returns:
            Decoded text.
        """
        pass
```

Pseudocode:

1. For `bytes_to_unicode`, follow OpenAI GPT-2's reversible byte mapping: keep visible bytes `!` through `~`, `¡` through `¬`, and `®` through `ÿ` as themselves, then map all remaining bytes to Unicode code points starting at `256`.
2. For `gpt2_pretokenize`, compile `GPT2_PRETOKEN_PATTERN` with the `regex` package and return all matches in order.
3. For `byte_encode_piece`, UTF-8 encode the piece and map each byte through `byte_encoder`.
4. For `bpe_merge_once`, find all adjacent pairs in `symbols`, choose the pair with the smallest rank, and merge every non-overlapping occurrence of that pair.
5. For `bpe_encode_piece`, byte-encode the piece, repeatedly call `bpe_merge_once`, and stop when a merge pass no longer changes the symbols.
6. For `load_merges`, skip the `#version:` header line, split each remaining line into two symbols, and assign increasing ranks starting at `0`.
7. In `GPT2Tokenizer.__init__`, check that both `vocab_path` and `merges_path` exist. Raise `FileNotFoundError` with a message mentioning `python scripts/download_openai_gpt2.py` if either file is missing.
8. Load `vocab.json` with `json.load` into `self._encoder: dict[str, int]`.
9. Build `self._decoder: dict[int, str]` by reversing the vocabulary.
10. Build `self._byte_encoder = bytes_to_unicode()` and `self._byte_decoder` by reversing that mapping.
11. Load merge ranks with `load_merges(merges_path)`.
12. For `vocab_size`, return `len(self._encoder)`.
13. For `encode`, pre-tokenize the text, BPE-encode each piece, look up every BPE token string in `self._encoder`, convert IDs to `np.int32`, and append `50256` when `add_eot` is true.
14. For `decode`, map token IDs to token strings with `self._decoder`, concatenate the token strings, map each Unicode symbol back to its original byte with `self._byte_decoder`, and decode the byte sequence as UTF-8 with `errors="replace"`.

Correctness checks:

- `gpt2_pretokenize("Hello, world! 123")` returns `["Hello", ",", " world", "!", " 123"]`.
- The byte encoder maps `ord("A")` to `"A"` and maps byte `0` to a Unicode string that is not the literal null character.
- With ranks `{("l", "o"): 0, ("lo", "w"): 1}`, `bpe_encode_piece("low", ranks)` returns `("low",)`.
- Encoding `"hello"` returns a one-dimensional integer array.
- Encoding `"hello"` with `add_eot=True` ends with `50256`.
- Decoding `encode("The quick brown fox")` contains the same words in the same order.

#### 1.c Download a cluster-covered ClimbMix sample

Create `scripts/download_climbmix.py` from the instructor-provided helper script at [`scripts/download_climbmix.py`](scripts/download_climbmix.py); use it unchanged.

Correctness checks:

- The output directory contains exactly 20 JSONL files.
- Each file has at least one row.
- Every row in `cluster_07.jsonl` has `"cluster_id": 7`.
- Re-running the command with the same seed produces identical SHA256 hashes.

#### 1.d Build token-balanced train, validation, and test arrays

Create `scripts/build_climbmix_subset.py` from the instructor-provided helper script at [`scripts/build_climbmix_subset.py`](scripts/build_climbmix_subset.py); use it unchanged. This script may use `tiktoken` because dataset preprocessing is support code, not student-authored tokenizer code.

Correctness checks:

- The manifest contains all 20 cluster IDs.
- Every cluster contributes exactly `tokens_per_cluster` tokens after truncation.
- The train, validation, and test token arrays are one-dimensional.
- `tokens.max() <= 50256` and `tokens.min() >= 0`.
- No split is empty.

### Tests

These are fixed pytest tests. Create them exactly as shown and use them unchanged.

Create `tests/test_tokenizer.py` from the provided file at [`tests/test_tokenizer.py`](tests/test_tokenizer.py).

Create `tests/test_no_transformers.py` from the provided file at [`tests/test_no_transformers.py`](tests/test_no_transformers.py).

Expected test behavior:

- `tests/test_tokenizer.py::test_gpt2_pretokenize_keeps_leading_spaces` uses the fixed text `"Hello, world! 123"` and expects `["Hello", ",", " world", "!", " 123"]`.
- `tests/test_tokenizer.py::test_gpt2_byte_encoder_is_reversible_for_all_bytes` expects `bytes_to_unicode()` to return 256 unique string values, with `mapping[ord("A")] == "A"` and `mapping[0] != "\x00"`.
- `tests/test_tokenizer.py::test_bpe_merge_loop_uses_lowest_rank_pair_first` uses ranks `{("l", "o"): 0, ("lo", "w"): 1}` and expects `bpe_encode_piece("low", ranks) == ("low",)`.
- `tests/test_tokenizer.py::test_gpt2_tokenizer_round_trip_contains_words` uses the fixed text `"The quick brown fox"` and expects the decoded string to contain the words `The`, `quick`, `brown`, and `fox` in order.
- `tests/test_tokenizer.py::test_gpt2_tokenizer_eot` uses the fixed text `"hello"` with `add_eot=True` and expects the final token ID to equal `50256`.
- `tests/test_no_transformers.py::test_source_does_not_import_forbidden_tokenizer_libraries` scans `src/` and `scripts/` for `transformers` imports, and scans `src/` for `tiktoken` or `tokenizers` imports. The provided `scripts/build_climbmix_subset.py` may import `tiktoken`.

Expected behavior:

- `pytest -q tests/test_tokenizer.py tests/test_no_transformers.py` passes.
- The no-transformers test scans `src/` and `scripts/` for `import transformers` and `from transformers`; it scans only `src/` for `import tiktoken`, `from tiktoken`, `import tokenizers`, and `from tokenizers`.

Before running tokenizer tests that instantiate `GPT2Tokenizer`, make sure the vocabulary and merge files exist:

```bash
python scripts/download_openai_gpt2.py
pytest -q tests/test_tokenizer.py tests/test_no_transformers.py
```

## Problem 2: Grain Data Loading for Language Modeling

### Context

JAX computations work best when batches are arrays with static shapes. Grain provides deterministic data loading and transformation pipelines. Your dataset object should convert a one-dimensional token array into many shifted blocks. A training block beginning at index `s` uses input tokens `tokens[s:s+block_size]` and target tokens `tokens[s+1:s+block_size+1]`.

The shift is the language-model objective. At position `t`, the model sees tokens up to `t` and predicts the token at `t + 1`. The target operation is not learnable and must not be differentiated.

### Tasks

Required files to edit: `src/data.py`.

Instructor-provided file to create unchanged: `tests/test_data.py`.

#### 2.a Implement a token block source

Create `src/data.py` and include:

```python
from typing import Iterator

import numpy as np


class TokenBlockDataset:
    """Indexable dataset of shifted language-model blocks.

    Each item is a dictionary with:
      `input_ids`: int32 array with shape `(block_size,)`;
      `target_ids`: int32 array with shape `(block_size,)`.
    """

    def __init__(self, tokens: np.ndarray, block_size: int, stride: int) -> None:
        """Store token data and define valid block start positions.

        Args:
            tokens: One-dimensional token array.
            block_size: Number of input tokens per example.
            stride: Distance between adjacent block starts.
        """
        pass

    def __len__(self) -> int:
        """Return the number of valid shifted blocks."""
        pass

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        """Return shifted input and target arrays for one block index."""
        pass
```

Pseudocode:

1. Check that `tokens` is one-dimensional and has at least `block_size + 1` elements.
2. Cast tokens to `np.int32`.
3. Compute starts `0, stride, 2 * stride, ...` where `start + block_size + 1 <= len(tokens)`.
4. For item `index`, slice `x = tokens[start:start + block_size]`.
5. Slice `y = tokens[start + 1:start + block_size + 1]`.
6. Return a dictionary with `input_ids` and `target_ids`.

Correctness checks:

- With tokens `[10, 11, 12, 13, 14]`, `block_size=3`, and `stride=1`, the first input is `[10, 11, 12]` and first target is `[11, 12, 13]`.
- The number of examples is `len(tokens) - block_size` when `stride=1`.

#### 2.b Build Grain loaders

Add to `src/data.py`:

```python
def load_tokens(path: str) -> np.ndarray:
    """Load a one-dimensional NumPy token array from disk as int32.

    Args:
        path: Path to a `.npy` file.

    Returns:
        One-dimensional int32 token array.
    """
    pass


def make_grain_loader(
    tokens: np.ndarray,
    block_size: int,
    batch_size: int,
    stride: int,
    shuffle: bool,
    seed: int,
) -> Iterator[dict[str, np.ndarray]]:
    """Create a Grain iterator over batched language-model examples.

    Args:
        tokens: One-dimensional token array.
        block_size: Number of input tokens per example.
        batch_size: Number of examples per batch.
        stride: Distance between adjacent block starts.
        shuffle: Whether to shuffle examples deterministically.
        seed: Seed used by Grain shuffling.

    Returns:
        Iterator yielding dictionaries with shapes `(batch_size, block_size)`.
    """
    pass
```

Pseudocode:

1. Wrap `TokenBlockDataset` with `grain.MapDataset.source`.
2. If `shuffle` is true, apply Grain shuffle with the seed.
3. Batch with `batch_size`, dropping incomplete batches for static shapes.
4. Return an iterator over dictionaries.
5. Ensure `input_ids` and `target_ids` are `np.int32`.

Correctness checks:

- A batch has keys exactly `input_ids` and `target_ids`.
- Both arrays have shape `(batch_size, block_size)`.
- With `shuffle=False`, the first batch preserves sequential shifted blocks.
- With a fixed seed, two shuffled loaders yield the same first batch.

### Tests

These are fixed pytest tests. Create them exactly as shown and use them unchanged.

Create `tests/test_data.py` from the provided file at [`tests/test_data.py`](tests/test_data.py).

Expected test behavior:

- `tests/test_data.py::test_token_block_dataset_shift` uses fixed tokens `[10, 11, 12, 13, 14]`, `block_size=3`, and `stride=1`; it expects `input_ids == [10, 11, 12]` and `target_ids == [11, 12, 13]` for the first item.
- `tests/test_data.py::test_grain_loader_batch_shapes` uses a small one-dimensional token array, `block_size=4`, and `batch_size=2`; it expects batch keys `input_ids` and `target_ids`, each with shape `(2, 4)` and dtype `int32`.
- `tests/test_data.py::test_grain_loader_seed_is_deterministic` creates two shuffled loaders from the same token array and seed; it expects their first batches to be identical.

After you implement the files in Problem 2, run:

```bash
pytest -q tests/test_data.py
```

## Problem 3: GPT-2 Modules in Equinox

### Context

GPT-2 is a stack of pre-layer-norm Transformer blocks. Each block has:

1. layer normalization;
2. masked multi-head self-attention;
3. residual addition;
4. layer normalization;
5. position-wise MLP;
6. residual addition.

For GPT-2-small, token embeddings have shape `(50257, 768)`, positional embeddings have shape `(1024, 768)`, and each of the 12 blocks preserves the residual stream shape `(sequence_length, 768)`. Multi-head attention splits the embedding dimension into `n_head` heads, so each head has width `head_dim = n_embd // n_head`.

Attention logits are:

```text
scores[h, q, k] = dot(Q[h, q, :], K[h, k, :]) / sqrt(head_dim)
```

Causal masking sets logits with `k > q` to a very negative value before softmax, so each query can attend only to previous positions and itself. The mask is not learnable.

### Tasks

Required files to edit: `src/model.py`.

Instructor-provided files to create unchanged: `tests/test_attention.py` and `tests/test_model_shapes.py`.

#### 3.a Implement GELU, layer normalization helpers, and causal masks

Create `src/model.py` and include:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from src.config import GPT2Config


def gelu(x: jax.Array) -> jax.Array:
    """Apply the GPT-2 GELU nonlinearity elementwise.

    Args:
        x: Input array of any shape.

    Returns:
        Array with the same shape as `x`.
    """
    pass


def causal_mask(sequence_length: int) -> jax.Array:
    """Return a boolean lower-triangular causal attention mask.

    Args:
        sequence_length: Number of query and key positions.

    Returns:
        Boolean array with shape `(sequence_length, sequence_length)` where
        true means the key position is visible to the query position.
    """
    pass
```

Pseudocode:

1. GELU uses the GPT-2 approximate formula with `tanh`, not ReLU.
2. The causal mask is lower triangular, including the diagonal.
3. Keep mask creation outside differentiable parameters.

Correctness checks:

- `causal_mask(3)` equals `[[True, False, False], [True, True, False], [True, True, True]]`.
- `gelu(0.0)` is exactly or very close to `0.0`.
- `gelu(1.0)` is between `0.83` and `0.85`.

#### 3.b Implement masked multi-head self-attention

Add to `src/model.py`:

```python
class CausalSelfAttention(eqx.Module):
    """GPT-2 masked multi-head self-attention.

    Input shape is `(sequence_length, n_embd)`.
    Output shape is `(sequence_length, n_embd)`.
    """

    c_attn: eqx.nn.Linear
    c_proj: eqx.nn.Linear
    n_head: int
    n_embd: int

    def __init__(self, config: GPT2Config, key: jax.Array) -> None:
        """Initialize fused QKV projection and output projection."""
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply causal self-attention to one sequence.

        Args:
            x: Residual stream with shape `(sequence_length, n_embd)`.

        Returns:
            Residual update with shape `(sequence_length, n_embd)`.
        """
        pass
```

Pseudocode:

1. Project `x` through one linear layer from `n_embd` to `3 * n_embd`.
2. Split the last dimension into query, key, and value.
3. Reshape each to `(sequence_length, n_head, head_dim)` and transpose to `(n_head, sequence_length, head_dim)`.
4. Compute scaled dot-product scores with shape `(n_head, sequence_length, sequence_length)`.
5. Apply the causal mask by replacing invisible scores with a large negative constant.
6. Apply softmax along the key dimension.
7. Multiply probabilities by values to get per-head outputs.
8. Transpose and reshape back to `(sequence_length, n_embd)`.
9. Apply `c_proj`.

Correctness checks:

- Output shape equals input shape.
- Changing token `x[future_position]` must not change attention output at any earlier query position when all other tokens are fixed.
- Attention probabilities for masked future positions must be numerically zero within `atol=1e-6`.

#### 3.c Implement the MLP, block, and full GPT-2 model

Add to `src/model.py`:

```python
class MLP(eqx.Module):
    """GPT-2 feed-forward network applied independently at each position."""

    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    def __init__(self, config: GPT2Config, key: jax.Array) -> None:
        """Initialize linear layers with hidden width `4 * n_embd`."""
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        """Return the MLP output with the same shape as `x`."""
        pass


class Block(eqx.Module):
    """One pre-layer-norm GPT-2 Transformer block."""

    ln_1: eqx.nn.LayerNorm
    attn: CausalSelfAttention
    ln_2: eqx.nn.LayerNorm
    mlp: MLP

    def __init__(self, config: GPT2Config, key: jax.Array) -> None:
        """Initialize layer norms, attention, and MLP."""
        pass

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply a Transformer block to one sequence."""
        pass


class GPT2(eqx.Module):
    """GPT-2 language model implemented as an Equinox PyTree.

    Input shape is `(sequence_length,)` for one sequence or
    `(batch_size, sequence_length)` through the `batched_call` helper.
    Logit shape is `(sequence_length, vocab_size)` per sequence.
    """

    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    blocks: tuple[Block, ...]
    ln_f: eqx.nn.LayerNorm
    config: GPT2Config

    def __init__(self, config: GPT2Config, key: jax.Array) -> None:
        """Initialize token embeddings, position embeddings, blocks, and final norm."""
        pass

    def __call__(self, input_ids: jax.Array) -> jax.Array:
        """Return logits for one token sequence.

        Args:
            input_ids: Integer token IDs with shape `(sequence_length,)`.

        Returns:
            Logits with shape `(sequence_length, vocab_size)`.
        """
        pass


def batched_call(model: GPT2, input_ids: jax.Array) -> jax.Array:
    """Apply GPT-2 to a batch of token sequences.

    Args:
        model: GPT-2 model.
        input_ids: Integer token IDs with shape `(batch_size, sequence_length)`.

    Returns:
        Logits with shape `(batch_size, sequence_length, vocab_size)`.
    """
    pass
```

Pseudocode:

1. For one sequence, create position IDs `0, 1, ..., sequence_length - 1`.
2. Embed token IDs and position IDs, then add them.
3. Pass the residual stream through every block in order.
4. Apply final layer norm.
5. Compute logits with tied weights: multiply hidden states by the transpose of the token embedding matrix.
6. Do not apply softmax in the model.
7. Use `jax.vmap` in `batched_call` to apply the single-sequence model over the batch dimension.

Correctness checks:

- For config `(vocab_size=100, block_size=8, n_layer=2, n_head=2, n_embd=16)`, input shape `(8,)` produces logits `(8, 100)`.
- Batched input shape `(3, 8)` produces logits `(3, 8, 100)`.
- Passing a sequence longer than `block_size` raises a clear error.
- The output projection uses tied token embedding weights, not a separate learned matrix.

### Tests

These are fixed pytest tests. Create them exactly as shown and use them unchanged.

Create `tests/test_attention.py` from the provided file at [`tests/test_attention.py`](tests/test_attention.py).

Create `tests/test_model_shapes.py` from the provided file at [`tests/test_model_shapes.py`](tests/test_model_shapes.py).

Expected test behavior:

- `tests/test_attention.py::test_causal_mask_values` uses `sequence_length=3` and expects the exact lower-triangular boolean mask `[[True, False, False], [True, True, False], [True, True, True]]`.
- `tests/test_attention.py::test_attention_output_shape` uses a tiny config with `n_embd=16`, `n_head=2`, and an input with shape `(4, 16)`; it expects an output with shape `(4, 16)`.
- `tests/test_attention.py::test_attention_cannot_see_future_tokens` compares attention outputs before and after changing only a future position; it expects earlier output positions to agree within `atol=1e-6`.
- `tests/test_model_shapes.py::test_gpt2_single_sequence_logits_shape` uses config `(vocab_size=100, block_size=8, n_layer=2, n_head=2, n_embd=16)` and input shape `(8,)`; it expects logits with shape `(8, 100)`.
- `tests/test_model_shapes.py::test_gpt2_batched_logits_shape` uses the same config and input shape `(3, 8)`; it expects logits with shape `(3, 8, 100)`.
- `tests/test_model_shapes.py::test_gpt2_rejects_too_long_sequence` uses an input longer than `block_size`; it expects a clear exception rather than silent truncation.

After you implement the files in Problem 3, run:

```bash
pytest -q tests/test_attention.py tests/test_model_shapes.py
```

## Problem 4: Loss, Optimization, and Training

### Context

The model returns logits, not probabilities. For target token `y`, cross-entropy is:

```text
loss(logits, y) = -log_softmax(logits)[y]
```

Use `jax.nn.log_softmax` or an equivalent stable log-sum-exp computation. Do not compute softmax and then take `log`, because large logits can overflow or underflow.

Training differentiates through model parameters and logits. It does not differentiate through integer token IDs, targets, random data loading, or metric printing. Equinox splits PyTrees into differentiable array leaves and static/non-array leaves so that JAX transformations work correctly.

### Tasks

Required files to edit: `src/loss.py`, `src/train.py`, and `src/generate.py`.

Instructor-provided file to create unchanged: `tests/test_loss.py`.

#### 4.a Implement cross-entropy and batch loss

Create `src/loss.py` and include:

```python
import equinox as eqx
import jax
import jax.numpy as jnp

from src.model import GPT2


def cross_entropy_loss(logits: jax.Array, targets: jax.Array) -> jax.Array:
    """Compute mean next-token cross-entropy.

    Args:
        logits: Array with shape `(..., vocab_size)`.
        targets: Integer array with shape `(...)`.

    Returns:
        Scalar mean negative log-likelihood.
    """
    pass


def batch_loss(model: GPT2, input_ids: jax.Array, target_ids: jax.Array) -> jax.Array:
    """Compute mean language-model loss for one batch.

    Args:
        model: GPT-2 model.
        input_ids: Integer array with shape `(batch_size, sequence_length)`.
        target_ids: Integer array with shape `(batch_size, sequence_length)`.

    Returns:
        Scalar mean cross-entropy over batch and sequence positions.
    """
    pass
```

Pseudocode:

1. Convert logits to log-probabilities with a numerically stable operation.
2. Gather the log-probability at each target token ID.
3. Negate and average over all target positions.
4. In `batch_loss`, call `batched_call` and pass logits to `cross_entropy_loss`.

Correctness checks:

- If logits are all zeros over a vocabulary of size 4, the loss is `log(4)`.
- If the correct class logit is much larger than all others, the loss is close to `0`.
- Loss is a scalar JAX array.

#### 4.b Implement one optimizer step

Create `src/train.py` and include:

```python
from typing import Iterator

import equinox as eqx
import jax
import optax

from src.config import GPT2Config, TrainConfig
from src.model import GPT2


def make_optimizer(config: TrainConfig) -> optax.GradientTransformation:
    """Create the AdamW optimizer used for GPT-2 training."""
    pass


@eqx.filter_jit
def train_step(
    model: GPT2,
    opt_state: optax.OptState,
    optimizer: optax.GradientTransformation,
    input_ids: jax.Array,
    target_ids: jax.Array,
) -> tuple[GPT2, optax.OptState, jax.Array]:
    """Run one differentiable optimizer step.

    Args:
        model: Current GPT-2 model.
        opt_state: Current optimizer state.
        optimizer: Optax optimizer.
        input_ids: Batch input IDs with shape `(batch_size, sequence_length)`.
        target_ids: Batch target IDs with shape `(batch_size, sequence_length)`.

    Returns:
        Updated model, updated optimizer state, and scalar loss before update.
    """
    pass


def estimate_loss(
    model: GPT2,
    loader: Iterator[dict[str, jax.Array]],
    num_batches: int,
) -> float:
    """Estimate mean loss over a fixed number of batches without updating parameters."""
    pass


def train(model_config: GPT2Config, train_config: TrainConfig) -> GPT2:
    """Train GPT-2 on processed ClimbMix token arrays.

    Args:
        model_config: GPT-2 architecture settings.
        train_config: Data and optimizer settings.

    Returns:
        The trained GPT-2 model.
    """
    pass


def main() -> None:
    """Parse arguments, load configs, and run training."""
    pass
```

Pseudocode:

1. Initialize a PRNG key from `train_config.seed`.
2. Create a `GPT2` model and an AdamW optimizer.
3. Load train and validation token arrays.
4. Create Grain loaders with `shuffle=True` for training and `shuffle=False` for validation.
5. For each step, get the next batch, convert arrays to JAX arrays, and call `train_step`.
6. In `train_step`, compute loss and gradients with `eqx.filter_value_and_grad`.
7. Use Optax to compute updates.
8. Apply updates with `eqx.apply_updates`.
9. At evaluation intervals, compute validation loss without applying updates.
10. Print step, train loss, and validation loss.

Correctness checks:

- At least one floating-point parameter changes after one `train_step`.
- The loss returned by `train_step` is finite.
- Running `configs/tiny.yaml` for 20 steps prints finite losses.
- On a very small repeated text subset, train loss decreases over 20 to 50 steps.

#### 4.c Implement generation

Create `src/generate.py` and include:

```python
import jax

from src.model import GPT2
from src.tokenizer import GPT2Tokenizer


def sample_next_token(
    logits: jax.Array,
    key: jax.Array,
    temperature: float,
    top_k: int | None,
) -> jax.Array:
    """Sample one token ID from final-position logits.

    Args:
        logits: Vocabulary logits with shape `(vocab_size,)`.
        key: JAX PRNG key.
        temperature: Positive softmax temperature.
        top_k: If not None, keep only the top-k logits before sampling.

    Returns:
        Scalar integer token ID.
    """
    pass


def generate(
    model: GPT2,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_new_tokens: int,
    key: jax.Array,
    temperature: float = 1.0,
    top_k: int | None = 40,
) -> str:
    """Generate text autoregressively from a prompt.

    Args:
        model: GPT-2 model.
        tokenizer: GPT-2 tokenizer wrapper.
        prompt: Initial text prompt.
        max_new_tokens: Number of tokens to sample.
        key: JAX PRNG key.
        temperature: Sampling temperature.
        top_k: Optional top-k truncation.

    Returns:
        Prompt plus generated continuation as decoded text.
    """
    pass


def main() -> None:
    """Parse arguments, optionally load pretrained weights, and print generated text."""
    pass
```

Pseudocode:

1. In `generate`, encode the prompt without adding end-of-text.
2. Repeatedly crop context to the model `block_size`.
3. Run the model on the current context.
4. Take logits from the final position only.
5. Divide logits by temperature.
6. If `top_k` is set, set logits outside the top-k values to a very negative number.
7. Sample the next token with `jax.random.categorical`.
8. Append the sampled token to the token list.
9. Decode all tokens back to text.
10. In `main`, parse `--checkpoint-dir`, `--prompt`, `--max-new-tokens`, `--seed`, `--temperature`, and `--top-k`.
11. Initialize a GPT-2-small model, load weights from `--checkpoint-dir` when it is provided, call `generate`, and print the generated string.

Correctness checks:

- With `max_new_tokens=0`, output equals the decoded prompt tokens.
- Generation never passes a context longer than `block_size` to the model.
- With the same key and deterministic model, generation is reproducible.
- `python -m src.generate --checkpoint-dir checkpoints/openai-gpt2 --prompt "In a shocking finding, scientists discovered" --max-new-tokens 40 --seed 1234` prints decoded text after you implement `src/generate.py` and download pretrained weights.

### Tests

These are fixed pytest tests. Create them exactly as shown and use them unchanged.

Create `tests/test_loss.py` from the provided file at [`tests/test_loss.py`](tests/test_loss.py).

Expected test behavior:

- `tests/test_loss.py::test_cross_entropy_uniform_logits` uses zero logits with shape `(2, 4)` and fixed targets `[0, 3]`; it expects loss `log(4)` within `atol=1e-6`.
- `tests/test_loss.py::test_cross_entropy_confident_correct_class_is_small` uses logits where the target class has a much larger value than all other classes; it expects loss below `1e-3`.
- `tests/test_loss.py::test_train_step_updates_parameter` uses a tiny GPT-2 config and a fixed synthetic batch; it expects finite loss and at least one differentiable floating-point leaf to change after one update.

After you implement the files in Problem 4, run:

```bash
pytest -q tests/test_loss.py
python -m src.train --config configs/tiny.yaml
```

## Problem 5: Load OpenAI GPT-2 Weights Without Transformers

### Context

OpenAI GPT-2-small has 12 layers, 12 heads, embedding width 768, context length 1024, and vocabulary size 50,257. HuggingFace hosts converted OpenAI GPT-2 weights in `safetensors` format. You may use `huggingface_hub` or `hf` CLI to download these files, but your model code and weight loader must not import `transformers`.

The checkpoint tensors use names that correspond to GPT-2 modules:

- `transformer.wte.weight`: token embeddings, shape `(50257, 768)`;
- `transformer.wpe.weight`: position embeddings, shape `(1024, 768)`;
- `transformer.h.{i}.ln_1.weight` and `.bias`: first layer norm in block `i`;
- `transformer.h.{i}.attn.c_attn.weight` and `.bias`: fused QKV projection;
- `transformer.h.{i}.attn.c_proj.weight` and `.bias`: attention output projection;
- `transformer.h.{i}.ln_2.weight` and `.bias`: second layer norm;
- `transformer.h.{i}.mlp.c_fc.weight` and `.bias`: MLP expansion;
- `transformer.h.{i}.mlp.c_proj.weight` and `.bias`: MLP projection;
- `transformer.ln_f.weight` and `.bias`: final layer norm.

Some checkpoint linear weights are stored with orientation `(in_features, out_features)`, while Equinox linear layers store weights as `(out_features, in_features)`. Your loader must transpose linear matrices when needed and must not transpose embedding or layer-norm vectors.

### Tasks

Required files to edit: `src/load_openai.py` and `src/check_pretrained.py`.

Instructor-provided files to create unchanged: `scripts/download_openai_gpt2.py` and `tests/test_weight_loading.py`.

#### 5.a Download checkpoint files

Create `scripts/download_openai_gpt2.py` from the instructor-provided helper script at [`scripts/download_openai_gpt2.py`](scripts/download_openai_gpt2.py); use it unchanged.

Equivalent CLI command for downloading the checkpoint files after setup:

```bash
huggingface-cli download openai-community/gpt2 \
  model.safetensors config.json vocab.json merges.txt \
  --local-dir checkpoints/openai-gpt2
```

Correctness checks:

- `checkpoints/openai-gpt2/model.safetensors` exists.
- `checkpoints/openai-gpt2/config.json` reports GPT-2-small dimensions.
- No `src/` file imports `transformers`, `tiktoken`, or `tokenizers`; no `scripts/` file imports `transformers`.

#### 5.b Implement a safetensors checkpoint reader

Create `src/load_openai.py` and include:

```python
from collections.abc import Mapping

import jax
import jax.numpy as jnp

from src.config import GPT2Config
from src.model import GPT2


def load_safetensors(path: str) -> dict[str, jax.Array]:
    """Load a safetensors file into CPU JAX arrays.

    Args:
        path: Path to `model.safetensors`.

    Returns:
        Dictionary mapping checkpoint tensor names to JAX arrays.
    """
    pass


def validate_openai_gpt2_config(config: GPT2Config, tensors: Mapping[str, jax.Array]) -> None:
    """Validate that model config and checkpoint tensor shapes are compatible.

    Args:
        config: Target GPT-2 config.
        tensors: Loaded checkpoint tensors.

    Raises:
        ValueError: If a required tensor is missing or has an incompatible shape.
    """
    pass


def orient_linear_weight(checkpoint_weight: jax.Array, expected_shape: tuple[int, int]) -> jax.Array:
    """Return a checkpoint linear weight in Equinox `(out_features, in_features)` orientation.

    Args:
        checkpoint_weight: Matrix from the OpenAI checkpoint.
        expected_shape: Shape expected by the destination Equinox linear layer.

    Returns:
        Matrix with shape `expected_shape`.

    Raises:
        ValueError: If neither the checkpoint matrix nor its transpose has the
            expected shape.
    """
    pass


def assign_openai_weights(model: GPT2, tensors: Mapping[str, jax.Array]) -> GPT2:
    """Return a copy of `model` with OpenAI GPT-2 weights assigned.

    Args:
        model: GPT-2-small Equinox model with matching architecture.
        tensors: Loaded safetensors checkpoint.

    Returns:
        New model PyTree containing checkpoint weights.
    """
    pass


def load_openai_gpt2(model: GPT2, checkpoint_dir: str) -> GPT2:
    """Load OpenAI GPT-2-small weights from a checkpoint directory.

    Args:
        model: Initialized GPT-2-small model.
        checkpoint_dir: Directory containing `model.safetensors`.

    Returns:
        Model with pretrained weights.
    """
    pass
```

Pseudocode:

1. Use `safetensors.numpy.load_file` or `safetensors.flax.load_file` to read arrays.
2. Convert arrays to JAX arrays with dtype matching your model parameters.
3. Check every required key before modifying the model.
4. Assign embeddings directly.
5. Assign layer-norm scale and bias directly.
6. Implement `orient_linear_weight` so it returns the checkpoint matrix directly when it already has `expected_shape`, returns its transpose when the transpose has `expected_shape`, and raises `ValueError` otherwise.
7. For each Equinox `Linear`, call `orient_linear_weight` before assigning the checkpoint matrix.
8. Use `eqx.tree_at` or a structured replacement helper so the returned model remains an Equinox PyTree.
9. Confirm tied output logits still use `wte.weight`.

Correctness checks:

- The loaded token embedding matrix has shape `(50257, 768)`.
- The loaded position embedding matrix has shape `(1024, 768)`.
- All 12 block indices are present.
- The model returns finite logits for the prompt `"Hello, my name is"`.
- Re-loading the same checkpoint into two freshly initialized models gives identical logits for the same input.

#### 5.c Verify pretrained behavior

Create `src/check_pretrained.py` and include:

```python
def main() -> None:
    """Load OpenAI GPT-2 weights and run deterministic correctness checks.

    The script must print top next-token predictions for a fixed prompt and
    verify that logits are finite and stable across repeated loads.
    """
    pass
```

Pseudocode:

1. Create `GPT2Config` for GPT-2-small.
2. Initialize a model with any seed.
3. Load weights from `checkpoint_dir`.
4. Tokenize `"Hello, my name is"`.
5. Run the model and inspect final-position logits.
6. Print the top 10 token IDs and decoded strings.
7. Load the same checkpoint into a second fresh model.
8. Assert that both models produce equal logits within `atol=1e-5`.

Correctness checks:

- All logits are finite.
- The top predictions are plausible English continuations or punctuation, not random high-index artifacts.
- Repeated load agreement is within `atol=1e-5`.
- `python -m src.check_pretrained --checkpoint-dir checkpoints/openai-gpt2` exits successfully.

### Tests

These are fixed pytest tests. Create them exactly as shown and use them unchanged.

Create `tests/test_weight_loading.py` from the provided file at [`tests/test_weight_loading.py`](tests/test_weight_loading.py).

Expected test behavior:

- `tests/test_weight_loading.py::test_openai_config_matches_gpt2_small_shapes` uses a `GPT2Config` with GPT-2-small dimensions and a minimal mapping of expected checkpoint tensor shapes; it expects validation to succeed.
- `tests/test_weight_loading.py::test_loader_rejects_missing_required_tensor` removes one required tensor key from the mapping; it expects `ValueError` with a clear missing-tensor message.
- `tests/test_weight_loading.py::test_linear_weight_orientation_rule` uses a fixed matrix with checkpoint shape `(in_features, out_features)` and an Equinox linear layer expecting `(out_features, in_features)`; it expects the assigned weight to equal the transpose within `atol=1e-6`.

After you create the provided downloader and implement the remaining files in Problem 5, run:

```bash
pytest -q tests/test_weight_loading.py
python scripts/download_openai_gpt2.py
python -m src.check_pretrained --checkpoint-dir checkpoints/openai-gpt2
```

## Problem 6: Integration and Training Behavior

### Context

A correct model can still fail as a system if data batches are shifted incorrectly, if the loss is computed over the wrong axis, if parameters are accidentally static, or if pretrained weights overwrite the wrong leaves. Integration checks should be small, deterministic, and targeted.

For a language model trained from scratch on a tiny subset, do not expect fluent text after a few steps. You should expect finite logits, finite loss, a decreasing training loss on repeated data, and generated strings that decode without tokenizer errors. For pretrained GPT-2, you should expect coherent short continuations before any fine-tuning.

### Tasks

Required files to edit: `README.md`.

Instructor-provided file to create unchanged: `tests/test_integration.py`.

#### 6.a Add a tiny overfit check

Create `tests/test_integration.py` from the provided file at [`tests/test_integration.py`](tests/test_integration.py).

Pseudocode:

1. Create a tiny config with vocabulary size at least 64, block size 16, 2 layers, 2 heads, and embedding width 32.
2. Create a repeated synthetic batch where targets are inputs shifted by one.
3. Run 25 to 50 optimizer steps with a fixed seed.
4. Record the first and final loss.
5. Assert that the final loss is lower than the first loss.

Correctness checks:

- The test is deterministic.
- The test runs on CPU in a reasonable time.
- The loss decrease is not required to be large, but it must be positive and finite.
- `tests/test_integration.py::test_tiny_model_can_overfit_repeated_batch` records the first and final loss on one fixed repeated synthetic batch; it expects the final finite loss to be lower than the first finite loss.

#### 6.b Run the real-data smoke test

Pseudocode:

1. Skip the test with a clear message if processed token files do not exist.
2. Load `train_tokens.npy`.
3. Create a Grain loader with `block_size=32` and `batch_size=2`.
4. Initialize a tiny GPT-2.
5. Compute one batch loss.
6. Assert that the loss is finite and positive.

Correctness checks:

- The test does not download data.
- The test fails only if files exist but are malformed or incompatible.
- The batch shape is exactly `(2, 32)`.
- `tests/test_integration.py::test_processed_climbmix_batch_smoke` uses `data/processed/climbmix_gpt2_1m/train_tokens.npy` when it exists; it expects one tiny-model batch loss to be finite and positive.

#### 6.c Document expected command sequence

Create or update `README.md` with the exact end-to-end sequence. This sequence is for a completed implementation, not for the initial skeleton:

```bash
python scripts/download_climbmix.py --dataset gvlassis/ClimbMix --split train --output-dir data/raw/climbmix --max-docs-per-cluster 2000 --seed 1234
python scripts/download_openai_gpt2.py
python scripts/build_climbmix_subset.py --input-dir data/raw/climbmix --output-dir data/processed/climbmix_gpt2_1m --manifest-dir data/manifests --tokens-per-cluster 50000 --block-size 1024 --val-fraction 0.01 --test-fraction 0.01 --seed 1234
pytest -q
python -m src.train --config configs/tiny.yaml
python -m src.check_pretrained --checkpoint-dir checkpoints/openai-gpt2
```

Correctness checks:

- The commands run in this order from a clean checkout after environment setup and after all referenced source files have been implemented.
- The full test suite does not require the large pretrained checkpoint unless the relevant files are already present.
- Training and pretrained checks are separate commands.

## End-to-End Checks

When your implementation is complete, run the following checks in order.

1. Static source check:

```bash
pytest -q tests/test_no_transformers.py
```

Expected behavior: the test passes, proving `src/` does not import `transformers`, `tiktoken`, or `tokenizers`, and `scripts/` does not import `transformers`.

2. Tokenizer and data checks:

```bash
python scripts/download_openai_gpt2.py
pytest -q tests/test_tokenizer.py tests/test_data.py
```

Expected behavior: tokenization round trips simple text, shifted blocks are correct, and Grain batches have static shapes.

3. Model and loss checks:

```bash
pytest -q tests/test_attention.py tests/test_model_shapes.py tests/test_loss.py
```

Expected behavior: causal masks are correct, attention cannot see future tokens, logits have expected shapes, cross-entropy matches fixed numerical properties, and at least one train step updates parameters.

4. ClimbMix subset construction:

```bash
python scripts/download_climbmix.py \
  --dataset gvlassis/ClimbMix \
  --split train \
  --output-dir data/raw/climbmix \
  --max-docs-per-cluster 2000 \
  --seed 1234

python scripts/build_climbmix_subset.py \
  --input-dir data/raw/climbmix \
  --output-dir data/processed/climbmix_gpt2_1m \
  --manifest-dir data/manifests \
  --tokens-per-cluster 50000 \
  --block-size 1024 \
  --val-fraction 0.01 \
  --test-fraction 0.01 \
  --seed 1234
```

Expected behavior: the manifest records all 20 clusters, each cluster contributes the target token count, token IDs are in GPT-2 range, and split files exist.

5. Tiny training run:

```bash
python -m src.train --config configs/tiny.yaml
```

Expected behavior: the script prints finite train and validation losses, and training loss decreases on repeated or sufficiently small data.

6. OpenAI GPT-2 weight loading:

```bash
python scripts/download_openai_gpt2.py
python -m src.check_pretrained --checkpoint-dir checkpoints/openai-gpt2
```

Expected behavior: the script loads `model.safetensors`, assigns every GPT-2-small tensor to your Equinox model, prints plausible top next-token predictions for `"Hello, my name is"`, and verifies repeated-load logits agree within `atol=1e-5`.

7. Optional pretrained generation:

```bash
python -m src.generate --checkpoint-dir checkpoints/openai-gpt2 --prompt "In a shocking finding, scientists discovered" --max-new-tokens 40 --seed 1234
```

Expected behavior: the output is valid decoded text. It should look like a GPT-2 continuation before any fine-tuning. If it is repetitive or low quality, that is not automatically a correctness failure; if logits are non-finite, token IDs are outside range, or generated text is empty despite positive `max_new_tokens`, your implementation is incorrect.
