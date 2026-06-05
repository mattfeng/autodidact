from pathlib import Path

import numpy as np
import pytest

from src.tokenizer import (
    GPT2Tokenizer,
    bpe_encode_piece,
    bytes_to_unicode,
    gpt2_pretokenize,
    load_merges,
)


def test_gpt2_pretokenize_keeps_leading_spaces() -> None:
    """GPT-2 pre-tokenization should attach leading spaces to word and number pieces."""
    assert gpt2_pretokenize("Hello, world! 123") == ["Hello", ",", " world", "!", " 123"]


def test_gpt2_byte_encoder_is_reversible_for_all_bytes() -> None:
    """GPT-2 byte encoding should assign one unique Unicode string to each byte."""
    mapping = bytes_to_unicode()
    assert len(mapping) == 256
    assert len(set(mapping.values())) == 256
    assert mapping[ord("A")] == "A"
    assert mapping[0] != "\x00"


def test_bpe_merge_loop_uses_lowest_rank_pair_first() -> None:
    """The BPE loop should repeatedly merge the best-ranked adjacent pair."""
    ranks = {("l", "o"): 0, ("lo", "w"): 1}
    assert bpe_encode_piece("low", ranks) == ("low",)


def test_gpt2_tokenizer_round_trip_contains_words() -> None:
    """Encoding and decoding a simple sentence should preserve readable words."""
    if not Path("checkpoints/openai-gpt2/vocab.json").exists():
        pytest.skip("run python scripts/download_openai_gpt2.py first")
    if not Path("checkpoints/openai-gpt2/merges.txt").exists():
        pytest.skip("run python scripts/download_openai_gpt2.py first")
    tokenizer = GPT2Tokenizer()
    decoded = tokenizer.decode(tokenizer.encode("The quick brown fox"))
    words = decoded.split()
    assert words[:4] == ["The", "quick", "brown", "fox"]


def test_gpt2_tokenizer_eot() -> None:
    """Encoding with add_eot=True should append token ID 50256."""
    if not Path("checkpoints/openai-gpt2/vocab.json").exists():
        pytest.skip("run python scripts/download_openai_gpt2.py first")
    if not Path("checkpoints/openai-gpt2/merges.txt").exists():
        pytest.skip("run python scripts/download_openai_gpt2.py first")
    tokenizer = GPT2Tokenizer()
    tokens = tokenizer.encode("hello", add_eot=True)
    assert tokens.ndim == 1
    assert tokens.dtype == np.int32
    assert int(tokens[-1]) == 50256
