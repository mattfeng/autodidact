import jax
import jax.numpy as jnp
import pytest

from src.config import GPT2Config
from src.model import GPT2, batched_call


def tiny_config() -> GPT2Config:
    """Return a small config for CPU shape tests."""
    return GPT2Config(
        vocab_size=100,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )


def test_gpt2_single_sequence_logits_shape() -> None:
    """A single input sequence should produce per-position vocabulary logits."""
    model = GPT2(tiny_config(), jax.random.PRNGKey(0))
    logits = model(jnp.arange(8, dtype=jnp.int32))
    assert logits.shape == (8, 100)


def test_gpt2_batched_logits_shape() -> None:
    """A batched input should produce batched per-position vocabulary logits."""
    model = GPT2(tiny_config(), jax.random.PRNGKey(1))
    input_ids = jnp.tile(jnp.arange(8, dtype=jnp.int32), (3, 1))
    logits = batched_call(model, input_ids)
    assert logits.shape == (3, 8, 100)


def test_gpt2_rejects_too_long_sequence() -> None:
    """The model should reject inputs longer than its positional embedding table."""
    model = GPT2(tiny_config(), jax.random.PRNGKey(2))
    with pytest.raises(ValueError):
        model(jnp.arange(9, dtype=jnp.int32))
