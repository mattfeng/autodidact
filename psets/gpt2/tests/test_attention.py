import jax
import jax.numpy as jnp
import numpy as np

from src.config import GPT2Config
from src.model import CausalSelfAttention, GPT2, batched_call, causal_mask


def tiny_config() -> GPT2Config:
    """Return a small config for CPU tests."""
    return GPT2Config(
        vocab_size=100,
        block_size=8,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )


def test_causal_mask_values() -> None:
    """The causal mask should be lower triangular with a visible diagonal."""
    expected = jnp.array(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    np.testing.assert_array_equal(np.asarray(causal_mask(3)), np.asarray(expected))


def test_attention_output_shape() -> None:
    """CausalSelfAttention should preserve `(sequence_length, n_embd)` shape."""
    attn = CausalSelfAttention(tiny_config(), jax.random.PRNGKey(0))
    x = jnp.ones((4, 16))
    assert attn(x).shape == (4, 16)


def test_attention_cannot_see_future_tokens() -> None:
    """Changing a future token should not change earlier attention outputs."""
    attn = CausalSelfAttention(tiny_config(), jax.random.PRNGKey(1))
    x = jnp.arange(4 * 16, dtype=jnp.float32).reshape(4, 16) / 100.0
    changed = x.at[3].add(1000.0)
    out = attn(x)
    changed_out = attn(changed)
    np.testing.assert_allclose(np.asarray(out[:3]), np.asarray(changed_out[:3]), atol=1e-6)
