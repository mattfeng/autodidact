import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.config import GPT2Config
from src.load_openai import orient_linear_weight, validate_openai_gpt2_config


def gpt2_small_config() -> GPT2Config:
    """Return the OpenAI GPT-2-small architecture config."""
    return GPT2Config(
        vocab_size=50257,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )


def fake_checkpoint_shapes() -> dict[str, jax.ShapeDtypeStruct]:
    """Return shape-only checkpoint tensors for GPT-2-small validation tests."""
    tensors: dict[str, jax.ShapeDtypeStruct] = {
        "transformer.wte.weight": jax.ShapeDtypeStruct((50257, 768), jnp.float32),
        "transformer.wpe.weight": jax.ShapeDtypeStruct((1024, 768), jnp.float32),
        "transformer.ln_f.weight": jax.ShapeDtypeStruct((768,), jnp.float32),
        "transformer.ln_f.bias": jax.ShapeDtypeStruct((768,), jnp.float32),
    }
    for i in range(12):
        prefix = f"transformer.h.{i}"
        tensors[f"{prefix}.ln_1.weight"] = jax.ShapeDtypeStruct((768,), jnp.float32)
        tensors[f"{prefix}.ln_1.bias"] = jax.ShapeDtypeStruct((768,), jnp.float32)
        tensors[f"{prefix}.attn.c_attn.weight"] = jax.ShapeDtypeStruct((768, 2304), jnp.float32)
        tensors[f"{prefix}.attn.c_attn.bias"] = jax.ShapeDtypeStruct((2304,), jnp.float32)
        tensors[f"{prefix}.attn.c_proj.weight"] = jax.ShapeDtypeStruct((768, 768), jnp.float32)
        tensors[f"{prefix}.attn.c_proj.bias"] = jax.ShapeDtypeStruct((768,), jnp.float32)
        tensors[f"{prefix}.ln_2.weight"] = jax.ShapeDtypeStruct((768,), jnp.float32)
        tensors[f"{prefix}.ln_2.bias"] = jax.ShapeDtypeStruct((768,), jnp.float32)
        tensors[f"{prefix}.mlp.c_fc.weight"] = jax.ShapeDtypeStruct((768, 3072), jnp.float32)
        tensors[f"{prefix}.mlp.c_fc.bias"] = jax.ShapeDtypeStruct((3072,), jnp.float32)
        tensors[f"{prefix}.mlp.c_proj.weight"] = jax.ShapeDtypeStruct((3072, 768), jnp.float32)
        tensors[f"{prefix}.mlp.c_proj.bias"] = jax.ShapeDtypeStruct((768,), jnp.float32)
    return tensors


def test_openai_config_matches_gpt2_small_shapes() -> None:
    """GPT-2-small config should match the expected OpenAI checkpoint shapes."""
    validate_openai_gpt2_config(gpt2_small_config(), fake_checkpoint_shapes())


def test_loader_rejects_missing_required_tensor() -> None:
    """The loader should raise a clear error if a required checkpoint tensor is missing."""
    tensors = fake_checkpoint_shapes()
    del tensors["transformer.h.0.attn.c_attn.weight"]
    with pytest.raises(ValueError, match="missing"):
        validate_openai_gpt2_config(gpt2_small_config(), tensors)


def test_linear_weight_orientation_rule() -> None:
    """Checkpoint linear matrices should be transposed when Equinox expects the opposite orientation."""
    checkpoint = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    oriented = orient_linear_weight(checkpoint, expected_shape=(3, 2))
    np.testing.assert_array_equal(np.asarray(oriented), np.asarray(checkpoint.T))
