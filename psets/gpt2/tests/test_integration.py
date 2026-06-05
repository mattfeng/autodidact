from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src.config import GPT2Config, TrainConfig
from src.data import load_tokens, make_grain_loader
from src.loss import batch_loss
from src.model import GPT2
from src.train import make_optimizer, train_step


def test_tiny_model_can_overfit_repeated_batch() -> None:
    """A tiny GPT-2 should reduce loss on one repeated batch after several updates."""
    model_config = GPT2Config(
        vocab_size=64,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )
    train_config = TrainConfig(
        data_dir="unused",
        batch_size=4,
        learning_rate=3e-3,
        weight_decay=0.0,
        max_steps=30,
        eval_interval=10,
        eval_batches=1,
        seed=1234,
    )
    model = GPT2(model_config, jax.random.PRNGKey(0))
    optimizer = make_optimizer(train_config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    row = jnp.arange(16, dtype=jnp.int32)
    input_ids = jnp.tile(row, (4, 1))
    target_ids = jnp.tile((row + 1) % model_config.vocab_size, (4, 1))

    first_loss = batch_loss(model, input_ids, target_ids)
    for _ in range(30):
        model, opt_state, _ = train_step(model, opt_state, optimizer, input_ids, target_ids)
    final_loss = batch_loss(model, input_ids, target_ids)

    assert jnp.isfinite(first_loss)
    assert jnp.isfinite(final_loss)
    assert float(final_loss) < float(first_loss)


def test_processed_climbmix_batch_smoke() -> None:
    """A processed ClimbMix batch should run through the tiny model and loss."""
    token_path = Path("data/processed/climbmix_gpt2_1m/train_tokens.npy")
    if not token_path.exists():
        pytest.skip("processed ClimbMix token file is not present")

    tokens = load_tokens(str(token_path))
    loader = make_grain_loader(tokens, block_size=32, batch_size=2, stride=32, shuffle=False, seed=1234)
    batch = next(iter(loader))
    assert batch["input_ids"].shape == (2, 32)
    assert batch["target_ids"].shape == (2, 32)

    model_config = GPT2Config(
        vocab_size=50257,
        block_size=32,
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )
    model = GPT2(model_config, jax.random.PRNGKey(0))
    loss = batch_loss(
        model,
        jnp.asarray(batch["input_ids"]),
        jnp.asarray(batch["target_ids"]),
    )
    assert jnp.isfinite(loss)
    assert float(loss) > 0.0
