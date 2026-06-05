import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from src.config import GPT2Config, TrainConfig
from src.loss import cross_entropy_loss
from src.model import GPT2
from src.train import make_optimizer, train_step


def test_cross_entropy_uniform_logits() -> None:
    """Uniform logits over four classes should have loss log(4)."""
    logits = jnp.zeros((2, 4), dtype=jnp.float32)
    targets = jnp.array([0, 3], dtype=jnp.int32)
    loss = cross_entropy_loss(logits, targets)
    np.testing.assert_allclose(np.asarray(loss), np.log(4.0), atol=1e-6)


def test_cross_entropy_confident_correct_class_is_small() -> None:
    """A very large correct logit should produce near-zero loss."""
    logits = jnp.array([[20.0, -5.0, -5.0], [-5.0, -5.0, 20.0]], dtype=jnp.float32)
    targets = jnp.array([0, 2], dtype=jnp.int32)
    assert float(cross_entropy_loss(logits, targets)) < 1e-3


def test_train_step_updates_parameter() -> None:
    """One train step should change at least one differentiable parameter."""
    model_config = GPT2Config(
        vocab_size=32,
        block_size=8,
        n_layer=1,
        n_head=2,
        n_embd=16,
        dropout=0.0,
        layer_norm_eps=1e-5,
    )
    train_config = TrainConfig(
        data_dir="unused",
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        max_steps=1,
        eval_interval=1,
        eval_batches=1,
        seed=1234,
    )
    model = GPT2(model_config, jax.random.PRNGKey(0))
    optimizer = make_optimizer(train_config)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    input_ids = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]], dtype=jnp.int32)
    target_ids = jnp.array([[1, 2, 3, 4, 5, 6, 7, 8], [6, 5, 4, 3, 2, 1, 0, 9]], dtype=jnp.int32)

    updated, _, loss = train_step(model, opt_state, optimizer, input_ids, target_ids)

    assert jnp.isfinite(loss)
    before = eqx.filter(model, eqx.is_inexact_array)
    after = eqx.filter(updated, eqx.is_inexact_array)
    changed = [
        jnp.any(a != b)
        for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after))
        if a.shape == b.shape
    ]
    assert any(bool(value) for value in changed)
