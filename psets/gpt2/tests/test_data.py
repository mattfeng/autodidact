import numpy as np

from src.data import TokenBlockDataset, make_grain_loader


def test_token_block_dataset_shift() -> None:
    """Targets should be inputs shifted left by one token."""
    dataset = TokenBlockDataset(np.array([10, 11, 12, 13, 14]), block_size=3, stride=1)
    item = dataset[0]
    np.testing.assert_array_equal(item["input_ids"], np.array([10, 11, 12], dtype=np.int32))
    np.testing.assert_array_equal(item["target_ids"], np.array([11, 12, 13], dtype=np.int32))
    assert len(dataset) == 2


def test_grain_loader_batch_shapes() -> None:
    """The Grain loader should return static batched arrays."""
    tokens = np.arange(20, dtype=np.int32)
    loader = make_grain_loader(tokens, block_size=4, batch_size=2, stride=1, shuffle=False, seed=1234)
    batch = next(iter(loader))
    assert set(batch) == {"input_ids", "target_ids"}
    assert batch["input_ids"].shape == (2, 4)
    assert batch["target_ids"].shape == (2, 4)
    assert batch["input_ids"].dtype == np.int32
    assert batch["target_ids"].dtype == np.int32


def test_grain_loader_seed_is_deterministic() -> None:
    """Two shuffled Grain loaders with the same seed should agree."""
    tokens = np.arange(40, dtype=np.int32)
    loader_a = make_grain_loader(tokens, block_size=4, batch_size=3, stride=1, shuffle=True, seed=7)
    loader_b = make_grain_loader(tokens, block_size=4, batch_size=3, stride=1, shuffle=True, seed=7)
    batch_a = next(iter(loader_a))
    batch_b = next(iter(loader_b))
    np.testing.assert_array_equal(batch_a["input_ids"], batch_b["input_ids"])
    np.testing.assert_array_equal(batch_a["target_ids"], batch_b["target_ids"])
