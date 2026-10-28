"""Document masking for packed PMD windows: loader segments, mask, positions, model."""

import importlib
from types import SimpleNamespace

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from theseus.base import ExecutionSpec
from theseus.config import build, configuration
from theseus.data import datasets
from theseus.data.datasets import DatasetConfig
from theseus.data.tokenizer import TokenizerConfig, get_tokenizer
from theseus.model.attention.base import SelfAttention
from theseus.training.flywheel import pmd
from theseus.training.flywheel.strategy import Sampling, Strategy

EOT = 7


def spec_at(path):
    return SimpleNamespace(hardware=SimpleNamespace(hosts=[
        SimpleNamespace(cluster=SimpleNamespace(data_dir=path))
    ]))


def pmd_reader(tmp_path, data, block_size, eot_token):
    path = tmp_path / "data"
    path.mkdir()
    np.asarray(data, dtype=np.uint32).tofile(path / "train.bin")
    return pmd.MemmapDataset(spec_at(tmp_path), block_size, "data", eot_token=eot_token)


def test_pmd_segments_split_after_each_eot(tmp_path):
    # Row 0 starts mid-document; row 1 opens on an <eos> and holds a doubled one.
    data = [1, 2, EOT, 3, 4, EOT, 5, 6, EOT, 8, 9, EOT, EOT, 3, 4, 5, 6]
    reader = pmd_reader(tmp_path, data, 8, EOT)

    rows = reader._read_rows(np.array([0, 1]), "train")

    np.testing.assert_array_equal(
        rows["segment_ids"],
        [[0, 0, 0, 1, 1, 1, 2, 2], [0, 1, 1, 1, 2, 3, 3, 3]],
    )
    assert rows["segment_ids"].dtype == np.int32


def test_pmd_without_eot_token_keeps_batch_format(tmp_path):
    reader = pmd_reader(tmp_path, np.arange(17), 8, None)

    assert set(reader._read_rows(np.array([0, 1]), "train")) == {
        "x", "y", "padding_mask",
    }


def masking_config():
    cfg = build(DatasetConfig, TokenizerConfig)
    OmegaConf.update(cfg, "data.document_masking", True)
    return configuration(cfg)


def local(dataset):
    return type("LocalDataset", (dataset,), {"DATASET_KEY": "local"})


def test_strategy_gives_pmd_readers_the_tokenizer_eot(tmp_path, monkeypatch):
    reader = Mock()
    monkeypatch.setattr("theseus.training.flywheel.pmd.MemmapDataset", reader)
    spec = ExecutionSpec.local(str(tmp_path))

    with masking_config():
        Strategy(spec, 8, [Sampling(local(datasets.FineWeb), 1.0)])
        eot_token = get_tokenizer().eot_token

    reader.assert_called_once_with(spec, 8, "local", "", eot_token)


def test_strategy_rejects_document_masking_for_padded_datasets(tmp_path):
    spec = ExecutionSpec.local(str(tmp_path))

    with masking_config(), pytest.raises(ValueError, match="only PMD"):
        Strategy(spec, 8, [Sampling(local(datasets.Dataset), 1.0)])


def test_build_mask_is_block_diagonal_causal():
    segment_ids = jnp.array([[0, 0, 1, 1, 1]])
    same_doc = segment_ids[0][:, None] == segment_ids[0][None, :]
    expected = np.tril(np.ones((5, 5), dtype=bool)) & np.asarray(same_doc)

    # Segments alone must still mask; padding masks combine with them.
    without_padding = SelfAttention.build_mask(
        SimpleNamespace(), 5, None, segment_ids=segment_ids
    )
    padding = jnp.array([[False, True, True, True, True]])
    with_padding = SelfAttention.build_mask(
        SimpleNamespace(), 5, padding, segment_ids=segment_ids
    )

    np.testing.assert_array_equal(without_padding[0, 0], expected)
    np.testing.assert_array_equal(
        with_padding[0, 0], expected & np.asarray(padding[0])[None, :]
    )


def test_positions_restart_at_each_segment():
    segment_ids = jnp.array([[0, 0, 0, 1, 1, 2, 2, 2]])

    positions = SelfAttention._positions_from_padding(None, 8, None, segment_ids)

    np.testing.assert_array_equal(positions, [[0, 1, 2, 0, 1, 0, 1, 2]])


def gpt_logits(tokens, **kwargs):
    from theseus.model.models.base import GPT

    shape = dict(
        n_layers=2, n_embd=64, rope=True, block_size=16, dropout=0.0, vocab_size=32
    )
    cfg = build(*GPT.gather())
    OmegaConf.set_struct(cfg, False)
    for key, value in {**shape, "n_head": 4, "bias": True}.items():
        OmegaConf.update(cfg, f"architecture.{key}", value)
    OmegaConf.update(cfg, "architecture.dtype.activation", "float32")
    OmegaConf.set_struct(cfg, True)
    with configuration(cfg):
        model = GPT(**shape)
        params = model.init(jax.random.PRNGKey(0), tokens[:, :1])["params"]
        logits, _ = model.apply({"params": params}, tokens, deterministic=True, **kwargs)
    return np.asarray(logits)


def test_packed_document_matches_document_alone():
    first, second = [3, 4, 5, EOT], [9, 10, 11, 12, 13]
    packed = jnp.array([first + second])
    segment_ids = jnp.array([[0, 0, 0, 0, 1, 1, 1, 1, 1]])
    padding_mask = jnp.ones_like(packed, dtype=jnp.bool_)

    packed_logits = gpt_logits(
        packed, padding_mask=padding_mask, segment_ids=segment_ids
    )
    alone_logits = gpt_logits(
        jnp.array([second]), padding_mask=jnp.ones((1, 5), dtype=jnp.bool_)
    )

    np.testing.assert_allclose(packed_logits[0, 4:], alone_logits[0], atol=1e-5)


def test_single_segment_is_bit_identical_to_unmasked():
    tokens = jnp.array([[3, 4, 5, EOT, 9, 10, 11]])
    padding_mask = jnp.ones_like(tokens, dtype=jnp.bool_)

    np.testing.assert_array_equal(
        gpt_logits(tokens, padding_mask=padding_mask),
        gpt_logits(
            tokens, padding_mask=padding_mask, segment_ids=jnp.zeros_like(tokens)
        ),
    )


@pytest.mark.parametrize(
    ("module", "model"),
    [
        ("theseus.model.models.contrib.llama", "Llama"),
        ("theseus.model.models.thoughtbubbles", "Thoughtbubbles"),
    ],
)
def test_trainer_rejects_document_masking_for_unsupported_models(
    tmp_path, module, model
):
    from theseus.training.base import BaseTrainer

    trainer = SimpleNamespace(MODEL=getattr(importlib.import_module(module), model))

    with masking_config(), pytest.raises(ValueError, match="does not support"):
        BaseTrainer._init_data(trainer, ExecutionSpec.local(str(tmp_path)))


def test_thoughtbubbles_refuses_segment_ids_when_called_directly():
    from theseus.model.models.thoughtbubbles import Thoughtbubbles

    with pytest.raises(NotImplementedError, match="segment_ids"):
        Thoughtbubbles.decode(None, None, segment_ids=jnp.zeros((1, 4)))
