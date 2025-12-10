import torch

from landauer_llm import utils


def test_build_vocab_encode_decode_roundtrip() -> None:
    text = "abca"
    stoi, itos = utils.build_vocab(text)

    tokens = utils.encode(text, stoi)
    decoded = utils.decode(tokens, itos)

    assert decoded == text
    assert set(stoi.keys()) == {"a", "b", "c"}
    assert set(itos.values()) == {"a", "b", "c"}


def test_split_dataset_respects_fraction() -> None:
    data = torch.arange(10)
    train, val = utils.split_dataset(data, val_fraction=0.2)

    assert len(train) == 8
    assert len(val) == 2
    assert torch.equal(train, torch.arange(8))
    assert torch.equal(val, torch.arange(8, 10))


def test_get_batch_shapes_and_shifted_targets() -> None:
    torch.manual_seed(0)
    data = torch.arange(30)

    batch_x, batch_y = utils.get_batch(data, batch_size=4, seq_len=5, device=torch.device("cpu"))

    assert batch_x.shape == (4, 5)
    assert batch_y.shape == (4, 5)
    assert torch.equal(batch_y, batch_x + 1)
