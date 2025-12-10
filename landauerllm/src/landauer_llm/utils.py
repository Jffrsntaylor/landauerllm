"""Utilities for text data handling."""

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch


def load_text(path: Path) -> str:
    """
    Read a text corpus from disk.

    Args:
        path: Path to the text file.

    Returns:
        The file contents as a single string.
    """

    return path.read_text(encoding="utf-8")


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build character-level vocab mappings.

    Returns:
        stoi (char->index), itos (index->char)
    """

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> List[int]:
    """Convert a string to index list."""
    return [stoi[c] for c in text]


def decode(tokens: Iterable[int], itos: Dict[int, str]) -> str:
    """Convert indices back to string."""
    return "".join(itos[i] for i in tokens)


def split_dataset(data: torch.Tensor, val_fraction: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split tensor into train/validation splits."""
    n = int((1 - val_fraction) * len(data))
    return data[:n], data[n:]


def get_batch(
    data: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of subsequences.

    Returns:
        input indices and target indices shaped ``(batch, seq_len)``.
    """

    idx = torch.randint(0, len(data) - seq_len - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + seq_len] for i in idx])
    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in idx])
    return x.to(device), y.to(device)
