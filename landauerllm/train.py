"""
Training CLI for the Landauer LLM.

Usage:
    python train.py --data-path tiny_shakespeare.txt --steps 2000
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as nn_utils

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from landauer_llm.model import LandauerLanguageModel  # type: ignore  # noqa: E402
from landauer_llm.losses import ThermodynamicInertiaLoss  # type: ignore  # noqa: E402
from landauer_llm import utils  # type: ignore  # noqa: E402


def load_corpus(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please supply a text corpus.")
    return utils.load_text(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Landauer LLM on character data.")
    parser.add_argument("--data-path", type=Path, default=Path("tiny_shakespeare.txt"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--phase-scaling", type=float, default=2.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="0 to disable clipping")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500, help="0 to skip validation logging")
    parser.add_argument("--save-dir", type=Path, default=Path("saved_models"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def train_loop(
    model: LandauerLanguageModel,
    train_data: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    inertia_loss: ThermodynamicInertiaLoss,
    index_to_char: dict,
    val_data: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    device = torch.device(args.device)
    model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    print(f"Training on {device} (cuda available: {torch.cuda.is_available()})")

    for step in range(1, args.steps + 1):
        xb, yb = utils.get_batch(train_data, args.batch_size, args.seq_len, device)
        logits, h_states, _ = model(xb)

        b, t, c = logits.shape
        content_loss = criterion(logits.view(b * t, c), yb.view(b * t))
        energy_total, _, energy_cost = inertia_loss(None, None, h_states)

        total_loss = content_loss + energy_total

        optimizer.zero_grad()
        total_loss.backward()
        if args.grad_clip > 0:
            nn_utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % 100 == 0:
            print(
                f"step {step:05d} | loss={total_loss.item():.4f} "
                f"(content={content_loss.item():.4f}, energy={energy_cost.item():.4f})"
            )

        if step % args.save_every == 0:
            save_checkpoint(model, args.save_dir, step, args, index_to_char)
        if args.eval_every > 0 and step % args.eval_every == 0:
            val_total, val_content, val_energy = evaluate(
                model, val_data, criterion, inertia_loss, args.batch_size, args.seq_len, device
            )
            print(
                f"[val] step {step:05d} | loss={val_total:.4f} "
                f"(content={val_content:.4f}, energy={val_energy:.4f})"
            )


def save_checkpoint(
    model: LandauerLanguageModel,
    save_dir: Path,
    step: int,
    args: argparse.Namespace,
    index_to_char: dict,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"landauer_llm_step{step}.pth"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": model.output_head.out_features,
                "embed_dim": model.token_embedding.embedding_dim,
                "hidden_dim": model.rnn.hidden_size,
                "num_layers": model.rnn.num_layers,
                "phase_scaling": model.rnn.scaling_factor,
            },
            "args": vars(args),
            "itos": index_to_char,
        },
        path,
    )
    print(f"Saved checkpoint to {path}")
    return path


def main() -> None:
    args = parse_args()
    raw_text = load_corpus(args.data_path)
    stoi, itos = utils.build_vocab(raw_text)
    encoded = torch.tensor(utils.encode(raw_text, stoi), dtype=torch.long)
    train_data, val_data = utils.split_dataset(encoded)

    model = LandauerLanguageModel(
        vocab_size=len(stoi),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        phase_scaling=args.phase_scaling,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    inertia_loss = ThermodynamicInertiaLoss(
        temperature=args.temperature, inertia_weight=args.energy_weight
    )

    train_loop(model, train_data, optimizer, criterion, inertia_loss, itos, val_data, args)


@torch.no_grad()
def evaluate(
    model: LandauerLanguageModel,
    data: torch.Tensor,
    criterion: nn.Module,
    inertia_loss: ThermodynamicInertiaLoss,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    xb, yb = utils.get_batch(data, batch_size, seq_len, device)
    logits, h_states, _ = model(xb)

    b, t, c = logits.shape
    content_loss = criterion(logits.view(b * t, c), yb.view(b * t))
    energy_total, _, energy_cost = inertia_loss(None, None, h_states)
    model.train()
    return float((content_loss + energy_total).item()), float(content_loss.item()), float(energy_cost.item())


if __name__ == "__main__":
    main()
