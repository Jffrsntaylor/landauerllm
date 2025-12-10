"""
Text generation CLI for the Landauer LLM.

Usage:
    python generate.py --checkpoint saved_models/landauer_llm_step2000.pth --prompt "To be"
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "src"))

from landauer_llm.model import LandauerLanguageModel  # type: ignore  # noqa: E402
from landauer_llm import utils  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with a trained Landauer LLM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", type=str, default=" ")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(
    checkpoint_path: Path, device: torch.device
) -> Tuple[LandauerLanguageModel, Dict[str, int], Dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = LandauerLanguageModel(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    itos = {int(k): v for k, v in checkpoint["itos"].items()}
    stoi = {ch: idx for idx, ch in itos.items()}
    return model, stoi, itos


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, stoi, itos = load_model(args.checkpoint, device)
    try:
        prompt_tokens = utils.encode(args.prompt, stoi)
    except KeyError as exc:
        missing = set(args.prompt) - set(stoi.keys())
        raise ValueError(f"Prompt contains out-of-vocab characters: {missing}") from exc
    context = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    generated = model.generate(context, max_new_tokens=args.max_new_tokens)
    text = utils.decode(generated[0].tolist(), itos)
    print(text)


if __name__ == "__main__":
    main()
