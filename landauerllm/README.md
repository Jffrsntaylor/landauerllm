## Landauer LLM (Phase-Space RNN)

Landauer LLM is a compact character-level language model built on an aperiodic recurrent stack (phase rotations instead of gates) plus a thermodynamic inertia regularizer. It is designed for long-horizon generation demos and teaching-friendly experiments on small GPUs/CPUs.

### Core Ideas
- Aperiodic RNN: maximal irrational phase rotation in each layer; deeper layers rotate more slowly to widen the horizon without attention.
- Thermodynamic inertia loss: penalizes rapid hidden-state velocity to keep trajectories smooth over long sequences.
- Minimal CLIs: `train.py` and `generate.py` for quick runs, checkpointing, and sampling.
- Batteries included utilities: vocab building, encoding/decoding, batching, and train/val splitting.
- Tiny Shakespeare corpus is bundled for zero-setup experiments.

### Repository Layout
- `src/landauer_llm/model.py` - AperiodicRNN cell and LandauerLanguageModel wrapper.
- `src/landauer_llm/losses.py` - ThermodynamicInertiaLoss regularizer.
- `src/landauer_llm/utils.py` - text helpers, batching, and splits.
- `train.py` / `generate.py` - CLI entrypoints for training and text generation.
- `tiny_shakespeare.txt` - sample dataset.
- `tests/` - smoke tests for model, loss, and data utilities.
- `BENCHMARKS.md` - reference runs and how to reproduce them.

### Setup
1) (Optional) create a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Run quick checks (CPU-only, fast):
```bash
pytest -q
```

### Train
Train on Tiny Shakespeare with defaults:
```bash
python train.py --data-path tiny_shakespeare.txt --steps 2000 --eval-every 500
```
Checkpoints are saved under `saved_models/` (created automatically). Key arguments:

| Flag | Meaning | Default |
| --- | --- | --- |
| `--data-path` | Text corpus path | `tiny_shakespeare.txt` |
| `--batch-size` | Batch size | `32` |
| `--seq-len` | Truncated BPTT length | `64` |
| `--embed-dim` | Embedding dimension | `64` |
| `--hidden-dim` | Hidden dimension per layer | `256` |
| `--num-layers` | Aperiodic RNN layers | `2` |
| `--phase-scaling` | Slower rotation per deeper layer | `2.0` |
| `--temperature` | Thermodynamic loss temperature | `0.05` |
| `--energy-weight` | Weight on inertia term | `0.1` |
| `--lr` | Adam learning rate | `3e-3` |
| `--grad-clip` | Gradient clipping max-norm (`0` disables) | `1.0` |
| `--steps` | Training steps | `2000` |
| `--save-every` | Checkpoint frequency | `500` |
| `--eval-every` | Validation frequency (`0` to disable) | `500` |
| `--device` | `cuda` if available else `cpu` | auto |

Validation logs print total loss plus the decomposed content/energy terms.
The training script also prints the active device at startup; override with `--device cuda` or `--device cpu` if you need to force it.

### Generate
Sample from a checkpoint:
```bash
python generate.py --checkpoint saved_models/landauer_llm_step2000.pth --prompt "To be" --max-new-tokens 200 --device cpu
```
Generation autoregressively feeds the evolving hidden state, enabling effectively unbounded context without KV caches.

### Data
- Works on any plain-text corpus; training builds a character-level vocabulary from the provided file.
- Replace `tiny_shakespeare.txt` with your own text and adjust `--seq-len` / `--steps` as needed.
- Keep text ASCII/UTF-8 for the default tokenizer.

### Benchmarks
See `BENCHMARKS.md` for a reference Tiny Shakespeare run (~1.71 val loss after 2000 steps) and reproduction steps.

### Development Notes
- The math (AperiodicRNN, ThermodynamicInertiaLoss, training loop) is intentionally unchanged; improvements focus on clarity and usability.
- The CLI falls back to CPU when CUDA is unavailable and uses gradient clipping by default to keep training stable.
