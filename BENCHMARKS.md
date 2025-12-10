# Benchmarks

Reference runs and reproduction notes for the Landauer LLM. These benchmarks keep the math unchanged and focus on transparent, repeatable settings.

## Methodology
- **Dataset:** `tiny_shakespeare.txt` (included). Character-level with a fixed vocab built from the file.
- **Command:** `python train.py --data-path tiny_shakespeare.txt --steps 2000 --eval-every 500 --save-every 500`
- **Metrics:** Validation cross-entropy (content loss) plus the thermodynamic inertia energy term. Total loss is `content + energy_weight * energy`.
- **Hardware:** The script auto-selects `cuda` when available; otherwise CPU. Log your device for new entries.

## Reference Results

**Used command for Benchmarks:** python train.py --data-path tiny_shakespeare.txt --steps 2000 --batch-size 32 --seq-len 256 --embed-dim 64 
--hidden-dim 512 --num-layers 2 --phase-scaling 1.618 --temperature 0.05 --energy-weight 0.1 --lr 3e-3 --grad-clip 1.0 
--eval-every 500 --save-every 500 --device cuda

| Run | Steps | Batch x Seq | Hidden / Layers | Val Loss (total) | Notes |

| 1   | 2000  | 32 x 64     | 256 hidden, 2 layers | ~ 1.7100 | RTX 3070 |

| 2   | 2000  | 32 x 64     | 256 hidden, 2 layers | = 1.7850 | RTX 3070 |

| 3   | 2000  | 32 x 64     | 256 hidden, 2 layers | = 1.8248 | AMD Ryzen 7 5800x 8 core |

| 4   | 2000  | 32 x 64     | 256 hidden, 2 layers | = 1.7637 | RTX 3070 | Turned down energy weight to .05 |

| 5   | 2000  | 32 x 256    | 512 hidden, 2 layers| = 1.6367  | RTX 3070 | Energy weight to .01, seq len to 256, hidden to 512, phase scaling to 1.618 |

| 6   | 2000 | 32 x 256     | 512 hidden, 4 layers| = 1.5129  | RTX 3070 | Energy weight to .01, phase scaling to 1.618, lowered the learning rate from 3e - 3 to 1e - 3 |


## Reproduce Locally
1) Install dependencies: `pip install -r requirements.txt`
2) Train: `python train.py --data-path tiny_shakespeare.txt --steps 2000 --eval-every 500 --save-every 500`
3) Watch logs for `[val]` lines to capture the validation loss and energy term.
4) Generate from the saved checkpoint for qualitative inspection:
```bash
python generate.py --checkpoint saved_models/landauer_llm_step2000.pth --prompt "To be" --max-new-tokens 200
```

## Adding Your Own Entries
- Record device (CPU/GPU model), PyTorch version, and command-line overrides.
- Keep the dataset and preprocessing unchanged for comparability.
- If you change hyperparameters (e.g., `hidden-dim`, `phase-scaling`, or `energy-weight`), note them in the table.

## Tips for Faster Runs
- Reduce `--steps` for smoke tests; the energy regularizer still stabilizes short runs.
- Lower `--seq-len` or `--batch-size` when constrained by memory.
- Set `--device cpu` explicitly to avoid silent GPU/CPU switching in shared environments.
