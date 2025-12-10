## Landauer LLM (Phase-Space RNN)

A compact character-level language model that uses an aperiodic recurrent stack (phase rotations instead of gates) plus a thermodynamic regularizer to keep hidden-state updates smooth. It runs on GPU or CPU and keeps the code simple for learning and demos.

### Features
- Aperiodic RNN with multi-layer “harmonic” stacking (deeper layers rotate more slowly).
- Thermodynamic inertia loss: penalizes rapid hidden swings to avoid noisy updates.
- Minimal text utils for encoding/decoding and batching.
- CLI scripts for training and generation, plus two small experiment scripts.

### Infinite Context Generation
Unlike Transformer-based models which rely on a fixed-size Key-Value cache and positional embeddings, the Landauer LLM is a pure Recurrent Neural Network (RNN).

- **O(1) Inference Memory:** The model maintains a fixed-size hidden state stack regardless of sequence length. It does not need to store the history of previous tokens to generate the next one.
- **Unbounded Horizon:** There is no architectural hard limit on `max_new_tokens`. While Transformers crash once they exceed their context window, this model can theoretically generate text indefinitely by continuously evolving its hidden state.
- **Stability:** The "Aperiodic" phase rotations and thermodynamic regularization are specifically designed to prevent the hidden state from collapsing into repetitive loops or degrading over long durations.


### Quickstart
1) Install deps:
```bash
pip install -r requirements.txt
```
2) Train on Tiny Shakespeare (saves checkpoints to `saved_models/`):
```bash
python train.py --data-path tiny_shakespeare.txt --steps 2000
```
3) Generate from a checkpoint:
```bash
python generate.py --checkpoint saved_models/landauer_llm_step2000.pth --prompt "To be" --max-new-tokens 200
```

### Results
Running the default training configuration on `tiny_shakespeare.txt` for 2000 steps yields a validation loss of **~1.71**, which is competitive with standard LSTM baselines for character-level modeling.

**Sample Generation (Step 2000):**
> "To beye thee, I devous confer namely of God my lord this;
> And sequal this,
> Her foul pother:
> ...
> For you friend thou rameomethou can fie, would the recked tell ather God, thou show no love, none"

**Observations:**
- **Vocabulary:** The model successfully learns complex words ("confer", "namely", "friend", "pother") purely from character-level probability.
- **Grammar:** It captures archaic sentence structures ("thou show no love") and dramatic formatting (line breaks and speaker attributions).
- **Stability:** The thermodynamic inertia loss kept training stable without the exploding gradients typical of vanilla RNNs.

### Notes
- If PyTorch sees your GPU, the scripts will print `Training on cuda...` and use it. Otherwise they fall back to CPU.
- Gradient clipping is on by default to reduce loss spikes. You can adjust with `--grad-clip`.
- The validation log in `train.py` (every `--eval-every` steps) helps you see when to stop training.

### Repo Layout
- `src/landauer_llm/`: model, loss, and utilities.
- `train.py` / `generate.py`: CLI for training and text generation.
- `experiments/`: small demo scripts (text and trajectory). Remove if you want a leaner repo.

