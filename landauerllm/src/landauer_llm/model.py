"""Model definitions for the Landauer LLM."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class AperiodicRNN(nn.Module):
    """
    Multi-layer recurrent cell using maximal irrational phase rotations.

    Each layer rotates its hidden state in phase space to discourage periodic
    attractors. Deeper layers apply a slower phase rotation (controlled by
    ``scaling_factor``) to mimic widening causal horizons across the stack.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        scaling_factor: float = 2.0,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.scaling_factor = scaling_factor

        self.maximal_irrational_phase = torch.tensor(2.39996322972865332)

        self.input_linears = nn.ModuleList(
            [
                nn.Linear(input_size if layer == 0 else hidden_size, hidden_size)
                for layer in range(num_layers)
            ]
        )
        self.hidden_scale = nn.Parameter(torch.ones(num_layers, hidden_size))

    def forward(
        self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run one recurrent step.

        Args:
            x: Input tensor of shape ``(batch, input_size)``.
            h_prev: Optional previous hidden states of shape
                ``(num_layers, batch, hidden_size)``.

        Returns:
            Tuple of:
            - top-layer hidden state of shape ``(batch, hidden_size)``
            - stacked hidden states of shape ``(num_layers, batch, hidden_size)``
        """

        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if h_prev is None:
            h_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_size, device=device, dtype=dtype
            )

        all_states: List[torch.Tensor] = []
        layer_input = x

        indices = torch.arange(self.hidden_size, device=device, dtype=dtype)
        base_phase = self.maximal_irrational_phase.to(device=device, dtype=dtype)

        for layer in range(self.num_layers):
            phase = base_phase / (self.scaling_factor ** layer)

            rotation = torch.cos(phase * indices + h_prev[layer])
            rotated_state = h_prev[layer] * rotation * self.hidden_scale[layer]

            h_next = torch.tanh(self.input_linears[layer](layer_input) + rotated_state)

            all_states.append(h_next)
            layer_input = h_next

        stacked_states = torch.stack(all_states, dim=0)
        top_state = stacked_states[-1]
        return top_state, stacked_states


class LandauerLanguageModel(nn.Module):
    """
    Character-level language model built on the aperiodic recurrent stack.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        phase_scaling: float = 2.0,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = AperiodicRNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            scaling_factor=phase_scaling,
        )
        self.output_head = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self, idx: torch.Tensor, h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """
        Args:
            idx: Token indices of shape ``(batch, seq_len)``.
            h: Optional hidden state stack shaped ``(num_layers, batch, hidden_size)``.

        Returns:
            logits: ``(batch, seq_len, vocab_size)``
            hidden_trace: list of top-layer hidden states for each timestep
            h: final stacked hidden states shaped ``(num_layers, batch, hidden_size)``
        """

        batch_size, seq_len = idx.shape
        embeddings = self.token_embedding(idx)

        hidden_trace: List[torch.Tensor] = []
        outputs: List[torch.Tensor] = []
        h_stack = h

        for t in range(seq_len):
            step_input = embeddings[:, t, :]
            top_state, h_stack = self.rnn(step_input, h_stack)
            logits = self.output_head(top_state)

            outputs.append(logits)
            hidden_trace.append(top_state)

        stacked_logits = torch.stack(outputs, dim=1)
        return stacked_logits, hidden_trace, h_stack

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Autoregressively generate new tokens from a context.

        Args:
            idx: Initial indices shaped ``(batch, context_len)``.
            max_new_tokens: Number of tokens to append.
            h: Optional hidden state stack for warm starts.

        Returns:
            Concatenated token indices shaped ``(batch, context_len + max_new_tokens)``.
        """

        for _ in range(max_new_tokens):
            logits, _, h = self(idx[:, -1:].contiguous(), h)
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx
