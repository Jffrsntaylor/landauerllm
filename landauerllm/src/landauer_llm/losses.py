"""Loss functions for Landauer LLM."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermodynamicInertiaLoss(nn.Module):
    """
    Penalizes rapid hidden-state fluctuations to model thermodynamic inertia.

    The energy term discourages high-frequency oscillations (destructive
    interference) while allowing slower, information-rich transitions.
    """

    def __init__(self, temperature: float = 0.1, inertia_weight: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.inertia_weight = inertia_weight
        self.entropy_cost = torch.log(torch.tensor((1 + 5**0.5) / 2))

    def forward(
        self,
        predictions: Optional[torch.Tensor],
        targets: Optional[torch.Tensor],
        hidden_states: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            predictions: Model outputs (logits or continuous values).
            targets: Ground-truth values.
            hidden_states: List of hidden states over time (top layer).

        Returns:
            total_loss, task_loss, energy_cost
        """

        device = None
        if predictions is not None:
            device = predictions.device
        elif hidden_states is not None and len(hidden_states) > 0:
            device = hidden_states[0].device
        else:
            device = torch.device("cpu")

        task_loss = torch.tensor(0.0, device=device)
        if predictions is not None and targets is not None:
            if predictions.shape == targets.shape:
                task_loss = F.mse_loss(predictions, targets)
            elif predictions.dim() >= 2:
                logits = predictions.view(-1, predictions.size(-1))
                task_targets = targets.view(-1).to(logits.device)
                task_loss = F.cross_entropy(logits, task_targets)

        energy_cost = torch.tensor(0.0, device=device)
        if hidden_states is not None and len(hidden_states) > 1:
            stacked = torch.stack(hidden_states)
            diffs = stacked[1:] - stacked[:-1]
            state_velocity = torch.mean(diffs.pow(2))
            energy_cost = self.temperature * self.entropy_cost.to(device) * state_velocity

        total = task_loss + self.inertia_weight * energy_cost
        return total, task_loss, energy_cost
