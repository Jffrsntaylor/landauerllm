import torch

from landauer_llm.losses import ThermodynamicInertiaLoss


def test_energy_penalizes_motion() -> None:
    loss_fn = ThermodynamicInertiaLoss(temperature=0.5, inertia_weight=1.0)

    static_hidden = [torch.zeros(2, 3) for _ in range(4)]
    moving_hidden = [torch.ones(2, 3) * 0.1 * i for i in range(4)]

    static_total, static_task, static_energy = loss_fn(None, None, static_hidden)
    moving_total, _, moving_energy = loss_fn(None, None, moving_hidden)

    assert torch.isclose(static_task, torch.tensor(0.0))
    assert moving_energy > static_energy
    assert moving_total >= moving_energy


def test_cross_entropy_path_outputs_positive_losses() -> None:
    torch.manual_seed(0)
    predictions = torch.randn(2, 3, 5)
    targets = torch.tensor([[1, 2, 3], [0, 4, 1]])
    hidden_states = [torch.zeros(2, 5) for _ in range(3)]

    loss_fn = ThermodynamicInertiaLoss(temperature=0.1, inertia_weight=0.2)
    total, task, energy = loss_fn(predictions, targets, hidden_states)

    assert total.item() >= task.item()
    assert energy.item() >= 0.0
    assert task.item() > 0.0
