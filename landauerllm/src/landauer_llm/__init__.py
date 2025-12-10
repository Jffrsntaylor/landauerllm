"""
Landauer LLM package.

Provides phase-space recurrent architectures, physics-inspired losses, and
utilities for text data handling.
"""

from .model import AperiodicRNN, LandauerLanguageModel
from .losses import ThermodynamicInertiaLoss
from . import utils

__all__ = [
    "AperiodicRNN",
    "LandauerLanguageModel",
    "ThermodynamicInertiaLoss",
    "utils",
]
