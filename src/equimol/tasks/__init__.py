"""Training tasks: the science that plugs into equimol.training.Trainer.

Each task implements the equimol.training.Task protocol (loss + evaluate):
how a batch becomes a loss, and how a loader becomes validation metrics.
The Trainer stays generic; everything about energies, forces, properties or
noise lives here.
"""

from .energy_force import EnergyForceTask, forces_from_energy

__all__ = [
    "EnergyForceTask",
    "forces_from_energy",
]
