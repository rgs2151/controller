"""Small online traces in controller coordinates, with one CPU transfer per prompt."""

from __future__ import annotations

import torch


class ReducedTrajectoryRecorder:
    """Observe interventions without calling or modifying a controller."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.layers = []
        self.values = {name: [] for name in (
            "state", "feedback", "control", "deviation_control", "reduced_intervention", "hidden_delta_energy",
        )}

    def append(self, layer_index: int, *, state: torch.Tensor, feedback: torch.Tensor,
               control: torch.Tensor, deviation_control: torch.Tensor, reduced_intervention: torch.Tensor,
               hidden_delta: torch.Tensor) -> None:
        self.layers.append(layer_index)
        for name, value in {"state": state, "feedback": feedback, "control": control, "deviation_control": deviation_control,
                            "reduced_intervention": reduced_intervention,
                            "hidden_delta_energy": hidden_delta.float().square().sum(dim=-1)}.items():
            self.values[name].append(value.detach().clone())

    def finish(self) -> dict:
        if not self.layers:
            raise ValueError("No controller interventions were recorded")
        tensors = {name: torch.stack(values).cpu() for name, values in self.values.items()}
        return {
            "schema_version": 1, "layer_index": torch.tensor(self.layers), **tensors,
            "control_energy": float(tensors["control"].double().square().sum()),
            "hidden_delta_energy_total": float(tensors["hidden_delta_energy"].double().sum()),
            "ordering": "forward-call order, then decoder layer; includes prefill and decode",
            "control_energy_definition": "sum over calls, layers, batch and channels of u squared; no R weighting",
            "hidden_delta_energy_definition": "squared norm after casting delta to model dtype, before addition rounding",
        }
