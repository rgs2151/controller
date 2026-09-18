"""Small online traces in controller coordinates, with one CPU transfer per prompt."""

from __future__ import annotations

import torch


class ReducedTrajectoryRecorder:
    """Observe interventions without calling or modifying a controller.

    ``full`` preserves every controller-coordinate tensor required by the
    H-infinity diagnostic handoff. ``norms`` preserves exact energy totals and
    per-step norm trajectories without copying model-width hidden states.
    """

    def __init__(self, detail: str = "full"):
        if detail not in {"full", "norms"}:
            raise ValueError("Recorder detail must be 'full' or 'norms'")
        self.detail = detail
        self.reset()

    def reset(self) -> None:
        self.layers = []
        names = (
            "state", "feedback", "control", "deviation_control", "reduced_intervention",
            "hidden_delta_energy",
        ) if self.detail == "full" else (
            "state_norm", "feedback_norm", "control_norm", "deviation_control_norm",
            "reduced_intervention_norm", "control_energy_step", "hidden_delta_energy",
        )
        self.values = {name: [] for name in names}

    def append(self, layer_index: int, *, state: torch.Tensor, feedback: torch.Tensor,
               control: torch.Tensor, deviation_control: torch.Tensor, reduced_intervention: torch.Tensor,
               hidden_delta: torch.Tensor) -> None:
        self.layers.append(layer_index)
        if self.detail == "full":
            values = {
                "state": state, "feedback": feedback, "control": control,
                "deviation_control": deviation_control,
                "reduced_intervention": reduced_intervention,
                "hidden_delta_energy": hidden_delta.float().square().sum(dim=-1),
            }
        else:
            values = {
                "state_norm": state.float().norm(dim=-1),
                "feedback_norm": feedback.float().norm(dim=-1),
                "control_norm": control.float().norm(dim=-1),
                "deviation_control_norm": deviation_control.float().norm(dim=-1),
                "reduced_intervention_norm": reduced_intervention.float().norm(dim=-1),
                "control_energy_step": control.double().square().sum(dim=-1),
                "hidden_delta_energy": hidden_delta.float().square().sum(dim=-1),
            }
        for name, value in values.items():
            self.values[name].append(value.detach().clone())

    def finish(self) -> dict:
        if not self.layers:
            raise ValueError("No controller interventions were recorded")
        tensors = {name: torch.stack(values).cpu() for name, values in self.values.items()}
        control_energy = (
            tensors["control"].double().square().sum()
            if self.detail == "full"
            else tensors["control_energy_step"].double().sum()
        )
        return {
            "schema_version": 1 if self.detail == "full" else 2,
            "storage": "full-controller-coordinates" if self.detail == "full" else "per-step-norms",
            "layer_index": torch.tensor(self.layers), **tensors,
            "control_energy": float(control_energy),
            "hidden_delta_energy_total": float(tensors["hidden_delta_energy"].double().sum()),
            "ordering": "forward-call order, then decoder layer; includes prefill and decode",
            "control_energy_definition": "sum over calls, layers, batch and channels of u squared; no R weighting",
            "hidden_delta_energy_definition": "squared norm after casting delta to model dtype, before addition rounding",
        }
