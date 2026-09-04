from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from robust_steerability.modeling.interventions import forward_with_policy
from robust_steerability.runtime.policy import SetpointLQRPolicy


class ToyLayer(torch.nn.Module):
    def __init__(self, matrix: torch.Tensor):
        super().__init__()
        self.register_buffer("matrix", matrix)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor]:
        return (hidden_states @ self.matrix.T,)


class ToyBody(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [ToyLayer(torch.eye(2)), ToyLayer(torch.eye(2))]
        )


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBody()
        self.config = SimpleNamespace(hidden_size=2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        hidden = torch.stack(
            [input_ids.float(), torch.ones_like(input_ids, dtype=torch.float32)],
            dim=-1,
        )
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=hidden)


class InterventionTests(unittest.TestCase):
    def test_forward_records_states_and_applies_policy(self) -> None:
        model = ToyModel()
        encoded = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        policy = SetpointLQRPolicy(
            tracking_gains=torch.eye(2).repeat(2, 1, 1),
            feature_unit=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            setpoints=torch.tensor([3.0, 3.0]),
        )
        output, states, controls = forward_with_policy(model, encoded, policy)
        self.assertEqual(states.shape, (3, 2, 2))
        self.assertEqual(controls.shape, (2, 2, 2))
        torch.testing.assert_close(controls[0, -1], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(controls[1, -1], torch.tensor([0.0, 0.0]))
        torch.testing.assert_close(output.logits[0, -1], torch.tensor([3.0, 1.0]))

    def test_baseline_records_zero_controls(self) -> None:
        model = ToyModel()
        encoded = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }
        output, states, controls = forward_with_policy(model, encoded, None)
        torch.testing.assert_close(output.logits[0, -1], torch.tensor([2.0, 1.0]))
        torch.testing.assert_close(controls, torch.zeros_like(controls))
        torch.testing.assert_close(states[-1, -1], torch.tensor([2.0, 1.0]))


if __name__ == "__main__":
    unittest.main()
