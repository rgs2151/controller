"""Targeted reference checks; no language-model downloads or benchmark runs."""

import importlib.util
from pathlib import Path

import torch

from robust_steerability.calibration.nominal import project_dynamics
from robust_steerability.control import PIDController, PIDGains, solve_identity_input_lqr
from robust_steerability.modeling.jacobians import layer_last_token_jacobian


def test_lqr_matches_preserved_reference():
    path = Path(__file__).resolve().parents[1] / "ref/lqr-activation-steering/steer/lqr_utils.py"
    spec = importlib.util.spec_from_file_location("reference_lqr_utils", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = torch.Generator().manual_seed(51)
    a = torch.randn(5, 4, 4, generator=rng) * 0.15
    eye = torch.eye(4).repeat(5, 1, 1)
    expected = reference.time_varying_lqr_noB_mem_efficient(a, 0.1 * eye, eye, torch.eye(4))
    actual = solve_identity_input_lqr(a, "cpu", 0.1, 1, 1)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_spid_reference_reset_and_previous_error_across_token_boundary():
    gains = PIDGains(0.7, 0.01, 0.1)
    controller = PIDController(gains)
    integral = torch.zeros(2, 3)
    previous = torch.zeros(2, 3)
    for step in range(26):
        layer = step % 13
        error = torch.full((2, 3), 1 + step / 10)
        integral += error
        if layer % 10 == 0:
            integral *= 0
        expected = gains.proportional * error + gains.integral * integral + gains.derivative * (error - previous)
        torch.testing.assert_close(controller.control(layer, -error), expected)
        previous = error
    controller.reset()
    torch.testing.assert_close(controller.control(0, -torch.ones(2, 3)), torch.full((2, 3), 0.8))


def test_last_token_derivative_does_not_perturb_prefix():
    class CoupledLayer(torch.nn.Module):
        def forward(self, hidden):
            return hidden.square() + 3 * hidden.sum(dim=1, keepdim=True)
    hidden = torch.tensor([[[1., 2.], [3., 4.], [5., 6.]]])
    actual = layer_last_token_jacobian(CoupledLayer(), hidden, {}, 1)
    torch.testing.assert_close(actual, torch.diag(torch.tensor([13., 15.])))


def test_projected_derivative_preserves_coordinate_transform():
    a = torch.tensor([[[1., 2.], [0., 3.]]])
    decoders = torch.stack([torch.diag(torch.tensor([2., 4.])), torch.diag(torch.tensor([3., 5.]))])
    encoders = torch.linalg.inv(decoders).transpose(-1, -2)
    projected = project_dynamics(a, encoders, decoders)
    z = torch.tensor([2., 1.])
    torch.testing.assert_close(projected[0] @ z, encoders[1].T @ a[0] @ decoders[0] @ z)
