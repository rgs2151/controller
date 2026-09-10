"""Cheap operator-level checks against the frozen upstream formulas."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from robust_steerability.source_methods.actadd import ActAddSteerer, fit_actadd_direction, positionwise_mean
from robust_steerability.source_methods.calibration import (
    fit_actadd_calibration,
    fit_iti_calibration,
    fit_odesteer_calibration,
    fit_transport_stack,
)
from robust_steerability.source_methods.control import fit_setpoint_calibration
from robust_steerability.source_methods.odesteer import ODESteerFit, polynomial_features, vector_field
from robust_steerability.source_methods.iti import fit_iti, register_iti_hooks
from robust_steerability.source_methods.modeling import source_model_spec
from robust_steerability.source_methods.protocol import (
    CALIBRATION_COUNTS,
    act_module_patterns,
    control_sweeps,
    odesteer_layers,
    protocol_manifest,
)
from robust_steerability.source_methods.transport import (
    apply_linear_transport,
    apply_mean_transport,
    apply_pid_transport,
    fit_linear_transport,
    fit_mean_transport,
    fit_pid_transport,
    matching_module_names,
)


def test_evaluation_count_does_not_change_source_calibration_or_sweeps():
    small = protocol_manifest("toxicity", "Qwen/Qwen2.5-14B", "revision", 50)
    full = protocol_manifest("toxicity", "Qwen/Qwen2.5-14B", "revision", 1000)
    assert small["evaluation_samples"] == 50
    assert full["evaluation_samples"] == 1000
    assert small["evaluation_repetitions"] == 5
    assert small["random_seed"] == 42
    assert "top_k" not in small["generation"]
    for key in small.keys() - {"evaluation_samples"}:
        assert small[key] == full[key]
    assert CALIBRATION_COUNTS["toxicity"].positive == 200
    assert CALIBRATION_COUNTS["toxicity"].jacobian == 50
    assert control_sweeps("toxicity", "Qwen/Qwen2.5-14B")[0].lambdas == (2.0, 2.5)
    assert odesteer_layers(32) == tuple(range(8, 25))
    assert small["actadd_selected"] == {"layer": 21, "strength": 4.0}
    assert act_module_patterns("Qwen/Qwen2.5-14B") == (
        r"model.layers.*.mlp.up_proj",
        r"model.layers.*.mlp.down_proj",
        r"model.layers.*.mlp.gate_proj",
    )


def test_unsupported_checkpoint_is_not_given_borrowed_parameters():
    with pytest.raises(ValueError, match="No source-defined protocol"):
        protocol_manifest("toxicity", "Qwen/Qwen2.5-0.5B", "revision", 50)

    with pytest.raises(ValueError, match="No source-defined AcT module protocol"):
        protocol_manifest("toxicity", "Qwen/Qwen2.5-32B", "revision", 50)


def test_source_calibration_sizes_are_enforced_before_model_execution():
    with pytest.raises(ValueError, match="ActAdd requires exactly 100"):
        fit_actadd_calibration(None, None, undesired_texts=[], desired_texts=[], batch_size=1)
    with pytest.raises(ValueError, match="ITI requires exactly 80"):
        fit_iti_calibration(None, None, undesired_texts=[], desired_texts=[], batch_size=1)
    with pytest.raises(ValueError, match="ODESteer toxicity requires exactly 5000"):
        fit_odesteer_calibration(
            None,
            None,
            behavior="toxicity",
            layer_index=0,
            undesired_texts=[],
            desired_texts=[],
            batch_size=1,
        )
    with pytest.raises(ValueError, match="AcT requires exactly 200"):
        fit_transport_stack(
            None,
            None,
            source_texts=[],
            target_texts=[],
            module_patterns=(".*",),
            method="mean_act",
            batch_size=1,
        )


def test_each_method_keeps_its_source_model_loading_protocol():
    alqr = source_model_spec("alqr", "toxicity", "Qwen/Qwen2.5-14B", "revision")
    truth_alqr = source_model_spec("alqr", "truthfulness", "Qwen/Qwen2.5-14B", "revision")
    iti = source_model_spec("iti", "toxicity", "Qwen/Qwen2.5-14B", "revision")
    linear = source_model_spec("linear_act", "toxicity", "Qwen/Qwen2.5-14B", "revision")
    actadd_lfs = source_model_spec("actadd_lfs", "toxicity", "Qwen/Qwen2.5-14B", "revision")
    assert alqr.quantized and alqr.dtype == "float32" and alqr.quantization_compute_dtype == "float16"
    assert truth_alqr.quantization_compute_dtype == "float32"
    assert not iti.quantized and iti.dtype == "float16"
    assert not linear.quantized and linear.dtype == "bfloat16"
    assert actadd_lfs.quantized and actadd_lfs.quantization_compute_dtype == "float16"
    assert alqr.attention_implementation is None


def test_setpoint_uses_positive_minus_negative_and_method_specific_thresholds():
    negative = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    positive = torch.tensor([[4.0, 6.0], [3.0 + 5e-7, 4.0]])
    fitted = fit_setpoint_calibration(negative, positive)
    torch.testing.assert_close(fitted.contrast, positive - negative)
    torch.testing.assert_close(fitted.unit_features(0.0)[0], torch.tensor([0.6, 0.8]))
    torch.testing.assert_close(fitted.unit_features(0.0)[1], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(fitted.unit_features(1e-6)[1], torch.zeros(2))
    torch.testing.assert_close(fitted.setpoints(2.5), 2.5 * fitted.feature_norm)


def test_actadd_position_average_and_first_call_only():
    activations = [torch.tensor([[[[1.0], [10.0]], [[3.0], [30.0]]]])]
    masks = [torch.tensor([[0, 1], [1, 1]])]
    torch.testing.assert_close(positionwise_mean(activations, masks), torch.tensor([[[3.0], [20.0]]]))

    negative = torch.tensor([[[1.0], [2.0]]])
    positive = torch.tensor([[[4.0]]])
    torch.testing.assert_close(
        fit_actadd_direction(negative, positive),
        torch.tensor([[[3.0], [-2.0]]]),
    )

    class Layer(torch.nn.Module):
        def forward(self, hidden):
            return (hidden,)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Model()
    steerer = ActAddSteerer(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), 0, 2.0)
    handles = steerer.register(model)
    first = model.model.layers[0](torch.zeros(1, 2, 2))[0]
    second = model.model.layers[0](torch.zeros(1, 2, 2))[0]
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(first, torch.tensor([[[2.0, 4.0], [6.0, 8.0]]]))
    torch.testing.assert_close(second, torch.zeros_like(second))


def test_act_operators_match_upstream_equations():
    responses = torch.tensor([
        [1.0, 2.0], [3.0, 4.0], [11.0, 12.0], [15.0, 18.0]
    ])
    source = torch.tensor([True, True, False, False])
    mean = fit_mean_transport(responses, source)
    output = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])
    expected = output + 0.5 * (mean.target_mean - mean.source_mean)
    torch.testing.assert_close(apply_mean_transport(output, mean, 0.5), expected)

    linear = fit_linear_transport(responses, source, random_state=np.random.RandomState(42))
    flat = output.float()
    mapped = flat * linear.slope + linear.intercept
    selected = (linear.lower < flat) & (flat < linear.upper)
    expected_linear = torch.where(selected, 0.5 * mapped + 0.5 * flat, flat)
    torch.testing.assert_close(apply_linear_transport(output, linear, 0.5), expected_linear)

    previous = (torch.tensor([2.0, 4.0]), torch.tensor([6.0, 8.0]))
    pid = fit_pid_transport(responses, source, previous)
    raw_difference = mean.target_mean - mean.source_mean
    expected_difference = raw_difference + 0.005 * (torch.cat(previous).mean() + raw_difference)
    torch.testing.assert_close(pid.difference, expected_difference)
    expected_pid = output + 0.7 * 0.5 * expected_difference
    torch.testing.assert_close(apply_pid_transport(output, pid, 0.5), expected_pid)


def test_act_module_matching_uses_full_regex_and_model_order():
    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.up_proj = torch.nn.Linear(2, 3)
            self.down_proj = torch.nn.Linear(3, 2)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = MLP()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer(), Layer()])

    names = matching_module_names(Model(), (r"model.layers.*.mlp.up_proj",))
    assert names == ("model.layers.0.mlp.up_proj", "model.layers.1.mlp.up_proj")


def test_iti_uses_probe_direction_and_negative_alpha_at_o_proj_input():
    generator = torch.Generator().manual_seed(12)
    labels = np.asarray([0, 1] * 20)
    activations = torch.randn(40, 1, 2, 3, generator=generator)
    activations[labels == 1, 0, 0, 0] += 4
    fitted = fit_iti(activations, labels, seed=42)
    assert fitted.selected_heads(1) == ((0, 0),)

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.o_proj = torch.nn.Linear(6, 6, bias=False)
            self.o_proj.weight.data.copy_(torch.eye(6))

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Model()
    handles = register_iti_hooks(model, fitted, top_heads=1, alpha=2.0)
    values = torch.zeros(1, 2, 6)
    output = model.model.layers[0].self_attn.o_proj(values)
    for handle in handles:
        handle.remove()
    expected = torch.zeros_like(output)
    selected_direction = torch.zeros_like(fitted.directions[0])
    selected_direction[0] = fitted.directions[0, 0]
    expected[:, -1] = -2 * selected_direction.reshape(-1)
    torch.testing.assert_close(output, expected)


def test_odesteer_vector_field_matches_source_polynomial_gradient_direction():
    generator = torch.Generator().manual_seed(7)
    fitted = ODESteerFit(
        indices=torch.randint(32, (2, 6), generator=generator),
        signs=(torch.randint(2, (2, 6), generator=generator) * 2 - 1).to(torch.int8),
        coefficient=torch.randn(32, generator=generator),
        intercept=torch.zeros(1),
    )
    values = torch.randn(3, 5, generator=generator, requires_grad=True)
    logit = polynomial_features(values, fitted) @ fitted.coefficient
    expected = torch.autograd.grad(logit.sum(), values)[0]
    expected = expected / (expected.norm(dim=-1, keepdim=True) + 1e-10)
    torch.testing.assert_close(vector_field(values.detach(), fitted), expected, atol=2e-5, rtol=2e-4)
