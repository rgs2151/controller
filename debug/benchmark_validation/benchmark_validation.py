"""Focused GPU checks against the preserved controller implementations."""

import argparse
import importlib.util
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

from robust_steerability.artifacts import implementation_hash
from robust_steerability.control import HInfinityController, HInfinityOptions, LQRController
from robust_steerability.control.h_infinity import _prepare_problem, _solve_prepared_gamma
from robust_steerability.control.types import FiniteHorizonControlProblem
from robust_steerability.experiments.diagnostics import sha256, write_json
from robust_steerability.modeling.huggingface import CausalModelLoadSpec, load_access_token, load_causal_model
from robust_steerability.modeling.interventions import _decoder_layers
from robust_steerability.modeling.jacobians import capture_layer_inputs, layer_last_token_jacobian


UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
MODELS = {
    "distil": CausalModelLoadSpec("distilgpt2", "2290a62682d06624634c1f46a6ad5be0f47f38aa", dtype="float32"),
    "qwen": CausalModelLoadSpec("Qwen/Qwen2.5-0.5B", "060db6499f32faf8b98477b0a26969ef7d8b9987", dtype="float32"),
}


def reference_hinf():
    spec = importlib.util.spec_from_file_location("hannah_validation_reference", ROOT / "ref/h_infinity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def controller_checks(device):
    generator = torch.Generator().manual_seed(913)
    identity = torch.eye(8).repeat(6, 1, 1)
    problem = FiniteHorizonControlProblem(
        dynamics=0.8 * identity + 0.02 * torch.randn(6, 8, 8, generator=generator),
        control_channels=identity, disturbance_channels=0.1 * identity,
        state_costs=0.1 * identity, control_costs=identity, terminal_cost=torch.eye(8))
    reference = reference_hinf()
    options = HInfinityOptions(gamma_upper=2.0, tolerance=1e-5, deployment_margin=0.01)
    ours = HInfinityController.synthesize(problem, device=device, options=options).solution()
    theirs = reference.HInfinityController.synthesize(
        problem, device=device, options=reference.HInfinityOptions(**asdict(options))).solution()
    assert ours.feasible and theirs.feasible
    assert abs(ours.gamma_star - theirs.gamma_star) <= options.tolerance
    torch.testing.assert_close(ours.gains, theirs.gains, rtol=2e-4, atol=2e-5)
    prepared = _prepare_problem(problem, device)
    statuses = []
    for gamma in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
        actual = _solve_prepared_gamma(prepared, gamma, numerical_tolerance=1e-7, collect=True)
        feasible, gains, _, diagnostics = reference._solve_for_gamma(
            problem, gamma, device=device, numerical_tolerance=1e-7)
        assert actual.feasible == feasible
        assert actual.diagnostics["failed_layer"] == diagnostics["failed_layer"]
        if feasible:
            torch.testing.assert_close(actual.gains, gains, rtol=2e-4, atol=2e-5)
        statuses.append({"gamma": gamma, "feasible": feasible})
    assert [row["feasible"] for row in statuses] == sorted(row["feasible"] for row in statuses)
    nominal = LQRController.synthesize(problem, device=device).solution().gains
    zero = replace(problem, disturbance_channels=torch.zeros_like(identity))
    zero_result = _solve_prepared_gamma(_prepare_problem(zero, device), 1.0,
                                        numerical_tolerance=1e-7, collect=True)
    large = _solve_prepared_gamma(prepared, 1e6, numerical_tolerance=1e-7, collect=True)
    assert zero_result.feasible and large.feasible
    torch.testing.assert_close(zero_result.gains.cpu(), nominal, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(large.gains.cpu(), nominal, rtol=2e-4, atol=2e-5)
    return {"gamma_star": ours.gamma_star, "reference_gamma_star": theirs.gamma_star,
            "maximum_gain_difference": float((ours.gains - theirs.gains).abs().max()),
            "zero_disturbance_lqr_parity": True, "large_gamma_lqr_parity": True,
            "feasibility_monotonicity": statuses}


def jacobian_checks(model_key, device):
    spec = MODELS[model_key]
    model, tokenizer = load_causal_model(spec, device, load_access_token(ROOT))
    prompt = "The neighbors helped each other and spoke kindly about their community."
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=24).to(device)
    captured = capture_layer_inputs(model, encoded)
    layers = _decoder_layers(model)
    results = []
    for index in (0, len(layers) - 1):
        hidden, kwargs = captured[index]
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        actual = layer_last_token_jacobian(layers[index], hidden, kwargs, 32)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        def full_block_last(full_input):
            result = layers[index](full_input, **kwargs)
            result = result[0] if isinstance(result, tuple) else result
            return result[0, -1]

        # Match ref/steer/lqr_utils.py: differentiate the full input, then select
        # the last-token columns (not a perturbation of every prefix token).
        expected = torch.autograd.functional.jacobian(
            full_block_last, hidden, vectorize=True)[:, 0, -1, :].detach().cpu()
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        results.append({"layer": index, "dimension": len(actual),
                        "maximum_absolute_difference": float((actual - expected).abs().max()),
                        "chunked_jacobian_seconds": elapsed})
    return {"model_loading": asdict(spec), "prompt": prompt,
            "input_token_ids": encoded["input_ids"][0].cpu().tolist(), "layers": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    if not args.device.startswith("cuda:"):
        raise ValueError("These model checks require an explicit GPU")
    device_index = int(args.device.split(":", 1)[1])
    torch.cuda.set_device(device_index)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats(device_index)
    reference_hash = sha256(ROOT / "ref/h_infinity.py")
    result = {"implementation_sha256": implementation_hash(),
              "script_sha256": sha256(Path(__file__)), "hannah_reference_sha256": reference_hash,
              "device": args.device, "gpu": torch.cuda.get_device_name(device_index),
              "torch_version": torch.__version__, "tolerance": {"rtol": 2e-4, "atol": 2e-5},
              "hinf": controller_checks(args.device),
              "jacobian": jacobian_checks(args.model, args.device)}
    assert sha256(ROOT / "ref/h_infinity.py") == reference_hash
    result["peak_allocated_gpu_bytes"] = torch.cuda.max_memory_allocated(device_index)
    result["passed"] = True
    output = UNIT / "cache" / (args.model + ".json")
    write_json(output, result)
    print(f"GPU reference checks passed: {output}", flush=True)


if __name__ == "__main__":
    main()
