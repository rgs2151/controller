"""Fit portable baseline parameters from the same isolated behavior fit split.

Algorithms: Turner et al. ActAdd; Apple ml-act sorted 1-D transport;
dungnvnus/pid-steering mean transport; honest_llama head selection; ODESteer.
The benchmark declares its common last-token, post-block intervention protocol.
ITI instead acts on attention heads before their output projection.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from robust_steerability.runtime.diagnostics import ReducedTrajectoryRecorder


def polynomial_features(values, indices, signs, components, gamma, bias):
    normalized = values / (values.norm(dim=-1, keepdim=True) + 1e-12)
    extended = torch.cat([normalized * gamma ** 0.5, values.new_full((len(values), 1), bias ** 0.5)], dim=1)
    sketches = []
    for index, sign in zip(indices, signs, strict=True):
        sketch = values.new_zeros(len(values), components)
        sketch.scatter_add_(1, index.expand(len(values), -1), extended * sign)
        sketches.append(sketch)
    sketches = torch.stack(sketches, dim=1)
    transformed = torch.fft.irfft(torch.fft.rfft(sketches).prod(dim=1), n=components)
    return transformed, sketches, normalized


def ode_gradient(values, parameters):
    """Analytical VJP of the normalized degree-two polynomial count sketch."""
    _, sketches, normalized = polynomial_features(
        values, parameters["indices"], parameters["signs"], parameters["components"],
        parameters["gamma"], parameters["bias"])
    other = sketches.flip(1)
    correlation = torch.fft.irfft(
        torch.fft.rfft(other).conj() * torch.fft.rfft(parameters["coef"])[None, None],
        n=parameters["components"])
    indices = parameters["indices"][:, :-1]
    signs = parameters["signs"][:, :-1]
    gradient = (correlation.gather(2, indices[None].expand(len(values), -1, -1)) * signs).sum(dim=1)
    gradient = gradient * parameters["gamma"] ** 0.5
    return (gradient - normalized * (gradient * normalized).sum(dim=-1, keepdim=True)) / (
        values.norm(dim=-1, keepdim=True) + 1e-12)


def fit_baselines(calibration: dict, *, seed: int, strengths: dict[str, float]) -> dict:
    hidden = calibration["fit_hidden_states"].detach().cpu().float()[:, 1:]
    labels = np.asarray(calibration["fit_labels"])
    negative, positive = hidden[labels == 0], hidden[labels == 1]
    horizon, width = hidden.shape[1:]
    layer = (horizon - 1) // 2
    difference = positive.double().mean(dim=0) - negative.double().mean(dim=0)
    mask = (negative.double().std(dim=0) > 1e-4) & (positive.double().std(dim=0) > 1e-4)
    source, target = negative.double().sort(dim=0).values, positive.double().sort(dim=0).values
    source_mean, target_mean = source.mean(dim=0), target.mean(dim=0)
    # Same sorted-pair OLS as Apple's LinearProj.optimize, with explicit RNG.
    noise = np.random.RandomState(seed).randn(*source.shape) * 1e-8
    centered = source + torch.from_numpy(noise) - source_mean
    slope = (centered * (target - target_mean)).sum(dim=0) / centered.square().sum(dim=0)
    intercept = target_mean - slope * source_mean
    result = {
        "layer": layer, "protocol": "last-token-post-block-fit-only-v1",
        "actadd_direction": (positive[0] - negative[0]).float(),
        "mean_direction": difference.float(), "mean_mask": mask,
        "linear_slope": slope.float(), "linear_intercept": intercept.float(),
        "fit_prompt_ids": [row["prompt_id"] for row in calibration["fit_records"]],
        "strengths": dict(strengths),
    }
    # ITI ranks heads using a held-out part of the fit split, never test outcomes.
    heads = calibration["fit_attention_heads"].detach().cpu().float()
    head_count = calibration["attention_head_count"]
    heads = heads.reshape(len(heads), horizon, head_count, -1)
    groups = [row.get("source_prompt_id", row["prompt_id"]) for row in calibration["fit_records"]]
    train, validation = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(hidden, labels, groups))
    if len(set(labels[train])) != 2 or len(set(labels[validation])) != 2:
        raise ValueError("ITI grouped probe split requires both classes in train and validation")
    accuracy, coefficients, intercepts = [], [], []
    for k in range(horizon):
        for head in range(head_count):
            x = heads[:, k, head].numpy()
            probe = LogisticRegression(random_state=seed, max_iter=1000).fit(x[train], labels[train])
            accuracy.append(float(probe.score(x[validation], labels[validation])))
            coefficients.append(torch.from_numpy(probe.coef_[0]).float())
            intercepts.append(float(probe.intercept_[0]))
    selected = np.argsort(accuracy)[::-1][:min(48, horizon * head_count)].copy()
    directions = torch.zeros_like(heads[0])
    for index in selected:
        k, head = divmod(int(index), head_count)
        direction = heads[labels == 1, k, head].mean(dim=0) - heads[labels == 0, k, head].mean(dim=0)
        direction = direction / direction.norm().clamp_min(1e-12)
        scale = (heads[:, k, head] @ direction).std(correction=0)
        directions[k, head] = scale * direction
    result["iti"] = {"directions": directions.reshape(horizon, width),
                     "head_accuracy": accuracy, "probe_coefficients": torch.stack(coefficients), "probe_intercepts": intercepts,
                     "selected_heads": selected.tolist(), "train_indices": train.tolist(),
                     "validation_indices": validation.tolist(), "head_count": head_count}
    generator = torch.Generator().manual_seed(seed)
    ode = {"components": 8000, "gamma": 0.1, "bias": 1.0, "steps": 10,
           "indices": torch.randint(8000, (2, width + 1), generator=generator),
           "signs": (torch.randint(2, (2, width + 1), generator=generator) * 2 - 1).float()}
    transformed, _, _ = polynomial_features(hidden[:, layer], ode["indices"], ode["signs"],
                                              ode["components"], ode["gamma"], ode["bias"])
    probe = LogisticRegression(max_iter=1000).fit(transformed.numpy(), labels)
    ode["coef"] = torch.from_numpy(probe.coef_[0]).float()
    ode["intercept"] = float(probe.intercept_[0])
    result["ode"] = ode
    return result


class BaselinePolicy:
    """Baseline transport in its declared physical coordinates, with saved traces."""

    def __init__(self, method, parameters, strength=1.0, record=False):
        self.method, self.parameters, self.strength = method, parameters, strength
        self.site = "attention_heads" if method == "iti" else "block_output"
        self.recorder = ReducedTrajectoryRecorder("norms") if record else None

    def prepare(self, device, dtype):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value
        self.parameters = move(self.parameters)

    def reset(self):
        if self.recorder is not None:
            self.recorder.reset()

    def activation_delta(self, layer_index, activation):
        p = self.parameters
        if self.method != "iti" and layer_index != p["layer"]:
            return torch.zeros_like(activation)
        x = activation.float()
        if self.method == "iti":
            delta = self.strength * p["iti"]["directions"][layer_index].expand_as(x)
        elif self.method == "actadd":
            delta = self.strength * p["actadd_direction"][layer_index].expand_as(x)
        elif self.method == "mean_act":
            delta = self.strength * p["mean_direction"][layer_index].expand_as(x) * p["mean_mask"][layer_index]
        elif self.method == "linear_act":
            transported = x * p["linear_slope"][layer_index] + p["linear_intercept"][layer_index]
            delta = self.strength * (transported - x)
        elif self.method == "pid_act":
            direction = p["mean_direction"][layer_index]
            if layer_index > 0:
                direction = direction + 0.005 * (p["mean_direction"][:layer_index].mean() + direction)
            delta = 0.7 * self.strength * direction.expand_as(x) * p["mean_mask"][layer_index]
        elif self.method == "odesteer":
            transported = x.clone()
            for _ in range(p["ode"]["steps"]):
                gradient = ode_gradient(transported, p["ode"])
                transported = transported + (self.strength / p["ode"]["steps"]) * gradient / (gradient.norm(dim=-1, keepdim=True) + 1e-10)
            delta = transported - x
        else:
            raise ValueError(f"Unknown baseline: {self.method}")
        if self.method in {"mean_act", "linear_act", "pid_act"}:
            delta = delta * ((x > -1e6) & (x < 1e6))
        delta = delta.to(activation.dtype)
        if self.recorder is not None:
            self.recorder.append(layer_index, state=x, feedback=x,
                                 control=delta.float(), deviation_control=delta.float(),
                                 reduced_intervention=delta.float(), hidden_delta=delta)
        return delta
