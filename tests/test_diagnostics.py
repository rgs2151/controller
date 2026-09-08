from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import torch

from robust_steerability.control import HInfinityController, HInfinityOptions
from robust_steerability.control.types import FiniteHorizonControlProblem
from robust_steerability.experiments import calibration as cal
from robust_steerability.experiments import diagnostics as diag
from robust_steerability.experiments.generation import generate_completions
from robust_steerability.experiments.methods import build_policy
from robust_steerability.runtime.diagnostics import ReducedTrajectoryRecorder
from robust_steerability.runtime.policy import ReducedStateSetpointPolicy


@pytest.fixture
def bundle():
    horizon, n = 2, 2
    eye = torch.eye(n).repeat(horizon, 1, 1)
    problem = FiniteHorizonControlProblem(
        dynamics=eye * 0.8, control_channels=eye, disturbance_channels=eye * 0.1,
        state_costs=eye * 0.1, control_costs=eye, terminal_cost=torch.eye(n),
    )
    return {
        "problem": asdict(problem), "options": asdict(HInfinityOptions(tolerance=1e-4)),
        "record": {"run_id": "toy", "model_id": "toy-model", "model_revision": "pinned",
                   "model_family": "toy", "parameter_count": 100, "behavior": "toxicity_mitigation",
                   "intervention_channel": "reduced", "protocol_id": "test", "synthetic": True},
        "splits": {"fit": ["toy:fit0", "toy:fit1"], "calibration": [f"toy:cal{i}" for i in range(4)]},
        "normalization": {"protocol_id": "test-normalized", "coordinates": "normalized",
                          "state_whitening": torch.eye(n).repeat(horizon + 1, 1, 1),
                          "control_std": torch.ones(horizon, n), "semantic_output_std": torch.ones(horizon + 1, 1),
                          "depth_increment": torch.full((horizon,), 1 / horizon),
                          "stage_costs_depth_weighted": True},
        "calibration": {"residuals": torch.randn(4, horizon, n, generator=torch.Generator().manual_seed(12)),
                        "state_basis": torch.eye(n).repeat(horizon + 1, 1, 1),
                        "target_readouts": torch.ones(horizon + 1, 1, n),
                        "protected_readouts": torch.empty(horizon + 1, 0, n),
                        "reference_states": torch.zeros(horizon + 1, n),
                        "reference_controls": torch.zeros(horizon, n),
                        "disturbance_construction": "synthetic test"},
        "predictors": {},
    }


def freeze(tmp_path, bundle):
    path = tmp_path / "input.pt"
    torch.save(bundle, path)
    return diag.score(path, "cpu", cache_root=tmp_path / "cache")


def evaluation_payload(run):
    stamp = datetime.fromisoformat(diag.read_json(run / "manifest.json")["created_at_utc"])
    return {"run_id": run.name, "evaluation_id": "ood", "controller": "hinf", "shift": "ood",
            "score_manifest_sha256": diag.sha256(run / "manifest.json"),
            "evaluation_started_at_utc": (stamp + timedelta(seconds=1)).isoformat(),
            "protocol_id": "test", "success_definition": "toy score < 0.5", "matching_rule": "same fixed gains",
            "generation_config": {"seed": 7}, "evaluator": "synthetic",
            "observations": [{"prompt_id": "toy:held-out", "seed": 7, "success": True,
                              "raw_score": 0.1, "control_energy": 2.0, "collateral_metrics": {}}]}


def test_export_matches_solver_and_portable_read(tmp_path, bundle):
    problem = FiniteHorizonControlProblem(**bundle["problem"])
    expected = HInfinityController.synthesize(problem, device="cpu", options=HInfinityOptions(**bundle["options"])).solution()
    run = freeze(tmp_path, bundle)
    loaded = diag.load_run(run)
    torch.testing.assert_close(loaded["controller"]["gains"], expected.gains, rtol=0, atol=0)
    assert loaded["score"]["gamma_star"] == expected.gamma_star
    assert loaded["score"]["s_rob"] == 1 / expected.gamma_star
    assert loaded["score"]["predictors"]["probe_accuracy"] is None
    payload = evaluation_payload(run)
    input_path = tmp_path / "evaluation.json"
    diag.write_json(input_path, payload)
    diag.evaluate(input_path, cache_root=tmp_path / "cache")
    archive = diag.pack_run(run, tmp_path / "share.zip")
    moved = tmp_path / "unpacked"
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(moved)
    shutil.move(run, tmp_path / "source-no-longer-there")
    relocated = moved / "toy"
    report = diag.share_report(relocated, tmp_path / "report.json")
    assert "toy:held-out" not in report.read_text()
    script = (
        "import sys; from pathlib import Path; from robust_steerability.experiments.diagnostics import load_run; "
        "x=load_run(Path(sys.argv[1])); assert x['controller']['gains'].device.type=='cpu'; "
        "assert x['evaluations']['ood']['observations'][0]['success']; assert 'transformers' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script, str(relocated)], check=True)
    (relocated / "controller.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="changed"):
        diag.load_run(relocated)


def test_supplied_solution_does_not_resynthesize(tmp_path, bundle, monkeypatch):
    solution = HInfinityController.synthesize(FiniteHorizonControlProblem(**bundle["problem"]), device="cpu").solution()
    monkeypatch.setattr(HInfinityController, "synthesize", lambda *a, **k: pytest.fail("extra synthesis"))
    source = tmp_path / "input.pt"
    torch.save(bundle, source)
    run = diag.score(source, "cpu", cache_root=tmp_path, solution=solution)
    assert diag.score(source, "cpu", cache_root=tmp_path, solution=solution) == run
    assert diag.load_run(run)["controller"]["gains"].device.type == "cpu"


def test_hannah_diagnostic_outputs_agree(tmp_path, bundle, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "ref/h_infinity_optimization.py"
    before = diag.sha256(path)
    spec = importlib.util.spec_from_file_location("hannah_diagnostic_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CACHE", tmp_path / "hannah")
    run = freeze(tmp_path, bundle)
    hannah_run = module.score(tmp_path / "input.pt", "cpu")
    ours = diag.load_run(run)
    reference_score = diag.read_json(hannah_run / "score.json")
    for key, value in reference_score.items():
        assert ours["score"][key] == value
    reference_controller = torch.load(hannah_run / "controller.pt", weights_only=True)
    for key, value in reference_controller.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(ours["controller"][key], value, rtol=0, atol=0)
        else:
            assert ours["controller"][key] == value
    assert diag.sha256(path) == before


@pytest.mark.parametrize("bad", ["split", "residual", "whitening"])
def test_invalid_inputs_rejected(tmp_path, bundle, bad):
    if bad == "split":
        bundle["splits"]["calibration"][0] = "toy:fit0"
    elif bad == "residual":
        bundle["calibration"]["residuals"][0, 0, 0] = float("nan")
    else:
        bundle["normalization"]["state_whitening"] *= 0
    with pytest.raises(ValueError):
        freeze(tmp_path, bundle)


@pytest.mark.parametrize("bad", ["overlap", "early", "hash", "duplicate"])
def test_evaluation_provenance_validation(tmp_path, bundle, bad):
    run = freeze(tmp_path, bundle)
    payload = evaluation_payload(run)
    if bad == "overlap":
        payload["observations"][0]["prompt_id"] = "toy:fit0"
    elif bad == "early":
        payload["evaluation_started_at_utc"] = "2000-01-01T00:00:00+00:00"
    elif bad == "hash":
        payload["score_manifest_sha256"] = "wrong"
    else:
        payload["observations"] *= 2
    source = tmp_path / "evaluation.json"
    diag.write_json(source, payload)
    with pytest.raises(ValueError):
        diag.evaluate(source, cache_root=tmp_path / "cache")


class ToyEncoding(dict):
    def to(self, device):
        return ToyEncoding({key: value.to(device) for key, value in self.items()})


class ToyTokenizer:
    eos_token_id = 31

    def __call__(self, texts, **kwargs):
        texts = [texts] if isinstance(texts, str) else texts
        return ToyEncoding(input_ids=torch.tensor([[1, 2, 3] for _ in texts]),
                           attention_mask=torch.ones(len(texts), 3, dtype=torch.long))

    def decode(self, tokens, **kwargs):
        return " ".join(map(str, tokens.tolist()))


@pytest.fixture
def calibrated(tmp_path, monkeypatch, request):
    from transformers import GPT2Config, GPT2LMHeadModel
    torch.manual_seed(44)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=32,
                                      n_positions=32, eos_token_id=31, pad_token_id=31)).eval()
    records = [{"prompt_id": f"rtp:{i}", "text": str(i)} for i in range(40)]
    monkeypatch.setattr(cal, "calibration_records", lambda *a: (records[:4], records[4:8], records[8:12], {"id": "toy", "revision": "pinned"}))

    def states(model, tokenizer, texts, **kwargs):
        hidden = torch.stack([torch.randn(3, 8, generator=torch.Generator().manual_seed(200 + int(text))) for text in texts])
        return {"hidden": hidden, "attention_heads": hidden[:, :-1]}

    monkeypatch.setattr(cal, "collect_last_token_states", states)
    behavior = "toxicity_mitigation" if getattr(request.node, "callspec", None) is not None and request.node.callspec.params.get("kind") in {"id_toxicity", "ood_toxicity"} else "truthfulness"
    settings = {"behavior": behavior, "disturbance_coverage": 0.95, "seed": 4, "fit_prompts_per_class": 4, "disturbance_prompts": 4,
                "calibration_max_length": 12, "activation_batch_size": 2, "state_rank": 2,
                "whitening_floor": 1e-4, "ridge": 1e-3, "disturbance_variance": 0.95,
                "setpoint_multiplier": 2.0, "q": 0.1, "r": 1.0, "q_final": 1.0,
                "gamma_lower": 0.01, "gamma_upper": 10.0, "gamma_tolerance": 1e-4,
                "gamma_max_iterations": 50, "gamma_deployment_margin": 1e-3,
                "kp": 1.0, "ki": 0.0, "kd": 0.0,
                "model_loading": {"revision": "pinned", "dtype": "float32", "quantized": False}}
    arguments = dict(model_label="toy", model_id="toy", cache_path=tmp_path / "controllers" / "toy.pt",
                     settings=settings, controller_device="cpu")
    artifact, metadata = cal.calibrate_controller(model, ToyTokenizer(), **arguments)
    return model, arguments, artifact, metadata


def test_calibration_bundle_and_missing_cache_rejection(calibrated, monkeypatch, tmp_path):
    model, arguments, artifact, metadata = calibrated
    source = cal.diagnostic_run(arguments["cache_path"], metadata["fingerprint"])
    loaded = diag.load_run(source)
    assert loaded["score"]["coordinates"] == "normalized"
    assert loaded["score"]["stage_costs_depth_weighted"] is True
    assert loaded["calibration"]["calibration"]["encoders"].shape == (3, 8, 2)
    assert loaded["calibration"]["calibration"]["protected_readouts"].shape == (3, 1, 2)
    cal_data = loaded["calibration"]["calibration"]
    residual = cal_data["calibration_reduced_states"][:, 1:] - torch.einsum(
        "lij,nlj->nli", loaded["calibration"]["problem"]["dynamics"], cal_data["calibration_reduced_states"][:, :-1])
    torch.testing.assert_close(residual, cal_data["residuals"], rtol=0, atol=0)
    before = diag.sha256(arguments["cache_path"])
    shutil.move(cal.diagnostic_root(arguments["cache_path"]), tmp_path / "saved-new-bundle")
    monkeypatch.setattr(cal, "collect_last_token_states", lambda *a, **k: pytest.fail("No automatic recalibration"))
    with pytest.raises(FileNotFoundError):
        cal.calibrate_controller(model, ToyTokenizer(), **arguments)
    assert diag.sha256(arguments["cache_path"]) == before


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    os.environ.get("HINF_TEST_CUDA") != "1" or not torch.cuda.is_available(), reason="opt-in CUDA test"))])
def test_recording_preserves_generation_and_prompt_resume(calibrated, device, monkeypatch, tmp_path):
    model, _, artifact, _ = calibrated
    model.to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    plain = build_policy("hinf", artifact, kp=1, ki=0, kd=0)
    recorded = build_policy("hinf", artifact, kp=1, ki=0, kd=0, record=True)
    kwargs = dict(seed=71, max_length=8, max_new_tokens=3, do_sample=True,
                  top_p=0.9, temperature=1.0, repetition_penalty=1.0)
    records = [{"prompt_id": "test:heldout1", "text": "example"}, {"prompt_id": "test:heldout2", "text": "second"}]
    baseline = generate_completions(model, ToyTokenizer(), records, policy=plain, **kwargs)
    folder = tmp_path / "traces"
    rows = generate_completions(model, ToyTokenizer(), records, policy=recorded, trace_directory=folder, **kwargs)
    assert [row["completion"] for row in rows] == [row["completion"] for row in baseline]
    assert [row["seed"] for row in rows] == [71, 72]
    trace = torch.load(folder / rows[0]["trace_file"], weights_only=True)
    assert trace["control"].device.type == "cpu"
    assert trace["control"].shape[-1] == 2
    k = trace["layer_index"]
    expected = torch.einsum("tij,tbj->tbi", artifact.hinf_gains[k], trace["feedback"])
    torch.testing.assert_close(trace["deviation_control"], expected)
    torch.testing.assert_close(trace["control"], expected + artifact.reference_controls[k, None])
    assert trace["control_energy"] == float(trace["control"].double().square().sum())
    monkeypatch.setattr(model, "generate", lambda *a, **k: pytest.fail("cached prompt regenerated"))
    assert generate_completions(model, ToyTokenizer(), records, policy=recorded, trace_directory=folder, **kwargs) == rows


def test_recorder_does_not_call_stateful_controller_twice():
    from robust_steerability.control import PIDController, PIDGains
    kwargs = dict(means=torch.zeros(1, 2), encoders=torch.eye(2).unsqueeze(0),
                  decoders=torch.eye(2).unsqueeze(0), feature_unit=torch.tensor([[1.0, 0.0]]), setpoints=torch.tensor([2.0]), reference_controls=torch.zeros(1, 2))
    original = ReducedStateSetpointPolicy(PIDController(PIDGains(1, 0.1, 0.3)), **kwargs)
    recorded = ReducedStateSetpointPolicy(PIDController(PIDGains(1, 0.1, 0.3)), recorder=ReducedTrajectoryRecorder(), **kwargs)
    for policy in [original, recorded]:
        policy.prepare(torch.device("cpu"), torch.float32)
        policy.reset()
    for x in [0.5, 1.0, 1.5]:
        activation = torch.tensor([[x, 0.0]])
        torch.testing.assert_close(original.activation_delta(0, activation), recorded.activation_delta(0, activation), rtol=0, atol=0)
    assert recorded.recorder.finish()["layer_index"].tolist() == [0, 0, 0]


@pytest.mark.parametrize("kind", ["id_toxicity", "ood_toxicity", "truthfulness"])
def test_benchmark_wires_all_method_caches(calibrated, tmp_path, monkeypatch, kind):
    from robust_steerability.experiments import runner
    from robust_steerability.experiments.manifest import load_manifest
    model, arguments, artifact, metadata = calibrated
    unit = tmp_path / "parking" / "test_benchmark"
    unit.mkdir(parents=True)
    config = {"schema_version": 1, "kind": kind, "seed": 12, "sample_count": 2,
              "revisions": {"synthetic": "pinned"}, "methods": ["original", "alqr", "spid", "hinf", "actadd", "iti", "mean_act", "linear_act", "pid_act", "odesteer"],
              "models": [{"label": "toy", "model_id": "toy", "revision": "pinned", "dtype": "float32", "quantized": False}],
              "controller": {k: v for k, v in arguments["settings"].items() if k != "model_loading"},
              "controller_cache": str(arguments["cache_path"].parent), "controller_cache_read_only": True,
              "generation": {"max_length": 8, "max_new_tokens": 2, "do_sample": True, "top_p": 0.9,
                             "temperature": 1.0, "repetition_penalty": 1.0},
              "mmlu_shots": 5, "mmlu_max_length": 8, "judge_batch_size": 2,
              "prompt_source": "prompts.json", "subsets": ["test_ood", "mmlu"],
              "subset_generation": {"mmlu": {"do_sample": False, "max_new_tokens": 2}}}
    diag.write_json(unit / "prompts.json", {"synthetic": True})
    manifest_path = unit / "manifest.json"
    diag.write_json(manifest_path, config)
    manifest = load_manifest(manifest_path)
    model_entry = manifest.models[0]
    monkeypatch.setattr(runner, "_token", lambda *a: "")
    monkeypatch.setattr(runner, "load_causal_model", lambda *a: (model, ToyTokenizer()))
    monkeypatch.setattr(cal, "collect_last_token_states", lambda *a, **k: pytest.fail("new calibration on read-only cache"))
    records = [{"prompt_id": f"test:heldout{i}", "text": f"prompt {i}", "question": f"question{i}", "answer_index": 0} for i in range(2)]
    monkeypatch.setattr(runner, "_score_toxicity_rows", lambda texts, **kwargs: [0.1] * len(texts))
    mmlu = [{**r, "prompt": str(r["text"]), "prompt_id": f"mmlu:heldout{i}"} for i, r in enumerate(records)]
    monkeypatch.setattr(runner, "_prompt_sets", lambda *a: {"test_ood": records, "mmlu": mmlu})
    monkeypatch.setattr(runner, "_load_truth_judge", lambda *a: (object(), object()))
    def judge(model, tokenizer, prompts, batch_size):
        assert all("\nA:" in prompt and "\\nA:" not in prompt for prompt in prompts)
        return [{"score": 1.0, "answer": "yes"} for prompt in prompts]
    monkeypatch.setattr(runner, "_judge_yes", judge)
    observed_methods = []
    original_builder = runner.build_policy

    def checked_builder(method, *args, **kwargs):
        observed_methods.append(method)
        return original_builder(method, *args, **kwargs)

    monkeypatch.setattr(runner, "build_policy", checked_builder)
    archived_calibration = diag.sha256(arguments["cache_path"])
    source_run = cal.diagnostic_run(arguments["cache_path"], metadata["fingerprint"])
    source_files = {str(p.relative_to(source_run)): diag.sha256(p) for p in source_run.rglob("*") if p.is_file()}
    runner.run_job(manifest_path, 0, "cpu")
    job = unit / "cache" / "jobs" / "manifest" / "toy"
    completed = diag.read_json(job / "result.json")
    assert {r["method"] for r in completed["rows"]} == set(manifest.methods)
    run = job / "diagnostics" / "runs" / source_run.name
    loaded = diag.load_run(run)
    assert set(loaded["evaluations"]) == {"test_ood", "mmlu"}
    for evaluation in loaded["evaluations"].values():
        assert len(evaluation["observations"]) == 2
        for row in evaluation["observations"]:
            trace = torch.load(run / row["trace_file"], weights_only=True)
            assert trace["prompt_id"] == row["prompt_id"]
            assert trace["control_energy"] == row["control_energy"]
    assert diag.sha256(arguments["cache_path"]) == archived_calibration
    assert source_files == {str(p.relative_to(source_run)): diag.sha256(p) for p in source_run.rglob("*") if p.is_file()}
    monkeypatch.setattr(runner, "load_causal_model", lambda *a: pytest.fail("completed job reloaded model"))
    runner.run_job(manifest_path, 0, "cpu")


def test_prediction_panels_exclude_raw_scores(tmp_path, bundle):
    path = Path(__file__).resolve().parents[1] / "parking/h_infinity_optimization/diagnostic_analysis.py"
    spec = importlib.util.spec_from_file_location("diagnostic_analysis", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle["normalization"]["coordinates"] = "raw"
    run = freeze(tmp_path, bundle)
    source = tmp_path / "evaluate.json"
    diag.write_json(source, evaluation_payload(run))
    diag.evaluate(source, cache_root=tmp_path / "cache")
    result = module.prepare_panels("toy-analysis", "hinf", "ood", "test", "test-normalized", True, cache_root=tmp_path / "cache")
    assert diag.read_json(result / "pair_table.json") == []
    assert len(diag.read_json(result / "excluded_runs.json")) == 1
