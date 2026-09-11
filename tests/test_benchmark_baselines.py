import torch
import importlib.util
from pathlib import Path

from robust_steerability.experiments.baselines import ode_gradient, polynomial_features
from robust_steerability.experiments.calibration import collect_last_token_states


def test_polynomial_vector_field_matches_derivative():
    generator = torch.Generator().manual_seed(4)
    p = {"components": 16, "gamma": 0.1, "bias": 1.0,
         "indices": torch.randint(16, (2, 6), generator=generator),
         "signs": (torch.randint(2, (2, 6), generator=generator) * 2 - 1).float(),
         "coef": torch.randn(16, generator=generator)}
    x = torch.randn(3, 5, generator=generator, requires_grad=True)
    features, _, _ = polynomial_features(x, p["indices"], p["signs"], 16, 0.1, 1.0)
    expected = torch.autograd.grad((features @ p["coef"]).sum(), x)[0]
    torch.testing.assert_close(ode_gradient(x.detach(), p), expected, atol=1e-6, rtol=1e-5)


def test_capture_uses_pre_final_norm_decoder_states():
    from transformers import GPT2Config, GPT2LMHeadModel
    from test_diagnostics import ToyTokenizer
    torch.manual_seed(6)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=8, n_head=2, vocab_size=32, n_positions=16)).eval()
    captured = collect_last_token_states(model, ToyTokenizer(), ["a", "b"], max_length=8, batch_size=2)
    encoded = ToyTokenizer()(["a", "b"])
    with torch.no_grad():
        normalized = model(**encoded, output_hidden_states=True).hidden_states[-1][:, -1]
        torch.testing.assert_close(model.transformer.ln_f(captured["hidden"][:, -1]), normalized)
    assert captured["hidden"].shape == (2, 3, 8)
    assert captured["attention_heads"].shape == (2, 2, 8)
    assert not torch.allclose(captured["hidden"][:, -1], normalized)


def test_quality_joint_tokenization_and_reference_truncation():
    path = Path(__file__).resolve().parents[1] / "ref/paper_benchmark_50/score_quality.py"
    spec = importlib.util.spec_from_file_location("score_quality", path)
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    calls = []
    def tokenizer(text, **kwargs):
        calls.append((text, kwargs))
        return {"input_ids": ([0] + list(range(1, len(text) + 1)))[:kwargs["max_length"]]}
    assert quality.perplexity_input(tokenizer, "context", "new") == list(range(11))
    assert calls[-1] == ("contextnew", {"add_special_tokens": True, "truncation": True, "max_length": 128})
    assert len(quality.perplexity_input(tokenizer, "x" * 200, "new")) == 128
    assert quality.perplexity_input(tokenizer, "context", "") == list(range(8))
