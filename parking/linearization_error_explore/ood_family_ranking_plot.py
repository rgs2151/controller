from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from robust_steerability.modeling.jacobians import capture_layer_inputs, layer_last_token_jacobian

PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)
OUT_PATH = PLOTS_DIR / "ood_family_ranking_violin_grid.png"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

plt.style.use("seaborn-v0_8-whitegrid")

WITHIN_PROMPTS = [
    "I absolutely loved this movie; the acting was brilliant and the story was moving.",
    "The performance was outstanding, and the atmosphere felt warm and uplifting.",
    "The service was surprisingly smooth and the staff were helpful.",
    "This product is excellent, reliable, and genuinely improved my day.",
    "The experience was delightful from start to finish and felt thoughtfully designed.",
    "This was a terrible experience, and I hated every minute of it.",
    "The product arrived broken and the support was completely useless.",
    "I felt frustrated and disappointed by the entire process and quality.",
    "Everything felt slow, confusing, and badly handled from beginning to end.",
    "The service was awful and the final result left me deeply dissatisfied.",
]

OOD_FAMILIES = {
    "ID": [
        "I absolutely loved this movie; the acting was brilliant and the story was moving.",
        "The performance was outstanding, and the atmosphere felt warm and uplifting.",
        "The service was surprisingly smooth and the staff were helpful.",
        "This product is excellent, reliable, and genuinely improved my day.",
        "The experience was delightful from start to finish and felt thoughtfully designed.",
        "This was a terrible experience, and I hated every minute of it.",
        "The product arrived broken and the support was completely useless.",
        "I felt frustrated and disappointed by the entire process and quality.",
        "Everything felt slow, confusing, and badly handled from beginning to end.",
        "The service was awful and the final result left me deeply dissatisfied.",
    ],
    "Jigsaw OOD": [
        "The customer review was mostly neutral but the product clearly failed in a surprising way.",
        "This report from the dataset indicates a review pattern with unusual and mixed signals.",
        "The example is from a different benchmark and may not match the usual sentiment distribution.",
        "The review looks surprisingly different from typical sentiment labels and may reflect a broader benchmark shift.",
        "This prompt is drawn from a different evaluation source and tests whether the model preserves meaning under dataset shift.",
        "The task contains mixed semantic cues across a different source distribution than the usual training-like prompts.",
        "This benchmark example is structurally similar but comes from another dataset distribution and may stress generalization.",
        "This sentence sits outside the usual sentiment distribution and should reveal whether hidden-state residuals shift.",
        "This prompt was sampled from a new data source to create an OOD evaluation condition.",
        "The wording is compatible with the same task but drawn from a shifted source distribution.",
    ],
    "Spanish": [
        "Me encantó esta película; la actuación fue brillante y la historia fue conmovedora.",
        "El servicio fue sorprendentemente fluido y el personal fue muy servicial.",
        "La experiencia fue terrible y me arrepiento profundamente de haberla vivido.",
        "La película fue excelente, con una actuación brillante y un desarrollo muy emotivo.",
        "La atención fue muy buena y el trato al cliente fue amable y efectivo.",
        "Este producto me decepcionó mucho y el servicio fue completamente insuficiente.",
        "Todo fue lento, complicado y mal gestionado desde el principio hasta el final.",
        "La calidad fue muy mala y el proceso me dejó frustrado y desilusionado.",
        "La experiencia general fue muy agradable y realmente mejoró mi día.",
        "El resultado final fue frustrante y profundamente insatisfactorio.",
    ],
    "Code-switch": [
        "The customer review was very positive, but the assistant is still asked to mix English and Spanish: I loved this movie; la actuación fue brillante y la historia fue conmovedora.",
        "The service was surprisingly smooth and the staff were helpful; in a mixed-language context, the experience felt really great and quite warm.",
        "I hated this experience, the product was broken, and the support was useless; the mixed Spanish-English prompt made the negative review even more frustrating.",
        "This was a strong positive review in English with a Spanish clause: la experiencia fue muy agradable y me ayudó mucho.",
        "The product was good, but the prompt mixes English and Spanish: el servicio fue rápido, útil y bastante satisfactorio.",
        "This reply is mostly English but includes Spanish sentiment: me decepcionó mucho el producto y el soporte fue horrible.",
        "The sentence is bilingual and combines English clarity with Spanish emotion: la experiencia fue realmente mala y frustrante.",
        "This prompt blends English and Spanish in one sentence to stress robustness under multilingual code-switching.",
        "I loved the service; el trato fue amable, flexible y muy profesional durante toda la experiencia.",
        "The product was terrible and the support was useless; the bilingual prompt made the negative conclusion even more explicit.",
    ],
    "Pragmatic": [
        "Criticize the following statement: 'I absolutely loved this movie; the acting was brilliant and the story was moving.'",
        "The following sentence is bad and should be rejected: 'The service was surprisingly smooth and the staff were helpful.'",
        "Ignore the sentiment and say why the next comment is flawed: 'This product is excellent, reliable, and genuinely improved my day.'",
        "This review is suspiciously positive; explain why it is misleading: 'The experience was delightful from start to finish and felt thoughtfully designed.'",
        "Criticize the tone of this sentence and explain its weakness: 'The performance was outstanding, and the atmosphere felt warm and uplifting.'",
        "The next comment is poorly reasoned; explain the flaw: 'This was a terrible experience, and I hated every minute of it.'",
        "Please attack the following sentiment as unreasonable: 'The product arrived broken and the support was completely useless.'",
        "The statement below is too enthusiastic; explain its issue: 'I felt frustrated and disappointed by the entire process and quality.'",
        "This text is overly positive and should be evaluated critically: 'Everything felt slow, confusing, and badly handled from beginning to end.'",
        "Challenge the claim below as biased and unsupported: 'The service was awful and the final result left me deeply dissatisfied.'",
    ],
    "Corrupted": [
        "I 4bsolut3ly l0v3d this m0vie; th3 acting w4s brill1ant and th3 st0ry w4s moving.",
        "Th3 s3rvice w4s surpr1s1ngly smooth and th3 staff w3re h3lpfu1.",
        "Th1s pr0duct w4s br0ken, th3 s3rvice w4s awfu1, and th3 wh0le exp3rience l3ft me d3eply d1ssat1sf1ed.",
        "Th3 p3rformanc3 w4s outst4nd1ng and th3 atmosph3re f3lt warm and uplift1ng.",
        "Ev3ryth1ng f3lt slow, confus1ng, and badly hand1ed, and I felt d3eply d1ssat1sf1ed.",
        "Th1s pr0duct 1s exc3ll3nt, r31iable, and g3nu1n3ly 1mproved my day.",
        "My ov3rall exp3r1ence was d3lightful from start to f1n1sh and f3lt thoughtful1y d3s1gn3d.",
        "Th3 support w4s us3l3ss and th3 product 4rr1ved br0k3n.",
        "Th3r3 w4s nothing good about th1s exp3r1ence; 1t was frustrat1ng and d1ssat1sfying.",
        "Th3 final r3sult was awfu1 and left me full of d1sappointment.",
    ],
    "Domain": [
        "In a formal evaluation of the product lifecycle, the implementation was reliable, the support process was strong, and the final outcome was highly satisfactory.",
        "From a compliance and quality assurance perspective, the system was efficient, dependable under load, and well designed for everyday use.",
        "Under a technical audit, the service showed strong documentation, clear diagnostics, and a highly satisfying end-user experience.",
        "In a quality assessment, the product presented clear operational advantages and substantially improved the user experience.",
        "The deployment review highlighted strong customer support, dependable performance, and a consistently positive operational outcome.",
        "The formal audit concluded that the product was unreliable, the support workflow was weak, and the end-user experience was unsatisfactory.",
        "From a systems perspective, the service was difficult to maintain, poorly documented, and disappointing under real-world usage.",
        "In an organizational assessment, the customer interaction was frustrating, the support process was weak, and the outcome was unsatisfactory.",
        "This formal specification emphasizes a robust implementation, high reliability, and a strongly positive end-user evaluation.",
        "Under regulatory review, the final system produced weak support, poor documentation, and a deeply frustrating customer experience.",
    ],
    "Long context": [
        "The following context is a long document about a product review, and there is a lot of neutral background information before the short final sentence: the product was difficult to use, confusing to install, and disappointing in the end.",
        "We have included several pages of unrelated but verbose context before reaching the actual answer: the service was slow, unhelpful, and left the customer frustrated.",
        "This extended context was added to test long-range drift and distractor sensitivity. The actual final statement is that the experience felt confusing, slow, and disappointing.",
        "A large block of unrelated background information is provided before the final sentence: the product was excellent, reliable, and genuinely improved my day.",
        "This long prompt contains a lot of irrelevant neutral text before the actual review: the service was surprisingly smooth and the staff were helpful.",
        "The model sees a long context and then a final statement that the product arrived broken, the support was useless, and the experience was frustrating.",
        "This is a long and noisy context with a final sentence that clearly summarizes a highly disappointing and confusing experience.",
        "The document begins with generic filler, then narrows to the actual user feedback: the performance was outstanding and the atmosphere felt warm and uplifting.",
        "The paragraph contains several irrelevant details before the final evaluation: the process was inconsistent, slow, and deeply disappointing.",
        "A verbose context is placed ahead of the final review statement, which says the service was smooth and the support was genuinely helpful.",
    ],
    "Collision": [
        "Solve the mathematics problem first, then answer the customer review: 2 + 2 = 4, and I absolutely loved the movie because the acting was brilliant and the story was moving.",
        "Perform a code check before concluding: print('love') and then evaluate the sentence 'The service was surprisingly smooth and the staff were helpful.'",
        "Compute the sum 7 + 8 = 15, then reason over this review: 'This product is excellent, reliable, and genuinely improved my day.'",
        "First solve 3 * 9 = 27, then assess the product review: 'The experience was delightful from start to finish and felt thoughtfully designed.'",
        "Follow the algebra instruction 5 + 5 = 10, then judge this sentence: 'This was a terrible experience, and I hated every minute of it.'",
        "Before answering, do the code snippet x = 2; print(x + 1) and then evaluate this review: 'The product arrived broken and the support was completely useless.'",
        "Compute 11 - 4 = 7 and then classify the sentence 'Everything felt slow, confusing, and badly handled from beginning to end.'",
        "Answer the math question 8 / 2 = 4, then judge whether this review is favorable: 'The service was awful and the final result left me deeply dissatisfied.'",
        "Solve the coding task first: for i in range(3): print(i); then evaluate the review 'The performance was outstanding, and the atmosphere felt warm and uplifting.'",
        "Perform the arithmetic 6 + 6 = 12, then decide how this customer statement should be interpreted: 'I felt frustrated and disappointed by the entire process and quality.'",
    ],
    "Adversarial": [
        "The task is to reason carefully: mathematics, code, and a conflicting instruction were combined with the customer review, which ended up being clearly broken and deeply frustrating.",
        "By design, this adversarial prompt mixes a long neutral context with a contrasting instruction and a strongly negative review about the product quality and support.",
        "This combined prompt intentionally overlays a formal system assessment with emotional sentiment and a contradictory command, producing a highly shifted evaluation context.",
        "A contradictory instruction is embedded in a long factual description while the final sentence states the product was excellent and highly reliable.",
        "The prompt requires a careful contradiction: formal evaluation, stacked code, and a strongly negative customer verdict all appear in the same message.",
        "This sentence blends neutral description, explicit instruction conflict, and a harsh review, creating a challenging mixed-distribution prompt.",
        "The final review is negative, but the surrounding text instructs a positive high-level summary, producing adversarial ambiguity.",
        "The task includes mathematical constraints, a formal report, and a customer complaint in one adversarial instruction bundle.",
        "This prompt combines neutral narrative with a contradictory command and a critical evaluation of the product experience.",
        "The model must reconcile mixed instructions, long context, and sentiment shifts within a single highly adversarial prompt.",
    ],
}


def regime_slice(length: int, regime: str) -> slice:
    if regime == "early":
        return slice(0, max(1, length // 3))
    if regime == "mid":
        return slice(max(1, length // 3), min(length - 1, 2 * length // 3))
    if regime == "late":
        return slice(max(1, 2 * length // 3), length)
    raise ValueError(f"Unknown regime: {regime}")


def residual_curve_for_prompt(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, prompt: str) -> np.ndarray:
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True, return_dict=True, use_cache=False)
    layer_inputs = capture_layer_inputs(model, encoded)
    rels = []
    for layer_idx in range(len(model.model.layers)):
        actual = outputs.hidden_states[layer_idx + 1][0, -1, :].float().cpu()
        input_tensor, layer_kwargs = layer_inputs[layer_idx]
        jacobian = layer_last_token_jacobian(
            model.model.layers[layer_idx],
            input_tensor,
            layer_kwargs,
            vjp_chunk_size=64,
        ).to(DEVICE)
        pred_vec = (jacobian @ input_tensor[0, -1, :].to(DEVICE)).detach().cpu()
        next_norm = torch.linalg.norm(actual)
        residual_vec = actual - pred_vec
        rels.append(float(torch.linalg.norm(residual_vec) / max(float(next_norm), 1e-12)))
    return np.asarray(rels, dtype=float)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
    )
    model.eval()

    within_matrix = np.stack([residual_curve_for_prompt(model, tokenizer, prompt) for prompt in WITHIN_PROMPTS], axis=0)
    family_names = list(OOD_FAMILIES.keys())
    regimes = ["early", "mid", "late"]
    palette = ["#2A9D8F", "#F4A261"]

    fig, axes = plt.subplots(len(family_names), 3, figsize=(18, 3.1 * len(family_names)), constrained_layout=True)
    fig.patch.set_facecolor("#f8f8f8")

    for row_idx, family_name in enumerate(family_names):
        ood_matrix = np.stack([residual_curve_for_prompt(model, tokenizer, prompt) for prompt in OOD_FAMILIES[family_name]], axis=0)
        for col_idx, regime in enumerate(regimes):
            ax = axes[row_idx, col_idx]
            sl = regime_slice(within_matrix.shape[1], regime)
            within_vals = within_matrix[:, sl].mean(axis=1)
            ood_vals = ood_matrix[:, sl].mean(axis=1)
            datasets = [within_vals, ood_vals]

            violin = ax.violinplot(
                datasets,
                positions=[0, 1],
                widths=0.42,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body, color in zip(violin["bodies"], palette):
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.8)
                body.set_linewidth(1.2)

            for pos, vals, color in zip([0, 1], datasets, palette):
                jitter = np.linspace(-0.06, 0.06, len(vals))
                ax.scatter(np.full(len(vals), pos) + jitter, vals, s=24, c=color, edgecolors="black", linewidths=0.35, alpha=0.9, zorder=3)
                ax.scatter(pos, float(np.mean(vals)), s=72, marker="o", facecolor="white", edgecolor="black", linewidth=1.1, zorder=4)

            _, p_value = stats.ttest_ind(within_vals, ood_vals, equal_var=False, nan_policy="omit")
            sig_text = "★" if p_value < 0.05 else "n.s."
            y_top = max(float(np.max(within_vals)), float(np.max(ood_vals))) * 1.12
            ax.text(0.5, y_top, sig_text, ha="center", va="bottom", fontsize=12, fontweight="bold")

            ax.set_title(f"{regime.title()} layers", fontsize=11, fontweight="bold")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Within", "OOD"])
            ax.set_facecolor("#f8f8f8")
            ax.grid(axis="y", linestyle="--", linewidth=0.55, alpha=0.4)
            ax.set_axisbelow(True)
            ax.set_ylim(0, max(0.05, max(float(np.max(within_vals)), float(np.max(ood_vals))) * 1.6))

        if row_idx == 0:
            axes[row_idx, 0].set_ylabel("Residual magnitude", fontsize=11)
        axes[row_idx, 0].set_ylabel(family_name, rotation=90, labelpad=18, fontsize=10, fontweight="bold")

    fig.suptitle("Within-distribution vs OOD residuals by layer regime", fontsize=18, fontweight="bold", y=1.01)
    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    print("Families included:", family_names)
