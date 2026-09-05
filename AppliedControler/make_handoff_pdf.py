"""Generate the remaining-experiments handoff PDF for the next developer.

Usage:
    python -m AppliedControler.make_handoff_pdf
Writes results_reports/remaining_experiments_handoff.pdf
"""

from __future__ import annotations

import os
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_reports", "remaining_experiments_handoff.pdf")

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("Bodyx", parent=styles["BodyText"], fontSize=9, leading=12)
CELL = ParagraphStyle("Cellx", parent=styles["BodyText"], fontSize=8, leading=10)
MONO = ParagraphStyle(
    "Monox", parent=styles["BodyText"], fontSize=7.5, leading=10, fontName="Courier",
    backColor=colors.whitesmoke, leftIndent=4, spaceBefore=2, spaceAfter=2,
)


def P(text: str, style=CELL) -> Paragraph:
    return Paragraph(text, style)


EXPERIMENTS = [
    # (#, experiment, model, status, what's needed, priority)
    ("1", "Truthfulness (paper-aligned, full)", "Qwen-2.5-0.5B",
     "Smoke only (100 TruthfulQA / 50 MMLU samples)",
     "Full run: all TruthfulQA + full MMLU 5-shot, quantized. "
     "<b>Must pass --device cuda:0</b> (bare 'cuda' crashes the device parser).",
     "High"),
    ("2", "RTP toxicity (calibrated, full)", "Qwen-2.5-0.5B",
     "Smoke only (25 prompts)",
     "Full-prompt calibrated steering run, then re-export "
     "paper_style_table_paper_like_calibrated.csv (replace smoke rows).",
     "High"),
    ("3", "OOD global-3 (full)", "DistilGPT-2",
     "Smoke only (25 prompts per subset)",
     "Full run of jigsaw_long, toxicchat_long, mmlu_ood_other_concepts x 3 "
     "controllers. Export without the _smoke suffix.",
     "High"),
    ("4", "OOD global-3 (full)", "Qwen-2.5-0.5B",
     "Smoke only (25 prompts per subset)",
     "Same as #3, using experiment_plan_ood_global3_qwen05b.json.",
     "High"),
    ("5", "RTP toxicity", "Qwen-2.5-1.5B",
     "Never run (N/A row in toxicity figure)",
     "Calibrated steering run on RealToxicityPrompts; fills the gray 1.5B row.",
     "Medium"),
    ("6", "OOD global-3", "Qwen-2.5-1.5B",
     "Never run (N/A rows in OOD figures)",
     "Steering run on the 3 global OOD subsets; fills the gray 1.5B rows.",
     "Medium"),
    ("7", "Gain / lambda sweep (A-LQR, H-infinity)", "Qwen-2.5-0.5B (cheapest)",
     "Deferred by project owner",
     "Sweep controller gains to fix over-steering collapse "
     "(True ~95%, Info ~1%) confirmed on 0.5B and 1.5B. Unblocks better "
     "truthfulness numbers for A-LQR / H-infinity everywhere.",
     "Medium"),
    ("8", "Truthfulness A-LQR / H-infinity", "Qwen-2.5-7B",
     "N/A", "Excluded by project owner decision (no more 7B/14B runs). "
     "Rows stay as red N/A markers.", "Off"),
    ("9", "All toxicity / OOD", "Qwen-2.5-7B / 14B",
     "N/A", "Excluded by project owner decision.", "Off"),
]

FIGURES = [
    ("final_summary_figure", "Truthfulness summary (bar, models on x) + DistilGPT-2 toxicity panel", "truthfulness CSV + calibrated toxicity CSV"),
    ("final_summary_model_dissected", "Truthfulness dissected: rows=models, cols=T-I/True/Info/MMLU", "paper_style_table_truthfulness_all_models_methods.csv"),
    ("final_summary_model_dissected_toxicity", "Toxicity dissected: rows=models, cols=rate(log)/reduction", "paper_style_table_paper_like_calibrated.csv"),
    ("final_summary_figure_ood[_smoke]", "OOD summary: 1x3 panels (one per OOD subset), models on x", "paper_style_table_ood_global3_smoke.csv"),
    ("final_summary_model_dissected_ood[_smoke]", "OOD dissected: rows=models, cols=3 subsets, toxicity % (log)", "same OOD table"),
    ("final_summary_model_dissected_ood_reduction[_smoke]", "OOD reduction dissected: same grid, -percent_change (linear)", "same OOD table"),
]


def build() -> None:
    doc = SimpleDocTemplate(
        OUT, pagesize=landscape(A4),
        leftMargin=14 * mm, rightMargin=14 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
        title="Remaining experiments - handoff",
    )
    story = []

    # ---------- Page 1: overview + table ----------
    story.append(Paragraph("LinearLLMDynamic — Remaining Experiments (Handoff)", H1))
    story.append(Paragraph(
        f"Status as of {date.today().isoformat()} · repo: LinearLLMDynamic (branch main) · "
        "contact figures: AppliedControler/results_reports/final_summary_*.png", BODY))
    story.append(Paragraph(
        "The project produces <b>6 summary figures</b> comparing 4 methods "
        "(Original, A-LQR, S-PID, H-infinity) across model scales on truthfulness, "
        "in-distribution toxicity (RealToxicityPrompts) and 3 out-of-distribution toxicity "
        "benchmarks. Models whose data is only <b>smoke-level</b> (reduced samples, "
        "provenance run_status=smoke_test) or missing are shown with <b>gray labels</b> in every "
        "figure; fully-run models are black. The table below lists what remains to make "
        "every label black.", BODY))
    story.append(Spacer(1, 6))

    header = ["#", "Experiment", "Model", "Current status", "What's needed", "Priority"]
    rows = [[P(f"<b>{h}</b>") for h in header]]
    for row in EXPERIMENTS:
        rows.append([P(c) for c in row])
    tbl = Table(rows, colWidths=[8 * mm, 48 * mm, 34 * mm, 48 * mm, 108 * mm, 16 * mm], repeatRows=1)
    prio_colors = {"High": colors.Color(1, 0.92, 0.92), "Medium": colors.Color(1, 0.98, 0.9),
                   "Off": colors.Color(0.93, 0.93, 0.93)}
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.15, 0.2, 0.3)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, row in enumerate(EXPERIMENTS, start=1):
        style.append(("BACKGROUND", (0, i), (-1, i), prio_colors[row[-1]]))
    tbl.setStyle(TableStyle(style))
    story.append(tbl)
    story.append(PageBreak())

    # ---------- Page 2: how to run ----------
    story.append(Paragraph("How to run the remaining experiments", H1))

    story.append(Paragraph("0. Environment", H2))
    story.append(Paragraph(
        "Lightning.ai studio, single H100 80GB. Interpreter: "
        "/teamspace/studios/this_studio/.venv/bin/python. Always run modules from the repo root:", BODY))
    story.append(Paragraph(
        "cd /teamspace/studios/this_studio/LinearLLMDynamic<br/>"
        "/teamspace/studios/this_studio/.venv/bin/python -u -m AppliedControler.&lt;module&gt; ...", MONO))
    story.append(Paragraph(
        "<b>Critical gotcha:</b> quantized model loading parses the device string as "
        "device.split(':')[1] — always pass <b>--device cuda:0</b>, never bare 'cuda' "
        "(IndexError otherwise). Push every commit to BOTH remotes: origin "
        "(erfanzabeh/LinearLLMDynamic) and upstream (rgs2151/controller).", BODY))

    story.append(Paragraph("1. Truthfulness full run (experiments #1)", H2))
    story.append(Paragraph(
        "Same command as the smoke run in results_reports/truthfulness_rerun_runs/qwen05b_smoke.log, "
        "but drop --truthful-samples / --mmlu-samples limits:", BODY))
    story.append(Paragraph(
        "python -u -m AppliedControler.eval_truthfulness_paper_aligned \\<br/>"
        "&nbsp;&nbsp;--model Qwen/Qwen2.5-0.5B --quantized --device cuda:0", MONO))
    story.append(Paragraph(
        "Then merge the output CSV into "
        "results_reports/paper_style_table_truthfulness_all_models_methods.csv "
        "(model key 'Qwen-2.5-0.5B-smoke' → rename to 'Qwen-2.5-0.5B-ours', set "
        "run_status=full, update TRUTH_MODEL_ORDER/LABELS in figure_config.py).", BODY))

    story.append(Paragraph("2. Steering runs: RTP + OOD full (experiments #2-6)", H2))
    story.append(Paragraph(
        "Experiment plans already exist: AppliedControler/experiment_plan_ood_global3.json "
        "(DistilGPT-2) and experiment_plan_ood_global3_qwen05b.json (0.5B; includes rtp_random). "
        "The smoke runs used --eval-prompts 25 --limit-experiments 4 — remove both flags for full runs:", BODY))
    story.append(Paragraph(
        "python -u -m AppliedControler.run_steering \\<br/>"
        "&nbsp;&nbsp;--experiment-plan AppliedControler/experiment_plan_ood_global3_qwen05b.json \\<br/>"
        "&nbsp;&nbsp;--device cuda:0 --output results_reports/ood_runs/steering_results_ood_global3_qwen05b_full", MONO))
    story.append(Paragraph(
        "For Qwen-2.5-1.5B (#5, #6): create a new plan JSON by copying the 0.5B plan and swapping "
        "model_name to Qwen/Qwen2.5-1.5B, label to 'Qwen-2.5-1.5B'. Then export the paper-style table:", BODY))
    story.append(Paragraph(
        "python -u -m AppliedControler.export_ood_paper_style \\<br/>"
        "&nbsp;&nbsp;--input results_reports/ood_runs/steering_results_..._full.csv \\<br/>"
        "&nbsp;&nbsp;--output results_reports/paper_style_table_ood_global3.csv --provenance full", MONO))
    story.append(Paragraph(
        "Toxicity metric: continuation toxicity % scored with s-nlp/roberta_toxicity_classifier. "
        "The 3 global OOD subsets are fixed across all models (chosen from residual analysis): "
        "jigsaw_long, toxicchat_long, mmlu_ood_other_concepts.", BODY))

    story.append(Paragraph("3. Gain / lambda sweep (experiment #7)", H2))
    story.append(Paragraph(
        "Both A-LQR and H-infinity over-steer on truthfulness (True ≈ 95%, Info ≈ 1%, so T·I "
        "collapses) — confirmed on 0.5B and 1.5B; S-PID does not collapse. Sweep the controller "
        "gain / lambda on the cheap 0.5B model, pick the value maximizing T·I without MMLU "
        "degradation, then re-run truthfulness for affected models. This was explicitly deferred "
        "by the project owner — confirm before spending GPU time.", BODY))

    story.append(Paragraph("4. After any full run: update figures", H2))
    story.append(Paragraph(
        "a) Merge new rows into the relevant CSV (keep run_status provenance column).<br/>"
        "b) In AppliedControler/figure_config.py add the model to TRUTH_FULLY_RUN / TOX_FULLY_RUN / "
        "OOD_FULLY_RUN so its label turns black. Layouts are append-only: add models/rows only via "
        "figure_config, never restructure figure scripts.<br/>"
        "c) Re-render all six figures:", BODY))
    story.append(Paragraph(
        "for s in make_final_summary_figure make_final_summary_model_dissected \\<br/>"
        "&nbsp;&nbsp;make_final_summary_model_dissected_toxicity make_final_summary_figure_ood \\<br/>"
        "&nbsp;&nbsp;make_final_summary_model_dissected_ood; do python -u -m AppliedControler.$s; done", MONO))
    story.append(PageBreak())

    # ---------- Page 3: figure map + key findings ----------
    story.append(Paragraph("Figure map (all in AppliedControler/results_reports/)", H1))
    frows = [[P("<b>Figure stem</b>"), P("<b>Content</b>"), P("<b>Data source</b>")]]
    for stem, content, src in FIGURES:
        frows.append([P(stem), P(content), P(src)])
    ftbl = Table(frows, colWidths=[78 * mm, 110 * mm, 74 * mm], repeatRows=1)
    ftbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.15, 0.2, 0.3)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.96, 0.96, 0.98)]),
    ]))
    story.append(ftbl)
    story.append(Paragraph(
        "[_smoke] suffix is applied automatically when the input table filename contains 'smoke'; "
        "it disappears once a full table is used.", BODY))

    story.append(Paragraph("Key findings so far (context for interpreting new runs)", H2))
    for txt in [
        "<b>S-PID</b> is the only controller that improves truthfulness without collapsing "
        "informativeness (0.5B smoke: T·I 46.2 vs Original 43.3; same pattern at 1.5B).",
        "<b>A-LQR / H-infinity over-steer</b> on truthfulness at current gains (Info ≈ 1%) — "
        "motivates experiment #7.",
        "<b>In-distribution toxicity:</b> DistilGPT-2 full run: large reductions all methods. "
        "Qwen-0.5B smoke: baseline 14.9% → A-LQR 6.43 (−56.7%), H-inf 9.88 (−33.5%), "
        "S-PID 14.1 (−5.3%) — A-LQR/S-PID ranking flips vs DistilGPT-2.",
        "<b>OOD transfer:</b> DistilGPT-2 controllers are flat on jigsaw_long (baseline 67.9%, "
        "±0.4%) — rtp-calibrated gains do not transfer; genuine finding, not a bug. Qwen-0.5B "
        "shows modest real transfer (72.7 → 67.3 A-LQR). toxicchat_long / mmlu_ood are near the "
        "classifier floor (~0.01%) for small models and are weakly discriminating.",
        "7B/14B rows are intentionally N/A (owner decision: no more large-model runs).",
    ]:
        story.append(Paragraph("• " + txt, BODY))

    doc.build(story)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
