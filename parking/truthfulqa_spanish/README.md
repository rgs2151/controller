# truthfulqa_spanish.py

## Method

- Read the immutable 817-question TruthfulQA ID cache and recover the canonical prompt-ID order.
- Translate every English question once with pinned `gpt-4.1-mini-2025-04-14` structured API responses, batching ten matched IDs per request.
- Audit every matched English–Spanish pair separately. Send disputed failures to pinned `gpt-4.1-2025-04-14` for final translation-only adjudication, then repair only confirmed errors.
- Freeze the dataset only after all 817 translations pass quality control.
- Preserve the five ID repetition permutations exactly. Store Spanish as model input and the original English question as judge input.

## Variables

- Data/input: `../bench_evaluations/cache/data/truthfulness.json`, containing all 817 pinned TruthfulQA questions.
- Sessions/groups: ten questions per API request and 12 concurrent requests.
- Labels/targets: translation-audit verdict `pass` or `fail`.
- Signals/features/measures: English source, Spanish translation, audit verdict, failure reason, API identity, token usage, and estimated cost.
- Parameters/thresholds: pinned GPT-4.1 mini translator/auditor, pinned GPT-4.1 adjudicator, temperature 0, batch size 10, and 817 required passes.
- Outputs: versioned frozen dataset at `data/truthfulqa_spanish.json`; ignored API caches under `cache/data/truthfulness_spanish/`; tracked audit report at `plots/spanish_translation_quality.md`.

## Statistics

- Tests/models: structured translation-fidelity audit plus exact structural checks; no inferential hypothesis test.
- Decision rule: require 817 unique nonempty translations, exact alignment with all five English permutations, and zero final audit failures.
- What it means: the final pass count records translations accepted under the frozen fidelity rubric.

## Interpretation

- The finalized dataset contains one fixed Spanish translation for every English TruthfulQA question.
- Translation changes only model-facing input. Evaluation later uses the unchanged English question, judge checkpoints, rubrics, and parsers.

## Notes

- Dataset: `data/truthfulqa_spanish.json`.
- Quality report: `plots/spanish_translation_quality.md`.
- The dataset is already frozen. Do not rerun construction unless the source or translation protocol is intentionally replaced.
- If reconstruction is intentionally required, run `python3 parking/truthfulqa_spanish/truthfulqa_spanish.py --stage prepare`.
- This unit constructs and audits data only. It does not generate model answers or run benchmark judges.

## References

- TruthfulQA revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490`.
- OpenAI GPT-4.1 mini documentation: `https://developers.openai.com/api/docs/models/gpt-4.1-mini`.
