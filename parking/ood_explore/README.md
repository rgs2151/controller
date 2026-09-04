# ood_explore

Status: research-scoping scaffold only. No dataset, analysis, statistic, or plot
has been selected or implemented.

## Method

- Define OOD relative to the prompts used to fit the semantic target and nominal
  dynamics and the separate prompts used to calibrate residual disturbances.
- Treat unseen prompts from the same sampling process as held-out ID, not OOD.
  The number of evaluation prompts does not determine distribution membership.
- Consider controlled shift families that preserve the steering objective while
  changing one input axis at a time, alongside deliberately compounded shifts.
- Compare candidate shifts by whether they change residual magnitude, residual
  direction, closed-loop amplification, and final steering failure.
- No candidate has been chosen for execution; the exact experiment will be
  specified later by the user.

## Variables

- Data/input: not selected. The current reference comparison is held-out
  RealToxicityPrompts as ID versus held-out Jigsaw prompts as source-shift OOD.
- Sessions/groups: a future ID group plus one or more explicitly defined OOD
  groups; equal group size alone will not define or validate OOD status.
- Labels/targets: preferably keep the controlled semantic objective fixed so
  observed failure measures robustness to input shift rather than an undefined
  new task.
- Signals/features/measures: candidate measures are residual norm, residual
  alignment with closed-loop sensitive directions, directional residual effect,
  residual amplification, nominal-dynamics drift, disturbance-subspace coverage,
  control effort, and final semantic tracking failure.
- Parameters/thresholds: not selected.
- Outputs: none yet.

Candidate OOD axes:

| Shift family | Example | What it isolates | Research value |
| --- | --- | --- | --- |
| Dataset/source | Fit and calibrate on RTP; test on Jigsaw | Collection, genre, and annotation-source shift | Direct extension of the current smoke test, but multiple factors remain confounded |
| Paired language | English prompt and meaning-preserving translations into Spanish, Arabic, Hindi, or another language | Language while approximately holding meaning fixed | Strong and interpretable test of whether equivalent semantics induce different residual dynamics |
| Code-switching | English mixed with another language within one prompt | Abrupt representational and tokenization shift | Potentially stronger than translation while retaining recognizable semantic content |
| Domain or genre | Social-media toxicity versus dialogue, forum, news-comment, or formal prose toxicity | Writing domain while preserving the non-toxicity objective | Tests whether the controller survives a new realization of the same concept |
| Topic | Fit on general insults; test on political, identity-related, medical, technical, or fictional topics | Content shift within a fixed steering objective | Can reveal topic-dependent dynamics without changing what success means |
| Toxicity subtype | Profanity versus identity attack, threat, sexual harassment, or implicit abuse | Concept subtype and severity | Tests whether a broad toxicity direction covers structurally different manifestations |
| Pragmatics | Direct abuse versus quotation, negation, counterspeech, sarcasm, or discussion of abusive language | Meaning changes despite lexical overlap | Especially useful for separating surface similarity from representation-dynamics shift |
| Surface corruption | Typos, slang, leetspeak, transliteration, homoglyphs, or spacing changes | Lexical and tokenization shift | Likely to stress local linearization without changing intended meaning |
| Prompt format | Statement versus question, instruction, role-play, few-shot, or multi-turn dialogue | Interaction-format shift | Tests whether controller behavior depends on the format used during fitting |
| Context length and position | Short prompts versus long contexts with the relevant content early or late | Length and positional shift | Tests depth- and position-dependent dynamics and accumulated model mismatch |
| Compositional | Toxic content combined with translation, summarization, coding, sentiment, or another instruction | Interaction between multiple concepts or tasks | A demanding robustness test that may expose residual directions absent from calibration |
| Intensity or prevalence | Mild-to-severe toxicity or a different class balance | Conditional or label shift | Useful as a graded severity ladder rather than a single binary OOD label |
| Adversarially selected | Search for prompts with high predicted residual amplification while excluding evaluation outcomes | Worst-case input shift | Most directly aligned with motivating robust control, but selection must remain independent of held-out outcomes |
| Model or decoding environment | Apply the same controller across model checkpoints, model sizes, or decoding regimes | Plant or environment shift rather than prompt OOD | Scientifically useful, but should be reported separately from input-distribution shift |
| Different controlled concept | Fit a toxicity controller and evaluate a sentiment, truthfulness, or unrelated target | Target/task shift | Useful as a boundary or negative-control test; not a fair primary OOD test unless the controller is defined for the new target |

## Statistics

- Tests/models: none selected; this unit currently contains research questions
  and candidate experimental factors only.
- Null hypothesis: not specified.
- Alternative hypothesis: not specified.
- Thresholds/decision rule: none.
- What the statistic means: not applicable until an experiment is chosen.
- Why this statistic is appropriate here: not applicable at the scoping stage.

For a later experiment, a matched transformation design would be preferable
where possible: compare each source prompt with its translated, corrupted,
reformatted, or pragmatically altered counterpart. This would help attribute a
residual change to the intended OOD axis rather than unrelated prompt content.

## Legends

- X axis: none yet.
- Y axis: none yet.
- Color/value: none yet.
- Grouping: candidate groups are ID and explicitly named OOD shift families.
- Ordering/sorting: a future severity ladder could order shifts from a matched
  source prompt through isolated and compounded transformations.
- Lines/markers/labels: none yet.
- Panels: none yet.

## Interpretation

- OOD is relational: it means shifted relative to the distributions used for
  fitting and calibration, not merely absent from a finite sample.
- The most informative primary tests preserve the steering target and alter the
  language, domain, pragmatics, surface form, format, or composition of inputs.
- A completely different controlled concept is a target/task shift. It can map
  the boundary of the method, but failure there would not by itself demonstrate
  that an H-infinity controller is needed for OOD robustness.
- The central mechanistic question is not only which shift makes residuals
  larger, but which shift rotates residuals toward directions that the
  closed-loop system amplifies into steering failure.
- A particularly strong future motivation would show a graded shift that leaves
  residual norm similar while increasing directional effect or amplification,
  followed by a robust controller reducing the resulting failure.

## Notes

- Recommended first candidates, without selecting the final experiment:
  meaning-preserving translation, code-switching, pragmatic reversal, and
  domain shift within the same controlled concept.
- Translation is the cleanest isolated language test; code-switching and
  compounded shifts may create a larger robustness gap.
- Pragmatic pairs are attractive because lexical content can remain similar
  while meaning changes, providing a hard test of representation dynamics.
- The current RTP-to-Jigsaw comparison remains useful as a source-shift
  baseline, but it does not identify which aspect of the dataset change caused
  the residual gap.
- No analysis or result claim has been made in this unit.

## References

- `parking/residual_checks/`
- `robust_steerability/control/README.md`
- No external literature has been reviewed for this scaffold yet.
