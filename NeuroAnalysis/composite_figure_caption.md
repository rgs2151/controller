# Composite figure caption (Overleaf / LaTeX)

Paste inside your `figure` environment:

```latex
\caption{\textbf{The human brain as a locally linear dynamical system with structured residuals: prediction under distribution shift (participant P12, MSIT).}
\textbf{(A)} Recording setup. Intracranial depth electrodes across prefrontal, cingulate, and temporal cortex (colored by lead); example broadband LFP traces from three simultaneously recorded channels.
\textbf{(B)} Task and distribution shift. In the Multi-Source Interference Task, each trial comprises fixation, a three-digit stimulus, and a button-press response. Low-conflict trials (target digit congruent with distractors) define the in-distribution (ID) regime; high-conflict trials define the out-of-distribution (OOD) regime. Models are fit exclusively on ID data and evaluated frozen on both regimes.
\textbf{(C)} Model. Top: schematic information-flow architecture of the recorded cortical network. Bottom: the neural state $x_t$ (per-channel $\theta$-power tokens, 20\,ms stride) is propagated by a locally linear map, $\hat{x}_{t+1} \approx A x_t + b + w_t$, where the residual $w_t$ is fit by a small nonlinear network $g(x_t)$ on top of the frozen linear dynamics.
\textbf{(D)} The residual helps exactly where distribution shift occurs. Reduction in median per-token residual RMS (baseline-SD units, z) contributed by $g(x)$, relative to the pure linear model, for ID (blue) and OOD (red) test data across prediction horizons $\Delta$. Under the conflict shift (left), $g(x)$ never reduces the typical error. Under the stimulation-context shift (right, NS1$\rightarrow$NS2), $g(x)$ leaves ID errors essentially unchanged but shrinks OOD errors increasingly with horizon, by $0.059$\,z at $\Delta = 200$\,ms---the learned residual encodes context-general structure that the linear map alone misses.
\textbf{(E)} Residual size scales with state size in raw error units. Median residual RMS of the linear model at $\Delta = 200$\,ms, across leads of different channel counts (left; color = anatomical region) and for random channel subsets of the LVF lead (right; mean $\pm$ SD over 10 draws). Absolute errors grow with state dimension $k$ because larger channel sets add signal variance; the corresponding variance-normalized accuracy ($R^2$) is flat in $k$, so linearization \emph{quality} is size-independent while the absolute error budget is not. Error bars shrink with $k$ as subset-to-subset variability averages out.
\textbf{(F)} Longer context delays, but does not remove, residual growth. Median ($\pm$IQR shading) residual RMS versus horizon for linear models conditioned on $k = 1$--$8$ past tokens (20--160\,ms of history). Longer context sharply suppresses short-horizon error (at $\Delta = 20$\,ms: $0.16$\,z for $k=1$ vs.\ $0.007$\,z for $k=8$), but all curves converge to the signal SD (dotted line) by $\Delta \approx 300$\,ms, beyond which added context slightly \emph{increases} error. Local linearity is a short-horizon property: context buys precision, not predictability horizon.
All quantitative panels: patient P12, LVF lead ($\theta$ band unless noted), whole-trial train/validation/test splits, frozen normalization for OOD evaluation.}
```

## Notes / honest caveats (not for the caption)

- Panel D's inner titles still read "D1/D2" from the standalone Panel D figure; if this composite keeps the letter **E** for that panel, relabel the sub-titles to E1/E2 in the panel script before final export.
- The conflict-shift result in D is a genuine negative (no median-error reduction ID or OOD); the caption states it plainly rather than hiding it.
- n = 1 participant (P12); all claims are within-subject.
