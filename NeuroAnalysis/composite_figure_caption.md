# Composite figure caption (Overleaf / LaTeX)

Paste inside your `figure` environment:

```latex
\caption{\textbf{Human intracranial dynamics are locally linear, but their residuals are structured --- growing exactly where control is hardest (P12, MSIT).}
\textbf{(A)} sEEG depth electrodes across prefrontal, cingulate, and temporal cortex; example broadband LFP traces.
\textbf{(B)} Cognitive-control task under two domains: models are fit on low-conflict trials (ID) and evaluated frozen on high-conflict trials (OOD).
\textbf{(C)} Locally linear model. The neural state $x_t$ ($\theta$-power tokens, 20\,ms stride) evolves as $\hat{x}_{t+1} \approx A x_t + b + w_t$; the residual $w_t$ is fit by a small network $g(x_t)$ on the frozen linear map.
\textbf{(D)} Residuals matter under distribution shift. Error reduction from $g(x)$ (median residual RMS, z; positive = smaller errors). Under the stimulation-context shift, $g(x)$ shrinks OOD errors by $0.059$\,z at $\Delta = 200$\,ms while leaving ID unchanged; under the conflict shift it never helps.
\textbf{(E)} Residuals grow with the sampled population ($0.74 \rightarrow 0.99$\,z from $k = 2$ to $15$ channels, $\Delta = 200$\,ms) but with a saturating trend; $R^2$ is flat in $k$, so linearization quality is size-independent.
\textbf{(F)} Longer context suppresses short-horizon residuals by over an order of magnitude ($0.16 \rightarrow 0.007$\,z at $\Delta = 20$\,ms, $k = 1 \rightarrow 8$ tokens), but all curves saturate at the signal SD (dotted) by $\Delta \approx 300$\,ms: context buys precision, not horizon.
Frozen models and normalization for all OOD evaluations; LVF lead unless noted.}
```

## Notes / honest caveats (not for the caption)

- Panel D's inner titles still read "D1/D2" from the standalone Panel D figure; if this composite keeps the letter **E** for that panel, relabel the sub-titles to E1/E2 in the panel script before final export.
- The conflict-shift result in D is a genuine negative (no median-error reduction ID or OOD); the caption states it plainly rather than hiding it.
- n = 1 participant (P12); all claims are within-subject.

---

# Results subsection (Overleaf / LaTeX)

```latex
\subsection{Locally linear dynamics of human intracranial recordings and the structure of their residuals}

To ground our control-theoretic framework in biological neural dynamics, we analyzed stereo-EEG (sEEG) recordings from depth electrodes spanning prefrontal, cingulate, and temporal cortex of a human participant performing the Multi-Source Interference Task (MSIT), a canonical cognitive-control paradigm \citep{basu2023}. Treating the simultaneously recorded channels of a lead as a state vector $x_t$ (per-channel $\theta$-power tokens at a 20\,ms stride), we asked how well the evolution of this deeply nested cortical network is captured by a locally linear estimation dynamic, $\hat{x}_{t+\Delta} = A x_t + b$, with a learned nonlinear residual $w_t \approx g(x_t)$ fit on top of the frozen linear map. Crucially, the task was performed under two distinct domains --- low- versus high-conflict trials, and separate stimulation-context sessions --- allowing us to fit all models in one domain (in-distribution, ID) and evaluate them frozen in the other (out-of-distribution, OOD).

Linearization is possible: the linear map predicts held-out neural states far above a persistence baseline at short horizons ($R^2 \approx 0.7$ at $\Delta = 20$\,ms on $\theta$-power; the same holds qualitatively on raw broadband voltage, where persistence fails by $\Delta = 40$\,ms while the linear map still tracks the waveform). However, a residual always exists, and its structure --- rather than the linear fit itself --- carries the lessons we transfer to the LLM analyses (Appendix). Three properties of the residual stand out. \textbf{(1) Residuals grow under distribution shift.} Frozen linear models degrade when the domain changes (e.g., a $\Delta R^2 \approx -0.07$ drop under the stimulation-context shift at $\Delta = 200$\,ms), and it is precisely in this OOD regime that the learned residual $g(x)$ recovers error: it shrinks the typical OOD prediction error by $0.059$\,z at $\Delta = 200$\,ms while leaving ID errors essentially unchanged (Fig.~D). \textbf{(2) Residuals depend on the context window.} Conditioning the linear map on longer state histories suppresses short-horizon residuals by more than an order of magnitude ($0.16 \rightarrow 0.007$\,z at $\Delta = 20$\,ms from one to eight tokens), yet beyond the predictability horizon ($\Delta \gtrsim 300$--$400$\,ms) all curves converge to the signal SD and longer context slightly \emph{increases} the residual (Fig.~F): context buys precision, not an extended horizon. \textbf{(3) Residuals grow with the size of the sampled neural population, but with a saturating trend.} As the state dimension increases from 2 to 15 channels, the absolute residual grows ($0.74 \rightarrow 0.99$\,z at $\Delta = 200$\,ms) --- larger channel sets carry more signal variance --- but the growth visibly decelerates toward the full lead while the variance-normalized accuracy ($R^2$) remains flat (Fig.~E). This saturating trend is consistent with a bounded residual budget in the fully sampled limit, which is encouraging for systems with full observability and directly accessible parameters at scale, such as large language models.

Together, these observations argue that open-loop linear prediction alone cannot govern such dynamics --- residuals are unavoidable, state-dependent, and inflated exactly where control is most needed (under domain shift and beyond short horizons) --- motivating closed-loop, state-dependent robust control. We repeat the same residual analysis on LLM hidden-state dynamics in the Appendix.
```

## Notes on the subsection (read before pasting)

- **Claim (2) as you stated it is inverted at short horizons.** Longer context *decreases* residuals below the predictability horizon (this is the dominant, order-of-magnitude effect); it only *increases* them slightly past ~400 ms. The text states both, leading with the true direction — please don't flip it back, the figure contradicts the inverted version.
- **Claim (3) caveat is included:** the raw-unit residual growth with population size is largely signal-variance scaling ($R^2$ is flat in $k$), so "saturating theoretical bound" is phrased as a *trend consistent with* a bound, not a demonstrated one.
- The conflict-shift axis shows **no** residual-gain benefit (honest negative); the OOD-rescue claim in (1) rests on the stimulation-context shift only.
- n = 1 participant; single lead (LVF) for panels D–F. Replace `\citep{basu2023}` with your bib key.

