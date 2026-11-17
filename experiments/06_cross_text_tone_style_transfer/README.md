# Experiment 06 — Cross-Text Tone Transfer via Activation Steering

**Status:** ✅ completed (negative result with a clean control) · **Runtime:** 93 s · **Model:** SmolLM-135M-Instruct (30 layers, d=576)

---

## 1. Idea

Transfer voice/rhythm without touching weights: compute a **difference-in-means**
style vector in the residual stream (human-written sources minus model rewrites
of *the same documents*, so topic is held constant), then inject
`α · ‖v‖ · unit(v)` after layer L during generation.

Key methodological fix found during the run: steering strength **must** be
expressed relative to the layer's residual norm (~150 at layer 6, ~740 at layer
24 in this model). The first sweep used absolute units (≤ 4) and measured
nothing — a 1 % perturbation is invisible.

## 2. Vector quality

| layer | separation (human vs model) | ‖v‖ | residual norm | ‖v‖ as % of residual |
|---|---|---|---|---|
| 6 | **1.60 σ** | 12.3 | 149 | 8.3 % |
| 12 | 1.14 σ | 244 | 345 | 71 % |
| 24 | 1.18 σ | 268 | 738 | 36 % |

Even at its best layer the direction separates the two corpora by only 1.6 σ —
weak for a "style axis". Data: `outputs/style_vector_projection.csv`.

## 3. Results (12 docs × 3 seeds)

| condition | style JSD | burst | copy₄ | entity | D2 | PPL | GLTR top-10 |
|---|---|---|---|---|---|---|---|
| no_steering | 0.315 | 0.393 | 0.505 | 0.264 | 0.926 | 12.9 | 0.992 |
| L6 α0.10 | 0.309 | 0.332 | 0.646 | 0.325 | 0.971 | 14.5 | 0.992 |
| L6 α0.25 | 0.315 | 0.319 | 0.657 | 0.327 | 0.975 | 23.2 | 0.991 |
| L6 α0.50 | 0.623 | 0.101 | 0.005 | 0.011 | 0.879 | ∞ | 0.900 |
| L12 α0.10 | 0.330 | 0.345 | 0.568 | 0.280 | 0.913 | 13.9 | 0.994 |
| L12 α0.25 | 0.451 | 0.341 | 0.105 | 0.068 | 0.734 | 20.4 | 0.979 |
| L12 α0.50 | 0.487 | 0.664 | 0.000 | 0.000 | 0.979 | 87.8 | 0.762 |
| L24 α0.10 | **0.269** | 0.334 | 0.650 | **0.349** | 0.966 | 13.3 | 0.997 |
| L24 α0.25 | 0.284 | 0.303 | 0.558 | 0.337 | 0.951 | 12.4 | 0.999 |
| L24 α0.50 | 0.563 | 0.129 | 0.001 | 0.009 | 0.217 | 5.5 | 0.994 |
| **L12 random α0.25 (control)** | 0.447 | 0.234 | **0.000** | **0.011** | 0.813 | 25.5 | 0.879 |

## 4. What the numbers say

1. **The steering vector is a damage knob, not a style knob.** Rand the control: a *random* vector of the same magnitude at layer 12/α0.25 destroys the output as thoroughly as the style vector (copy 0.000 vs 0.105, entity 0.011 vs 0.068, PPL 25.5 vs 20.4). Any 25 % residual-norm perturbation breaks a 135M model. The style direction has no privileged effect.
2. **Small strengths do move behaviour — in the wrong direction.** L6/L24 at α0.10–0.25 raise entity recall to 0.325–0.349 (the best non-greedy recall in the repo) but only by *increasing copying* (0.505 → 0.65) and PPL (12.9 → 13–23). The "human style" vector, extracted from the human sources, encodes *"be like the source"* — i.e. copy it.
3. **The style metric does not reward any condition.** Function-word JSD improves only at L24/α0.10 (0.315 → 0.269) and that condition also copied more; at every strength where style JSD clearly improves, PPL has already exploded (0.487 @ PPL 87.8). No point on the sweep is a win.
4. **Layer matters for *what* breaks, not *whether*.** Late layers (24) at low α preserve fluency and push toward the source; middle layers (12) at 0.25 collapse recall first; layer 6 at 0.5 kills the run outright (17 words, EOS).

**Verdict:** inference-time difference-in-means steering is not a viable humanization path for this model. The confound is now explicit (below), and the fix is training, not test-time vectors.

## 5. Gaps & how to close them

| Gap | Why it matters | Fix |
|---|---|---|
| **Style vector is confounded with source-copying** | the "human" corpus *is* the text being rewritten, so the vector's main axis is "reproduce the input" (visible as copy↑ at low α) | Extract from an **external** human corpus (public-domain prose, e.g. Gutenberg) that shares no wording with the benchmark. This is the single most important fix. |
| **Difference-in-means is weak (1.6 σ)** | weak axes need destructive magnitudes to show | Use **contrastive/pairwise** extraction: same content, two styles (mechanical vs human), take per-pair difference then average; or PCA on paired differences (better SNR). |
| **Global constant injection** | adding the same vector to every token destabilises syntax | **Per-token gating**: steer only at low-confidence positions (E12's high-entropy steps), or only at the first 3 tokens of each sentence (cadence control). |
| **No AdaIN-style variant** | the original idea proposed AdaIN on residual streams | Match mean **and** variance: `h' = ((h - μ_h)/σ_h)·σ_ref + μ_ref` on the style subspace. Direction-only steering ignores σ, which is what burstiness lives in. |
| **Only 12 documents, one prompt** | the model may just be prompt-brittle | Re-run at α0.10 with 3 prompt variants; check that the recall gain is not a prompt artefact. |
| **No attention/MLP localisation** | we don't know which sub-layer carries style | Ablate: steer only `self_attn` output vs only `mlp` output at L24. Two extra conditions, ~40 s. |

## 6. Cross-experiment links

* **E12** explains the cliff: perturbations that push the model off its own trajectory flatten the distribution (PPL↑) long before they change the style profile.
* **E07 (recursion)** is the *behavioural* counterpart: repeated rewriting drifts out of distribution the same way steering does, so E07's drift curve can be read as "steering by iteration".
* **E09 (DPO)** is the constructive version of this idea: instead of adding a vector, move the weights along the human-style direction with a preference objective and measure the same JSD/burstiness metrics.

## 7. Files

`outputs/style_vector_projection.csv` (axis quality), `outputs/steering_raw.csv` (396 rows), `outputs/steering_summary.{csv,md}`, `outputs/samples.json`, `outputs/steering.png`, `outputs/env.json`.
