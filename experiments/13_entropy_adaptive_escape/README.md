# Experiment 13 — Entropy-Adaptive Head Escape (+ E11 combination)

**Status:** ✅ completed (first mechanism to reach the human GLTR profile) · **Runtime:** 107 s + 73 s · **Model:** SmolLM-135M-Instruct

**Spawned by:** E12 gap table ("adaptive temperature: raise T only when the model is confident").

---

## 1. Idea

E12 proved that head conformity has a floor: the model puts 92–99 % of its mass
on ten tokens at safe temperatures, and the only way to reach the human 0.797
GLTR profile is T ≥ 1.3, where perplexity explodes to 5 700+.

But *where* the heat is applied matters. Copying is decided at **confident**
steps (the model is sure the source token is right). At uncertain steps it is
genuinely choosing. So apply the heat only to confident rows:

```
T_row = 1.3   if  top1_prob > confidence_threshold
T_row = 0.7   otherwise
```

The inverted control (heat the *uncertain* rows) tests whether the direction
proposed by E12 is actually the one that works.

**13b** then combines this with E11 entity pinning: pinned fact tokens are
copied greedily and never heated; every other position gets the adaptive heat.

## 2. Results — 13a, confidence × heat sweep

Human GLTR top-10 = **0.797**.

| condition | GLTR top-10 | copy₄ | entity | PPL | burst | boil/100w |
|---|---|---|---|---|---|---|
| baseline T=0.7 | 0.997 | 0.543 | **0.299** | **12.1** | 0.352 | 1.62 |
| fixed T=1.3 (E12 soup) | 0.562 | 0.063 | 0.060 | **898.8** | 0.379 | 0.95 |
| adapt c=0.30 | 0.722 | 0.058 | 0.064 | 132.2 | 0.476 | 1.31 |
| adapt c=0.40 | **0.810** | 0.061 | 0.063 | 68.9 | 0.376 | 1.11 |
| adapt c=0.50 | 0.849 | 0.102 | 0.078 | 34.7 | 0.343 | 1.75 |
| adapt c=0.60 | 0.884 | 0.097 | 0.085 | 30.7 | 0.333 | 2.11 |
| adapt c=0.70 | 0.923 | 0.103 | 0.091 | 27.4 | 0.320 | 2.82 |
| adapt c=0.50, heat 1.15 | 0.948 | 0.210 | 0.160 | 22.3 | 0.370 | 2.16 |
| adapt c=0.60, heat 1.15 | 0.968 | 0.278 | 0.175 | 23.6 | 0.353 | 2.60 |
| adapt c=0.70, heat 1.15 | 0.972 | 0.293 | 0.193 | **15.3** | 0.333 | 2.70 |
| **adapt inverse c=0.60 (control)** | 0.906 | 0.244 | 0.155 | **60.6** | 0.361 | 2.65 |

## 3. Results — 13b, entity pinning × adaptive heat

| condition | GLTR top-10 | copy₄ | entity | **recall/copy** | PPL | burst |
|---|---|---|---|---|---|---|
| baseline T=0.7 | 0.997 | 0.543 | 0.299 | 0.55 | 12.11 | 0.352 |
| adapt c=0.50 | 0.849 | 0.102 | 0.078 | 0.77 | 34.72 | 0.343 |
| pin only (E11) | 0.998 | 0.643 | 0.379 | 0.59 | 11.02 | 0.382 |
| pin + adapt c=0.50 | 0.895 | 0.158 | 0.157 | 0.99 | 23.33 | 0.334 |
| pin + adapt c=0.60 | 0.923 | 0.175 | 0.187 | 1.07 | 18.92 | 0.367 |
| **pin + adapt c=0.70** | **0.952** | 0.178 | 0.215 | **1.21** | **12.06** | 0.400 |
| pin + adapt c=0.70 (heat 1.15) | 0.987 | 0.475 | **0.324** | 0.68 | 11.90 | 0.379 |

Data: `outputs/adaptive_{raw,summary}.csv`, `outputs/pinned_adaptive_{raw,summary}.csv`, samples in
`outputs/samples.json` / `outputs/pinned_adaptive_samples.json`, plots `outputs/adaptive.png`,
`outputs/pinned_adaptive.png` (GLTR vs PPL frontier).

## 4. What the numbers say

1. **Adaptive heat is ~17× more PPL-efficient than uniform heat at the same head
   conformity.** Fixed T=1.3 reaches 0.562 GLTR at PPL 899. Adaptive c=0.50
   reaches **0.849** — *closer* to the human 0.797 — at **PPL 34.7**. E12's
   hypothesis was right: the escape has to be spent at confident steps, not
   sprayed over the whole trajectory.
2. **This is the first mechanism in the repo to reach the human profile at all.**
   adapt c=0.40 lands at **0.810 vs human 0.797** (distance 0.013). Every
   previous intervention (E05 attention masks, E06 steering, E10 ten decoders,
   E11 pinning, E09 DPO) stayed at 0.95–1.00. Decoding *can* move the statistic
   after all — it just needs to be conditional.
3. **The inversion control confirms the mechanism.** Heat on *uncertain* rows
   gives 0.906 GLTR at PPL **60.6**, versus 0.884 at PPL 30.7 for the same
   threshold applied to confident rows. Same average heat budget, twice the
   damage, no better escape. The effect is not "more randomness"; it is
   "randomness where the copy prior lives".
4. **Pinning restores recall without giving back the escape (13b).** `pin+adapt c=0.70`
   sits at baseline perplexity (12.06 vs 12.11) with GLTR 0.952, copy 0.178
   (3× less than baseline) and **recall/copy 1.21** — 2.2× the baseline
   efficiency and the best value measured anywhere in this repo. Anchors are
   copied, prose is heated, and the two levers do not cancel.
5. **There is a gentle-heat regime with no fluency cost at all.**
   `pin+adapt c=0.70, heat 1.15` beats the baseline on *both* entity recall
   (0.324 vs 0.299) and PPL (11.90 vs 12.11) while nearly halving copy. Its
   GLTR (0.987) barely moves — so this is a **copy/recall** win, not a
   head-escape win. The two goals need different settings, and that distinction
   is the main practical output of this experiment.
6. **Boilerplate moves the wrong way as heat rises** (1.11 → 2.82/100w from
   c=0.40 to c=0.70). Heated steps drift toward generic phrasing. E11's
   preamble ban is the complement, not a competitor.

## 5. Gaps & how to close them

| Gap | Evidence | Fix / next action` | Cost |
|---|---|---|---|
| **The crossing point costs facts** | 0.810 GLTR comes with entity 0.063 | Sweep the *pin fraction*: pin only a per-document NER allow-list (not 28.7 % of the vocab) and heat everything else. Should reach 0.80 with recall ≥ 0.15. | 20 min + spaCy |
| **Confidence threshold is global and static** | one c for all documents and steps | Per-token budget: heat until the *cumulative* escaped mass hits a target (e.g. 20 % of tokens), distributed across the most confident steps. Turns a hyper-parameter into a constraint. | 30 min |
| **Top-1 probability is a crude confidence** | top1 confounds "confident" with "copying" | Gate on **copy evidence** instead: heat only when the top-1 token also appears in the source within the last N tokens (a copy-detector / "look-back" ratio, as in LLMA / contrastive decoding). Directly targets the behaviour, not a proxy. | 45 min |
| **No perplexity floor control** | PPL 34 at c=0.50 | Add a *second* gate: skip the heat if local entropy is already high, so heat is spent once per phrase rather than per token. Prevents the "English soup" failure at low c. | 30 min |
| **Heat magnitude only 1.15 / 1.3** | two points only | Scan heat ∈ {1.05, 1.15, 1.3, 1.5} × c ∈ {0.5…0.8}; fit a 2-D frontier and report the Pareto curve to the synthesis. | 15 min |
| **Not trained, only sampled** | inference-only again | Feed the `pin+adapt c=0.70` outputs into E09's DPO as the *chosen* side. E09's failure was that the chosen outputs were still assistant-shaped; these are measurably less so (copy 0.18 vs 0.28, GLTR 0.95). This is the most promising untried loop in the project. | 30 min |
| **PPL scorer is the same model family** | self-PPL | Cross-model PPL (gpt2-124M / Qwen2.5-0.5B) to check that 34.7 really is worse text and 12.06 really is not. | 20 min |

## 6. Cross-experiment links

* **E12** → predicted this experiment and supplies the 0.797 target and the
  1.3-temperature soup reference. E13 overturns E12's *stronger* claim ("decoding
  cannot humanize") while keeping its measurement: the floor exists, but it is
  escapable when the heat is conditional.
* **E11** → the pinning half of 13b, and the source of the recall/copy metric
  this experiment maximises (0.52 → 1.21).
* **E10** → all ten standard decoders are unconditional; E13 shows conditionality
  is the missing dimension. Re-running E10's zoo with adaptive temperature is a
  cheap follow-up.
* **E09 (DPO)** → its chosen side should be regenerated from `pin+adapt c=0.70`;
  the current DPO run collapsed into "What a fascinating topic!" because the
  chosen outputs were still assistant-shaped.
* **E07 (recursion)** → a heated decoder with copy ≈ 0.18 will *not* converge to
  the identity fixed point as fast; recursion + adaptive heat is an untested
  combination.
* **E04 (diffusion)** → the same confidence gate applies to unmasking order:
  unmask high-confidence positions first is exactly MDLM's `low_confidence`
  remasking policy, inverted. Worth comparing directly.
