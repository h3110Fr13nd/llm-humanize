# Experiment 12 — The Head-Conformity Floor: can decoding alone humanize a 135M model?

**Status:** ✅ completed · **Runtime:** 57 s · **Model:** SmolLM-135M-Instruct

---

## 1. Question

E10 showed ten truncation families all land between 0.98 and 1.00 GLTR top-10
conformity while human text sits at 0.797. Two explanations are possible:

* **(a) sampler-side**: the truncation rules are all too conservative, and a
  better rule reaches 0.80;
* **(b) model-side**: the model puts ~92 % of its mass on ten tokens, so *no*
  sampler that preserves the model's distribution can go below that floor.

This experiment separates (a) from (b) by removing truncation entirely and
sweeping temperature, while computing the theoretical floor analytically.

## 2. Method

1. **Analytic floor.** For each of the 12 documents, generate a 60-token greedy
   draft (a realistic, coherent trajectory), then teacher-force prompt+draft and
   record, at every position, how much probability mass the model puts on its own
   **top-10 tokens** at temperature T. Averaged → `predicted_top10_mass(T)`.
   For an unbiased sampler at temperature T this *is* the expected GLTR fraction.
2. **Measured.** Sample with temperature T and **no truncation at all**
   (full 49 152-token vocabulary), 3 seeds × 12 docs × 100 tokens; measure the
   realised GLTR rank profile, perplexity, copy, entity recall.
3. Human reference: **0.797** top-10 (from E10).

## 3. Results

| T | predicted top-10 mass | measured GLTR top-10 | measured − predicted | copy₄ | entity | D2 | PPL |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.997 | 0.999 | +0.002 | 0.64 | 0.34 | 0.96 | 14.2 |
| 0.7 | 0.983 | 0.986 | +0.003 | 0.47 | 0.26 | 0.95 | 13.3 |
| 1.0 | 0.915 | **0.872** | −0.043 | 0.20 | 0.17 | 0.97 | **29.1** |
| 1.3 | 0.764 | 0.282 | −0.482 | 0.01 | 0.02 | 0.999 | **5 736** |
| 1.6 | 0.560 | 0.068 | −0.492 | 0.001 | 0.005 | 1.000 | 35 235 |
| 2.0 | 0.307 | 0.018 | −0.289 | 0.000 | 0.004 | 1.000 | 71 100 |
| 3.0 | 0.051 | 0.004 | −0.047 | 0.000 | 0.005 | 1.000 | 128 039 |

Human: top-10 = 0.797, PPL ≈ 12–15 for the same corpus. Plot: `outputs/head_floor.png`.

## 4. What the numbers say

1. **The floor is real at safe temperatures.** At T ≤ 1.0, measured ≈ predicted
   (within 0.004–0.043). The model genuinely allocates 92–99 % of its mass to ten
   tokens; a sampler cannot manufacture human rank statistics out of that.
2. **You *can* reach and undershoot the human profile — but only by leaving English.**
   The human 0.797 point is crossed between T=1.0 and T=1.3. At T=1.0 the profile
   is 0.872 with PPL already 29 (2.3× the coherent range). By T=1.3 the profile is
   0.282 (far *below* human) while PPL is 5 736 and entity recall 0.019. Distinct-2
   saturates at 0.999–1.000 — that is token soup, and it is exactly E01's failure
   mode rediscovered from the other direction.
3. **The measured curve falls *below* the prediction above T=1.0** (gap −0.48 at
   T=1.3). Mechanism: the prediction was taken along a *greedy* (coherent)
   trajectory, but hot sampling pushes the context out-of-distribution, which
   flattens the model's own next-token distribution. Head conformity is therefore
   **trajectory-dependent**: incoherence feeds head escape, not the reverse.
4. **Consequence for the whole project.** The premise "choose a better decoding
   rule and get human text" is *measurably false* for SmolLM-135M. Decoding can
   move along the copy↔hallucination frontier (E10/E11) but cannot create a
   human-shaped distribution. The human 0.80 profile with coherent text must come
   from **weights** (E09 DPO/RL), from **conditioning** (E06 steering, E04
   global refinement), or from a stronger base model (base-model scaling gap).

## 5. Gaps & how to close them

| Gap | Evidence | Fix / next action |
|---|---|---|
| **Floor measured on one model size** | 135M only | Repeat on SmolLM2-360M and Qwen2.5-0.5B (both fit 4 GB). Prediction: the floor rises with model quality — this is the single most informative scaling datapoint available locally. |
| **Prediction uses a greedy trajectory** | gap −0.48 at T=1.3 | Compute the prediction along *each condition's own* trajectory (teacher-force the generated text). Then predicted should match measured at all T and the "trajectory-dependent" claim becomes a clean identity. ~10 min of code. |
| **No entropy accounting** | top-1 prob / entropy are captured in the trace but unused | Correlate per-step entropy with chosen rank: is head escape concentrated in high-entropy steps? This tells us *where* a router could intervene without wrecking the text. |
| **Temperature is the crudest head-escape** | T=1.3 = soup | Try **cosine/adaptive temperature**: raise T only when the model is *confident* (top1 > 0.6), which is where copying is decided, and keep T=0.7 elsewhere. Cheap next test. |
| **No calibrated comparison to human entropy** | we only compare rank profile | Compute per-token entropy of human text under the model and compare to generated text. If human text sits at ~3 nats and generation at ~1.2 nats, that is the real target, not temperature. |
| **PPL uses the same model family** | self-perplexity biases toward model-like text | Add a small, independent scorer (`gpt2` 124M or `Qwen2.5-0.5B`) for cross-model PPL. Two models fit in 4 GB fp16 if loaded sequentially. |

## 6. Cross-experiment links

* **E10**: supplied the 10 decoding points that all clustered on the floor; E12 explains *why* they cluster.
* **E05**: attention surgery cannot change the head either — consistent with E12's model-side explanation.
* **E11**: entity pinning raises recall but stays at 0.99 head conformity — pinning *protects* head tokens by construction.
* **E09 (DPO/RL)**: is now the only remaining local intervention that can move the distribution itself. E12 gives it its target metric (GLTR top-10 → 0.80 at PPL < 20) and its baseline (0.99 @ 12.9).
* **E04 (global refinement)**: conditioning on a full draft can lower per-step confidence legitimately (the model is checking work rather than guessing), so it may shift the floor without soup — worth measuring with the same GLTR protocol.
