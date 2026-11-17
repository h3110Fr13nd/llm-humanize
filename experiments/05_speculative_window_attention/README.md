# Experiment 05 — Speculative / Asymmetric Window Attention

**Status:** ✅ completed (negative result, mechanism isolated) · **Runtime:** 67 s (216 generations) · **Model:** SmolLM-135M-Instruct

---

## 1. Hypothesis

> Restricting generated-token self-attention to a local sliding window, while
> keeping 100 % of the source prefix visible ("speculative window attention"),
> stops past generated tokens from dominating the heads and kills the
> repetitive syntactic echoes and plagiarism of left-to-right decoding.

Implementation: a per-step 4-D attention mask passed through
`humanize_lab.processors.window_mask_fn` into a manual KV-cached sampling loop.
No weights change — this is pure inference-time attention surgery.

## 2. Conditions

| condition | source attention | generated self-attention |
|---|---|---|
| `full_attention` | full | full (baseline) |
| `win4/8/16/32/64_src_full` | full | last W tokens |
| `win16_all` | last 16 tokens only | last 16 tokens |
| `win16_no_src` | **masked out** | last 16 tokens |

3 seeds × 12 documents, 100 new tokens, T=0.8 / top-p 0.9.

## 3. Results

| condition | words | copy₄ | rep₅ | entity | D2 | burst | PPL | GLTR top-10 |
|---|---|---|---|---|---|---|---|---|
| full_attention | 66.9 | 0.505 | 0.025 | 0.264 | 0.926 | 0.393 | 12.9 | 0.992 |
| win4_src_full | 48.5 | **0.356** | 0.085 | 0.155 | 0.841 | 0.342 | 15.2 | 0.990 |
| win8_src_full | 63.2 | 0.380 | **0.169** | 0.197 | 0.753 | 0.319 | 10.9 | 0.992 |
| win16_src_full | 68.5 | 0.389 | 0.065 | 0.242 | 0.871 | 0.398 | 13.2 | 0.990 |
| win32_src_full | 64.9 | 0.447 | 0.029 | 0.253 | 0.922 | 0.356 | 12.2 | 0.990 |
| win64_src_full | 66.7 | 0.493 | 0.024 | 0.262 | 0.925 | 0.372 | 12.6 | 0.992 |
| win16_all | 17.8 | 0.056 | 0.117 | 0.005 | 0.823 | 0.101 | ∞ (EOS collapse) | 0.967 |
| win16_no_src | 57.7 | 0.000 | **0.430** | 0.000 | 0.457 | 0.419 | 8.6 | 0.953 |

Data: `outputs/window_attention_raw.csv`, `outputs/window_attention_summary.{csv,md}`, `outputs/samples_all_conditions.json`, `outputs/window_attention.png`.

## 4. What the numbers say

1. **The window does exactly what it says on the tin and it is not enough.** Copy falls monotonically as the window shrinks (0.505 → 0.356 at W=4), but 5-gram self-repetition rises 3.4× (0.025 → 0.085) and entity recall falls 0.264 → 0.155. The plagiarism is traded for local loops, not for humanization.
2. **W=8 is the pathological midpoint**: lowest D2 (0.753) and highest rep₅ (0.169) of any "working" condition. A window that small is long enough to lose the argument and short enough to get stuck in a phrase.
3. **Source attention is the coherence anchor (the important control).** Remove it (`win16_no_src`) and copying goes to exactly 0.000, repetition explodes to 0.430, D2 halves to 0.457, and boilerplate doubles to 5.7/100w. The model's stability in a rewrite task is *borrowed from the source*; take it away and a 135M model has no plan.
4. **`win16_all` is an instruction-loss artefact**, not a clean uniform-window result: the system + user instruction sits at the head of the prompt and a 16-token window never sees it, so the model emits EOS after 18 words. The ablation needs an instruction-preserving variant (see gaps).
5. **GLTR top-10 never moves (0.990–0.992)** — attention surgery does not change the shape of the output distribution head. This confirms E10's conclusion: head conformity is a *logit-level* property, so it must be attacked at the logits or the residual stream, not at the attention mask. (E12 later shows even logits-level attacks hit a floor.)

**Verdict:** hypothesis falsified in its inference-only form. Window attention is a copy-rate knob, not a humanization mechanism.

## 5. Gaps & how to close them

| Gap | Evidence | Fix / next action |
|---|---|---|
| **Masking context ≠ changing the copy prior** | copy drops but repetition rises | The copy distribution comes from pretraining data and the prompt format. Attack it with: (a) an instruction that forbids reuse + `BanPhrases`, (b) entity-pinned head escape (E11), (c) fine-tuning. |
| **Attention sink / instruction loss** | `win16_all` collapses in 18 words | Keep the first K tokens fully visible (attention sinks, Xiao et al. 2024) and window only the body. One extra mask clause. |
| **Window is fixed, not speculative** | no adaptivity in W | *Speculative* variant: expand the window when top-1 entropy is high (uncertain step needs more context), shrink when confident. Implement as a `step`-dependent mask. |
| **Not trained with the mask** | Mistral/Gemma sliding-window models are *pretrained* with it | Fine-tune SmolLM-135M for ~1 h on the cached FineWeb-Edu subset with the asymmetric mask, then re-measure. This is the version the original idea actually proposed; the POC shows inference-only cannot work. |
| **Copy metric is length-confounded** | W=4 outputs are 27 % shorter | Report copy per 100 tokens; or fix length by forcing `min_new_tokens`. |
| **Source attention measured only by ablation** | no attention-mass statistics | Capture per-head attention mass on source vs generated tokens; test whether heads that "look at the source" are the ones that copy. Cheap with `output_attentions=True` on a 30-layer/9-head model. |

## 6. Cross-experiment links

* **E10** supplied the metric that killed the hypothesis (GLTR head conformity unchanged) and the copy/recall trade-off axis.
* **E07** (recursion) uses the same `win16_no_src` as a "context-free degeneration" reference: 0.43 rep₅ is numerically the same regime recursion collapses into.
* **E11** (spawned) takes the only useful part of this result — source attention must stay full — and moves the intervention to the logits.
