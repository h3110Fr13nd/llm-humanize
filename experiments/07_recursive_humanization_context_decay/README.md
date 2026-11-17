# Experiment 07 — Recursive Humanization & Context Decay

**Status:** ✅ completed · **Runtime:** 137 s (648 generations, 6 rounds × 3 decoders × 3 seeds) · **Model:** SmolLM-135M-Instruct

---

## 1. Question

If one pass is not enough to humanize, what happens when the output is fed back
in? Track facts, churn, plagiarism, fluency and "slop" over
`T0 → T1 → … → T6` for three decoders.

Metrics per round (mean over 12 docs × 3 seeds): entity recall vs the **original**
source, `churn` (1 − difflib similarity to the previous round), 4-gram copy vs
previous output, 4-gram copy vs original, similarity to original, Distinct-2,
burstiness, boilerplate markers/100w, PPL under the base model.

## 2. Results

| decoder | round | entity | churn | copy→prev | copy→orig | sim→orig | D2 | boil | PPL |
|---|---|---|---|---|---|---|---|---|---|
| greedy | 1 | **0.424** | 0.442 | 0.796 | 0.796 | 0.558 | 0.989 | 1.8 | 12.2 |
| greedy | 2 | 0.413 | 0.036 | 0.967 | 0.771 | 0.542 | 0.981 | 2.5 | 12.5 |
| greedy | 3 | 0.413 | 0.017 | 0.992 | 0.771 | 0.541 | 0.970 | 2.9 | 12.3 |
| greedy | 4 | 0.390 | 0.051 | **1.000** | 0.729 | 0.512 | 0.920 | 4.2 | 11.9 |
| greedy | 5 | 0.390 | **0.000** | 1.000 | 0.729 | 0.512 | 0.920 | 4.2 | 11.9 |
| greedy | 6 | 0.390 | **0.000** | 1.000 | 0.729 | 0.512 | 0.920 | 4.2 | 11.9 |
| top-p | 1 | 0.264 | 0.672 | 0.505 | 0.505 | 0.328 | 0.926 | 2.2 | 12.9 |
| top-p | 2 | 0.228 | 0.334 | 0.672 | 0.403 | 0.264 | 0.924 | 1.8 | 11.4 |
| top-p | 3 | 0.223 | 0.166 | 0.865 | 0.392 | 0.258 | 0.930 | 2.2 | 11.7 |
| top-p | 4 | 0.225 | 0.065 | 0.923 | 0.388 | 0.257 | 0.927 | 2.3 | 11.3 |
| top-p | 5 | 0.229 | 0.077 | 0.937 | 0.387 | 0.256 | 0.903 | 2.7 | 11.2 |
| top-p | 6 | 0.229 | **0.033** | 0.976 | 0.387 | 0.257 | 0.908 | 2.8 | 11.2 |
| hybrid | 1 | 0.227 | 0.718 | 0.369 | 0.369 | 0.282 | 0.970 | 1.9 | 12.9 |
| hybrid | 2 | 0.142 | 0.724 | 0.337 | 0.173 | 0.128 | 0.902 | 3.9 | 7.7 |
| hybrid | 3 | 0.135 | 0.594 | 0.436 | 0.116 | 0.108 | 0.875 | 4.3 | 7.0 |
| hybrid | 4 | 0.134 | 0.590 | 0.509 | 0.096 | 0.103 | 0.864 | 6.4 | **6.5** |
| hybrid | 5 | 0.107 | 0.516 | 0.600 | 0.127 | 0.109 | 0.886 | 8.1 | 8.4 |
| hybrid | 6 | 0.107 | 0.469 | 0.535 | 0.101 | **0.097** | 0.893 | 4.9 | 6.6 |

Data: `outputs/recursion_raw.csv` (648 rows), `outputs/recursion_by_round.csv`, `outputs/entity_half_life.json`, `outputs/trajectories.json`, `outputs/recursion.png`.

## 3. What the numbers say

1. **Greedy recursion converges to an identity copier in 4 rounds.** Churn falls 0.442 → 0.036 → 0.017 → 0.051 → **0.000**, and from round 4 the output is a 100 % 4-gram subset of its own input. The loop has a stable fixed point and the fixed point is *"repeat the previous text verbatim"*. Iteration cannot humanize; it can only converge to copying. 51 % of the original wording survives to the fixed point.
2. **Sampling recursion converges too, just slower** (churn 0.672 → 0.033 by round 6, copy→prev 0.976). It loses 22 % of the original wording and 13 % of entity recall in the first round, then stabilises.
3. **The post-nucleus (hybrid) decoder does not converge at all — it collapses.** Churn stays at 0.47–0.72 for six rounds, similarity to the original falls 0.282 → **0.097** (a 66 % loss of the original text), entity recall halves (0.227 → 0.107) and boilerplate grows 1.9 → 8.1/100w. This is textbook model collapse: the decoder's own tail sampling becomes the next round's distribution, and the tail is uncalibrated.
4. **Perplexity is an anti-indicator of collapse — the most important methodological result here.** PPL *falls* in every decoder as information is destroyed (hybrid 12.9 → 6.5, greedy 12.2 → 11.9), because repeated self-copying makes text maximally predictable. Any objective that minimises perplexity will actively select the collapsed text. The corpus-level metrics (entity recall, similarity-to-original, boilerplate) must lead.
5. **Entity half-life:** hybrid hits 50 % of its own round-1 recall at round 5; greedy and top-p never halve (they converge above 50 %). Half-life therefore measures *drift*, not *quality* — a decoder can be stable and still bad (greedy: stable at 80 % copy).

## 4. Gaps & how to close them

| Gap | Evidence | Fix / next action |
|---|---|---|
| **Fixed point is copying, because the prompt says "rewrite"** | copy→prev 1.000 | Add an explicit anti-copy constraint per round (E11 ban + entity pin) and re-run: does the loop still converge to identity? This is the cheapest test of whether recursion is fundamentally doomed or only doomed under this prompt. |
| **No stopping rule** | churn collapses but nobody knows when to stop | Define a stopping criterion grounded in information: stop when `entity_recall` or `similarity_to_original` falls below a budget (e.g. 90 % of round 1), then report the round at which it happens. |
| **Perplexity misleads** | PPL ↓ while everything else ↓ | Replace PPL with **cross-model** PPL (`gpt2`-124M or Qwen2.5-0.5B) and/or a compression-based information measure. Self-PPL is structurally biased toward collapsed text. |
| **Only 6 rounds, 12 docs** | collapse may continue | Run 15 rounds on 4 docs to see whether hybrid reaches a fixed point or keeps degrading, and whether greedy's fixed point is genuinely stable. ~1 min. |
| **Recursion measured only on "rewrite" prompts** | the prompt may be the whole story | Same loop with `humanize` and `summarize` templates; and with the source re-injected each round (rewrite the *original* using round N-1 as a style example) — that variant should *not* collapse. |
| **No human realism check** | "collapsed" is inferred from proxies | 10 paired samples (round 1 vs round 6) sent for human preference: does a human see degradation? |

## 5. Cross-experiment links

* **E12**: the collapse in hybrid recursion is the head-escape cliff reached by iteration instead of temperature — same mechanism (distribution flattens, text degrades, PPL misleads).
* **E06**: steering and recursion are two ways to push the model off its trajectory; both produce incoherence long before they produce style.
* **E10**: greedy is the most stable recursive base *because* it copies most; this is why recursion results must be reported alongside copy rate.
* **E11**: the anti-copy processors (preamble ban, entity pin) are the natural next lever to test against the copying fixed point.
