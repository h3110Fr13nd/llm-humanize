# Experiment 08 — Position-Free Thought Streams (feasible POC)

**Status:** ✅ completed (negative result, two clean measurements) · **Runtime:** 52 s · **Model:** SmolLM-135M-Instruct

---

## 1. What was tested

The original idea (IDEA family "position-free thought streams") proposes that a
model could think in a *bag of concepts* with no left-to-right order, and only
linearize at the end — which would remove the autoregressive copy-prior that
E10/E12 proved is the bottleneck.

Two parts were feasible on a 4 GB laptop; the third needs training and was
deliberately skipped.

| part | question | status |
|---|---|---|
| **08a** | How much does SmolLM-135M depend on *absolute position*? | ✅ measured |
| **08b** | Temporal-JEPA concept streams (predict concept trajectories, not tokens) | ❌ out of scope — needs a trained predictor, not a POC |
| **08c** | Can the model unscramble a bag of words back into a sentence? | ✅ measured |

## 2. Method

**08a — position ablation (teacher-forced).** Take the 12 human source
documents, run them through the model with four position-id regimes: normal
RoPE positions, all-zero positions, reversed positions, and per-row shuffled
positions. Perplexity is computed on the *same tokens*, so only positional
information changes.

**08c — lexical unscrambling.** One mid-document sentence per document (12
sentences, 8–25 words). Swap 0 % / 15 % / 30 % / 50 % of word positions, then
prompt: *"Reorder these words into the original grammatical sentence."*
Because the model is autoregressive it sees the scrambled sentence as a
sequence; if a position-free syntax prior existed, it should still recover the
order.

Critical addition (missed on the first run): **net reordering**, i.e.
`τ(output, original) − τ(input, original)`. Without it the metric is confounded
— a model that simply *copies the scrambled input* keeps most of its order when
only 15 % of positions were swapped.

## 3. Results

### 08a — position ablation

| position regime | mean PPL | × normal |
|---|---|---|
| normal RoPE | **17.36** | 1.0× |
| reversed | 236.67 | 13.6× |
| zero (all positions = 0) | 258.91 | 14.9× |
| shuffled (per-row permutation) | **489.78** | 28.2× |

### 08c — unscrambling (12 sentences)

| shuffled | τ_input | τ_output | **Δτ (net recovery)** | sim_input | sim_output | word recall |
|---|---|---|---|---|---|---|
| 0 % (intact control) | 0.791 | 0.225 | **−0.565** | 0.901 | 0.317 | 0.648 |
| 15 % | 0.718 | 0.443 | **−0.275** | 0.867 | 0.502 | 0.901 |
| 30 % | 0.471 | 0.291 | **−0.180** | 0.729 | 0.387 | 0.848 |
| 50 % | 0.366 | 0.172 | **−0.194** | 0.597 | 0.142 | 0.589 |

Exact recoveries: **0 / 48**.

## 4. What the numbers say

1. **Position is load-bearing, not decorative.** Destroying position costs
   13–28× perplexity on text the model would otherwise find easy. Random
   shuffling is worse than *reversing* (490 vs 237), i.e. a consistent wrong
   order is more usable to the model than an inconsistent one. SmolLM-135M has
   no position-independent syntax representation; its grammar lives in the
   RoPE channels.
2. **The model cannot unscramble at all — and it does not try.** Δτ is negative
   in every condition: the output is *less* ordered with respect to the original
   than the scrambled input was. At 50 % it stops attempting the task entirely
   and emits Python (`def word_order_to_original_sentence(words):`) or
   meta-commentary (`"the word "Deep" is the first word…"`).
3. **The 0 % control is the most informative row.** Given an *already correct*
   sentence, the model returns something only 0.317 similar — it paraphrases
   instead of copying. Combined with the 15 % row (recall 0.901), the picture is:
   the model copies a *bag* of content words when the input looks like a list,
   but it never reasons about order. So earlier "unscrambling works a bit"
   readings would have been pure artefact of input-order preservation.
4. **Together with E10/E12 this closes the door on the position-free route for
   an AR model.** Order-free *representation* is not something you can prompt
   into a left-to-right transformer; the quantity you would need to remove
   (position dependence) is the same quantity that makes its text coherent.

**Verdict:** both sub-questions answer negatively. 08a shows position is
irreplaceable; 08c shows order recovery is zero-to-negative. 08b remains the
only untested part, and it requires training a separate predictor — not a POC.

## 5. Gaps & how to close them

| Gap | Evidence | Fix / next action | Cost |
|---|---|---|---|
| **No true position-free model to compare against** | we only ablate an AR model's positions | Use a **real diffusion LM** (E04, `dllm-hub/Qwen3-0.6B-diffusion-mdlm`) and ask the *same* unscrambling question. A masked-diffusion model denoises all positions jointly, so it is the only local model where the question is well-posed. | E04 run (~5 min) |
| **08b (concept streams) untested** | needs a predictor | Minimal viable version: train a 2-layer MLP to predict the *next sentence embedding* (MiniLM) from the current one, on the 12 docs + a public corpus; measure whether the predicted trajectory stays on the manifold. That is a real (if tiny) JEPA proxy. | ~2 h + 22 MB model |
| **Task framing caused instruction-following collapse** | 50 % → Python code | Prompt engineering alone: give 3 in-context examples of scrambled→correct pairs. If Δτ is still ≤ 0 with demonstrations, the failure is representational, not instructional. | 10 min |
| **Only 12 sentences, one seed** | τ noise ±0.1 | 100 sentences from a public corpus, 3 seeds, bootstrap CI. | 15 min |
| **Absolute position only ablated coarsely** | 4 regimes | Separate *distance* from *absolute index*: use relative-only attention (no RoPE offset) or monotonic-but-shifted positions (`pos + k`). Distinguishes "needs distance" from "needs absolute index". | 20 min |
| **Metric τ breaks on duplicate words** | τ_input = 0.791 for an *identical* string | Use position-aware alignment (LCS-based τ or edit-distance order score) instead of a first-occurrence dict. Fixes the 0 % row's ceiling too. | 20 min |

## 6. Cross-experiment links

* **E12 (head-conformity floor)** — the same conclusion from a different angle:
  the constraint is in the weights. Position-free prompting cannot create a
  distribution the model does not have.
* **E04 (diffusion)** — the *only* architecture in this repo where a
  position-free/bidirectional pass is real. E08's unscramble protocol transfers
  directly: run it on the MDLM and compare Δτ. This is the natural follow-up.
* **E05 (window attention)** — E05 masked *which* tokens a query can see; E08
  masks *where* they are. Both leave GLTR head conformity at ~0.99, reinforcing
  that attention/position surgery does not change the output distribution.
* **E10/E11** — the copy behaviour E08 exposes (copy the bag, ignore order) is
  the same copy prior measured quantitatively there; E08 adds the *mechanism*
  (order is what the model cannot do without).
* **E09 (RL post-training)** — if order-free reasoning is ever wanted, it has to
  be *learned*: a training objective that rewards recovering order from
  permutations. E09's DPO harness is the place to add it.
