# Experiment 11 — Entity-Pinned Head Escape

**Status:** ✅ completed (mixed result, one clear mechanism found) · **Runtime:** 73 s · **Spawned by:** E10 gap table

---

## 1. Idea

E10: truncation families are head-locked (GLTR ≈ 0.99 vs 0.80 human), and the only head-escaping decoder (repetition penalty) destroys entity recall. E02/E03 gated on **confidence**; E11 gates on **token class**:

* capitalised / numeric tokens that occur in the source → **force greedy** (copy the anchor, never paraphrase a name or a date);
* every other token → sample from a **mixture** of the nucleus `(1-q)` and the post-nucleus boundary token `(q)`.

Plus an orthogonal lever: hard-ban the assistant preamble (`Here is`, `Here's`, `Sure,`, `**`, `Step 1`, …) with `BanFirst` + `BanPhrases`, which E10 showed no decoder can avoid.

Implementation: `src/humanize_lab/routers.py` (`EntityPinnedRouter`, `fact_token_ids`, `preamble_ban`). 14 090 of 49 152 vocab ids (28.7 %) classify as fact carriers.

## 2. Results (12 docs × 3 seeds × 100 tokens)

| condition | words | copy₄ | entity | recall/copy | D2 | burst | boil/100w | PPL | GLTR top-10 |
|---|---|---|---|---|---|---|---|---|---|
| top-p 0.9 (baseline) | 66.9 | 0.505 | 0.264 | 0.52 | 0.926 | 0.393 | 2.17 | 12.9 | 0.992 |
| ban only | 71.5 | **0.191** | 0.181 | **0.95** | 0.891 | 0.482 | 2.30 | **9.8** | 0.990 |
| pin only (q=0) | 72.2 | 0.660 | **0.393** | 0.60 | 0.957 | 0.422 | 1.58 | 11.3 | 0.997 |
| pin + q=0.25 | 72.1 | 0.584 | 0.372 | **0.64** | 0.957 | 0.412 | 1.51 | 10.7 | 0.996 |
| pin + q=0.50 | 71.6 | 0.582 | 0.365 | 0.63 | 0.931 | 0.432 | **1.22** | **10.3** | 0.996 |
| pin + q=1.00 | 58.8 | 0.021 | 0.075 | 3.6 | 0.948 | 0.435 | 0.91 | **45.6** | 0.983 |
| pin + q=0.5 + ban | 70.0 | **0.167** | 0.166 | **0.99** | 0.947 | 0.472 | 1.97 | **9.7** | 0.990 |
| pin + q=1.0 + ban | 44.0 | 0.004 | 0.042 | 10.5 | 0.964 | 0.313 | 0.43 | 27.0 | 0.989 |

`recall/copy` = entity recall divided by 4-gram copy rate: how many facts you keep per unit of plagiarism.

## 3. What the numbers say

1. **Pinning works as designed but inherits the copy it protects.** q=0 (pin only) gives the best entity recall of any condition measured in this repo so far short of greedy: **0.393 vs 0.424 for greedy, at copy 0.66 vs 0.80**. On the recall-per-copy axis it beats both baselines (0.60 vs 0.52–0.53). Anchors are protected; their surrounding words come along for the ride.
2. **The preamble is a plagiarism primer.** Banning the assistant preamble halves verbatim copying (0.505 → **0.191**) and *lowers* PPL (12.9 → 9.8) while raising burstiness (0.39 → 0.48). The "Here's a rewritten version:" opener is not cosmetic — once the model emits it, it continues in copy-the-input mode. This is the single cheapest humanization win found so far and it is a 15-line processor.
3. **Combining banning with pinning does not add up.** `pin + q0.5 + ban` reaches the best recall/copy efficiency (0.99) but drops absolute recall to 0.166, because the ban removes the summary framing the model was using to restate facts. The two levers both reduce copy; their combination over-reduces it.
4. **Mixture q is not a free knob.** q=0.5 is safe (PPL 10.3); q=1.0 collapses (PPL 45.6, recall 0.075). There is no smooth head-escape dial here — same cliff as E12.
5. **GLTR stays ≥ 0.98 in every condition.** Even when *half or all* of the non-anchor tokens are forced to the post-nucleus boundary, the chosen tokens remain inside the model's top-10. This is E12's floor showing up inside a router.

**Verdict:** entity pinning is a genuine improvement on the copy/recall frontier, and preamble banning is a free win, but neither moves head conformity.

## 4. Gaps & how to close them

| Gap | Evidence | Fix / next action |
|---|---|---|
| **Anchor detection is vocabulary-shaped, not semantic** | 28.7 % of vocab flagged; every capitalised word qualifies | Use a proper NER (spaCy `en_core_web_sm`, 12 MB) over the *source* to build a per-document anchor allow-list, then pin only those. Should cut copy a lot while keeping recall. |
| **Pinning forces the model's chosen token, not the source's** | recall 0.39 < greedy 0.42 | If the top-1 fact token is not the source's token, **overwrite** it with the source token (constrained decoding over the source span). Requires a copy-pointer mechanism (see E04 canvas / E05 attention). |
| **Ban list is hand-written** | 18 phrases | Learn it: take the top-100 5-gram openers of the model's own outputs and ban the ones absent from the human corpus. Data-driven, no guessing. |
| **No semantic similarity metric** | copy falling could just mean content loss | Add MiniLM cosine to source; recall/copy is a proxy until then. |
| **Mixture q has a cliff** | q=1.0 → PPL 45.6 | Make q **entropy-adaptive**: q high only when the model is confident (top1 > 0.5), which is exactly where copying happens. This is E12's "adaptive temperature" gap with a different knob. |
| **Only measured on rewrite** | pinning may hurt summarization/open generation | Run the same router on `corpus.USER_TEMPLATES['summarize']` and `['continue']` to see whether anchors are task-dependent. |

## 5. Cross-experiment links

* **E10** → this experiment (head-lock + recall/copy frontier).
* **E12** → explains why q=1.0 collapses: same "leave the head, leave English" cliff, now with an explicit prediction curve.
* **E09 (DPO)** → the preamble ban is a *rule*; DPO could *learn* the same suppression, which is how you would deploy it without a processor.
* **E04 (global canvas)** → pinning is a local version of "anchor first, then diffuse the prose"; the canvas experiment tests the global version.
