# Sub-Experiment 09a: DPO Anti-Boilerplate Distillation

**Parent Experiment**: [`../README.md`](../README.md)  
**Thought Node**: [`ideas/nodes/IDEA-10-rl-post-training-humanization.md`](ideas/nodes/IDEA-10-rl-post-training-humanization.md)

---

## 🎯 Research Objective
Distill the stylistic synthesis of the Phase 03 Tuned Hybrid Decoder directly into model weights via Direct Preference Optimization (DPO).

## 🧪 Experimental Questions:
1. **Boilerplate Suppression**: Can a DPO-aligned model avoid generating cliché transition phrases (*"Here's a rewritten summary..."*) even when evaluated at standard greedy ($k=1$) decoding?
2. **Entity Retention**: Does preference training on synthesized rewrites preserve the 16.7% entity recall while eliminating verbatim copying?
3. **Inference Latency**: Quantify the speed advantage of running an aligned model with native greedy decoding vs. runtime `LogitsProcessor` overhead.

---

## 3. 🔧 What Was Built

| Piece | Path | Note |
| :--- | :--- | :--- |
| Hand-rolled LoRA | `src/humanize_lab/tuning.py` | r=16, α=32 on q/k/v/o — **1,843,200 params = 1.35 %** of SmolLM-135M; `peft` is not installed and a dependency-free 120-module swap is 200 lines |
| DPO trainer | `src/humanize_lab/tuning.py` | the adapter *is* the reference policy: `handle.set_enabled(False)` gives the frozen model, so no second copy of the weights in 4 GB VRAM |
| Run | `scripts/run_dpo.py` | builds 48 passage-level pairs (40 train / 8 held out), 25 steps, 52 s |
| Artifact re-check | `scripts/eval_adapter.py` | reloads `outputs/lora_adapter.pt` and regenerates the eval text (`outputs/eval_samples.json`) |

**Pair construction** (no reward model, no human labels): chosen = top-p(0.9)
output of the *tuned hybrid* decoder with the E11 preamble ban applied; rejected =
plain greedy output. So 09a is a distillation of E11's runtime ban into weights,
which is exactly the question: *can a runtime crutch become a learned habit?*

Training: 25 steps, loss 0.692 → 0.587, reward margin 0.025 → 2.243, batch
preference accuracy 1.000, **held-out preference accuracy 0.875** (7/8).

---

## 4. 📊 Results (greedy decoding, 12 documents, SmolLM-135M-Instruct)

| Stage | words | copy₄ | entity recall | boilerplate | burstiness | distinct-2 | rep₄ | style JSD | PPL | GLTR top-10 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| before (base greedy) | 71.5 | 0.796 | 0.424 | 1.79 | 0.349 | 0.989 | 0.000 | 0.267 | 12.21 | **1.000** |
| after (LoRA + DPO) | 75.7 | **0.024** | **0.116** | 2.32 | **0.573** | 0.896 | 0.034 | 0.298 | **4.57** | **1.000** |

`eval_adapter.py` reproduces both rows exactly from the committed adapter.

**What improved:** verbatim copying collapses 0.796 → 0.024 — a 33× reduction with
*no* decoding penalty, no ban, no logit processor. Sentence-length variance doubles
(burstiness 0.349 → 0.573), which is the human-like property E10 could not buy.

**What broke:** entity recall 0.424 → 0.116 (−73 %). The model stops copying and
also stops saying what the document said.

**What the metrics missed — and this is the real finding.** The boilerplate score
went *up*, but the eye sees the opposite: 10 of the 12 "after" samples open with the
identical phrase.

```
before  openers:  6/12 distinct 4-grams  ("here's a rewritten version" x7)
after   openers:  3/12 distinct 4-grams  ("what a fascinating topic!"      x10)
```

| sample | first 88 chars |
| :--- | :--- |
| 01_space_telescopes | `What a fascinating topic! The James Webb Space Telescope (JWST) is a groundbreaking spac…` |
| 02_crispr_gene_editing | `What a fascinating topic! CRISPR-Cas9 is a revolutionary technology that has the potenti…` |
| 03_quantum_computing | `What a fascinating topic! Quantum computing has the potential to revolutionize the way w…` |

PPL 4.57 is the tell: the model has become *more* predictable, not more human. It
swapped one formula for another. The `BOILERPLATE_MARKERS` lexicon in
`metrics.py` only knows "here is / rewritten / version / summary", so a new opener
family scores zero and the aggregate even rises. **The metric was measuring the old
failure, not boilerplate as such.**

And the head-conformity floor did not move: GLTR top-10 stays at **1.000** before and
after, exactly as E12 predicts. Preference training changes *what* the model says,
not *how surprising* each token is to the base model.

---

## 5. 🧩 What Is Still Missing, and How to Close Each Gap

| # | Gap | Fix |
| :--- | :--- | :--- |
| 1 | **Chosen side is weak** — it is a *banned* greedy output, not a good rewrite. The model learned "no preamble" and nothing else, so it invented a new preamble. | Build the pair from E13's `pin+adapt c=0.70` output (GLTR 0.952, copy 0.178, recall/copy 2.2× baseline) as *chosen* and E13's `pin+adapt c=0.30` as *rejected*. The preference then carries accuracy, not just a ban. |
| 2 | **Data is 40 pairs from 12 documents** — the opener uniformity is overfitting to 48 passages in one register. | Generate pairs from a larger, multi-register corpus (news, technical, narrative), or augment by passage permutation. 40 pairs cannot teach a style. |
| 3 | **Opener diversity is unmeasured.** The obvious failure (10/12 identical first sentence) has no column in the table. | Add `opener_diversity` (distinct first-4-grams / n) and `self_bleu` against the other outputs to `metrics.py`; make it a merge gate for any style run. |
| 4 | **DPO has no accuracy term.** DPO maximises preference, and the rejected side was already more accurate, so accuracy was the thing being optimised *away*. | 09b's reward design (`R = entity_recall − detector_penalty`) is the right shape; with 4 GB the feasible version is RLOO/GRPO on LoRA with `metrics` as the reward, 1 sample per prompt. |
| 5 | **8 test pairs cannot distinguish 0.875 from 0.75.** | Report a binomial interval, or hold out 12+ pairs; cheap because the run is 52 s. |
| 6 | **The model got blander, not humaner.** PPL fell to 4.57, GLTR stayed 1.000. | Human-likeness must be *rewarded*, not assumed: add the E13 adaptive-escape sampler during pair generation so the chosen side already has GLTR ≈ 0.8–0.9. |

---

## 6. 🔗 How This Combines With the Other Experiments

* **E13 → 09a (the strongest combination in the repo).** E13's `pin+adapt c=0.70`
  is the only configuration that reaches the human GLTR profile (0.952) while
  keeping copy at 0.178 and recall/copy at 2.2× baseline. Distilling *that* instead
  of E11's ban would give DPO both a style target and an accuracy target. The
  infrastructure already exists on both sides.
* **E11 → 09a (what happened here).** 09a is literally E11's `ban_only` distilled.
  The lesson generalises: **distilling a suppression teaches a suppression.** The
  banned phrase disappears verbatim and returns as a paraphrase, which is why
  opener diversity is the metric that must gate this class of run.
* **E12 → 09a (a negative result that constrains everything).** GLTR 1.000 → 1.000
  means weight updates at 1.35 % of parameters cannot move the head-conformity
  floor. If humanisation requires moving it, DPO on a LoRA is the wrong instrument;
  E13's sampler is the only thing that has moved it so far.
* **E04 → 09a (same symptom, cheaper fix).** Diffusion removed boilerplate
  *structurally* (0.00 against 2.47) without losing recall (0.174 vs 0.168) and
  without any training. If the goal is only "no assistant preamble", E04 already
  achieves it; DPO is the more expensive route to a different problem.
* **E07 → 09a.** Recursive rewriting showed the model's self-copy fixed point. An
  adapter trained to rewrite differently should be re-tested under recursion: the
  interesting question is whether DPO moves the fixed point or just the first step.

---

## 7. ▶️ Reproduction

```bash
# retrain from scratch (~1 min; needs the scorer parked on CPU for VRAM)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python experiments/09_rl_post_training_humanization/09a_dpo_anti_boilerplate/scripts/run_dpo.py

# or re-evaluate the committed adapter and dump the text (~40 s)
uv run python experiments/09_rl_post_training_humanization/09a_dpo_anti_boilerplate/scripts/eval_adapter.py
```

Outputs: `dpo_eval_summary.md`, `dpo_history.csv` (loss/margin curve),
`pair_quality.csv`, `pairs_preview.json`, `eval_samples.json` (the text),
`lora_adapter.pt` (1.8 M params, 7 MB), `dpo.png`, `env.json`.

---

## 8. Decision Log

| # | Change | Evidence | Effect on the hypotheses |
| ---: | :--- | :--- | :--- |
| 1 | Hand-rolled LoRA + DPO instead of `peft`/`trl` | 1.35 % of params, 120 modules, `outputs/env.json` | unblocked training in 4 GB — the adapter doubles as its own reference policy |
| 2 | Chose `ban_only` top-p as the *chosen* side | `outputs/pair_quality.csv`: chosen copy 0.00 vs rejected 0.897 | **H-boilerplate confirmed** (copy 0.796 → 0.024) |
| 3 | Added `eval_adapter.py` because the run saved no text | `outputs/eval_samples.json` | exposed the failure the metrics missed (10/12 identical openers) |
| 4 | Measured opener diversity after seeing the samples | 6/12 → 3/12 distinct openers | **H-transfer rejected**: the preamble was replaced, not removed |
| 5 | Kept GLTR in the eval | 1.000 before and after | **H-floor rejected**: DPO does not move head conformity, consistent with E12 |
