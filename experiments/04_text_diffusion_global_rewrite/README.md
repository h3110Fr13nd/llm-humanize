# Experiment 04: Global Non-Autoregressive Text Diffusion for In-Place Rewriting

## 1. 📝 Raw User Idea (Verbatim Transcription)

> *"I want to utilise a text diffion model like diffusion gemma for this. since rewrite will be much better if we knew and adapted the full context in place completely instead of cumulative divergence context based on word choices coming next."*

---

## 2. 🧠 Synthesized & Descriptive Engineering Formulation

### A. The Core Problem: Autoregressive Exposure Bias & Cumulative Drift
In standard autoregressive generation (AR), tokens are generated strictly left-to-right:
$$P(y_1, y_2, \dots, y_T \mid x) = \prod_{t=1}^T P(y_t \mid y_{<t}, x)$$

* **The Cumulative Divergence Trap**: If step $t=15$ introduces an exploratory synonym or post-boundary token, steps $t > 15$ are conditioned on that specific divergence. The model must constantly adapt forward to its own previous choices without the ability to look ahead or backtrack. Over a 500–1,000 word rewrite, small early stylistic decisions compound, leading to semantic drift, loss of factual anchors, or abrupt syntactic compromises.
* **Why Rewriting is Inherently Non-Causal**: Unlike open-ended story generation, **rewriting already has a defined global semantic scope**. A human writer rewriting a paragraph knows the ending, theme, and key facts before writing sentence one; they draft, revise in-place, adjust word choices to fit the whole paragraph rhythm, and resolve dependencies globally.

---

### B. The Proposed Solution: Discrete / Masked Text Diffusion
Instead of generating sequentially, treat rewriting as **iterative denoising of a full sequence canvas**:

```
[Prompt: Source Text (Full Context)]
                 │ Conditioning
                 ▼
Step T   : [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] ... [MASK]   (Pure Noise Canvas)
Step t   : [The]  [MASK] [pivotal] [MASK] [Cas9] [MASK] ... [DNA]   (Global Anchor Settlement)
Step 0   : [The]  [team] [utilized] [Cas9] [to]  [edit] ... [DNA]   (Fluently Polished Rewrite)
```

In discrete diffusion (e.g., **Diffusion-Gemma**, **LLaDA**, **SEDD**, or **Plaid**):
1. **Bidirectional Attention at Every Step**: Every token position $i$ attends to all other positions $j \in [1, T]$ simultaneously.
2. **In-Place Adaptation**: If the model chooses a creative or post-boundary word at position $i$, positions $1 \dots i-1$ and $i+1 \dots T$ adjust their choices in the next denoising step to harmonize with that word.
3. **No Left-to-Right Hallucination Cascades**: Factual anchors (dates, names, metrics) can be established first across the document, and connective prose/style words are filled in around them in subsequent sampling steps.

---

### C. Proposed Architecture & Exploration Matrix

| Component | Target Implementation / Framework |
| :--- | :--- |
| **Candidate Models** | Diffusion-Gemma, LLaDA (Large Language Diffusion with Autoregressive-level performance), SEDD (Score Entropy Discrete Diffusion), or Plaid. |
| **Conditioning Mode** | Classifier-Free Guidance (CFG) conditioning on the source text: $\hat{\epsilon}_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$ |
| **Sampling Regimes** | - Uniform schedule vs. Cosine schedule denoising<br>- Infilling / Constrained In-Place Diffusion (pinning verified key entity tokens and diffusing only around them) |
| **Evaluation Metrics** | - Entity Retention % vs. Word-for-Word Plagiarism Rate<br>- Global Coherence & Burstiness vs. Autoregressive Hybrid Baseline |

---

### D. Planned Directory Structure

```
experiments/04_text_diffusion_global_rewrite/
├── README.md              # This document (raw voice + technical plan)
├── data/                  # Source articles (1K context benchmark dataset)
├── scripts/               # Diffusion inference & unmasking scheduling scripts
├── notebooks/             # Independent runnable interactive denoising visualization
└── outputs/               # Diffusion rewrite samples, side-by-side comparisons, step traces
```

---

### E. Milestones & Next Steps
1. **Milestone 1**: Set up local inference harness for an open-weights discrete text diffusion model (e.g., LLaDA or Diffusion-Gemma checkpoint).
2. **Milestone 2**: Implement classifier-free guidance conditioning with the 1,000-word source prompt.
3. **Milestone 3**: Run side-by-side qualitative and quantitative comparison against Experiment 03's Tuned Hybrid AR baseline.

---

## 3. 🔧 What Was Actually Built

| Piece | Path | Note |
| :--- | :--- | :--- |
| MDLM sampler | `src/humanize_lab/diffusion.py` | block/step schedule, low-confidence remasking, optional CFG, 4-D bidirectional mask |
| Model loader | `load_diffusion_model()` | loads `dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1` into **stock** `Qwen3ForCausalLM` — 0 missing / 0 unexpected tensors |
| AR baseline | `src/humanize_lab/sampling.py` | SmolLM-135M-Instruct, same prompts, same metrics |
| Comparison | `scripts/run_diffusion_vs_ar.py` | 2 AR conditions + 5 diffusion conditions × 12 docs |
| Diagnostics | `scripts/probe_mask_semantics.py`, `probe_denoising.py`, `collapse_probe.py`, `numeral_probe.py` | each answers one question, output committed |
| Numeral fix | `src/humanize_lab/textfix.py` | `verbalise_numerals()` |

**Why stock `Qwen3ForCausalLM`:** the checkpoint ships `A2DQwen3Model`, which only
subclasses `Qwen3Model` (MDLM changes the *training objective*, not the
transformer), and the state dict uses stock keys. `dllm` itself is not on PyPI,
so avoiding the custom code removes the only install blocker. The checkpoint's own
`forward` replaces the causal mask with a padding-only mask, which is exactly the
4-D all-ones mask the sampler passes — verified, not assumed.

### Hardware reality (RTX 3050 Laptop, 4 GB)

| Attempt | Peak VRAM | Outcome |
| :--- | ---: | :--- |
| batch 2, float64 Gumbel noise (upstream recipe) | OOM (888 MiB request) | 2 of 3 diffusion arms never ran |
| batch 2, float32 noise | OOM (444 MiB request) | first arm only |
| **batch 1, float32 noise, `logsumexp` confidence, `empty_cache` per condition** | **1.78 GB (T=0) / 2.71 GB (T>0)** | all 5 arms complete in 65 s each |

The peak was never the 1.2 GB model. `add_gumbel_noise` upcast the logits to float64:
at batch 2 / 224 tokens / 151 k vocab that is **545 MB per tensor, two of them**, and
`softmax` added another 272 MB of float32. Dropping to float32 plus a `logsumexp`
confidence removes 1.3 GB of transient memory, and the numbers do not change
(we `argmax` immediately; float32 has ample mantissa for logits in [-30, 30]).

---

## 4. 📊 Results (12 documents, 64 new tokens, RTX 3050)

| Condition | words | copy₄ | entity recall | boilerplate | PPL | seconds |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| AR SmolLM-135M top-p 0.9 / T 0.8 | 43.6 | 0.516 | 0.168 | 2.47 | 14.0 | 1.9 |
| AR SmolLM-135M greedy | 46.6 | 0.768 | 0.259 | 2.75 | 12.2 | 1.6 |
| diffusion, global canvas, T=0 | 10.4 | 0.355 | 0.047 | **0.00** | ∞ | 53.5 |
| diffusion, global canvas, T=1 | 21.8 | 0.407 | 0.132 | **0.00** | 1.1 × 10⁷ | 64.9 |
| diffusion, block 16, T=1 | 33.5 | 0.369 | 0.205 | **0.00** | 1.1 × 10⁷ | 65.0 |
| diffusion, global, T=1, **numerals verbalised** | 25.5 | 0.515 | 0.112 | **0.00** | 36.6 | 64.3 |
| **diffusion, block 16, T=1, numerals verbalised** | **41.1** | **0.420** | **0.174** | **0.00** | **23.4** | 64.7 |

Per-document, after the numeral fix, **every one of the 12 canvas runs lands in the
normal PPL band (6.4 – 49.1)**; before it, 1–5 of 12–24 collapsed into a comma run.

The honest reading:

* **Boilerplate is structurally eliminated.** Every diffusion condition scores 0.00
  against 2.47–2.75 for the AR baseline. There is no "Here's a rewritten version"
  because nothing is generated left-to-right from a chat prefix; the commas that do
  appear come from the numeral attractor, not from a preamble habit.
* **Copying improves without any penalty term.** `block16 + verbalised` copies 0.420
  versus 0.516 for AR top-p (−19 %) and 0.768 for greedy (−45 %), while E10 showed AR
  only achieves that with a 1.3 repetition penalty that costs 90 % of entity recall.
* **Entity recall matches the AR baseline**, 0.174 vs 0.168 — the global canvas does
  not lose facts, it just places them differently.
* **Fluency is worse and cost is 34×.** PPL 23.4 vs 14.0, 65 s vs 1.9 s for 12
  documents. At 0.6 B params on a 4 GB card this is a research instrument, not a
  drop-in replacement.

---

## 5. 🔬 The Four Diagnostics (why the numbers look like this)

### 5a. Mask semantics — is the 4-D mask real? (`probe_mask_semantics.py`)
Causal and bidirectional logits differ (max |Δ| 24.4), so the mask is honoured, not
silently ignored. But a **fully masked suffix** produces junk under every mask
variant, while a causal mask predicts `' Paris'` correctly for the same prompt. The
plumbing is right; the *regime* is wrong.

### 5b. Denoising — can the model cold-start a canvas? (`probe_denoising.py`)
| Case | Result |
| :--- | :--- |
| T1 — mask 15 % of tokens **mid-sequence** (infilling) | top-1 `' Paris'` — correct |
| T2, T3 — mask the **entire** suffix, greedy unmask | `' a'` then `'_'`; full unmask → `' a_ of___'` |

So the checkpoint **can** infill and **cannot** cold-start a fully masked block when
driven by the plain `Qwen3ForCausalLM` path. This is the single biggest gap.

### 5c. Block geometry and prompt length (`collapse_probe.py`)
Neither explains the collapse: `01_space_telescopes` fails at block 8, 16, 32 and 64,
and still fails when its prompt is truncated to 100 tokens. `12_calvin_cycle` is
unstable (PPL 3.5 at block 8, 1.4 × 10⁸ at block 32), `03_quantum` is stable
everywhere (6.6 – 10.7). Both hypotheses (length, canvas geometry) are **rejected**.

### 5d. The numeral attractor (`numeral_probe.py`) — the actual cause
The degenerating canvases emit digits and commas (`1,2,3,4,5,6,7,8,9,1,1,2,…`), and
the two documents that ever collapsed are the number-dense ones.

| Case | commas | PPL | output |
| :--- | ---: | ---: | :--- |
| doc 01, raw numerals (`2.4-meter`, `540 kilometers`, `April 24, 1990`) | 63 | 2770.3 | `Request,,,,,,,,,…` |
| doc 01, numerals verbalised | 0 | 26.8 | fluent |
| doc 01, numerals deleted | 0 | 21.3 | fluent |
| doc 03, few numerals (control) | 0 | 4.9 | fluent |
| doc 03, numerals **injected** | 19 | 23.0 | `The concept of a name is,, in,,,,,, and beyond. Values: 1,2,3,…` |

Causal in both directions. The fix (`verbalise_numerals`) is what turns the table in
§4 from "1.1 × 10⁷" into "23.4", and it costs the digit anchors, which is why entity
recall stays at 0.174 rather than climbing.

---

## 6. 🧩 What Is Still Missing, and How to Close Each Gap

| # | Gap | Why it matters | Fix |
| :--- | :--- | :--- | :--- |
| 1 | **Cold-start block is unsupported** by the stock path (5b) | forces `block_size=16`, i.e. semi-autoregressive — the "global in-place rewrite" claim is only partly tested | install the upstream `dllm` package (GitHub only, `pip install git+…`) and re-run `block_size=64`; it ships the A2D sampler with the mask-token handling this path approximates |
| 2 | **Better: redesign as infilling** | the ideas note asks for *in-place* rewriting, which is exactly 15 % masking of the source, not generating from a blank canvas — and infilling provably works (5b, T1) | mask the source tokens in place, denoise `steps≈10`, keep the rest fixed; length stays the source's, so no 64-token ceiling |
| 3 | **64-token ceiling** | 41 words is not a rewrite of a 150-word article | `block_size=16` + slice logits to the active block (`logits[:, block_start:block_end]`), which cuts the logit tensor by 4× and buys 256 tokens |
| 4 | **No GLTR / human-likeness measurement** | we cannot say whether bidirectionality escapes E12's head-conformity floor — the most interesting open question in the repo | score diffusion outputs with the same `metrics.gltr` on the base-model distribution (implementation exists in `metrics.py`) |
| 5 | **CFG never exercised** | `cfg_scale` is implemented and untested; it is the diffusion analogue of E06's steering, but at the *denoising* level | run `cfg_scale ∈ {1, 2, 4}` on the verbalised prompt; measure style JSD and copy |
| 6 | **Digits are lost** | 5d fixes the attractor by deleting the anchors | verbalise → diffuse → restore digits by alignment against the source (the entities are known per document) |
| 7 | **Single seed where it matters** | block16 has 1 seed; 12 docs is a small sample | 3 seeds × 12 docs ≈ 3.5 min, fits the budget |

---

## 7. 🔗 How This Combines With the Other Experiments

* **E11 / E13 (entity pinning) — the strongest combination, untried.** Pinning is
  *free* in a canvas: never mask the entity token positions, and the model must
  denoise around them. In AR, pinning cost a preamble ban and lowered copy only to
  0.19; here the entity anchor is structural rather than a logit constraint. Expected
  next run: `block16 + verbalised + entity-pinned canvas`.
* **E13 (adaptive escape) — complementary, not overlapping.** E13 reached the human
  GLTR profile by *raising temperature where the model is uncertain*; diffusion has no
  temperature schedule over a sequence but a per-step confidence (0.72 – 0.90 here).
  Gating the unmask order on that confidence is the diffusion version of the same idea.
* **E10 (decoder zoo) — explains why the AR side cannot do this.** Every truncation
  family stayed head-locked at GLTR 0.98–1.00 and only the rep-penalty escaped, by
  destroying recall. Diffusion reaches copy 0.420 with no penalty at all, so the two
  escape routes are genuinely different mechanisms.
* **E12 (head-conformity floor) — the open question.** If diffusion's outputs also
  score GLTR ≈ 1.0 on the base model, the floor is a property of the 135 M/0.6 B
  models rather than of left-to-right decoding, and no decoder can fix it.
* **E09a (DPO) — a different fix for the same symptom.** DPO removed boilerplate by
  weight updates and lost recall (0.424 → 0.116); diffusion removes it structurally
  and keeps recall (0.174). Combining them would test whether DPO on *diffusion*
  outputs is even meaningful.

---

## 8. ▶️ Reproduction

```bash
# all 7 conditions, 12 docs  (~2 min)
uv run python experiments/04_text_diffusion_global_rewrite/scripts/run_diffusion_vs_ar.py

# diagnostics, each < 1 min
uv run python experiments/04_text_diffusion_global_rewrite/scripts/probe_mask_semantics.py
uv run python experiments/04_text_diffusion_global_rewrite/scripts/probe_denoising.py
uv run python experiments/04_text_diffusion_global_rewrite/scripts/collapse_probe.py
uv run python experiments/04_text_diffusion_global_rewrite/scripts/numeral_probe.py
```

Outputs: `diffusion_vs_ar_summary.md` (main table), `diffusion_vs_ar_raw.csv`,
`samples.json`, `run_log.txt`, `collapse_probe.json`, `numeral_probe.json`, `env.json`.

---

## 9. Decision Log

| # | Change | Evidence | Effect on the hypotheses |
| ---: | :--- | :--- | :--- |
| 1 | Load the MDLM checkpoint into stock `Qwen3ForCausalLM` instead of installing `dllm` | 0 missing / 0 unexpected tensors; `outputs/run_log.txt` | unblocked the experiment without a dependency that is not on PyPI |
| 2 | Pass a 4-D bidirectional mask (`bidirectional=True`) | checkpoint's own `A2DQwen3Model.forward` builds a padding-only mask | made the sampler match the model's training regime |
| 3 | float64 → float32 Gumbel noise, `softmax` → `logsumexp`, batch 2 → 1, `empty_cache` per condition | 1.09 GB transient → peak 2.71 GB, all 5 arms complete | **H-OOM** (memory) resolved; 2 of 3 arms that never ran now report |
| 4 | Add `collapse_probe.py` (block sweep + prompt truncation) | `outputs/collapse_probe.json` | **H1 (length) rejected**, **H2 (canvas geometry) rejected** |
| 5 | Add `numeral_probe.py` (verbalise / delete / inject) | `outputs/numeral_probe.json` | **H3 (numeral attractor) confirmed**, causally in both directions |
| 6 | Add `diff_*_verb` conditions using `verbalise_numerals` | PPL 1.1 × 10⁷ → 23.4; 12/12 documents survive | turns the diffusion arm from "degenerate" into a usable baseline |
| 7 | Report per-document PPL alongside the mean | `diffusion_vs_ar_raw.csv` | the "11 words, PPL ∞" headline was 2–5 collapsed documents, not the whole arm |
