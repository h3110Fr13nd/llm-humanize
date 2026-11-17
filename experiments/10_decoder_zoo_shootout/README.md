# Experiment 10 — Decoder Zoo Shootout (135M, local)

**Status:** ✅ completed · **Device:** RTX 3050 Laptop (4 GB) + Ryzen 7 6800H · **Model:** `HuggingFaceTB/SmolLM-135M-Instruct` · **Runtime:** 85 s for 360 generations

---

## 1. Why this experiment exists

Experiments 01–03 tuned only two hand-built decoders (pure post-top-p, hybrid
routing). The literature (Experiments' `research/references.md`) proposes at
least eight truncation families, and nobody had checked **which of them actually
moves a tiny model out of its probability head** on the same benchmark.

This is the control experiment for everything else: if standard decoders cannot
beat the plagiarism trap, architectural interventions (04–09, 14) have a clear
target to beat.

## 2. Method

* Corpus: the 12-document / 12-domain, entity-annotated benchmark from Experiment 03 (`data/dataset.json`).
* 10 decoders × 3 seeds × 12 documents = **360 generations**, 100 new tokens each, greedy-batched (no sampling seed reused across conditions).
* Metrics per generation: verbatim 4-gram copy rate, entity recall, Distinct-2, 4-gram repetition, sentence-length burstiness (CV), boilerplate markers /100 words, function-word JSD vs. the human corpus, perplexity under the **base** model (`SmolLM-135M`), and GLTR head conformity from the per-step rank trace.
* **Human baseline:** the same GLTR rank profile computed on the real next-tokens of the 12 human-written sources. This is the number every decoder should be compared to.

Reproduce: `uv run python experiments/10_decoder_zoo_shootout/scripts/run_decoder_shootout.py`

## 3. Results

Human GLTR baseline: **top-10 = 0.797**, 10–100 = 0.123, 100–1000 = 0.061, tail = 0.019.

| decoder | words | copy₄ | entity | D2 | burst | boil/100w | style JSD | PPL | GLTR top-10 |
|---|---|---|---|---|---|---|---|---|---|
| hybrid τ0.5/x0.4 | 66.8 | **0.37** | 0.23 | 0.97 | **0.53** | 1.88 | 0.310 | 12.9 | **0.997** |
| greedy | 71.5 | 0.80 | **0.42** | 0.99 | 0.35 | 1.79 | 0.267 | 12.2 | 1.000 |
| top-h 2.0 | 62.5 | 0.46 | 0.25 | 0.97 | 0.39 | 1.92 | 0.319 | 13.4 | 0.986 |
| η 1e-4 | 66.0 | 0.47 | 0.27 | 0.96 | 0.37 | 1.76 | 0.312 | 13.9 | 0.980 |
| min-p 0.05 | 64.9 | 0.52 | 0.30 | 0.94 | 0.36 | 1.89 | 0.316 | 12.9 | 0.998 |
| top-p 0.9 + rep 1.3 | 73.6 | **0.01** | **0.04** | 0.98 | 0.46 | **2.94** | 0.493 | **30.7** | **0.850** |
| DRY 0.8 + top-p | 64.8 | 0.54 | 0.28 | 0.94 | 0.37 | 1.91 | 0.322 | 13.4 | 0.993 |
| top-p 0.9 | 66.9 | 0.51 | 0.26 | 0.93 | 0.39 | 2.17 | 0.315 | 12.9 | 0.992 |
| typical 0.9 | 66.9 | 0.51 | 0.26 | 0.93 | 0.39 | 2.17 | 0.315 | 12.9 | 0.992 |
| mirostat v2 τ3 | 64.2 | 0.51 | 0.28 | 0.93 | 0.35 | 2.04 | 0.317 | 12.9 | 0.999 |

> **Measurement note (important).** The first run computed the GLTR rank *after* truncation, where the chosen token is always rank 0 — that reported 1.000 for everything and hid the effect. The rank is now computed on the **raw model distribution before any processor** (`sampling.py`), which is the correct GLTR definition and matches the human baseline computation. All numbers above are post-fix.

Full data: `outputs/decoder_shootout_raw.csv` (360 rows), summary `outputs/decoder_shootout_summary.csv|md`, samples `outputs/samples_all_conditions.json`, plot `outputs/decoder_shootout.png`.

## 4. What the numbers say

1. **Every standard truncation family is head-locked.** GLTR top-10 fraction is 0.98–1.00 for eight of ten decoders versus **0.797 for human text**. Truncating the head harder (top-p, min-p, η, top-h, typical, mirostat) changes *which* head token is picked, not *whether* the model stays in the head. This is the strongest quantitative confirmation of the project's core premise to date, and it was measured on-device in 85 s.
2. **The only decoder that leaves the head destroys the facts.** Repetition penalty 1.3 drags top-10 to 0.850 but entity recall collapses 0.42 → 0.038 and PPL triples (12.2 → 30.7). It is forcing the model into the tail, which is exactly Experiment 01's failure mode with extra steps.
3. **Even the project's own post-nucleus router barely leaves the head (0.997).** At τ=0.5/x=0.4 the "boundary token" is only rank ~3–8 in the raw distribution, because a 135M model's nucleus at x=0.4–0.6 already covers only 2–5 tokens. Post-nucleus sampling escapes the *nucleus*, not the *top-10 head*. This revises the interpretation of Experiments 01–03 and motivates **E12** (head-conformity floor).
3. **Plagiarism and recall are the same axis.** Greedy has the best entity recall (0.42) *because* it copies 80 % of its 4-grams from the source. Any metric that rewards recall without penalising copy is measuring the copy.
4. **Hybrid routing is the only decoder that trades along the frontier instead of collapsing**: −54 % copy vs greedy while keeping moderate recall and the highest burstiness (0.53) — but it still sits at GLTR 1.000, so its "diversity" is prompt-following, not head escape.
5. **Boilerplate is universal and model-level, not decoder-level.** Every condition emits ~2 markers per 100 words; the raw samples all open with *"Here's a rewritten version of the sentence:"*. A 135M Instruct model has "assistant preamble" burned into its chat tuning. No decoder fixes that — it has to be banned or trained out.
6. **Typical decoding is a no-op at this scale.** Typical(0.9) and Top-p(0.9) at T=0.8 keep *identical* token sets (Jaccard = 1.0 at 4/4 probe positions; kept-set size 1–7 tokens). With a head that narrow the two criteria coincide. Reported as a finding, not a bug.

## 5. Gaps & how to close them

| Gap | Evidence | Fix / next action | Cost |
|---|---|---|---|
| **No head-escape decoder that keeps facts** | GLTR 0.98–1.00 vs human 0.80, recall collapses when forced out | Two-part router: force-copy protected spans (capitalised tokens, numbers, dates) and post-nucleus-sample everything else. Entity tokens are ~15 % of positions, so the head gap has room to move. → carried into **E11 (entity-pinned head escape)** | 1 run |
| **Boilerplate is a chat-template artefact** | every sample starts "Here's a rewritten…" | `BanPhrases` processor already implemented — ban `Here is`, `Here's`, `Below is`, `Sure,` at step 0 and require the first token to be a capitalised word. Measure boil/100w drop. | 10 min |
| **Metrics have no semantics** | entity recall is substring matching, no paraphrase/semantic check | Add a 22 MB `all-MiniLM-L6-v2` embedding cosine + a tiny NLI contradiction check. Both fit in 4 GB alongside SmolLM. | ~1 h |
| **12 docs × 3 seeds = wide CIs** | ±0.1 swings on copy between seeds | 50-doc corpus + 5 seeds + bootstrap CIs (the harness already batches). | 15 min compute |
| **Only one model size** | 135M may be uniquely degenerate | Scaling curve: SmolLM2-360M-Instruct, Qwen2.5-0.5B-Instruct (both fit fp16 in 4 GB). Same script, one flag. | 30 min |
| **No human judgement** | all metrics are proxies | 3-rater preference on 60 paired samples. | manual |

## 6. What this unblocks (cross-experiment)

* **E05 (window attention)** must be judged on `gltr_top10`, not Distinct-n — E10 shows Distinct-n is saturated (0.93–0.99) and cannot separate decoders.
* **E06 (steering)** gets a target: push GLTR top-10 from 0.99 → 0.80 *without* losing entity recall. Steering acts on the residual stream, so it can shift the head shape rather than truncate it.
* **E07 (recursion)** should use greedy as the decay baseline because greedy is the highest-recall decoder here, i.e. the strongest starting point to watch degrade.
* **E11** (new, spawned by this experiment) = entity-pinned head escape + phrase banning: combine the protected-span idea with `BanPhrases`.

## 7. Files

```
data/dataset.json                     # frozen copy of the E03 benchmark
scripts/run_decoder_shootout.py       # full reproduction (85 s)
outputs/decoder_shootout_raw.csv      # 360 rows, every metric per generation
outputs/decoder_shootout_summary.{csv,md}
outputs/samples_all_conditions.json   # 120 full texts for qualitative reading
outputs/human_gltr_baseline.json      # the human reference profile
outputs/decoder_shootout.png
outputs/env.json                      # device + versions + git commit
```
