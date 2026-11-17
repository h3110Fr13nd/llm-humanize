# Experiment 01: Pure Inverted Nucleus Sampling & Rewriting Breakdown

## Overview
Investigates the behavior of deterministically selecting the **very first token immediately outside the nucleus boundary** (the first rejected token $x_{k^* + 1}$ where cumulative probability mass exceeds $p$).

---

## Structure
* **`data/`**:
  * `prompts.json`: 3 evaluation prompts (Factual, Open-ended, Technical).
  * `source_text_1k.txt`: 876-word historical article on space telescopes.
  * `key_entities.json`: 33 ground-truth factual entities.
* **`scripts/`**:
  * `run_experiment.py`: Standalone CLI execution script.
* **`notebooks/`**:
  * `post_top_p_exploration.ipynb`: Self-contained, pre-executed notebook with live outputs.
* **`outputs/`**:
  * `step1_distribution.png`: Step 1 probability distribution comparing greedy top-1 vs. rank #58 boundary token.
  * `short_prompts_results.csv`: Metrics across greedy, standard nucleus, and post-top-p cutoffs.
  * `perplexity_and_repetition.png`: Log-scale perplexity explosion and Distinct-1 diversity charts.
  * `rewrite_breakdown_results.csv`: 10 decoding strategies on 1,000-word rewriting context.
  * `rewrite_failure_boundaries.png`: Factual entity recall collapse and token rank drift curves ($t=1\dots 35$).

---

## Key Conclusions
1. **Loop Avoidance**: Completely eliminates greedy repetitive loops ($Distinct\text{-}1 \ge 0.82$, $Distinct\text{-}2 = 1.00$).
2. **Exponential Perplexity Explosion**: $\text{PPL}$ jumps from $\approx 2$ (greedy) to $>60,000$ at $p \ge 0.95$.
3. **Factual Rewriting Breakdown**:
   * **Factual unreliability ($p^* \ge 0.10$)**: Because proper nouns have $>50\%$ mass, skipping the nucleus forces hallucinating fictional entities.
   * **Total non-usable collapse ($p^* \ge 0.85 - 0.90$)**: Boundary lands beyond rank #100, generating unreadable token soup ($\text{PPL} > 15,000$).
