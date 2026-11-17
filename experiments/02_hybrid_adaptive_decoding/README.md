# Experiment 02: Confidence-Gated Hybrid Decoding

## Overview
Introduces and validates a **Dynamic Token Routing** mechanism to solve the factual collapse observed in Experiment 01:

$$w_t = \begin{cases} 
\text{argmax}_{w} P(w \mid x_{<t}) & \text{if } \max_w P(w \mid x_{<t}) \ge \tau \quad \textbf{(Factual Anchor)} \\ 
\text{First token outside } x\text{-nucleus} & \text{if } \max_w P(w \mid x_{<t}) < \tau \quad \textbf{(Exploration / Anti-Loop)} 
\end{cases}$$

---

## Structure
* **`data/`**:
  * `source_text_1k.txt`: 876-word historical space telescope text.
  * `key_entities.json`: 33 ground-truth factual entities.
* **`scripts/`**:
  * `run_hybrid_grid.py`: Standalone CLI execution script.
* **`notebooks/`**:
  * `hybrid_rewrite_matrix.ipynb`: Self-contained, pre-executed notebook with live outputs.
* **`outputs/`**:
  * `baselines_comparison.csv`: Pure greedy, standard nucleus, and pure post-top-p baselines.
  * `hybrid_grid_results.csv`: Full 16-point matrix sweep across $\tau \in [0.25, 0.70] \times x \in [0.30, 0.90]$.
  * `parameter_matrix_heatmaps.png`: 3-panel heatmaps of Entity Recall %, Generation Perplexity, and Greedy Anchor Trigger Frequency %.
  * `microscopic_trace.csv`: Token-by-token routing log for the optimal configuration ($\tau=0.55, x=0.50$).

---

## Key Conclusions
1. **Factual Grounding Breakthrough**: At $\tau=0.55, x=0.50$, Entity Recall reached **9.38%**, surpassing both Pure Greedy (6.25%) and Standard Top-P (6.25%).
2. **Fluency & Perplexity**: Preserved low generation perplexity ($\text{PPL} = 2.74$).
3. **Lexical Diversity**: Reached a Distinct-1 ratio of **0.73**, avoiding repetitive loops.
4. **Humanized Tone**: Bypassed canned AI boilerplate opening phrases (*"Here's a summary of the key points:"*) in favor of natural conversational explanations (*"Please allow me to explain Hubble and JWST. Let's break down what they are and their differences..."*).
