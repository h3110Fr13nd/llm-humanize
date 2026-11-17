# Experiment 03: Multi-Domain Tuning & Side-by-Side Comparison

## Overview
Evaluates and fine-tunes the **Hybrid Confidence-Gated Decoding** strategy across a diverse dataset of **12 real-world factual documents** across distinct fields:
Astronomy, Biotechnology, Quantum Physics, History, AI, Space Exploration, Medicine, Geology, Aerospace, Materials Science, Computer Networking, and Biochemistry.

---

## Directory Structure
* **`data/`**:
  * `dataset.json`: 12 comprehensive source documents with domain tags and 15–20 ground-truth tracking entities each.
* **`scripts/`**:
  * `run_sweet_spot_benchmark.py`: Complete CLI benchmark running the 2D parameter sweep and generating all side-by-side files.
* **`notebooks/`**:
  * `sweet_spot_tuning_and_comparison.ipynb`: 100% self-contained, pre-executed interactive notebook with tuning heatmaps and an interactive side-by-side comparison viewer.
* **`outputs/`**:
  * `parameter_tuning_results.csv`: Grid evaluation metrics over all 12 documents.
  * `sweet_spot_tuning_plots.png`: 2-panel heatmaps of average Entity Recall % and Generation Perplexity across the 12 documents.
  * `all_examples_summary.csv`: Summary metrics table for all 12 examples $\times$ 4 strategies.
  * `all_examples_comparison.json`: Complete JSON recording full text generations.
  * `side_by_side_comparisons/`: 12 individual Markdown comparison files:
    * `01_space_telescopes.md`
    * `02_crispr_gene_editing.md`
    * `03_quantum_computing.md`
    * `04_printing_press.md`
    * `05_transformer_architecture.md`
    * `06_voyager_mission.md`
    * `07_penicillin_antibiotics.md`
    * `08_plate_tectonics.md`
    * `09_apollo_guidance_computer.md`
    * `10_roman_concrete.md`
    * `11_arpanet_to_internet.md`
    * `12_calvin_cycle_photosynthesis.md`

---

## Key Conclusions from the 12-Document Benchmark

### 1. The Statistically Tuned Sweet Spot
Across 12 distinct domains, the optimal balance between factual grounding and lexical exploration converged to:
$$\tau^* = 0.50, \quad x^* = 0.40$$
* **Average Entity Recall**: **16.72%** (vs. pure post-top-p's 1.54%).
* **Average Generation Perplexity**: **1.88** (fluent and stable).
* **Average Greedy Trigger Frequency**: **78.5%** (factual grounding is maintained).

### 2. The Surprising Insight on Greedy vs. Hybrid Rewriting
* **Top-1 (Greedy)** achieves higher entity recall primarily by **copying the source prompt verbatim word-for-word**, producing rigid, repetitive text with standard AI opening boilerplate (*"Here's a rewritten summary of the article..."*).
* **Tuned Hybrid** forces the model away from verbatim plagiarism, generating **genuine high-level paraphrasing and natural prose** (*"The breakthrough was a pivotal moment in the history of genetics, revolutionizing the field..."*) while preserving key facts.
