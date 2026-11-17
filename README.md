# 🧬 Humanize-Experiments: Adaptive Decoding & Non-Autoregressive Paradigms for Text Humanization

[![GitHub](https://img.shields.io/badge/Repository-Private-blue?logo=github)](https://github.com/h3110Fr13nd/humanize-experiments)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Project Mission & Core Goals

Current Large Language Models (LLMs) suffer from two distinct failure modes when asked to rewrite, summarize, or humanize text:
1. **The Plagiarism & Boilerplate Trap (Top-1 / Standard Nucleus)**: Standard decoding strategies (Greedy, Beam Search, Top-$P \ge 0.9$) sample strictly from the high-probability probability head ($>85\%$ top-10 tokens). On rewrite tasks, smaller models default to **verbatim word-for-word copy-pasting** from the prompt, accompanied by robotic AI boilerplate (*"Here's a rewritten summary of the article..."*).
2. **The Hallucination & Error Cascade Trap (Unconstrained Tail Sampling)**: Attempting to inject diversity by raw sampling or inverted boundaries forces the model into the uncalibrated tail of the vocabulary, hallucinating fictional entities and rapidly degrading syntax.

### The Objective
**Humanize-Experiments** is a rigorous research pipeline dedicated to developing, benchmarking, and tuning novel decoding algorithms and model architectures that **humanize text**—maximizing human-like stylistic burstiness, natural phrasing, and synthetic variance while strictly preserving factual anchors (names, dates, entities, technical relationships).

```
                                  [Source Context (1,000+ Words)]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
     [Standard Top-P / Greedy]                                      [Tuned Hybrid Decoder]
  - Verbatim text copy-pasting                                   - Genuine conceptual paraphrasing
  - "Here is a rewritten summary..."                             - Zero robotic boilerplate
  - Flat, robotic perplexity                                     - Humanized surprisal & burstiness
  - 100% Top-10 head conformity                                  - Confident anchors + creative boundaries
```

---

## 🧪 Completed Research Phases

All experimental data, scripts, runnable Jupyter notebooks, and output artifacts are strictly isolated in [`experiments/`](experiments/):

### [Phase 01: Pure Inverted Nucleus Sampling](experiments/01_pure_post_top_p_boundary/)
* **Hypothesis**: Forcing the decoder to pick the first rejected token outside the Top-$P$ nucleus at every step will break repetition loops and inject lateral creativity.
* **Findings**:
  * Completely eliminates repetitive n-gram loops ($Distinct\text{-}1 \ge 0.82$, $Distinct\text{-}2 = 1.00$).
  * On dense 1,000-word rewrites, factual fidelity collapses immediately ($p \ge 0.10$) because proper nouns command $>50\%$ mass; bypassing the nucleus forces hallucinating fictional entities.
  * At $p \ge 0.85$, perplexity spikes exponentially past $15,000$, resulting in token soup.

### [Phase 02: Confidence-Gated Hybrid Routing](experiments/02_hybrid_adaptive_decoding/)
* **Hypothesis**: Gating between greedy selection ($P(\text{top-1}) \ge \tau$) and post-$x$ boundary selection ($P(\text{top-1}) < \tau$) protects factual anchors while humanizing connective prose.
* **Findings**:
  * Mapped a $4 \times 4$ parameter grid ($\tau \in [0.45, 0.60] \times x \in [0.30, 0.60]$).
  * Identified that high-confidence steps ($P \ge \tau$) are syntax pivots and named entities, whereas low-confidence steps ($P < \tau$) are stylistic transitions where human writers exercise variety.

### [Phase 03: 12-Domain Sweet Spot Tuning & Full-Text Comparisons](experiments/03_sweet_spot_tuning_and_comparison/)
* **Methodology**: Scaled evaluation across 12 diverse academic, technical, and historical domains (Biotechnology, Quantum Computing, History, AI Architecture, Deep Space, Geology, Aerospace, etc.).
* **The Empirical Sweet Spot**: Converged to **$\tau^* = 0.50,\; x^* = 0.40$**.

| Decoding Strategy | Generation PPL | Entity Recall | Greedy Anchor Usage | Distinct-1 Diversity | Behavioral Characteristic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Top-1 (Greedy)** | $1.36$ | $27.9\%$ | $100.0\%$ | $0.76$ | Verbatim prompt plagiarism; AI boilerplate |
| **Standard Top-P ($0.90$)** | $1.41$ | $28.4\%$ | N/A | $0.78$ | Cliché transitions; copies source phrasing |
| **Pure Post-Top-P ($0.50$)** | $19.46$ | $3.8\%$ | $0.0\%$ | $0.84$ | Severe hallucination; made-up entities |
| **Tuned Hybrid ($\tau=0.50, x=0.40$)** | **$1.88$** | **$16.7\%$** | **$78.5\%$** | **$0.76$** | **Humanized synthesis; zero boilerplate** |

*Side-by-side full text comparisons are available in [`experiments/03_sweet_spot_tuning_and_comparison/outputs/side_by_side_comparisons/`](experiments/03_sweet_spot_tuning_and_comparison/outputs/side_by_side_comparisons/).*

---

## 🔮 Active & Upcoming Research Tracks

We are actively expanding beyond autoregressive post-top-p sampling into architectural and representation-level text humanization:

| Experiment Track | Folder | Core Architectural Mechanism | Primary Humanization Benefit |
| :--- | :--- | :--- | :--- |
| **04. Global Text Diffusion** | [`04_text_diffusion_global_rewrite/`](experiments/04_text_diffusion_global_rewrite/) | Non-autoregressive discrete diffusion (Diffusion-Gemma / LLaDA) on an in-place canvas. | Eliminates exposure bias and cumulative drift; adapts all sentence choices globally. |
| **05. Speculative Window Attention** | [`05_speculative_window_attention/`](experiments/05_speculative_window_attention/) | Asymmetric dual attention: sliding-window self-attention ($W=128$) + 100% full uncompressed source cross-attention ($\alpha > 1$). | Prevents past generated tokens from dominating attention heads; kills repetitive syntactic echoes. |
| **06. Cross-Text Tone Transfer** | [`06_cross_text_tone_style_transfer/`](experiments/06_cross_text_tone_style_transfer/) | Latent tone steering vectors (Representation Engineering / AdaIN in residual streams). | Transfers rhythm, cadence, and vocabulary from reference texts (e.g., Hemingway, academic) onto content rewrites. |
| **07. Recursive Context Decay** | [`07_recursive_humanization_context_decay/`](experiments/07_recursive_humanization_context_decay/) | Multi-turn autophagic loop ($T_0 \to T_1 \dots \to T_N$) tracking context atrophy, slop index, and entity half-life. | Identifies which humanization model avoids collapsing into AI slop and reliably preserves core facts across $N$ repeated rewrites. |
| **08. Position-Free Thought Streams** | [`08_position_free_thought_streams/`](experiments/08_position_free_thought_streams/) | Decoupled generation: Position-free concept encoding (`08a`), temporal JEPA scene streams (`08b`), and diffusion syntax linearizing (`08c`). | Bypasses sequential word token bottlenecks; models the human mental imagery stream before enforcing surface grammar. |
| **09. RL Post-Training & Persona Modeling** | [`09_rl_post_training_humanization/`](experiments/09_rl_post_training_humanization/) | Weight-level alignment via DPO distillation (`09a`), anti-detector RL (`09b`), and personal conversational transcript cloning (`09c`). | Bypasses test-time inference overhead; embeds natural burstiness and authentic colloquial human voice directly into weights. |
| **14. GNNN Modular Cyclic Architecture** | [`14_gnnn_modular_cyclic_architecture/`](experiments/14_gnnn_modular_cyclic_architecture/) | Neuromorphic Graph Neural Node Network where nodes are autonomous sub-networks with cyclic recurrence, neurogenesis, and dynamic pruning. | Brain-like continual learning without catastrophic forgetting; decouples cyclic reasoning loops from surface syntax decoding. |

---

## 🕸️ Zettelkasten Thought Graph

All research hypotheses, cognitive intuitions, and experimental links are mapped into an interconnected graph topology in [`ideas/GRAPH.md`](ideas/GRAPH.md). Atomic concept nodes live under [`ideas/nodes/`](ideas/nodes/).

---

## 📚 Research Foundations & PDF Library

A comprehensive research bibliography supporting this project is cataloged in [`research/references.md`](research/references.md). Key research literature and citations:

1. **GLTR: Statistical Detection and Visualization of Generated Text** (Gehrmann et al., ACL 2019) — *Demonstrates that humans systematically sample outside the top-10 head, whereas LLMs are trapped in it.*
2. **Typical Decoding for Natural Language Generation** (Meister et al., TACL 2023) — *Proves that human text aligns with Uniform Information Density, explicitly truncating the head.*
3. **Mirostat: A Perplexity-Controlled Neural Text Generation Algorithm** (Basu et al., ICLR 2021) — *Formulates the "Boredom Trap" vs. "Confusion Trap" in generative perplexity.*
4. **The Curious Case of Neural Text Degeneration** (Holtzman et al., ICLR 2020) — *Foundational study on nucleus sampling and the unreliable probability tail.*
5. **Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs** (Nguyen et al., ICLR 2025) — *Dynamically scales candidate filtering based on top-1 probability.*
6. **DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature** (Mitchell et al., ICML 2023) — *Shows that LLM text resides on local probability peaks, while human text does not.*
7. **Large Language Models can be Guided to Evade AI-Generated Text Detection** (Lu et al., TMLR 2023) — *Analyzes perplexity variance and burstiness in evading AI text classifiers.*
8. **Top-H Decoding: Adapting Creativity and Coherence with Bounded Entropy in Text Generation** (Preprint 2025) — *Entropy-budget optimization for adaptive token generation.*

---

## 💻 Quickstart & Reproduction

### Prerequisites
* Linux x86_64 with CUDA support
* Python $\ge 3.11$ managed via [`uv`](https://github.com/astral-sh/uv)

```bash
# Clone the repository
git clone https://github.com/h3110Fr13nd/humanize-experiments.git
cd humanize-experiments

# Sync dependencies using uv
uv sync

# Run Phase 03 sweet-spot tuning benchmark
uv run python experiments/03_sweet_spot_tuning_and_comparison/scripts/run_sweet_spot_benchmark.py
```

### Interactive Notebooks
Every notebook in `experiments/<XX>_<slug>/notebooks/` is 100% self-contained and independently executable:
* Phase 01: [`experiments/01_pure_post_top_p_boundary/notebooks/post_top_p_exploration.ipynb`](experiments/01_pure_post_top_p_boundary/notebooks/post_top_p_exploration.ipynb)
* Phase 02: [`experiments/02_hybrid_adaptive_decoding/notebooks/hybrid_rewrite_matrix.ipynb`](experiments/02_hybrid_adaptive_decoding/notebooks/hybrid_rewrite_matrix.ipynb)
* Phase 03: [`experiments/03_sweet_spot_tuning_and_comparison/notebooks/sweet_spot_tuning_and_comparison.ipynb`](experiments/03_sweet_spot_tuning_and_comparison/notebooks/sweet_spot_tuning_and_comparison.ipynb)

---

## 📜 Repository Standards
All future experiments adhere to the reproducible research standards established across the benchmark suites.
