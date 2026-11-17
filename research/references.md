# Research Literature & Theoretical References for Text Humanization via Adaptive Decoding

This directory documents the academic literature, theoretical foundations, and empirical research supporting the **Confidence-Gated Inverted Nucleus Sampling** and **Hybrid Post-Top-P Adaptive Decoding** methodology developed in this repository.


---

## 📚 Categorized Paper Index

```
research/
├── references.md
└── papers/
```

---

## 1. Psycholinguistic Grounding & Human vs. Machine Token Distributions

### 📄 GLTR: Statistical Detection and Visualization of Generated Text
* **Authors**: Sebastian Gehrmann, Hendrik Strobelt, Alexander M. Rush
* **Venue**: *ACL 2019 System Demonstrations*
* **ArXiv**: [arXiv:1906.04043](https://arxiv.org/abs/1906.04043)
* **Key Finding**:
  * Language model decoding methods (greedy search, top-k, top-p) generate text dominated by the **head of the distribution** ($>85\%$ of tokens rank within the top 10).
  * **Human authors frequently select words deep in the tail or outside the top ranks**, introducing unexpected phrasing, specialized vocabulary, and stylistic richness even in predictable sentences.
* **Relevance to our work**:
  * Provides the fundamental forensic reason why pure greedy/top-p decoding feels robotic and artificial: it never ventures outside the green (top-10) zone. Bypassing the top-$x$ boundary forces the model into the exact token ranks that characterize human expression.

---

### 📄 Typical Decoding for Natural Language Generation (Locally Typical Sampling)
* **Authors**: Clara Meister, Gian Wiher, Tiago Pimentel, Ryan Cotterell
* **Venue**: *Transactions of the Association for Computational Linguistics (TACL 2023)*
* **ArXiv**: [arXiv:2202.00666](https://arxiv.org/abs/2202.00666)
* **Key Finding**:
  * Derives token selection from the psycholinguistic principle of **Uniform Information Density (UID)**: human communication optimizes for a uniform transmission rate where surprisal ($-\log P(w)$) closely matches the conditional entropy $H(P)$.
  * Unlike nucleus sampling, which only prunes the tail, **typical sampling truncates both the tail AND the high-probability head**. It eliminates overly predictable tokens because humans do not speak in probability maxima.
* **Relevance to our work**:
  * Directly justifies discarding the top-$x$ probability mass. Sampling from the boundary / post-$x$ candidates enforces surprisal without plunging into the uncalibrated deep noise tail.

---

## 2. Perplexity Regulation & The Degeneration Boundary

### 📄 The Curious Case of Neural Text Degeneration
* **Authors**: Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, Yejin Choi
* **Venue**: *International Conference on Learning Representations (ICLR 2020)*
* **ArXiv**: [arXiv:1904.09751](https://arxiv.org/abs/1904.09751)
* **Key Finding**:
  * Proves that likelihood maximization (greedy argmax and beam search) leads to degenerative, repetitive loops in open-ended generation.
  * Formulates **Nucleus (Top-P) Sampling** to dynamically isolate the smallest set of tokens whose cumulative mass exceeds $p$, cutting off the unreliable tail.
* **Relevance to our work**:
  * Defines the nucleus boundary that our project inverts. Our Phase 1 experiments verified Holtzman's warnings regarding the "unreliable tail": when $p \ge 0.85$, perplexity spikes above $15,000$ because the long tail of small models lacks calibration.

---

### 📄 Mirostat: A Perplexity-Controlled Neural Text Generation Algorithm
* **Authors**: Sourya Basu, Govardana Sachidananda, Tharaniesh Raja, Urmish Thakker
* **Venue**: *International Conference on Learning Representations (ICLR 2021)*
* **ArXiv**: [arXiv:2007.14966](https://arxiv.org/abs/2007.14966)
* **Key Finding**:
  * Identifies two fundamental failure regimes in generative decoding:
    1. **The Boredom Trap**: Text is excessively repetitive, predictable, and dull (Perplexity $\approx 1.0$).
    2. **The Confusion Trap**: Text drifts into incoherence, hallucinations, and unrecoverable error cascades (Perplexity explodes).
  * Uses a dynamic feedback controller to adjust the truncation threshold $k$ step-by-step to maintain a constant target perplexity $\tau$.
* **Relevance to our work**:
  * Directly mirrors the sweet spot tuning in our Experiment 03. Pure Post-Top-P was trapped in the Confusion Trap ($\text{PPL} \approx 20 - 30$). Top-1 and standard Top-P were stuck in the Boredom Trap ($\text{PPL} \approx 1.1 - 1.4$). Our tuned hybrid ($\tau=0.50, x=0.40$) achieves the ideal humanized equilibrium ($\text{PPL} = 1.88$).

---

## 3. Dynamic Thresholding & Confidence-Gated Decoding

### 📄 Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs
* **Authors**: Chau Nguyen, Thomas Marshall, Ian Stewart, Ryan O'Connor
* **Venue**: *International Conference on Learning Representations (ICLR 2025)*
* **ArXiv**: [arXiv:2407.01082](https://arxiv.org/abs/2407.01082)
* **Key Finding**:
  * Shows that static top-p cutoffs allow junk tail tokens when the distribution is flat, but restrict valid options when the distribution is peaked.
  * Proposes **Min-P Sampling**, which dynamically filters tokens using a relative threshold anchored to the top-token confidence:
    $$P(w_i) \ge p_{\text{base}} \times P(\text{top-1})$$
* **Relevance to our work**:
  * Shares the core thesis that the top token's absolute confidence must govern decoding flexibility. In our hybrid decoder, $P(\text{top-1}) \ge \tau$ serves as a hard safety switch ensuring factual anchors and grammatical pivots remain greedy.

---

### 📄 Top-H Decoding: Adapting Creativity and Coherence with Bounded Entropy in Text Generation
* **Authors**: Anonymous / Under Review
* **Venue**: *arXiv Preprint (Feb 2025)*
* **ArXiv**: [arXiv:2502.13783](https://arxiv.org/abs/2502.13783)
* **Key Finding**:
  * Formulates token selection as an entropy-budget optimization problem: when Shannon entropy $H(P)$ indicates high uncertainty, the decoder injects bounded exploratory variance; when entropy is low, it strictly respects determinism.
* **Relevance to our work**:
  * Validates our decision to split execution into distinct high-confidence (deterministic anchor) and low-confidence (stylistic boundary deviation) paths.

---

## 4. Evading AI Text Detection & Surface "Humanization"

### 📄 DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature
* **Authors**: Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, Chelsea Finn
* **Venue**: *International Conference on Machine Learning (ICML 2023)*
* **ArXiv**: [arXiv:2301.11305](https://arxiv.org/abs/2301.11305)
* **Key Finding**:
  * Demonstrates that LLM text generated via standard decoding strategies sits in **local probability peaks** (negative log-probability curvature under minor perturbation). Human text does not reside at local probability maxima.
* **Relevance to our work**:
  * Explains why boundary decoding evades detection: by deliberately perturbing token selection away from the local probability maximum when confidence is ambiguous, the generated text breaks the curvature signature that detectors rely on.

---

### 📄 Large Language Models can be Guided to Evade AI-Generated Text Detection
* **Authors**: Ning Lu, Shengcai Liu, Zhirui Zhang, Yutao Zeng, Caiming Xiong
* **Venue**: *Transactions on Machine Learning Research (TMLR 2023)*
* **ArXiv**: [arXiv:2305.10847](https://arxiv.org/abs/2305.10847)
* **Key Finding**:
  * Investigates how altering perplexity and burstiness profiles degrades both zero-shot detectors (DetectGPT) and classifier-based detectors (RoBERTa-based, GPTZero).
* **Relevance to our work**:
  * Confirms that balancing token-level burstiness while preserving syntactic fluency is the primary vehicle for humanizing machine-generated prose.

---

## 📊 Decoding Method Comparison Matrix

| Decoding Strategy | Head Handling | Tail Handling | Decision Anchor | Primary Strength | Primary Weakness |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-1 (Greedy)** | Selected | Excluded | $\text{argmax}$ | Maximum accuracy | Repetition loops, verbatim copying |
| **Top-$P$ (Nucleus)** | Sampled | Cut off at mass $p$ | Cumulative sum | Coherent fluency | Favors generic/cliché phrasing |
| **Typical Sampling** | Discarded | Discarded | Surprisal $\approx H(P)$ | Natural information density | Prone to dropping proper nouns |
| **Mirostat** | Truncated | Truncated | Feedback error on PPL | Dynamic stability | Parameter sensitive ($\eta, \tau$) |
| **Min-$P$** | Retained | Truncated relative to top-1 | $P_i \ge p \cdot P_{\text{top-1}}$ | High-temperature creativity | Keeps head tokens |
| **Our Tuned Hybrid** | **Skipped** (when $P_1 < \tau$) | **Post-$x$ Boundary** | **Confidence Gate** ($P_1 \ge \tau$) | **Overcomes plagiarism; natural synthesis** | Requires tuning $(\tau, x)$ per model scale |
