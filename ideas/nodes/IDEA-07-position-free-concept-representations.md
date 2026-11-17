# IDEA-07: Position-Free Concept Representations & Cognitive Invariance

**ID**: `IDEA-07`  
**Status**: `brainstorm / active`  
**Cluster**: `#cognitive-architecture` `#position-free` `#representation-theory`  
**Related Nodes**: [[IDEA-08-temporal-mental-stream-jepa]], [[IDEA-09-multi-stage-decoupled-syntax-linearizer]], [[IDEA-03-full-canvas-text-diffusion]]  
**Target Experiment**: [`experiments/08_position_free_thought_streams/`](experiments/08_position_free_thought_streams/)

---

## 1. 📝 Raw User Voice (Verbatim)

> *"add one  more like moving away from positional embeddings. how we can encode or have same context without or with least positional information. like human not necessarily stores the positional things in mind, only when they want to be eloquent or deterministic they learn or mug things otherwise they do mess up with position and later come up with like no before that this happened and after that  etc. the position only matters to preserve grammar near correctness, to make sense  of jumbled words. these thing could have it's own subset of experiments especially. with diffusion model."*

---

## 2. 🧠 Descriptive Engineering & Cognitive Formulation

### A. The Cognitive Reality vs. Modern Transformers
* **The Transformer Assumption**: Standard architectures (using RoPE, NoPE, or learned sinusoidal embeddings) inject a strict coordinate $i \in [1, N]$ into every token vector before the first self-attention layer. The model's representation of `"Einstein was born in Ulm in 1879"` is completely different if the sentence is transposed to `"In 1879, Einstein was born in Ulm"`.
* **The Human Cognitive Reality**: Human working memory does **not** store sequential index integers for facts. A person comprehending a story extracts an unordered conceptual proposition:
  $$\mathcal{M} = \{\text{Entity: Einstein}, \text{Event: Born}, \text{Location: Ulm}, \text{Time: 1879}\}$$
  Positional ordering is only instantiated at the final output boundary to satisfy linear acoustic/phonological constraints (mouths can only speak one word at a time) and local grammatical syntax. When humans recall events casually, they frequently retrieve concepts out of order (*"Oh wait, before that happened, X actually occurred"*).

### B. Architectural Research Questions
1. **How much positional information is actually required for semantic understanding?**
   - Can a transformer encoder process documents with **Zero Positional Embeddings (pure permutation-invariant self-attention)** and still preserve 95%+ of entity relationships?
2. **Decoupling Semantics from Word Order**:
   - Represent a document as a set of unordered semantic vectors (Set Transformer / DeepSets).
   - Position is only injected at the final linear decoding stage.
