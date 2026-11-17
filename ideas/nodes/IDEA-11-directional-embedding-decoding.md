# IDEA-11: Directional Embedding Vector Decoding

**ID**: `IDEA-11`  
**Status**: `brainstorm / active`  
**Cluster**: `#directional-decoding` `#vector-steering` `#sampling`  
**Related Nodes**: [[IDEA-01-inverted-nucleus-boundary]], [[IDEA-02-confidence-gated-hybrid]], [[IDEA-05-latent-tone-style-transfer]]  
**Target Sub-Experiment**: `experiments/10_directional_and_distillation_decoding/10a_directional_token_steering/`

---

## 1. 📝 Raw User Voice (Verbatim)

> `[03:38, 5/11/2026] Researcher: Directional decoding tokens. Like tokens must have embeddings aligned or closer to the direction we need it to be. While still being relevant and continuing to text.`

---

## 2. 🧠 Descriptive Engineering Formulation

### A. The Core Concept: Geometric Logit Re-Weighting via Embedding Angles
Standard decoding methods (Greedy, Top-P, Temperature) treat the vocabulary as an unorganized set of categorical indices, ranking tokens solely by scalar logits $z_i \in \mathbb{R}$.

**Directional Decoding** projects candidate tokens into the semantic vector space of the embedding matrix $E \in \mathbb{R}^{|V| \times d}$:

```
                              Target Steering Vector v_target (e.g. informal, technical, concise)
                                            ▲
                                           / 
                                          /  θ (Cosine Similarity)
                                         /
         Origin ───────────────────────> e(w_i) Candidate Token Embedding
```

### B. Mathematical Mechanics
Given candidate token $w_i$ with embedding vector $\mathbf{e}(w_i)$ and a target steering vector $\vec{v}_{\text{target}}$:
1. **Semantic Alignment Score**:
   $$\text{Sim}(w_i, \vec{v}_{\text{target}}) = \frac{\mathbf{e}(w_i) \cdot \vec{v}_{\text{target}}}{\|\mathbf{e}(w_i)\| \|\vec{v}_{\text{target}}\|}$$
2. **Directional Logit Formulation**:
   Adjust the raw model logit $z_i$ by a directional projection bonus:
   $$\tilde{z}_i = z_i + \beta \cdot \text{Sim}(w_i, \vec{v}_{\text{target}})$$
   where $\beta \ge 0$ is the directional intensity parameter.
3. **Relevance Filter Constraint**:
   To ensure the text remains grammatically coherent and contextually relevant, directional re-weighting is applied **only within a candidate pool** (e.g., within top-$K$ or post-$x$ boundary tokens), preventing catastrophic out-of-context injection.
