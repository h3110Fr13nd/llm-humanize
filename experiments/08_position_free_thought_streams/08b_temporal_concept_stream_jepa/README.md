# Sub-Experiment 08b: Temporal Concept Stream Predictor (JEPA)

**Parent Experiment**: [`../README.md`](../README.md)  
**Thought Node**: [`ideas/nodes/IDEA-08-temporal-mental-stream-jepa.md`](ideas/nodes/IDEA-08-temporal-mental-stream-jepa.md)

---

## 🎯 Research Objective
Model the human stream of mental imagery using a Joint Embedding Predictive Architecture (JEPA), generating temporal sequences of latent concept vectors rather than discrete subword tokens.

## 🧪 Experimental Questions:
1. **Latent Trajectory Modeling**: Train/test a latent predictor that projects transitions between concept states:
   $$\mathbf{z}_{t+1} = \text{Predictor}(\mathbf{z}_t, \mathbf{c})$$
   (e.g., $[E_{\text{cat}}] \to [E_{\text{sitting}}] \to [E_{\text{mat}}]$) directly in embedding space.
2. **Representation Stability**: Does generating in abstract latent space prevent the lexical collapse and repetition traps common to autoregressive token decoders?
3. **Cross-Lingual & Stylistic Invariance**: Can the same generated latent concept stream condition multiple surface language realizations (formal academic vs. casual conversational) without drift?
