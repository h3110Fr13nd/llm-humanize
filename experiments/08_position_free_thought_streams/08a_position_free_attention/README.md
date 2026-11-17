# Sub-Experiment 08a: Position-Free Conceptual Attention & Order Ablation

**Parent Experiment**: [`../README.md`](../README.md)  
**Thought Node**: [`ideas/nodes/IDEA-07-position-free-concept-representations.md`](ideas/nodes/IDEA-07-position-free-concept-representations.md)

---

## 🎯 Research Objective
Evaluate how much positional information is strictly necessary for a model to retain context and key entities.

## 🧪 Experimental Questions:
1. **Positional Ablation**: What happens to entity recall and semantic similarity when rotary/absolute position embeddings are scaled by $\gamma \in [0.0, 1.0]$?
2. **Positional Permutation Resistance**: If input sentences are shuffled at the clause level or token level, can a model with relaxed positional attention still identify core factual triples?
3. **Bag-of-Embeddings Baseline**: Can we represent a 1,000-word source document as an unordered set of concept centroids (via Set Attention Blocks) without losing fine-grained factual relationships?
