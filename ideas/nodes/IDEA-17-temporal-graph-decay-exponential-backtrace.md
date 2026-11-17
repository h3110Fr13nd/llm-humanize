# IDEA-17: Temporal Relevancy & Exponential Back-Trace Decay in Graph Memory

**ID**: `IDEA-17`  
**Status**: `brainstorm / active`  
**Cluster**: `#temporal-memory` `#graph-relevance` `#recommendation-systems`  
**Related Nodes**: [[IDEA-15-conversational-latency-temporal-reward]], [[IDEA-07-position-free-concept-representations]]  
**Target Sub-Experiment**: `experiments/12_attention_forensics_and_temporal_memory/12b_temporal_backtrace_memory/`

---

## 1. 📝 Raw User Voice (Verbatim)

> `[04:41, 12/11/2026] Researcher: Temporal relevancy of nodes. For recommendation systems.
Data for exponential back trace time specificness of time. Specific news of time range based on context. Etc.`

---

## 2. 🧠 Descriptive Engineering Formulation

### A. The Problem in Static Memory & Recommendation Graphs
Standard knowledge graphs and vector databases (RAG) calculate semantic similarity without intrinsic temporal awareness:
* A node representing an event or recommendation from 3 years ago has the exact same dot-product score as one from yesterday if their text descriptions match.
* In reality, information relevancy decays exponentially over time, but is subject to **selective temporal reactivation** when the context explicitly triggers an earlier historical era.

### B. Mathematical Model: Exponential Decay with Contextual Temporal Back-Trace

```
Relevance(Node i, Current Time t, Context c)
                    ┌── Default Recency Decay: exp(-λ · (t - t_i))
                    ▼
Relevance Curve  ────────────\
                              \
                               \      Context mentions: "Recall the 2022 policy changes"
                                \           ▲
                                 \_________/ \────────  (Temporal Back-Trace Spike)
```

1. **Base Exponential Temporal Decay**:
   $$\mathcal{W}_{\text{base}}(i, t) = \exp(-\lambda \cdot (t - t_i))$$
   where $t_i$ is the timestamp of node $i$ and $\lambda$ is the natural forgetting rate.
2. **Context-Specific Exponential Back-Trace**:
   When the dialogue or query context $c$ contains a temporal anchor $t_{\text{target}}$ (e.g., *"news from last summer"*, *"in Q3 2023"*):
   $$\mathcal{W}_{\text{backtrace}}(i, t_{\text{target}}) = \exp\left(-\gamma \cdot |t_i - t_{\text{target}}|\right)$$
3. **Composite Node Activation**:
   $$\text{Score}(i) = \text{SemanticSimilarity}(\mathbf{q}, \mathbf{e}_i) \times \left[ \alpha \cdot \mathcal{W}_{\text{base}}(i, t) + (1 - \alpha) \cdot \mathcal{W}_{\text{backtrace}}(i, t_{\text{target}}) \right]$$
   This guarantees that recommendations and retrieved memories naturally prioritize fresh recent context, yet instantly pivot backward in time with mathematical precision when historical context is queried.
