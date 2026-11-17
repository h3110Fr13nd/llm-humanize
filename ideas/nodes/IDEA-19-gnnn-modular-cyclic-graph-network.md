# IDEA-19: Graph Neural Node Network (GNNN) — Modular Cyclic Brain Architecture with Neurogenesis & Pruning

**ID**: `IDEA-19`  
**Status**: `brainstorm / active`  
**Cluster**: `#neuromorphic` `#modular-networks` `#continual-learning` `#gnnn` `#cyclic-graphs`  
**Related Nodes**: [[IDEA-07-position-free-concept-representations]], [[IDEA-08-temporal-mental-stream-jepa]], [[IDEA-09-multi-stage-decoupled-syntax-linearizer]], [[IDEA-18-autonomous-idea-generator-stepping-stones]]  
**Target Experiment**: [`experiments/14_gnnn_modular_cyclic_architecture/`](experiments/14_gnnn_modular_cyclic_architecture/)

---

## 1. 📝 Raw User Voice (Verbatim)

> *"an experiment on building the decoupled GNNN (Graph neural node network)  kind model, where each node is a small neural network arch generalised but specific for some things (probably kind of moe but in graph there can be cycles so we actually wanna allow that to see how it works out, with one of the nodes can be the decoder as mentioned in previous ideas. this is to create a human brain like model for intelligence and the idea of increasing the numberof neurons in nodes as i mentioned in one of previus ideas can be utilised here to grow the nodes or parts of brain and probably have a pruner somewhere in the process, this is probably one of those continual learning pardigm."*

---

## 2. 🧠 Descriptive Engineering & Neuromorphic Formulation

### A. The Core Paradigm Shift: From Vector GNNs to Neural Node Graphs
* **Traditional GNNs**: Nodes are passive feature vectors $\mathbf{h}_v \in \mathbb{R}^d$; edges are fixed adjacency matrices; message passing updates vector states via a single shared weight matrix.
* **Traditional Mixture-of-Experts (MoE)**: Experts are isolated feedforward blocks in a flat feedforward layer; a top-$k$ gating router selects 2 out of $E$ experts per token. No cycles, no memory, strictly feedforward.
* **The GNNN (Graph Neural Node Network)**:
  * **Every node $v_i$ is an autonomous neural sub-network $f_{\theta_i}$** with its own internal parameters and specialized functional capacity (e.g., temporal relation module, entity memory module, concept reasoning module).
  * **The graph topology allows directed cycles ($v_i \to v_j \to v_k \to v_i$)**, enabling iterative re-entrant processing and internal mental simulation loops before emitting a surface response.
  * **Decoupled Decoder Node**: One specialized node in the graph acts as the surface syntactic realization engine (connecting directly to [[IDEA-09-multi-stage-decoupled-syntax-linearizer]]).

---

### B. Architectural Topology & Cyclic Message Passing

```
             ┌────────────────────────────────────────────────────────┐
             │            CYCLIC CORTICAL CONCEPT GRAPH               │
             │                                                        │
             │   ┌────────────────┐      Message M_ij   ┌───────────────┐   │
             │   │ Node 1:        │ ──────────────────> │ Node 2:       │   │
             │   │ Temporal Actor │ <────────────────── │ Relational    │   │
             │   │ Sub-Net f_θ1   │     Recurrent Loop  │ Sub-Net f_θ2  │   │
             │   └────────────────┘                     └───────────────┘   │
             │           │                                      │           │
             │           │ Feedback               Feedforward   │           │
             │           ▼                                      ▼           │
             │   ┌────────────────┐                     ┌───────────────┐   │
             │   │ Node 3:        │ <────────────────── │ Node 4:       │   │
             │   │ Episodic Anchor│ ──────────────────> │ Causal Checker│   │
             │   │ Sub-Net f_θ3   │      Cycle          │ Sub-Net f_θ4  │   │
             │   └────────────────┘                     └───────────────┘   │
             └────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                     ┌──────────────────────────────────────┐
                     │ Specialized Node K: Decoder Node     │
                     │ (Diffusion Syntax Linearizer)        │
                     └──────────────────────────────────────┘
                                         │
                                         ▼
                             [Linear Human Output]
```

#### Cyclic Message Formulation:
At recurrent iteration $\tau \in [1, \mathcal{T}_{\text{settle}}]$:
$$\mathbf{m}_{j \to i}^{(\tau)} = \text{MessageNet}_{ji}\left(\mathbf{h}_j^{(\tau-1)}, \mathbf{e}_{ji}\right)$$
$$\mathbf{h}_i^{(\tau)} = f_{\theta_i}\left(\mathbf{h}_i^{(\tau-1)}, \sum_{j \in \mathcal{N}_{\text{in}}(i)} \mathbf{m}_{j \to i}^{(\tau)}\right)$$
Processing continues until graph dynamical states reach an equilibrium: $\|\mathbf{h}^{(\tau)} - \mathbf{h}^{(\tau-1)}\| < \epsilon$, allowing the system to "think" in iterative loops.

---

### C. Continual Learning: Neurogenesis & Synaptic Pruning

Standard models suffer from **catastrophic forgetting** because new gradient updates overwrite existing orthogonal weights. GNNN addresses the *Plasticity-Stability Dilemma* via biological structural plasticity:

1. **Dynamic Node & Neuron Expansion (Neurogenesis)**:
   * When a new domain or novel concept produces a persistently high reconstruction error across the current graph:
     $$\mathcal{E}_{\text{novelty}} = \|\mathbf{x} - \hat{\mathbf{x}}\| > \delta_{\text{expand}}$$
   * The network triggers **neurogenesis**:
     - *Local Growth*: Expands the hidden dimension of relevant node sub-networks ($d_i \leftarrow d_i + \Delta d$), initializing new weights orthogonally.
     - *Modular Growth*: Spawns a brand-new specialized node $v_{\text{new}}$ with dynamic learnable routing edges connected to the active subgraph.
2. **Dynamic Synaptic Pruning**:
   * Nodes and inter-node channels that register low gradient activity or redundant informational mutual information over time are pruned:
     $$\text{Prune edge } e_{ij} \text{ if } \frac{1}{T} \sum_t |\mathbf{m}_{j \to i}^{(t)}| < \tau_{\text{prune}}$$
   * Prevents runaway parameter bloat and keeps the active computation graph sparse and energy-efficient.
