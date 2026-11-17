# Experiment 14: Graph Neural Node Network (GNNN) — Modular Cyclic Brain Architecture

## 1. 📝 Raw User Idea (Verbatim Transcription)

> *"an experiment on building the decoupled GNNN (Graph neural node network)  kind model, where each node is a small neural network arch generalised but specific for some things (probably kind of moe but in graph there can be cycles so we actually wanna allow that to see how it works out, with one of the nodes can be the decoder as mentioned in previous ideas. this is to create a human brain like model for intelligence and the idea of increasing the numberof neurons in nodes as i mentioned in one of previus ideas can be utilised here to grow the nodes or parts of brain and probably have a pruner somewhere in the process, this is probably one of those continual learning pardigm."*

---

## 2. 🧠 Synthesized & Descriptive Engineering Formulation

### The Neuromorphic Shift: From Monolithic Layers to Cyclic Neural Node Graphs
Instead of an identical stack of $L$ feedforward transformer layers, GNNN models intelligence as a **directed cyclic graph of autonomous neural sub-networks**:
* **Autonomous Neural Nodes**: Each node $v_i$ is a mini-network $f_{\theta_i}$ with specialized functional roles.
* **Cyclic Re-entrant Loops**: Allows recurrent feedback loops ($v_i \to v_j \to v_i$) where internal thoughts iterate until dynamic equilibrium before emitting an action.
* **Continual Learning via Structural Plasticity**: Dynamically expands neurons/nodes during novel tasks (neurogenesis) and prunes redundant connections to prevent catastrophic forgetting.
* **Surface Decoder Node**: A specialized node in the graph dedicated strictly to linearizing internal conceptual states into humanized grammar (connecting to [[IDEA-09]]).

---

## 📂 Sub-Experimentation Tracks

| Track | Directory | Objective & Methodology | Deliverables |
| :--- | :--- | :--- | :--- |
| **14a** | [`14a_cyclic_message_passing_dynamics/`](./14a_cyclic_message_passing_dynamics/) | **Cyclic Convergence**: Build a minimal cyclic graph of PyTorch sub-networks; evaluate dynamical settling stability vs. chaotic oscillation. | Settling curve plots; contraction mapping proof-of-concept scripts. |
| **14b** | [`14b_neurogenesis_and_pruning_continual_learning/`](./14b_neurogenesis_and_pruning_continual_learning/) | **Continual Learning**: Test sequential multi-domain tasks with dynamic neuron growth (neurogenesis) and magnitude-based edge pruning. | Catastrophic forgetting benchmarks; parameter count vs. task accuracy curves. |
| **14c** | [`14c_decoupled_decoder_node_synthesis/`](./14c_decoupled_decoder_node_synthesis/) | **Surface Realization**: Route the stabilized equilibrium states of the cyclic concept graph into a designated syntax linearizer node. | End-to-end concept-to-text generation traces; grammatical fluency vs. settling steps. |

---

## 🕸️ Zettelkasten Knowledge Graph Links
* **Thought Node**: [`ideas/nodes/IDEA-19-gnnn-modular-cyclic-graph-network.md`](ideas/nodes/IDEA-19-gnnn-modular-cyclic-graph-network.md)
* **Master Topology**: [`ideas/GRAPH.md`](ideas/GRAPH.md)
