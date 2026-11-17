# Sub-Experiment 14b: Neurogenesis & Synaptic Pruning for Continual Learning

**Parent Experiment**: [`../README.md`](../README.md)  
**Thought Node**: [`ideas/nodes/IDEA-19-gnnn-modular-cyclic-graph-network.md`](ideas/nodes/IDEA-19-gnnn-modular-cyclic-graph-network.md)

---

## 🎯 Research Objective
Overcome catastrophic forgetting by implementing biological structural plasticity: dynamically adding neurons/nodes when encountering novel distribution shifts and pruning inactive connections.

## 🧪 Experimental Questions:
1. **Novelty-Triggered Neurogenesis**: Can an empirical loss threshold automatically trigger orthogonal neuron expansion without destabilizing previously learned parameters?
2. **Zero-Forgetting Retention**: Benchmark accuracy on Task A after sequentially training on Tasks B, C, and D with GNNN growth vs. static feedforward baselines.
3. **Dynamic Sparsity Pruning**: Does magnitude-based edge pruning maintain an energy-efficient sparse computation graph across continuous operation?
