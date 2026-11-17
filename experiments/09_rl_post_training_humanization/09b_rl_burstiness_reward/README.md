# Sub-Experiment 09b: Reinforcement Learning with Burstiness & Anti-Detector Rewards

**Parent Experiment**: [`../README.md`](../README.md)  
**Thought Node**: [`ideas/nodes/IDEA-10-rl-post-training-humanization.md`](ideas/nodes/IDEA-10-rl-post-training-humanization.md)

---

## 🎯 Research Objective
Train a policy using GRPO / PPO where the reward function explicitly optimizes for human-like burstiness and penalizes the local probability curvature detected by zero-shot AI detectors (e.g., DetectGPT, Binoculars).

## 🧪 Experimental Questions:
1. **Curvature Evasion**: Can policy gradient optimization learn to produce outputs that avoid sitting at local probability maxima without becoming incoherent?
2. **Reward Balance**: What is the optimal Pareto coefficient between the entity preservation reward $\lambda_{\text{fact}}$ and the detector evasion reward $\lambda_{\text{ai}}$?
