# IDEA-10: RL Post-Training for Humanization & Conversational Persona Modeling

**ID**: `IDEA-10`  
**Status**: `brainstorm / active`  
**Cluster**: `#post-training` `#rlhf` `#dpo` `#persona-modeling`  
**Related Nodes**: [[IDEA-02-confidence-gated-hybrid]], [[IDEA-06-recursive-autophagy-slop-decay]], [[IDEA-05-latent-tone-style-transfer]]  
**Target Experiment**: [`experiments/09_rl_post_training_humanization/`](experiments/09_rl_post_training_humanization/)

---

## 1. 📝 Raw User Voice (Verbatim)

> *"another experiment to rl train a model for the best humanizer we create later or post training directly on humanized text. not sure. this necessarily not about the rewrite but in general giving the best next humanized responsess and stuff. (for fun i could also utilise my text conversations with ai agents and use it to train a model that predicts my next sayings.)  ahha just note it as one of the experiments."*

---

## 2. 🧠 Descriptive Engineering Formulation

### A. The Evolution: From Test-Time Decoding to Weight-Level Alignment
Our earlier experiments (Phases 01–03) intervene at **test-time decoding** via `LogitsProcessor` without altering model weights. While effective, decoding interventions add computational overhead and can fight against the underlying model's prior training.

This direction explores **weight-level post-training**:
1. **Direct Preference Optimization (DPO / ORPO / KTO)**:
   - Construct contrastive pairs:
     $$y_w = \text{Tuned Hybrid Humanized Rewrite (High burstiness, natural phrasing)}$$
     $$y_l = \text{Greedy / Standard Nucleus Output (Robotic boilerplate, verbatim copying)}$$
   - Optimize the model's policy directly to prefer natural phrasing without needing a separate reward model:
     $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

2. **Reinforcement Learning with Anti-Detector & Burstiness Rewards (GRPO / PPO)**:
   - Reward composite:
     $$R(y) = \lambda_{\text{fact}} \cdot \text{EntityRecall}(y) - \lambda_{\text{ai}} \cdot \text{AIClassifierScore}(y) + \lambda_{\text{burst}} \cdot \text{PerplexityVariance}(y)$$
   - Guides the policy to naturally avoid local log-probability peaks without manually masking logits at inference.

3. **Personal User Persona Cloning & Next-Utterance Prediction**:
   - Collect authentic conversational turns from user-agent logs (`transcript.jsonl`, messaging histories).
   - Fine-tune a lightweight model (`SmolLM-135M` or `SmolLM2-360M`) on the causal task:
     $$P(\text{User Utterance}_{t} \mid \text{Agent Context}_{<t})$$
   - Evaluates whether fine-tuning on real, colloquial user conversation captures authentic human cadence, informal shorthand, sentence fragment rhythms, and idiosyncratic thinking patterns that generic RLHF strips away.
