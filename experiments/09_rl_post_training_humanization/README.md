# Experiment 09: RL Post-Training for Humanization & Conversational Persona Modeling

## 1. 📝 Raw User Idea (Verbatim Transcription)

> *"another experiment to rl train a model for the best humanizer we create later or post training directly on humanized text. not sure. this necessarily not about the rewrite but in general giving the best next humanized responsess and stuff. (for fun i could also utilise my text conversations with ai agents and use it to train a model that predicts my next sayings.)  ahha just note it as one of the experiments."*

---

## 2. 🧠 Synthesized & Descriptive Engineering Formulation

### The Shift from Test-Time Logits Tweaking to Permanent Model Behavior
While LogitsProcessors modify sampling at runtime, **post-training (SFT / DPO / RL)** permanently reshapes the model's policy distribution:

```
[Phase 03 Tuned Hybrid Outputs] ─────────────┐
                                              ▼
[Standard Greedy Boilerplate Outputs] ──> [Pairwise DPO Dataset] ──> [Fine-Tuned Humanizer Policy]
                                                                                   │
[User-Agent Chat Transcripts] ──────────> [Conversational SFT] ─────────────> [Personal Voice Clone]
```

### The Three Experimental Pathways:
1. **DPO Distillation (`09a`)**: Distill the outputs of our best test-time decoder into weights so standard greedy generation naturally produces humanized rewrites without inference overhead.
2. **Anti-Slop & Burstiness RL (`09b`)**: Train with a reward function penalizing low perplexity variance and high AI detector certainty.
3. **Personal Conversational Next-Utterance Prediction (`09c`)**: Train a small model on authentic user-agent dialogue histories to model the user's natural cadence and predict their next response.

---

## 📂 Sub-Experimentation Tracks

| Track | Directory | Objective & Methodology | Deliverables |
| :--- | :--- | :--- | :--- |
| **09a** | [`09a_dpo_anti_boilerplate/`](./09a_dpo_anti_boilerplate/) | **Direct Preference Optimization**: Train on paired dataset (Chosen = Tuned Hybrid rewrite, Rejected = Greedy boilerplate). | LoRA adapter / weights; evaluation of boilerplate elimination at standard greedy decoding. |
| **09b** | [`09b_rl_burstiness_reward/`](./09b_rl_burstiness_reward/) | **Reinforcement Learning with Detector Penalty**: Optimize policy using GRPO/PPO with reward $R = \text{EntityRecall} - \text{DetectGPTScore}$. | Reward curves; distribution plots of probability curvature and perplexity variance. |
| **09c** | [`09c_personal_transcript_persona_clone/`](./09c_personal_transcript_persona_clone/) | **Personal Persona Modeling**: Fine-tune on multi-turn user agent logs (`transcript.jsonl`) to predict real user responses. | Perplexity on held-out user turns; qualitative persona response samples. |

---

## 🕸️ Zettelkasten Knowledge Graph Links
* **Thought Node**: [`ideas/nodes/IDEA-10-rl-post-training-humanization.md`](ideas/nodes/IDEA-10-rl-post-training-humanization.md)
* **Master Topology**: [`ideas/GRAPH.md`](ideas/GRAPH.md)
