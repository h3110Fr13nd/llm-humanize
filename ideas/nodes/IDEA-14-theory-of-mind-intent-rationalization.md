# IDEA-14: Theory-of-Mind Intent Rationalization & Next-Human Prediction

**ID**: `IDEA-14`  
**Status**: `brainstorm / active`  
**Cluster**: `#theory-of-mind` `#next-human-prediction` `#cognitive-cot`  
**Related Nodes**: [[IDEA-10-rl-post-training-humanization]], [[IDEA-13-zero-shot-persona-cloning-llm]]  
**Target Sub-Experiment**: `experiments/11_persona_cloning_and_temporal_dynamics/11b_intent_rationalization_cot/`

---

## 1. 📝 Raw User Voice (Verbatim)

> `[03:44, 5/11/2026] Researcher: Another thing is to add a thinking step in training on why a human says what he says or simply training a model to predict the next human message.`

---

## 2. 🧠 Descriptive Engineering Formulation

### A. The Cognitive Gap in Conversational AI
Current chat models optimize directly for the conditional likelihood of the assistant's reply:
$$P(\text{Response} \mid \text{Dialogue History})$$
* This causes models to jump straight to answering surface queries without understanding the human's underlying psychological, emotional, or practical motivations.
* Humans in conversation constantly run **Theory of Mind (ToM)** inferences:
  * *Why did they ask this specific question right now?*
  * *What unstated context, emotional frustration, or unspoken goal prompted this phrasing?*
  * *What are they going to say next if I answer with X vs. Y?*

### B. The Intermediate Intent Thinking Step (CoT for Human Psychology)
Before generating a response (or predicting the next human turn), train an explicit latent rationalization step:

```
[User Input Utterance]
           │
           ▼
┌────────────────────────────────────────────────────────┐
│ <thought_intent> (Latent / Visible Rationalization)    │
│ - Underlying Goal: User wants concise code, not theory │
│ - Emotional Valence: Impatient / Time-constrained      │
│ - Prior Implicit Context: Previous run crashed with OOM│
│ - Anticipated Next Turn: If code works, will ask deploy│
└────────────────────────────────────────────────────────┘
           │
           ▼
[Output: Grounded Empathetic Response OR Predicted Next Human Turn]
```

### C. Training Method: Dual-Task Loss
1. **Next-Human Utterance Prediction**:
   Train the model to anticipate what the *human* will say next based on the interaction history:
   $$\mathcal{L}_{\text{Human}} = -\log P(\text{User Utterance}_{t+1} \mid \text{History}_{\le t})$$
2. **Synthetic Intent Distillation**:
   Use frontier models to annotate dialogue turns with inferred intent rationales, then fine-tune smaller models on the structured tuple:
   $$(\text{History}, \text{Intent CoT}, \text{Response})$$
