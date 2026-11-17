# Sub-Experiment 09c: Personal Conversational Transcript Persona Modeling

**Parent Experiment**: [`../README.md`](../README.md)  
**Thought Node**: [`ideas/nodes/IDEA-10-rl-post-training-humanization.md`](ideas/nodes/IDEA-10-rl-post-training-humanization.md)

---

## 🎯 Research Objective
Investigate authentic human conversational dynamics by fine-tuning a small model on actual user-agent interaction transcripts (`transcript.jsonl`), evaluating its ability to predict the user's authentic next utterance and capture colloquial human cadence.

## 🧪 Experimental Questions:
1. **Next-Utterance Accuracy**: Does fine-tuning on real user dialogue achieve low perplexity on held-out user prompt distributions?
2. **Informal Cadence Modeling**: Does the model learn authentic human conversational idiosyncrasies—such as sentence fragments, rapid topic shifts, and pragmatic colloquialisms—that standard corporate RLHF systematically trains out?
3. **Privacy-Preserving Local Tuning**: Can this be run entirely locally on private logs using a lightweight parameter footprint (e.g. `SmolLM-135M` + LoRA)?
