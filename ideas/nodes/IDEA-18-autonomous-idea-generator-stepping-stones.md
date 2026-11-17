# IDEA-18: Autonomous Idea Generator via Cognitive Stepping Stones & Cross-Domain Blending

**ID**: `IDEA-18`  
**Status**: `brainstorm / active`  
**Cluster**: `#novelty-search` `#idea-generator` `#conceptual-blending` `#cognitive-architecture`  
**Related Nodes**: [[IDEA-07-position-free-concept-representations]], [[IDEA-08-temporal-mental-stream-jepa]], [[IDEA-14-theory-of-mind-intent-rationalization]]  
**Target Sub-Experiment**: `experiments/13_autonomous_idea_generation/13a_stepping_stone_search/`

---

## 1. 📝 Raw User Voice (Verbatim)

> *"add an entry on how we can create an idea generator like current llms don't even genrate new ideas. i as a human you can ee have generated the ideas which you can only validate that these are unheard of or unique. idea generator is the thing that can build on the stepping stones on how or what human thinks. temporal thing is a perpective to it."*

---

## 2. 🧠 Descriptive Engineering & Epistemological Formulation

### A. The Fundamental Bottleneck: Why LLMs Cannot Generate New Ideas
Autoregressive Large Language Models are mathematically formulated as maximum likelihood density estimators over internet corpora:
$$\mathcal{L}(\theta) = -\sum_{t} \log P_\theta(w_t \mid w_{<t})$$

* **The Convex Hull Trap**: An LLM can only interpolate within the convex hull of its training distribution. When prompted for "innovative ideas", it retrieves the statistical centroid of what the internet considers "innovative" (e.g., standard combinations of popular keywords).
* **The Role of the Human vs. AI**: As observed in this repository, **the human originates radical, cross-domain stepping stones** (e.g., mapping speech voice-cloning into LLM residual streams; inverting positional embeddings into diffusion unscramblers). The LLM can validate, formalize, and execute these ideas, but would never generate them autonomously under standard sampling.

---

### B. The Architecture of a True "Idea Generator"

```
[Knowledge Base / Current Paradigms]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Axiom Inversion & Assumption Mining           │
│ - Mines the consensus assumptions of a field           │
│ - Inverts core axioms: "Transformers MUST have RoPE"   │
│   ──> "What if position is purely a vocal artifact?"   │
└────────────────────────────────────────────────────────┘
                 │ Inverted Premise
                 ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Cross-Domain Conceptual Analogy Engine        │
│ (Fauconnier & Turner Conceptual Blending)              │
│ - Topology mapping: Source Domain (TTS, Biology, Audio)│
│   into Target Domain (Token Decoding, Memory Graphs)   │
│ - e.g. Voice Cloning Speaker Vector ──> Persona Vector │
└────────────────────────────────────────────────────────┘
                 │ Raw Concept Seed
                 ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Temporal Stepping-Stone Evolution             │
│ (Kenneth Stanley's Novelty Search / Non-Objective)     │
│ - Evaluates candidate ideas NOT by likelihood,         │
│   but by Behavioral Novelty & Distance from Corpus     │
│ - Incubates ideas over temporal state transitions      │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
[High-Novelty, Unheard-of Hypothesis Ready for Empirical Testing]
```

---

### C. The Four Core Algorithmic Pillars

#### 1. Novelty Search over Conceptual Graph Distance (Kenneth Stanley)
In *Why Greatness Cannot Be Planned*, Stanley demonstrated that ambitious discoveries cannot be achieved by optimizing an objective function because intermediate **stepping stones** do not resemble the final goal.
* Instead of optimizing for "likelihood" or "expected utility", the Idea Generator scores candidates using a **Novelty Metric**:
  $$\mathcal{N}(\mathbf{c}) = \frac{1}{K} \sum_{i=1}^K \|\mathbf{e}(\mathbf{c}) - \mathbf{e}(\mu_i)\|$$
  where $\mu_i$ are the $K$-nearest existing concepts in the literature.

#### 2. Cross-Domain Latent Analogy Projection (Hofstadter's *Surfaces and Essences*)
Human breakthrough ideas almost always involve importing an established relational structure from an unrelated field:
$$\text{RelationalStructure}(\text{Domain } A) \xrightarrow{\Phi} \text{Target Problem}(\text{Domain } B)$$
* Example: Acoustic Style Transfer (Vision/Audio) $\xrightarrow{\Phi}$ Transformer Hidden State Steering (NLP).
* The Idea Generator explicitly maintains a relational knowledge graph of structural patterns across disciplines and queries cross-domain topological matches.

#### 3. Temporal Perspective & Conceptual Incubation
Human creativity requires **temporal intervals**: a thought conceived on Day 1 sits in background memory; an unrelated observation on Day 7 acts as a catalyst, merging them into a breakthrough on Day 10.
* A static single-turn prompt cannot simulate incubation.
* An Idea Generator must run as an **asynchronous temporal agent** that maintains an associative memory graph, continually introducing stochastic perturbations and cross-pollinating thoughts across time.

#### 4. Axiomatic Anti-Consensus Sampling
Standard temperature sampling explores random low-probability noise. Axiomatic inversion explicitly targets **structural negations**:
* Identify the highest-confidence token/concept in a discipline ($P \approx 0.99$, e.g., *"LLMs require left-to-right causal attention"*).
* Construct a constraint forcing $P = 0$ on the consensus axiom, and task the reasoning engine with constructing a viable alternative paradigm (e.g., discrete diffusion infilling).
