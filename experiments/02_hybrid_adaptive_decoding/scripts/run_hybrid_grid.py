import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
model.eval()

with open(DATA_DIR / "source_text_1k.txt") as f:
    source_text = f.read()

with open(DATA_DIR / "key_entities.json") as f:
    key_entities = json.load(f)

class HybridPostTopPLogitsProcessor(LogitsProcessor):
    def __init__(self, confidence_threshold: float = 0.55, top_p: float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.top_p = top_p
        self.decisions = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        top1_p = probs[:, 0]

        use_greedy = top1_p >= self.confidence_threshold
        self.decisions.append(bool(use_greedy.item() if top1_p.numel() == 1 else use_greedy[0].item()))

        cumulative_probs = torch.cumsum(probs, dim=-1)
        exceeds_p = cumulative_probs > self.top_p
        first_crossing_idx = exceeds_p.to(torch.int64).argmax(dim=-1)
        post_p_idx = torch.clamp(first_crossing_idx + 1, max=scores.shape[-1] - 1)

        target_sorted_idx = torch.where(use_greedy, torch.zeros_like(post_p_idx), post_p_idx)
        target_vocab_idx = sorted_indices.gather(-1, target_sorted_idx.unsqueeze(-1))

        filtered_scores = torch.full_like(scores, float("-inf"))
        filtered_scores.scatter_(-1, target_vocab_idx, scores.gather(-1, target_vocab_idx))
        return filtered_scores

def compute_generation_perplexity(model, full_ids: torch.LongTensor, prompt_len: int) -> float:
    if full_ids.shape[1] <= prompt_len:
        return 0.0
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        targets = full_ids[:, 1:]
        gen_logits = logits[:, prompt_len - 1 :, :]
        gen_targets = targets[:, prompt_len - 1 :]
        loss = F.cross_entropy(gen_logits.reshape(-1, gen_logits.size(-1)), gen_targets.reshape(-1))
        return torch.exp(loss).item()

def compute_distinct_ngrams(tokens, n: int) -> float:
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)

def evaluate_entity_retention(text: str, entities: list) -> float:
    text_lower = text.lower()
    return sum(1 for e in entities if e.lower() in text_lower) / len(entities)

system_prompt = (
    "You are an expert editor. Rewrite the following text clearly and concisely, "
    "faithfully summarizing all key historical facts, dates, numbers, and technical specifications."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Please rewrite this 1,000-word article into a concise, professional summary:\n\n{source_text}"},
]
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted_input, return_tensors="pt").to(device)
prompt_len = inputs["input_ids"].shape[1]

# 1. Baselines
baselines = []
for b_name, b_kwargs in [
    ("Pure Greedy Baseline", {"do_sample": False}),
    ("Standard Top-P (p=0.9)", {"do_sample": True, "top_p": 0.9, "top_k": 0}),
    ("Pure Post-Top-P (p=0.5)", {"do_sample": False, "logits_processor": LogitsProcessorList([HybridPostTopPLogitsProcessor(confidence_threshold=1.01, top_p=0.5)])}),
]:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    out = model.generate(**inputs, max_new_tokens=85, pad_token_id=tokenizer.eos_token_id, **b_kwargs)
    gen_tokens = out[0, prompt_len:].tolist()
    gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    gen_ppl = compute_generation_perplexity(model, out, prompt_len)
    dist1 = compute_distinct_ngrams(gen_tokens, 1)
    dist2 = compute_distinct_ngrams(gen_tokens, 2)
    recall = evaluate_entity_retention(gen_text, key_entities)
    baselines.append({
        "Configuration": b_name,
        "Entity_Recall_%": recall * 100,
        "Gen_PPL": gen_ppl,
        "Dist_1": dist1,
        "Dist_2": dist2,
        "Snippet": gen_text[:140] + ("..." if len(gen_text) > 140 else "")
    })

df_baselines = pd.DataFrame(baselines)
df_baselines.to_csv(OUTPUTS_DIR / "baselines_comparison.csv", index=False)

# 2. 2D Parameter Grid
tau_values = [0.25, 0.40, 0.55, 0.70]
x_values = [0.30, 0.50, 0.70, 0.90]

grid_records = []
for tau in tau_values:
    for x in x_values:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        proc = HybridPostTopPLogitsProcessor(confidence_threshold=tau, top_p=x)
        out = model.generate(
            **inputs,
            max_new_tokens=85,
            do_sample=False,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=tokenizer.eos_token_id,
        )

        gen_tokens = out[0, prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        gen_ppl = compute_generation_perplexity(model, out, prompt_len)
        dist1 = compute_distinct_ngrams(gen_tokens, 1)
        dist2 = compute_distinct_ngrams(gen_tokens, 2)
        recall = evaluate_entity_retention(gen_text, key_entities)
        greedy_pct = (sum(proc.decisions) / len(proc.decisions)) * 100 if proc.decisions else 0.0

        grid_records.append({
            "Confidence_Threshold_tau": tau,
            "Post_x_Threshold": x,
            "Entity_Recall_%": recall * 100,
            "Gen_PPL": gen_ppl,
            "Greedy_Trigger_%": greedy_pct,
            "Dist_1": dist1,
            "Dist_2": dist2,
            "Snippet": gen_text[:140] + ("..." if len(gen_text) > 140 else "")
        })

df_grid = pd.DataFrame(grid_records)
df_grid.to_csv(OUTPUTS_DIR / "hybrid_grid_results.csv", index=False)

# 3. Heatmap Visualizations
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4.8), dpi=130)

pivot_recall = df_grid.pivot(index="Confidence_Threshold_tau", columns="Post_x_Threshold", values="Entity_Recall_%")
pivot_ppl = df_grid.pivot(index="Confidence_Threshold_tau", columns="Post_x_Threshold", values="Gen_PPL")
pivot_trigger = df_grid.pivot(index="Confidence_Threshold_tau", columns="Post_x_Threshold", values="Greedy_Trigger_%")

sns.heatmap(pivot_recall, annot=True, fmt=".1f", cmap="YlGn", cbar=True, ax=ax1, linewidths=0.5)
ax1.set_title("Entity Recall % (Factual Fidelity)\n[Higher is Better]", fontsize=11, fontweight='bold')
ax1.set_xlabel("Post-x Cutoff Threshold (x)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Confidence Threshold (tau)", fontsize=10, fontweight='bold')

sns.heatmap(pivot_ppl, annot=True, fmt=".2f", cmap="YlOrRd_r", cbar=True, ax=ax2, linewidths=0.5)
ax2.set_title("Generation Perplexity (PPL)\n[Lower is More Fluent]", fontsize=11, fontweight='bold')
ax2.set_xlabel("Post-x Cutoff Threshold (x)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Confidence Threshold (tau)", fontsize=10, fontweight='bold')

sns.heatmap(pivot_trigger, annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=ax3, linewidths=0.5)
ax3.set_title("Greedy Anchor Frequency % (P >= tau)\n[Factual Routing Ratio]", fontsize=11, fontweight='bold')
ax3.set_xlabel("Post-x Cutoff Threshold (x)", fontsize=10, fontweight='bold')
ax3.set_ylabel("Confidence Threshold (tau)", fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(OUTPUTS_DIR / "parameter_matrix_heatmaps.png", bbox_inches='tight')
plt.close(fig)

# 4. Microscopic Trace
trace_rows = []
curr_ids = inputs["input_ids"].clone()
for step in range(16):
    with torch.no_grad():
        outputs = model(curr_ids)
        logits = outputs.logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)

        top1_p = sorted_probs[0, 0].item()
        top1_tok = tokenizer.decode([sorted_indices[0, 0].item()])

        if top1_p >= 0.55:
            decision = "GREEDY (Anchor)"
            chosen_idx = sorted_indices[0, 0].item()
            chosen_tok = top1_tok
            chosen_p = top1_p
            chosen_rank = 1
        else:
            decision = "POST-X (Explore)"
            exceeds = cumsum > 0.50
            fc = exceeds.to(torch.int64).argmax(dim=-1)[0].item()
            target_pos = min(fc + 1, sorted_indices.shape[-1] - 1)
            chosen_idx = sorted_indices[0, target_pos].item()
            chosen_tok = tokenizer.decode([chosen_idx])
            chosen_p = sorted_probs[0, target_pos].item()
            chosen_rank = target_pos + 1

        trace_rows.append({
            "Step": step + 1,
            "Top1_Token": repr(top1_tok),
            "Top1_Prob": f"{top1_p:.3f}",
            "Decision": decision,
            "Selected_Token": repr(chosen_tok),
            "Selected_Rank": f"#{chosen_rank}",
            "Selected_Prob": f"{chosen_p:.3f}"
        })

        next_t = torch.tensor([[chosen_idx]], device=device)
        curr_ids = torch.cat([curr_ids, next_t], dim=-1)

df_trace = pd.DataFrame(trace_rows)
df_trace.to_csv(OUTPUTS_DIR / "microscopic_trace.csv", index=False)

print("Experiment 02 script finished successfully. All outputs saved to:", OUTPUTS_DIR)
