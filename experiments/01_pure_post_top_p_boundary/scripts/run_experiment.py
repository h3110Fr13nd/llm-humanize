import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
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

with open(DATA_DIR / "prompts.json") as f:
    short_prompts = json.load(f)

with open(DATA_DIR / "source_text_1k.txt") as f:
    source_text = f.read()

with open(DATA_DIR / "key_entities.json") as f:
    key_entities = json.load(f)

class PostTopPLogitsProcessor(LogitsProcessor):
    def __init__(self, top_p: float = 0.9):
        if not (0.0 < top_p < 1.0):
            raise ValueError("top_p must be strictly between 0 and 1.")
        self.top_p = top_p

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        exceeds_p = cumulative_probs > self.top_p
        first_crossing_idx = exceeds_p.to(torch.int64).argmax(dim=-1)
        target_sorted_idx = torch.clamp(first_crossing_idx + 1, max=scores.shape[-1] - 1)

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

# 1. Step 1 Distribution Visualization
prompt_step1 = "The capital of France is"
inp_s1 = tokenizer(prompt_step1, return_tensors="pt").to(device)
with torch.no_grad():
    l_s1 = model(inp_s1["input_ids"]).logits[:, -1, :]
    p_s1 = F.softmax(l_s1, dim=-1)
sp_s1, si_s1 = torch.sort(p_s1, descending=True, dim=-1)
cs_s1 = torch.cumsum(sp_s1, dim=-1)
ex_s1 = cs_s1 > 0.90
fc_s1 = ex_s1.to(torch.int64).argmax(dim=-1)[0].item()
bi_s1 = min(fc_s1 + 1, si_s1.shape[-1] - 1)

tokens_s1 = [repr(tokenizer.decode([si_s1[0, i].item()])) for i in range(12)]
p_vals_s1 = sp_s1[0, :12].cpu().numpy()
bt_s1 = repr(tokenizer.decode([si_s1[0, bi_s1].item()]))
bp_s1 = sp_s1[0, bi_s1].item()
tokens_s1.append(f"{bt_s1}\n(Rank #{bi_s1+1})")
p_vals_s1 = np.append(p_vals_s1, bp_s1)

fig1, ax1 = plt.subplots(figsize=(11, 4.5), dpi=130)
x_pos = np.arange(len(tokens_s1))
colors_s1 = ['#1f77b4'] * (len(tokens_s1) - 1) + ['#d62728']
ax1.bar(x_pos, p_vals_s1, color=colors_s1, alpha=0.85, edgecolor='black', linewidth=0.8)
ax1.set_ylabel("Token Probability P(w)", color='#1f77b4', fontsize=11, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(tokens_s1, rotation=35, ha='right', fontsize=9.5)
ax1.set_title(f"Step 1 Probability Distribution for '{prompt_step1}'\n(Standard Nucleus vs. Post-Top-P Boundary)", fontsize=12, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
fig1.savefig(OUTPUTS_DIR / "step1_distribution.png", bbox_inches='tight')
plt.close(fig1)

# 2. Run Short Prompts Suite
short_configs = [
    ("Greedy (Baseline)", {"do_sample": False}, 0.0),
    ("Standard Top-P (0.9)", {"do_sample": True, "top_p": 0.9, "top_k": 0}, 0.9),
    ("Post-Top-P (p=0.50)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(0.50)])}, 0.50),
    ("Post-Top-P (p=0.70)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(0.70)])}, 0.70),
    ("Post-Top-P (p=0.90)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(0.90)])}, 0.90),
    ("Post-Top-P (p=0.95)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(0.95)])}, 0.95),
]

short_rows = []
for p_text in short_prompts:
    p_inp = tokenizer(p_text, return_tensors="pt").to(device)
    p_len = p_inp["input_ids"].shape[1]
    for c_name, c_kwargs, c_val in short_configs:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        c_out = model.generate(**p_inp, max_new_tokens=35, pad_token_id=tokenizer.eos_token_id, **c_kwargs)
        g_toks = c_out[0, p_len:].tolist()
        g_text = tokenizer.decode(g_toks, skip_special_tokens=True).strip()
        g_ppl = compute_generation_perplexity(model, c_out, p_len)
        d1 = compute_distinct_ngrams(g_toks, 1)
        d2 = compute_distinct_ngrams(g_toks, 2)
        short_rows.append({
            "Prompt": p_text,
            "Strategy": c_name,
            "Cutoff_p": c_val,
            "Gen_PPL": g_ppl,
            "Dist_1": d1,
            "Dist_2": d2,
            "Generated_Text": g_text
        })

df_short = pd.DataFrame(short_rows)
df_short.to_csv(OUTPUTS_DIR / "short_prompts_results.csv", index=False)

# Plot Perplexity & Anti-loop Diversity
fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(15, 4.8), dpi=130)
post_data = df_short[df_short["Strategy"].str.startswith("Post-Top-P")]
for p_name in short_prompts:
    sub = post_data[post_data["Prompt"] == p_name]
    ax2_1.plot(sub["Cutoff_p"], sub["Gen_PPL"], marker='o', linewidth=2.2, label=f"{p_name[:20]}...")
ax2_1.set_yscale('log')
ax2_1.set_xlabel("Top-P Boundary Threshold (p)", fontsize=11, fontweight='bold')
ax2_1.set_ylabel("Generated Perplexity (Log)", fontsize=11, fontweight='bold')
ax2_1.set_title("Perplexity Explosion Across Boundary Thresholds", fontsize=12, fontweight='bold')
ax2_1.grid(True, which="both", linestyle="--", alpha=0.5)
ax2_1.legend(fontsize=9)

strat_order = ["Greedy (Baseline)", "Post-Top-P (p=0.50)", "Post-Top-P (p=0.70)", "Post-Top-P (p=0.90)"]
p_france = df_short[df_short["Prompt"] == "The capital of France is"].set_index("Strategy").loc[strat_order]
p_python = df_short[df_short["Prompt"] == "Python is a high-level programming language that"].set_index("Strategy").loc[strat_order]
x_bar = np.arange(len(strat_order))
w = 0.35
ax2_2.bar(x_bar - w/2, p_france["Dist_1"], w, label='France Prompt', color='#3498db', edgecolor='black')
ax2_2.bar(x_bar + w/2, p_python["Dist_1"], w, label='Python Prompt', color='#e67e22', edgecolor='black')
ax2_2.set_ylabel("Distinct-1 Token Ratio", fontsize=11, fontweight='bold')
ax2_2.set_title("Repetition Resistance: Greedy vs. Post-Top-P", fontsize=12, fontweight='bold')
ax2_2.set_xticks(x_bar)
ax2_2.set_xticklabels(["Greedy\n(Loops)", "Post-0.50\n(Diverse)", "Post-0.70\n(Diverse)", "Post-0.90\n(Diverse)"], fontsize=10)
ax2_2.set_ylim(0, 1.15)
ax2_2.grid(axis='y', linestyle='--', alpha=0.5)
ax2_2.legend(fontsize=10)
plt.tight_layout()
fig2.savefig(OUTPUTS_DIR / "perplexity_and_repetition.png", bbox_inches='tight')
plt.close(fig2)

# 3. Run 1,000-Word Rewriting Suite
system_prompt = (
    "You are an expert editor. Rewrite the following text clearly and concisely, "
    "faithfully summarizing all key historical facts, dates, numbers, and technical specifications."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"Please rewrite this 1,000-word article into a concise, professional summary:\n\n{source_text}"},
]
formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs_1k = tokenizer(formatted_input, return_tensors="pt").to(device)
prompt_len_1k = inputs_1k["input_ids"].shape[1]

rewrite_strategies = [
    ("Greedy Baseline", None, 0.0),
    ("Standard Top-P (p=0.9)", "sample_0.9", 0.90),
    ("Post-Top-P (p=0.10)", 0.10, 0.10),
    ("Post-Top-P (p=0.25)", 0.25, 0.25),
    ("Post-Top-P (p=0.40)", 0.40, 0.40),
    ("Post-Top-P (p=0.50)", 0.50, 0.50),
    ("Post-Top-P (p=0.60)", 0.60, 0.60),
    ("Post-Top-P (p=0.70)", 0.70, 0.70),
    ("Post-Top-P (p=0.80)", 0.80, 0.80),
    ("Post-Top-P (p=0.90)", 0.90, 0.90),
]

rewrite_rows = []
for name, p_val, p_num in rewrite_strategies:
    torch.cuda.empty_cache()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    if p_val == "sample_0.9":
        out = model.generate(**inputs_1k, max_new_tokens=75, do_sample=True, top_p=0.9, top_k=0, pad_token_id=tokenizer.eos_token_id)
    elif p_val is None:
        out = model.generate(**inputs_1k, max_new_tokens=75, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    else:
        out = model.generate(**inputs_1k, max_new_tokens=75, do_sample=False, logits_processor=LogitsProcessorList([PostTopPLogitsProcessor(p_val)]), pad_token_id=tokenizer.eos_token_id)

    gen_toks = out[0, prompt_len_1k:].tolist()
    gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
    gen_ppl = compute_generation_perplexity(model, out, prompt_len_1k)
    recall = evaluate_entity_retention(gen_text, key_entities)

    if p_num == 0.0 or p_val == "sample_0.9":
        status = "Factual Grounding (Accurate)"
    elif p_num <= 0.30:
        status = "Stylistic Diversion (Fact Evasion)"
    elif p_num <= 0.75:
        status = "Hallucination / Modality Drift"
    else:
        status = "Complete Token Soup / Non-usable"

    rewrite_rows.append({
        "Strategy": name,
        "Cutoff_p": p_num,
        "Gen_PPL": gen_ppl,
        "Entity_Recall_%": recall * 100,
        "Status": status,
        "Generated_Text": gen_text
    })

df_rewrites = pd.DataFrame(rewrite_rows)
df_rewrites.to_csv(OUTPUTS_DIR / "rewrite_breakdown_results.csv", index=False)

# 4. Drift Tracking & Breakdown Plot
drift_tracking = {}
for p_track in [None, 0.25, 0.50, 0.70, 0.90]:
    label = f"Post-Top-P (p={p_track})" if p_track is not None else "Greedy Baseline"
    curr_ids = inputs_1k["input_ids"].clone()
    ranks = []
    for step in range(35):
        with torch.no_grad():
            outputs = model(curr_ids)
            logits = outputs.logits[:, -1, :]
            prob_dist = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            if p_track is None:
                chosen_idx = sorted_indices[0, 0].item()
                chosen_rank = 1
            else:
                exceeds = cumsum > p_track
                fc = exceeds.to(torch.int64).argmax(dim=-1)[0].item()
                pos = min(fc + 1, sorted_indices.shape[-1] - 1)
                chosen_idx = sorted_indices[0, pos].item()
                chosen_rank = pos + 1

            ranks.append(chosen_rank)
            next_tok = torch.tensor([[chosen_idx]], device=device)
            curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
    drift_tracking[label] = ranks

fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(16, 5), dpi=130)
post_sub = df_rewrites[df_rewrites["Cutoff_p"] > 0]
ax3_1.plot(post_sub["Cutoff_p"], post_sub["Entity_Recall_%"], marker='s', color='#2ca02c', linewidth=2.5, label='Entity Recall %')
ax3_1.set_xlabel("Post-Top-P Threshold (p)", fontsize=11, fontweight='bold')
ax3_1.set_ylabel("Entity Recall %", color='#2ca02c', fontsize=11, fontweight='bold')
ax3_1.set_ylim(-2, 20)
ax3_1.grid(True, linestyle="--", alpha=0.5)

ax3_1_twin = ax3_1.twinx()
ax3_1_twin.plot(post_sub["Cutoff_p"], post_sub["Gen_PPL"], marker='o', color='#d62728', linewidth=2.5, linestyle='--', label='Gen Perplexity')
ax3_1_twin.set_yscale('log')
ax3_1_twin.set_ylabel("Generated Perplexity (Log)", color='#d62728', fontsize=11, fontweight='bold')
ax3_1.set_title("Factual Fidelity & Perplexity vs. Cutoff (p)", fontsize=12, fontweight='bold')

steps_x = np.arange(1, 36)
colors_map = {
    "Greedy Baseline": ("black", "-", 2.5),
    "Post-Top-P (p=0.25)": ("#1f77b4", "-", 1.8),
    "Post-Top-P (p=0.5)": ("#ff7f0e", "--", 1.8),
    "Post-Top-P (p=0.7)": ("#9467bd", "-.", 1.8),
    "Post-Top-P (p=0.9)": ("#d62728", ":", 2.2),
}

for label, ranks_list in drift_tracking.items():
    c, ls, lw = colors_map.get(label, ("gray", "-", 1.5))
    ax3_2.plot(steps_x, ranks_list, label=label, color=c, linestyle=ls, linewidth=lw)

ax3_2.set_yscale('log')
ax3_2.set_xlabel("Generation Step (Token Position t)", fontsize=11, fontweight='bold')
ax3_2.set_ylabel("Selected Token Vocabulary Rank (Log)", fontsize=11, fontweight='bold')
ax3_2.set_title("Sequence Degradation: Rank Drift Over 35 Steps", fontsize=12, fontweight='bold')
ax3_2.grid(True, which="both", linestyle="--", alpha=0.5)
ax3_2.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Drift Warning (Rank 10)')
ax3_2.axhline(100, color='red', linestyle='--', alpha=0.5, label='Collapse (Rank 100+)')
ax3_2.legend(loc='upper right', fontsize=8.5)

plt.tight_layout()
fig3.savefig(OUTPUTS_DIR / "rewrite_failure_boundaries.png", bbox_inches='tight')
plt.close(fig3)

print("Experiment 01 script finished successfully. All outputs saved to:", OUTPUTS_DIR)
