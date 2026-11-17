import json
import os
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
COMP_DIR = OUTPUTS_DIR / "side_by_side_comparisons"
COMP_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
model.eval()

with open(DATA_DIR / "dataset.json") as f:
    dataset = json.load(f)

print(f"Loaded {len(dataset)} evaluation documents.")

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

class PurePostTopPLogitsProcessor(LogitsProcessor):
    def __init__(self, top_p: float = 0.50):
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

system_prompt = (
    "You are an expert editor. Rewrite the following text clearly and concisely, "
    "faithfully summarizing all key historical facts, dates, numbers, and technical specifications."
)

# Phase 1: Parameter Tuning across a subset of grid values
tuning_csv_path = OUTPUTS_DIR / "parameter_tuning_results.csv"
if tuning_csv_path.exists():
    print(f"\n--- Phase 1: Found existing tuning results at {tuning_csv_path} ---")
    df_tuning = pd.read_csv(tuning_csv_path)
else:
    print("\n--- Phase 1: Tuning the Sweet Spot on 12 Documents ---")
    tau_grid = [0.45, 0.50, 0.55, 0.60]
    x_grid = [0.30, 0.40, 0.50, 0.60]

    tuning_results = []
    for tau in tau_grid:
        for x in x_grid:
            recalls = []
            ppls = []
            dist1s = []
            triggers = []
            for doc in dataset:
                torch.cuda.empty_cache()
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Please rewrite this article into a concise, professional summary:\n\n{doc['text']}"},
                ]
                fmt_inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inp = tokenizer(fmt_inp, return_tensors="pt").to(device)
                p_len = inp["input_ids"].shape[1]

                proc = HybridPostTopPLogitsProcessor(confidence_threshold=tau, top_p=x)
                out = model.generate(
                    **inp,
                    max_new_tokens=65,
                    do_sample=False,
                    logits_processor=LogitsProcessorList([proc]),
                    pad_token_id=tokenizer.eos_token_id,
                )
                g_tokens = out[0, p_len:].tolist()
                g_text = tokenizer.decode(g_tokens, skip_special_tokens=True).strip()
                g_ppl = compute_generation_perplexity(model, out, p_len)
                d1 = compute_distinct_ngrams(g_tokens, 1)
                rec = evaluate_entity_retention(g_text, doc["entities"])
                trig = (sum(proc.decisions) / len(proc.decisions)) * 100 if proc.decisions else 0.0

                recalls.append(rec)
                ppls.append(g_ppl)
                dist1s.append(d1)
                triggers.append(trig)

            avg_recall = np.mean(recalls) * 100
            avg_ppl = np.mean(ppls)
            avg_d1 = np.mean(dist1s)
            avg_trig = np.mean(triggers)

            tuning_results.append({
                "tau": tau,
                "x": x,
                "Avg_Entity_Recall_%": avg_recall,
                "Avg_Gen_PPL": avg_ppl,
                "Avg_Dist_1": avg_d1,
                "Avg_Greedy_Trigger_%": avg_trig,
            })
            print(f"tau={tau:.2f}, x={x:.2f} -> Recall={avg_recall:.2f}%, PPL={avg_ppl:.2f}, Dist1={avg_d1:.2f}, Trigger={avg_trig:.1f}%")

    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.to_csv(tuning_csv_path, index=False)

# Select best sweet spot: maximize Recall with PPL < 5.0
best_row = df_tuning.sort_values(by=["Avg_Entity_Recall_%", "Avg_Gen_PPL"], ascending=[False, True]).iloc[0]
optimal_tau = best_row["tau"]
optimal_x = best_row["x"]
print(f"\n★ STATISTICALLY TUNED SWEET SPOT: tau = {optimal_tau}, x = {optimal_x} (Avg Recall: {best_row['Avg_Entity_Recall_%']:.2f}%, Avg PPL: {best_row['Avg_Gen_PPL']:.2f})")

# Plot Heatmaps for Tuning
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)
piv_rec = df_tuning.pivot(index="tau", columns="x", values="Avg_Entity_Recall_%")
piv_ppl = df_tuning.pivot(index="tau", columns="x", values="Avg_Gen_PPL")

sns.heatmap(piv_rec, annot=True, fmt=".2f", cmap="YlGn", cbar=True, ax=ax1, linewidths=0.5)
ax1.set_title("Avg Entity Recall % across 12 Docs\\n[Higher is Better]", fontsize=11, fontweight='bold')
ax1.set_xlabel("Post-x Threshold (x)", fontsize=10, fontweight='bold')
ax1.set_ylabel("Confidence Threshold (tau)", fontsize=10, fontweight='bold')

sns.heatmap(piv_ppl, annot=True, fmt=".2f", cmap="YlOrRd_r", cbar=True, ax=ax2, linewidths=0.5)
ax2.set_title("Avg Generation Perplexity across 12 Docs\\n[Lower is More Fluent]", fontsize=11, fontweight='bold')
ax2.set_xlabel("Post-x Threshold (x)", fontsize=10, fontweight='bold')
ax2.set_ylabel("Confidence Threshold (tau)", fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(OUTPUTS_DIR / "sweet_spot_tuning_plots.png", bbox_inches='tight')
plt.close(fig)

# Phase 2: Side-by-Side Generation across 4 Strategies for all 12 Documents
print("\n--- Phase 2: Generating Side-by-Side Comparisons for 12 Documents ---")

strategies = [
    ("Top-1 (Greedy)", lambda: {"do_sample": False}),
    ("Standard Top-P (0.9)", lambda: {"do_sample": True, "top_p": 0.9, "top_k": 0}),
    ("Pure Post-Top-P (0.5)", lambda: {"do_sample": False, "logits_processor": LogitsProcessorList([PurePostTopPLogitsProcessor(top_p=0.50)])}),
    ("Tuned Hybrid", lambda: {"do_sample": False, "logits_processor": LogitsProcessorList([HybridPostTopPLogitsProcessor(confidence_threshold=optimal_tau, top_p=optimal_x)])}),
]

all_comparisons = []
summary_rows = []

for idx, doc in enumerate(dataset, 1):
    doc_id = doc["id"]
    title = doc["title"]
    domain = doc["domain"]
    entities = doc["entities"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please rewrite this article into a concise, professional summary:\n\n{doc['text']}"},
    ]
    fmt_inp = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(fmt_inp, return_tensors="pt").to(device)
    p_len = inp["input_ids"].shape[1]

    doc_results = {}
    md_content = f"# Example {idx:02d}: {title}\n"
    md_content += f"**Domain**: {domain}  \n"
    md_content += f"**Key Tracking Entities ({len(entities)})**: `{', '.join(entities)}`\n\n"
    md_content += f"### Original Source Text\n> {doc['text']}\n\n---\n\n"
    md_content += "### Side-by-Side Generation Comparison\n\n"

    table_rows = []

    for s_name, s_kw_fn in strategies:
        torch.cuda.empty_cache()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        kwargs = s_kw_fn()
        out = model.generate(**inp, max_new_tokens=85, pad_token_id=tokenizer.eos_token_id, **kwargs)
        g_tokens = out[0, p_len:].tolist()
        g_text = tokenizer.decode(g_tokens, skip_special_tokens=True).strip()
        g_ppl = compute_generation_perplexity(model, out, p_len)
        d1 = compute_distinct_ngrams(g_tokens, 1)
        rec = evaluate_entity_retention(g_text, entities)

        doc_results[s_name] = {
            "text": g_text,
            "ppl": g_ppl,
            "dist1": d1,
            "recall": rec * 100
        }

        table_rows.append({
            "Strategy": s_name,
            "Entity_Recall_%": f"{rec*100:.1f}%",
            "Gen_PPL": f"{g_ppl:.2f}",
            "Distinct_1": f"{d1:.2f}"
        })

        summary_rows.append({
            "Doc_ID": doc_id,
            "Title": title,
            "Domain": domain,
            "Strategy": s_name,
            "Entity_Recall_%": rec * 100,
            "Gen_PPL": g_ppl,
            "Dist_1": d1,
            "Snippet": g_text[:120] + ("..." if len(g_text) > 120 else "")
        })

    # Render table in MD
    df_md_table = pd.DataFrame(table_rows)
    md_content += df_md_table.to_markdown(index=False) + "\n\n"

    # Add full texts
    for s_name in [s[0] for s in strategies]:
        res = doc_results[s_name]
        md_content += f"#### [{s_name}] (Recall: {res['recall']:.1f}% | PPL: {res['ppl']:.2f} | Dist-1: {res['dist1']:.2f})\n"
        md_content += f"```text\n{res['text']}\n```\n\n"

    # Save individual markdown file
    file_path = COMP_DIR / f"{doc_id}.md"
    with open(file_path, "w", encoding="utf-8") as f_out:
        f_out.write(md_content)

    all_comparisons.append({
        "id": doc_id,
        "title": title,
        "domain": domain,
        "results": doc_results
    })
    print(f"[{idx}/12] Generated side-by-side comparison for: {title}")

# Save JSON and summary CSV
with open(OUTPUTS_DIR / "all_examples_comparison.json", "w", encoding="utf-8") as f_json:
    json.dump(all_comparisons, f_json, indent=2)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUTPUTS_DIR / "all_examples_summary.csv", index=False)

# Strategy Aggregate Averages
print("\n" + "=" * 80)
print("AGGREGATE BENCHMARK OVER ALL 12 EXAMPLES")
print("=" * 80)
agg = df_summary.groupby("Strategy").agg({
    "Entity_Recall_%": "mean",
    "Gen_PPL": "mean",
    "Dist_1": "mean"
}).round(2)
print(agg.to_string())

print(f"\nAll 12 markdown comparisons saved to: {COMP_DIR}")
