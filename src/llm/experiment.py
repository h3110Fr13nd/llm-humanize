import math
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
)


class PostTopPLogitsProcessor(LogitsProcessor):
    """
    Masks everything EXCEPT the very first token immediately outside the top_p nucleus.
    In standard top-p, tokens whose cumulative mass <= top_p (plus the token crossing top_p)
    are included. This processor isolates the very first token AFTER that crossing token.
    """

    def __init__(self, top_p: float = 0.9):
        if not (0.0 < top_p < 1.0):
            raise ValueError("top_p must be strictly between 0 and 1.")
        self.top_p = top_p

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Sort logits descending
        sorted_logits, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Standard Top-P finds where cumulative_probs > top_p.
        # The token that crosses the threshold is included in the nucleus,
        # so the very next token (first_crossing_idx + 1) is the first one outside Top-P.
        exceeds_p = cumulative_probs > self.top_p
        first_crossing_idx = exceeds_p.to(torch.int64).argmax(dim=-1)
        target_sorted_idx = torch.clamp(first_crossing_idx + 1, max=scores.shape[-1] - 1)

        # Map back to original vocabulary indices
        target_vocab_idx = sorted_indices.gather(-1, target_sorted_idx.unsqueeze(-1))

        # Mask all logits to -inf except our target boundary token
        filtered_scores = torch.full_like(scores, float("-inf"))
        filtered_scores.scatter_(-1, target_vocab_idx, scores.gather(-1, target_vocab_idx))
        return filtered_scores


def compute_sequence_perplexity(model, input_ids: torch.LongTensor) -> float:
    """Computes the model's perplexity over the entire sequence (prompt + output)."""
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()


def compute_generation_perplexity(model, full_ids: torch.LongTensor, prompt_len: int) -> float:
    """Computes conditional perplexity exclusively on the newly generated tokens."""
    if full_ids.shape[1] <= prompt_len:
        return 0.0
    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits[:, :-1, :]
        targets = full_ids[:, 1:]

        gen_logits = logits[:, prompt_len - 1 :, :]
        gen_targets = targets[:, prompt_len - 1 :]

        loss = F.cross_entropy(
            gen_logits.reshape(-1, gen_logits.size(-1)),
            gen_targets.reshape(-1),
        )
        return torch.exp(loss).item()


def compute_distinct_ngrams(tokens: List[int], n: int) -> float:
    """Computes ratio of unique n-grams (distinct-n metric to detect repetition loops)."""
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def analyze_step_by_step(
    model, tokenizer, prompt: str, top_p: float = 0.9, num_steps: int = 5, device: str = "cpu"
) -> List[Dict]:
    """Inspects step-by-step distributions showing top-1 greedy token vs post-top-p boundary token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    curr_ids = inputs["input_ids"]
    rows = []

    for step in range(num_steps):
        with torch.no_grad():
            outputs = model(curr_ids)
            next_token_logits = outputs.logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            # Greedy (top-1)
            greedy_idx = sorted_indices[0, 0].item()
            greedy_prob = sorted_probs[0, 0].item()
            greedy_token = tokenizer.decode([greedy_idx])

            # Post-top-p boundary token
            exceeds_p = cumsum > top_p
            first_cross = exceeds_p.to(torch.int64).argmax(dim=-1)[0].item()
            boundary_sorted_pos = min(first_cross + 1, sorted_indices.shape[-1] - 1)
            boundary_idx = sorted_indices[0, boundary_sorted_pos].item()
            boundary_prob = sorted_probs[0, boundary_sorted_pos].item()
            boundary_token = tokenizer.decode([boundary_idx])
            boundary_cum_prob = cumsum[0, boundary_sorted_pos].item()

            rows.append({
                "step": step + 1,
                "greedy_token": greedy_token,
                "greedy_prob": greedy_prob,
                "boundary_token": boundary_token,
                "boundary_rank": boundary_sorted_pos + 1,
                "boundary_prob": boundary_prob,
                "boundary_cum_prob": boundary_cum_prob,
            })

            # Append boundary token for next iteration
            next_token = torch.tensor([[boundary_idx]], device=device)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    return rows


def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_id = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    model.eval()

    prompts = [
        "Artificial intelligence will change the world because",
        "The capital of France is",
        "Python is a high-level programming language that",
    ]

    configs = [
        ("Greedy (Baseline)", {"do_sample": False}),
        ("Standard Top-P (p=0.9, do_sample=True)", {"do_sample": True, "top_p": 0.9, "top_k": 0}),
        ("Post-Top-P (p=0.5, Mild)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(top_p=0.5)])}),
        ("Post-Top-P (p=0.7, Moderate)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(top_p=0.7)])}),
        ("Post-Top-P (p=0.9, Aggressive)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(top_p=0.9)])}),
        ("Post-Top-P (p=0.95, Extreme)", {"do_sample": False, "logits_processor": LogitsProcessorList([PostTopPLogitsProcessor(top_p=0.95)])}),
    ]

    for p_idx, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 80)
        print(f"PROMPT #{p_idx}: \"{prompt}\"")
        print("=" * 80)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        # First, display step-by-step token selection for p=0.9 (first 4 tokens)
        print("\n--- Step-by-Step Nucleus Bypass Inspection (p=0.9) ---")
        steps = analyze_step_by_step(model, tokenizer, prompt, top_p=0.9, num_steps=4, device=device)
        print(f"{'Step':<5} | {'Greedy Token':<16} | {'Greedy P':<10} | {'Bypass Token':<16} | {'Rank':<6} | {'Bypass P':<10} | {'Cum P':<10}")
        print("-" * 85)
        for s in steps:
            print(
                f"{s['step']:<5} | {repr(s['greedy_token']):<16} | {s['greedy_prob']:<10.4f} | "
                f"{repr(s['boundary_token']):<16} | #{s['boundary_rank']:<5} | {s['boundary_prob']:<10.4f} | {s['boundary_cum_prob']:<10.4f}"
            )

        print("\n--- Full Generation & Perplexity Comparison (max_new_tokens=40) ---")
        for name, gen_kwargs in configs:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

            out = model.generate(
                **inputs,
                max_new_tokens=40,
                pad_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )

            seq_ppl = compute_sequence_perplexity(model, out)
            gen_ppl = compute_generation_perplexity(model, out, prompt_len)

            gen_tokens = out[0, prompt_len:].tolist()
            dist1 = compute_distinct_ngrams(gen_tokens, 1)
            dist2 = compute_distinct_ngrams(gen_tokens, 2)

            text = tokenizer.decode(out[0], skip_special_tokens=True)
            gen_only_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            print(f"\n[{name}]")
            print(f"  Seq PPL: {seq_ppl:.2f} | Gen PPL: {gen_ppl:.2f} | Dist-1: {dist1:.2f} | Dist-2: {dist2:.2f}")
            print(f"  Generated Text: \"{gen_only_text}\"")


if __name__ == "__main__":
    run_experiment()
