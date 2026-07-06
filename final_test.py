import sys
sys.path.insert(0, r"C:\WS_Internship_Project\transformer_scratch")

import torch
import torch.nn.functional as F
import math
from collections import Counter
from modules.transformer_model import TransformerModel
from data.data_loader import create_data_loaders

# ============================================================
# LOAD MODEL
# ============================================================
model = TransformerModel(
    vocab_size=50257, dim=768, num_heads=12, num_layers=12,
    hidden_dim=2048, tie_weights=True
)
model.load_state_dict(torch.load("final_model.pt", map_location="cpu"))
model.eval()
print("Model loaded successfully.\n")


def evaluate_model(model, dataloader, device="cpu", max_batches=30, vocab_size=50257):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total_tokens = 0
    total_loss = 0.0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]

            predictions = logits.argmax(dim=-1)
            correct_top1 += (predictions == targets).sum().item()

            top5_preds = logits.topk(5, dim=-1).indices
            correct_top5 += (top5_preds == targets.unsqueeze(-1)).any(dim=-1).sum().item()

            total_tokens += targets.numel()

            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.contiguous().view(-1),
                reduction='sum'
            )
            total_loss += loss.item()

            if (i + 1) % 5 == 0:
                print(f"  Batch {i+1}/{max_batches} processed...")

    avg_loss = total_loss / total_tokens
    return {
        "top1_accuracy": (correct_top1 / total_tokens) * 100,
        "top5_accuracy": (correct_top5 / total_tokens) * 100,
        "avg_loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "tokens_evaluated": total_tokens
    }


def compute_train_loss_sample(model, dataloader, device="cpu", max_batches=10, vocab_size=50257):
    """Quick check on TRAIN split loss, to compare against validation for an overfitting signal."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            logits = model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.contiguous().view(-1),
                reduction='sum'
            )
            total_loss += loss.item()
            total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return avg_loss, math.exp(avg_loss)


def distinct_n(text: str, n: int) -> float:
    """
    Distinct-n: fraction of n-grams in generated text that are unique.
    Standard NLG repetition metric — lower values indicate more repetition.
    """
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def generate_sample(model, prompt_tokens, max_length=80, temperature=0.8, top_k=50):
    model.eval()
    input_ids = prompt_tokens.clone()
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_logits = logits[0, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices[next_token_idx].unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    return input_ids


# ============================================================
# RUN EVALUATION
# ============================================================
print("Loading WikiText-103 data...")
train_loader, val_loader = create_data_loaders(batch_size=4, seq_len=1024, num_workers=0)

print("\nRunning validation evaluation (30 batches)...")
results = evaluate_model(model, val_loader, device="cpu", max_batches=30)

print("\nComputing train-split loss for overfitting check (10 batches)...")
train_loss, train_ppl = compute_train_loss_sample(model, train_loader, device="cpu", max_batches=10)

# ============================================================
# GENERATE SAMPLES + REPETITION METRIC
# ============================================================
from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

print("\nGenerating fresh samples...")
prompts = ["The history of", "In recent years, scientists have", "The film was released"]
samples = []
for prompt_text in prompts:
    prompt_ids = torch.tensor([tokenizer.encode(prompt_text)])
    generated = generate_sample(model, prompt_ids, max_length=60)
    samples.append(tokenizer.decode(generated[0]))

distinct_1_scores = [distinct_n(s, 1) for s in samples]
distinct_2_scores = [distinct_n(s, 2) for s in samples]
avg_distinct_1 = sum(distinct_1_scores) / len(distinct_1_scores)
avg_distinct_2 = sum(distinct_2_scores) / len(distinct_2_scores)

# ============================================================
# FINAL REPORT (markdown-ready)
# ============================================================
gap = results['perplexity'] - train_ppl

print("\n" + "=" * 70)
print("FINAL MODEL EVALUATION REPORT")
print("=" * 70)
print(f"""
| Metric                    | Value          |
|---------------------------|-----------------|
| Parameters                | 123.6M          |
| Architecture              | dim=768, heads=12, layers=12, tied embeddings |
| Dataset                   | WikiText-103    |
| Training steps            | 36,621 (~3 epochs) |
| Tokens evaluated (val)    | {results['tokens_evaluated']:,} |
| Next-token accuracy       | {results['top1_accuracy']:.2f}% |
| Top-5 accuracy            | {results['top5_accuracy']:.2f}% |
| Validation loss           | {results['avg_loss']:.4f} |
| Validation perplexity     | {results['perplexity']:.2f} |
| Train perplexity (sample) | {train_ppl:.2f} |
| Train/val gap             | {gap:+.2f} |
| Distinct-1 (generation)   | {avg_distinct_1:.3f} |
| Distinct-2 (generation)   | {avg_distinct_2:.3f} |
""")
print("=" * 70)
print("\nSample Generations:")
for i, (s, d1, d2) in enumerate(zip(samples, distinct_1_scores, distinct_2_scores), 1):
    print(f"\n[{i}] (distinct-1: {d1:.2f}, distinct-2: {d2:.2f})")
    print(f"    {s}")
print("\n" + "=" * 70)
print("\nInterpretation notes:")
print(f"- Train/val gap of {gap:+.2f} perplexity points indicates "
      f"{'healthy generalization, minimal overfitting' if abs(gap) < 3 else 'some overfitting - worth noting'}")
print("- Distinct-n closer to 1.0 = less repetitive; scores here reflect")
print("  the expected small-model repetition pattern observed during training")
print("=" * 70)