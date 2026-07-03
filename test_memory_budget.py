"""
Memory smoke test: run BEFORE committing to a long training run.
"""

import torch
import torch.nn as nn
from modules.transformer_model import TransformerModel


def run_memory_test(batch_size=16, seq_len=1024, vocab_size=50257,
                     dim=768, num_heads=12, num_layers=12, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, falling back to CPU.")
        device = "cpu"

    print("=" * 60)
    print("MEMORY BUDGET SMOKE TEST")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size} | Seq len: {seq_len}")
    print(f"dim={dim}, heads={num_heads}, layers={num_layers}")
    print()

    model = TransformerModel(vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers)
    model.to(device)

    num_params = model.num_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    batch = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    print("\nRunning forward pass...")
    logits = model(batch[:, :-1])

    print("Computing loss...")
    loss = loss_fn(logits.reshape(-1, vocab_size), batch[:, 1:].contiguous().view(-1))
    print(f"Initial loss: {loss.item():.4f}")

    print("Running backward pass...")
    optimizer.zero_grad()
    loss.backward()

    print("Running optimizer step...")
    optimizer.step()

    if device == "cuda":
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)

        print("\n" + "=" * 60)
        print("MEMORY RESULTS")
        print("=" * 60)
        print(f"Peak allocated:  {peak_mb:,.0f} MB ({peak_mb / 1024:.2f} GB)")
        print(f"Peak reserved:   {reserved_mb:,.0f} MB ({reserved_mb / 1024:.2f} GB)")
        print(f"GPU total:       {total_mb:,.0f} MB ({total_mb / 1024:.2f} GB)")
        print(f"Headroom:        {total_mb - reserved_mb:,.0f} MB ({(total_mb - reserved_mb) / 1024:.2f} GB)")
        print(f"Utilization:     {reserved_mb / total_mb * 100:.1f}%")
        print("=" * 60)

        if reserved_mb / total_mb > 0.85:
            print("\n[WARNING] Using >85% of GPU memory. Consider reducing batch_size/seq_len.")
        else:
            print("\nComfortable headroom for this batch_size/seq_len.")
    else:
        print("\n[INFO] Ran on CPU - no GPU memory stats available.")

    print("\nNo NaNs in loss:", not torch.isnan(loss).item())
    print("Memory smoke test PASSED (ran without OOM or errors).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_memory_test(batch_size=args.batch_size, seq_len=args.seq_len, dim=args.dim,
                     num_heads=args.num_heads, num_layers=args.num_layers, device=args.device)