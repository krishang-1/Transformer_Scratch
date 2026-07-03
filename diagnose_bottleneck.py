"""
Bottleneck diagnostic: measures CPU-side (tokenization/dataloading) and
GPU-side (model forward+backward) throughput INDEPENDENTLY.
"""

import torch
import torch.nn as nn
import time
from modules.transformer_model import TransformerModel


def measure_cpu_dataloader_throughput(create_loader_fn, num_batches=20, warmup_batches=3):
    print("Measuring CPU-side dataloader throughput (no GPU involved)...")
    loader = create_loader_fn()
    it = iter(loader)

    for _ in range(warmup_batches):
        batch = next(it)

    batch_shape = None
    start = time.perf_counter()
    total_tokens = 0

    for _ in range(num_batches):
        batch = next(it)
        batch_shape = batch.shape
        total_tokens += batch.numel()

    elapsed = time.perf_counter() - start
    tokens_per_sec = total_tokens / elapsed

    print(f"  Batches measured: {num_batches} (shape each: {batch_shape})")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  CPU throughput: {tokens_per_sec:,.0f} tokens/sec")
    return tokens_per_sec


def measure_gpu_model_throughput(vocab_size, dim, num_heads, num_layers,
                                   batch_size, seq_len, num_iters=30, warmup_iters=5, device="cuda"):
    assert torch.cuda.is_available(), "This measurement requires a CUDA GPU."

    print("Measuring GPU-side model throughput (no data loading involved)...")
    model = TransformerModel(vocab_size=vocab_size, dim=dim, num_heads=num_heads, num_layers=num_layers).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    batch = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    for _ in range(warmup_iters):
        logits = model(batch[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), batch[:, 1:].contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    start = time.perf_counter()
    total_tokens = 0

    for _ in range(num_iters):
        logits = model(batch[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), batch[:, 1:].contiguous().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_tokens += batch.numel()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens_per_sec = total_tokens / elapsed

    num_params = model.num_parameters()
    print(f"  Model: {num_params:,} params (dim={dim}, heads={num_heads}, layers={num_layers})")
    print(f"  Iterations measured: {num_iters} (batch={batch_size}, seq_len={seq_len})")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  GPU throughput: {tokens_per_sec:,.0f} tokens/sec")
    return tokens_per_sec


def run_bottleneck_diagnostic(create_loader_fn, vocab_size, dim, num_heads, num_layers,
                                batch_size, seq_len, target_total_tokens=2_000_000_000, device="cuda"):
    print("=" * 70)
    print("BOTTLENECK DIAGNOSTIC: CPU (data) vs GPU (compute)")
    print("=" * 70)
    print()

    cpu_tps = measure_cpu_dataloader_throughput(create_loader_fn)
    print()
    gpu_tps = measure_gpu_model_throughput(vocab_size, dim, num_heads, num_layers, batch_size, seq_len, device=device)
    print()

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"CPU (dataloader) throughput: {cpu_tps:,.0f} tokens/sec")
    print(f"GPU (model compute) throughput: {gpu_tps:,.0f} tokens/sec")
    print()

    bottleneck_tps = min(cpu_tps, gpu_tps)
    bottleneck_name = "CPU (data pipeline)" if cpu_tps < gpu_tps else "GPU (model compute)"

    print(f"BOTTLENECK: {bottleneck_name}")
    print(f"Real-world expected throughput: ~{bottleneck_tps:,.0f} tokens/sec")
    print()

    if cpu_tps < gpu_tps:
        headroom_pct = (gpu_tps - cpu_tps) / gpu_tps * 100
        print(f"The GPU has {headroom_pct:.0f}% idle headroom, waiting on data.")
        print("Recommendation: prioritize MORE vCPUs / num_workers over a faster GPU.")
    else:
        headroom_pct = (cpu_tps - gpu_tps) / cpu_tps * 100
        print(f"The data pipeline has {headroom_pct:.0f}% idle headroom, waiting on GPU.")
        print("Recommendation: prioritize a FASTER GPU over more vCPUs.")

    print()
    estimated_hours = target_total_tokens / bottleneck_tps / 3600
    print(f"Projected time for {target_total_tokens/1e9:.1f}B tokens: {estimated_hours:.1f} hours")
    print("=" * 70)

    return {"cpu_tokens_per_sec": cpu_tps, "gpu_tokens_per_sec": gpu_tps,
            "bottleneck": bottleneck_name, "estimated_hours_for_target": estimated_hours}


if __name__ == "__main__":
    from data.data_loader import create_data_loaders

    def make_loader():
        train_loader, _ = create_data_loaders(batch_size=16, seq_len=1024, num_workers=4)
        return train_loader

    run_bottleneck_diagnostic(
        create_loader_fn=make_loader,
        vocab_size=50257, dim=768, num_heads=12, num_layers=12,
        batch_size=16, seq_len=1024,
        target_total_tokens=2_000_000_000
    )