"""
Phase 4 required deliverable: benchmark report comparing PyTorch's
native attention (Phase 1 implementation) against the custom CUDA
kernel, on identical shapes, on the same GPU.
"""

import torch
import time
from modules.attention import MultiHeadAttention
from modules.attention_cuda import MultiHeadAttentionCUDA
from modules.positional import RoPEPositionalEmbedding


def benchmark_attention(attn_module, x, num_warmup=10, num_iters=50, backward=True):
    for _ in range(num_warmup):
        x_in = x.clone().requires_grad_(backward)
        out = attn_module(x_in)
        if backward:
            out.sum().backward()

    torch.cuda.synchronize()

    forward_times = []
    for _ in range(num_iters):
        x_in = x.clone().requires_grad_(False)
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = attn_module(x_in)
        torch.cuda.synchronize()
        forward_times.append(time.perf_counter() - start)

    results = {
        "forward_mean_ms": sum(forward_times) / len(forward_times) * 1000,
        "forward_min_ms": min(forward_times) * 1000,
    }

    if backward:
        fb_times = []
        for _ in range(num_iters):
            x_in = x.clone().requires_grad_(True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = attn_module(x_in)
            out.sum().backward()
            torch.cuda.synchronize()
            fb_times.append(time.perf_counter() - start)

        results["fwd_bwd_mean_ms"] = sum(fb_times) / len(fb_times) * 1000
        results["fwd_bwd_min_ms"] = min(fb_times) * 1000

    return results


def run_benchmark(batch_size=4, seq_len=512, dim=768, num_heads=12, device="cuda"):
    assert torch.cuda.is_available(), "Benchmark requires a CUDA GPU."

    print("=" * 70)
    print("PHASE 4 BENCHMARK: PyTorch Attention vs Custom CUDA Kernel")
    print("=" * 70)
    print(f"Shape: batch={batch_size}, seq_len={seq_len}, dim={dim}, heads={num_heads}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print()

    torch.manual_seed(42)
    rope = RoPEPositionalEmbedding(dim).to(device)

    pytorch_attn = MultiHeadAttention(dim, num_heads, rope_module=rope).to(device)
    cuda_attn = MultiHeadAttentionCUDA(dim, num_heads, rope_module=rope).to(device)

    with torch.no_grad():
        cuda_attn.W_q.weight.copy_(pytorch_attn.W_q.weight)
        cuda_attn.W_k.weight.copy_(pytorch_attn.W_k.weight)
        cuda_attn.W_v.weight.copy_(pytorch_attn.W_v.weight)
        cuda_attn.W_o.weight.copy_(pytorch_attn.W_o.weight)

    x = torch.randn(batch_size, seq_len, dim, device=device)

    print("Benchmarking PyTorch native attention...")
    pytorch_results = benchmark_attention(pytorch_attn, x)

    print("Benchmarking custom CUDA kernel attention...")
    cuda_results = benchmark_attention(cuda_attn, x)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"{'Metric':<25} {'PyTorch (ms)':<18} {'Custom CUDA (ms)':<20} {'Ratio (CUDA/PyTorch)'}")
    print("-" * 70)

    for key, label in [
        ("forward_mean_ms", "Forward (mean)"),
        ("forward_min_ms", "Forward (min)"),
        ("fwd_bwd_mean_ms", "Forward+Backward (mean)"),
        ("fwd_bwd_min_ms", "Forward+Backward (min)"),
    ]:
        pt_val = pytorch_results[key]
        cu_val = cuda_results[key]
        ratio = cu_val / pt_val
        print(f"{label:<25} {pt_val:<18.4f} {cu_val:<20.4f} {ratio:.2f}x")

    print("=" * 70)
    slower_or_faster = "SLOWER" if cuda_results["fwd_bwd_mean_ms"] > pytorch_results["fwd_bwd_mean_ms"] else "FASTER"
    ratio = cuda_results["fwd_bwd_mean_ms"] / pytorch_results["fwd_bwd_mean_ms"]
    print(f"\nHONEST SUMMARY: Custom CUDA kernel is {ratio:.2f}x PyTorch runtime ({slower_or_faster}).")
    print("=" * 70)

    return pytorch_results, cuda_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    args = parser.parse_args()

    run_benchmark(batch_size=args.batch_size, seq_len=args.seq_len, dim=args.dim, num_heads=args.num_heads)