"""
Phase 4 verification: does the custom CUDA kernel produce the same
forward output AND the same gradients as Phase 1's PyTorch attention?

This must run on a CUDA-capable GPU (not local CPU).
"""

import torch
from modules.attention import MultiHeadAttention          # Phase 1 reference
from modules.attention_cuda import MultiHeadAttentionCUDA  # Phase 4 kernel
from modules.positional import RoPEPositionalEmbedding


def run_cuda_attention_forward_test(
    batch_size=2, seq_len=64, dim=64, num_heads=8, device="cuda"
):
    torch.manual_seed(42)

    rope = RoPEPositionalEmbedding(dim).to(device)

    ref_attn = MultiHeadAttention(dim, num_heads, rope_module=rope).to(device)
    cuda_attn = MultiHeadAttentionCUDA(dim, num_heads, rope_module=rope).to(device)

    with torch.no_grad():
        cuda_attn.W_q.weight.copy_(ref_attn.W_q.weight)
        cuda_attn.W_k.weight.copy_(ref_attn.W_k.weight)
        cuda_attn.W_v.weight.copy_(ref_attn.W_v.weight)
        cuda_attn.W_o.weight.copy_(ref_attn.W_o.weight)

    x = torch.randn(batch_size, seq_len, dim, device=device)

    ref_out = ref_attn(x)
    cuda_out = cuda_attn(x)

    max_error = torch.max(torch.abs(ref_out - cuda_out)).item()
    print(f"Forward max abs error: {max_error:.2e}")
    return max_error


def run_cuda_attention_backward_test(
    batch_size=2, seq_len=64, dim=64, num_heads=8, device="cuda"
):
    torch.manual_seed(42)

    rope = RoPEPositionalEmbedding(dim).to(device)

    ref_attn = MultiHeadAttention(dim, num_heads, rope_module=rope).to(device)
    cuda_attn = MultiHeadAttentionCUDA(dim, num_heads, rope_module=rope).to(device)

    with torch.no_grad():
        cuda_attn.W_q.weight.copy_(ref_attn.W_q.weight)
        cuda_attn.W_k.weight.copy_(ref_attn.W_k.weight)
        cuda_attn.W_v.weight.copy_(ref_attn.W_v.weight)
        cuda_attn.W_o.weight.copy_(ref_attn.W_o.weight)

    x_ref = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
    x_cuda = x_ref.clone().detach().requires_grad_(True)

    ref_out = ref_attn(x_ref)
    cuda_out = cuda_attn(x_cuda)

    ref_loss = ref_out.pow(2).sum()
    cuda_loss = cuda_out.pow(2).sum()

    ref_loss.backward()
    cuda_loss.backward()

    results = {}

    def relative_error(a, b, name):
        err = torch.max(torch.abs(a - b)).item()
        results[name] = err
        print(f"  {name:20s} max abs error: {err:.2e}")

    relative_error(x_ref.grad, x_cuda.grad, "dL/dx")
    relative_error(ref_attn.W_q.weight.grad, cuda_attn.W_q.weight.grad, "dL/dW_q")
    relative_error(ref_attn.W_k.weight.grad, cuda_attn.W_k.weight.grad, "dL/dW_k")
    relative_error(ref_attn.W_v.weight.grad, cuda_attn.W_v.weight.grad, "dL/dW_v")
    relative_error(ref_attn.W_o.weight.grad, cuda_attn.W_o.weight.grad, "dL/dW_o")

    return results


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This test requires a CUDA GPU."

    print("=" * 60)
    print("PHASE 4 GATE: CUDA Attention Kernel Verification")
    print("=" * 60)

    print("\n--- Forward Pass Test ---")
    fwd_error = run_cuda_attention_forward_test()
    fwd_status = "PASSED" if fwd_error < 1e-3 else "FAILED"
    print(f"Forward test: {fwd_status} (tolerance: 1e-3)")

    print("\n--- Backward Pass Test ---")
    bwd_results = run_cuda_attention_backward_test()
    bwd_pass = all(v < 1e-3 for v in bwd_results.values())
    bwd_status = "PASSED" if bwd_pass else "FAILED"
    print(f"Backward test: {bwd_status} (tolerance: 1e-3)")

    print("\n" + "=" * 60)
    if fwd_status == "PASSED" and bwd_status == "PASSED":
        print("GATE 4A: CUDA KERNEL VERIFICATION - PASSED")
    else:
        print("GATE 4A: CUDA KERNEL VERIFICATION - FAILED")
    print("=" * 60)