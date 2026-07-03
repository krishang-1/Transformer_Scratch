import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# JIT-compile the CUDA extension on first import. This will take
# 30-90 seconds the first time (nvcc compilation), then it's cached.
_kernel_dir = os.path.dirname(os.path.abspath(__file__))
_cuda_src = os.path.join(_kernel_dir, "..", "kernels", "attention_cuda.cu")

print("Compiling custom CUDA attention kernel (first run only)...")
custom_attention_cuda = load(
    name="custom_attention_cuda",
    sources=[_cuda_src],
    verbose=True
)
print("CUDA kernel compiled successfully.")


class CausalAttentionCUDAFunction(torch.autograd.Function):
    """
    Autograd bridge between PyTorch and our hand-written CUDA kernel.

    forward() calls the CUDA forward kernel and saves what backward() needs.
    backward() calls the CUDA backward kernel we derived by hand.
    """

    @staticmethod
    def forward(ctx, Q, K, V, scale):
        Q = Q.contiguous()
        K = K.contiguous()
        V = V.contiguous()

        O, P = custom_attention_cuda.forward(Q, K, V, scale)

        ctx.save_for_backward(Q, K, V, P)
        ctx.scale = scale
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, P = ctx.saved_tensors
        dQ, dK, dV = custom_attention_cuda.backward(dO, Q, K, V, P, ctx.scale)
        # 4th return is for `scale`, which has no gradient
        return dQ, dK, dV, None


class MultiHeadAttentionCUDA(nn.Module):
    """
    Drop-in replacement for Phase 1's MultiHeadAttention, with the
    score -> softmax -> weighted-sum step replaced by our custom CUDA kernel.

    Q/K/V projections, RoPE, and output projection remain in PyTorch -
    those are just GEMMs / cheap elementwise ops, not the "attention hot path."

    Same __init__ signature and forward() interface as Phase 1's
    MultiHeadAttention, so it can be swapped in without touching
    TransformerBlock or TransformerModel.
    """

    def __init__(self, dim: int, num_heads: int, rope_module=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.dim = dim
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.scale = 1.0 / (self.d_k ** 0.5)
        self.rope_module = rope_module

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)
        return x.reshape(batch_size, seq_len, self.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if self.rope_module:
            Q = self.rope_module(Q)
            K = self.rope_module(K)

        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # This line is the entire point of Phase 4: the score/softmax/
        # weighted-sum computation now runs through our hand-written
        # CUDA kernel instead of torch.matmul + torch.softmax.
        attn_out = CausalAttentionCUDAFunction.apply(Q, K, V, self.scale)

        attn_out = self._merge_heads(attn_out)
        return self.W_o(attn_out)