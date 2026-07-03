import torch
import torch.nn as nn


class ReferenceMultiHeadAttention(nn.Module):
    """
    Ground-truth multi-head attention using explicit nested loops.
    
    For each batch, head, and position:
    - Compute attention scores manually
    - Apply causal mask
    - Compute softmax
    - Weight values
    
    This reference is intentionally un-optimized for mathematical transparency.
    """
    
    def __init__(self, dim: int, num_heads: int, rope_module=None):
        super().__init__()
        assert dim % num_heads == 0
        self.rope_module = rope_module
        
        self.dim = dim
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        self.scale = 1.0 / (self.d_k ** 0.5)
        
        # Same projections as custom implementation
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply attention via nested loops (one head, one query at a time).
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            attn_out: shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Project: (batch, seq_len, dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if self.rope_module:
            Q = self.rope_module(Q)
            K = self.rope_module(K)
        
        # Initialize output
        attn_out = torch.zeros(batch_size, seq_len, dim, device=device, dtype=dtype)
        
        # Iterate over batch
        for b in range(batch_size):
            # Iterate over each head
            for h in range(self.num_heads):
                # Extract head slices: (seq_len, d_k)
                Q_h = Q[b, :, h * self.d_k:(h + 1) * self.d_k]
                K_h = K[b, :, h * self.d_k:(h + 1) * self.d_k]
                V_h = V[b, :, h * self.d_k:(h + 1) * self.d_k]
                
                # Initialize head output: (seq_len, d_k)
                head_out = torch.zeros(seq_len, self.d_k, device=device, dtype=dtype)
                
                # Iterate over each query position
                for i in range(seq_len):
                    q_i = Q_h[i]  # (d_k,)
                    
                    # Compute attention scores for this query
                    scores = torch.zeros(seq_len, device=device, dtype=dtype)
                    
                    for j in range(seq_len):
                        # Causal mask: can only attend to j <= i
                        if j <= i:
                            k_j = K_h[j]  # (d_k,)
                            score = torch.dot(q_i, k_j) * self.scale
                            scores[j] = score
                        else:
                            scores[j] = float('-inf')
                    
                    # Softmax with numerical stability
                    scores_safe = scores.clone()
                    scores_safe[~torch.isfinite(scores_safe)] = 0.0
                    max_score = torch.max(scores_safe)
                    scores_shifted = scores - max_score
                    scores_shifted[~torch.isfinite(scores)] = float('-inf')
                    
                    exp_scores = torch.exp(scores_shifted)
                    exp_scores[~torch.isfinite(scores)] = 0.0
                    attn_weights = exp_scores / (torch.sum(exp_scores) + 1e-10)
                    
                    # Weighted sum of values
                    for j in range(seq_len):
                        if j <= i:
                            v_j = V_h[j]  # (d_k,)
                            head_out[i] += attn_weights[j] * v_j
                
                # Place head output into the full output
                attn_out[b, :, h * self.d_k:(h + 1) * self.d_k] = head_out
        
        # Output projection
        attn_out = attn_out.view(batch_size, seq_len, dim)
        attn_out = self.W_o(attn_out)
        
        return attn_out


def run_attention_test() -> float:
    """
    Verify multi-head attention against reference baseline.
    
    Returns:
        max_error: Maximum absolute error between custom and reference outputs.
    """
    torch.manual_seed(42)
    
    # Test configuration
    batch_size, seq_len, hidden_dim, num_heads = 2, 8, 64, 8
    
    # Generate random input
    input_x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Initialize RoPE module
    from modules.positional import RoPEPositionalEmbedding
    rope_module = RoPEPositionalEmbedding(hidden_dim)

    # Custom attention implementation
    from modules.attention import MultiHeadAttention
    custom_attn = MultiHeadAttention(hidden_dim, num_heads)
    
    # Reference attention implementation
    ref_attn = ReferenceMultiHeadAttention(hidden_dim, num_heads)
    
    # Mirror parameters for fair comparison
    with torch.no_grad():
        ref_attn.W_q.weight.copy_(custom_attn.W_q.weight)
        ref_attn.W_k.weight.copy_(custom_attn.W_k.weight)
        ref_attn.W_v.weight.copy_(custom_attn.W_v.weight)
        ref_attn.W_o.weight.copy_(custom_attn.W_o.weight)
    
    # Forward pass
    custom_out = custom_attn(input_x)
    ref_out = ref_attn(input_x)
    
    # Calculate maximal discrepancy
    max_error = torch.max(torch.abs(custom_out - ref_out)).item()
    
    return max_error


if __name__ == "__main__":
    error = run_attention_test()
    print(f"Component 1C (Multi-Head Attention) Max Error: {error:.2e}")
    assert error < 1e-4, f"Gate Violation! Error {error} matches or exceeds 1e-4."
    print("Verification Gate: PASSED.")