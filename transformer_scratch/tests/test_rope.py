import torch
import torch.nn as nn


class ReferenceRoPEValidation(nn.Module):
    """
    Ground-truth RoPE baseline using explicit nested loops.
    
    For each position m and dimension pair (2i, 2i+1):
    - Compute θ_i = base^(-2i/d)
    - Apply rotation: [cos(m·θ_i), -sin(m·θ_i); sin(m·θ_i), cos(m·θ_i)]
    
    This reference is intentionally un-optimized to be mathematically transparent.
    """
    
    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE via nested loops (one position, one dimension pair at a time).
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            x_rotated: shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize output
        x_rotated = torch.zeros_like(x)
        
        # Iterate over each batch
        for b in range(batch_size):
            # Iterate over each position
            for m in range(seq_len):
                # Iterate over each dimension pair (0,1), (2,3), (4,5), ...
                for i in range(0, dim, 2):
                    # Frequency for this pair
                    theta_i = 1.0 / (self.base ** (i / dim))
                    
                    # Angle for this position
                    angle = m * theta_i
                    
                    # Precompute sin and cos
                    cos_angle = torch.cos(torch.tensor(angle, device=device, dtype=dtype))
                    sin_angle = torch.sin(torch.tensor(angle, device=device, dtype=dtype))
                    
                    # Extract the pair
                    x_even = x[b, m, i]          # x_{2i}
                    x_odd = x[b, m, i + 1]       # x_{2i+1}
                    
                    # Apply rotation
                    x_rotated[b, m, i] = x_even * cos_angle - x_odd * sin_angle
                    x_rotated[b, m, i + 1] = x_even * sin_angle + x_odd * cos_angle
        
        return x_rotated


def run_rope_test() -> float:
    """
    Verify RoPE implementation against reference baseline.
    
    Returns:
        max_error: Maximum absolute error between custom and reference outputs.
    """
    torch.manual_seed(42)
    
    # Test configuration
    batch_size, seq_len, hidden_dim = 2, 8, 64
    base = 10000.0
    
    # Generate random embeddings (as if from token embedding layer)
    input_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Custom RoPE implementation
    from modules.positional import RoPEPositionalEmbedding
    custom_rope = RoPEPositionalEmbedding(hidden_dim, base=base)
    
    # Reference RoPE implementation
    ref_rope = ReferenceRoPEValidation(hidden_dim, base=base)
    
    # Forward pass
    custom_out = custom_rope(input_embeddings)
    ref_out = ref_rope(input_embeddings)
    
    # Calculate maximal discrepancy
    max_error = torch.max(torch.abs(custom_out - ref_out)).item()
    
    return max_error


if __name__ == "__main__":
    error = run_rope_test()
    print(f"Component 1B (RoPE) Max Error: {error:.2e}")
    assert error < 1e-4, f"Gate Violation! Error {error} matches or exceeds 1e-4."
    print("Verification Gate: PASSED.")