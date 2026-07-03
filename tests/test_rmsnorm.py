import torch
import torch.nn as nn


class ReferenceRMSNormValidation(nn.Module):
    """
    Ground-truth RMSNorm using explicit nested loops.
    
    For each position in sequence:
    - Compute RMS over the embedding dimension
    - Normalize each value
    - Scale by learnable weight
    
    This reference is intentionally un-optimized for mathematical transparency.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm via nested loops.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            x_norm: shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize output
        x_norm = torch.zeros_like(x)
        
        # Iterate over batch
        for b in range(batch_size):
            # Iterate over sequence
            for s in range(seq_len):
                # Extract position: (dim,)
                x_pos = x[b, s, :]
                
                # Compute RMS: sqrt(mean(x^2))
                x_squared = x_pos ** 2
                mean_x_squared = torch.mean(x_squared)
                rms = torch.sqrt(mean_x_squared + self.eps)
                
                # Normalize: x / rms
                x_norm_pos = x_pos / rms
                
                # Scale: x_norm * weight
                x_norm_pos = x_norm_pos * self.weight
                
                x_norm[b, s, :] = x_norm_pos
        
        return x_norm


def run_rmsnorm_test() -> float:
    """
    Verify RMSNorm against reference baseline.
    
    Returns:
        max_error: Maximum absolute error between custom and reference outputs.
    """
    torch.manual_seed(42)
    
    # Test configuration
    batch_size, seq_len, hidden_dim = 2, 8, 64
    
    # Generate random input
    input_x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Custom RMSNorm implementation
    from modules.rmsnorm import RMSNorm
    custom_norm = RMSNorm(hidden_dim)
    
    # Reference RMSNorm implementation
    ref_norm = ReferenceRMSNormValidation(hidden_dim)
    
    # Mirror parameters for fair comparison
    with torch.no_grad():
        ref_norm.weight.copy_(custom_norm.weight)
    
    # Forward pass
    custom_out = custom_norm(input_x)
    ref_out = ref_norm(input_x)
    
    # Calculate maximal discrepancy
    max_error = torch.max(torch.abs(custom_out - ref_out)).item()
    
    return max_error


if __name__ == "__main__":
    error = run_rmsnorm_test()
    print(f"Component 1D (RMSNorm) Max Error: {error:.2e}")
    assert error < 1e-4, f"Gate Violation! Error {error} matches or exceeds 1e-4."
    print("Verification Gate: PASSED.")