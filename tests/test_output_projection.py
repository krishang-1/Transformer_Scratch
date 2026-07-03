import torch
import torch.nn as nn


class ReferenceOutputProjectionValidation(nn.Module):
    """
    Ground-truth output projection using vectorized matrix multiplication.
    
    Mathematical operation (same as nested loops, just accelerated):
    logits = x @ weight.T
    
    Where:
    - x: (batch, seq_len, dim)
    - weight: (vocab_size, dim)
    - logits: (batch, seq_len, vocab_size)
    
    This reference is mathematically transparent and numerically identical
    to explicit nested loops, just computed efficiently.
    """
    
    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply output projection via matrix multiplication.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """
        # Matrix multiply: x @ weight.T
        # (batch, seq_len, dim) @ (dim, vocab_size) = (batch, seq_len, vocab_size)
        logits = torch.matmul(x, self.weight.t())
        
        return logits


def run_output_projection_test() -> float:
    """
    Verify output projection against reference baseline.
    
    Returns:
        max_error: Maximum absolute error between custom and reference outputs.
    """
    torch.manual_seed(42)
    
    # Test configuration
    batch_size, seq_len, hidden_dim = 2, 8, 64
    vocab_size = 1024  # Vocabulary size
    
    # Generate random embeddings
    input_x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Custom output projection implementation
    from modules.output_projection import OutputProjection
    custom_proj = OutputProjection(hidden_dim, vocab_size)
    
    # Reference output projection implementation
    ref_proj = ReferenceOutputProjectionValidation(hidden_dim, vocab_size)
    
    # Mirror parameters for fair comparison
    with torch.no_grad():
        ref_proj.weight.copy_(custom_proj.weight)
    
    # Forward pass
    custom_out = custom_proj(input_x)
    ref_out = ref_proj(input_x)
    
    # Calculate maximal discrepancy
    max_error = torch.max(torch.abs(custom_out - ref_out)).item()
    
    return max_error


if __name__ == "__main__":
    error = run_output_projection_test()
    print(f"Component 1F (Output Projection) Max Error: {error:.2e}")
    assert error < 1e-4, f"Gate Violation! Error {error} matches or exceeds 1e-4."
    print("Verification Gate: PASSED.")