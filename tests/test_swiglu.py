import torch
import torch.nn as nn


class ReferenceSwiGLUValidation(nn.Module):
    """
    Ground-truth SwiGLU using explicit nested loops.
    
    For each position:
    - Compute W_1(x) branch
    - Compute W_2(x) branch with swish activation
    - Element-wise multiply
    - Project back to original dimension
    
    This reference is intentionally un-optimized for mathematical transparency.
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        self.W_1 = nn.Linear(dim, hidden_dim, bias=False)
        self.W_2 = nn.Linear(dim, hidden_dim, bias=False)
        self.W_out = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU via explicit computation (no optimizations).
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            output: shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        dtype = x.dtype
        
        # Initialize output
        output = torch.zeros(batch_size, seq_len, dim, device=device, dtype=dtype)
        
        # Iterate over batch
        for b in range(batch_size):
            # Iterate over sequence
            for s in range(seq_len):
                # Extract position: (dim,)
                x_pos = x[b, s, :]
                
                # First branch: W_1(x) → (hidden_dim,)
                x_1 = self.W_1(x_pos)
                
                # Second branch: W_2(x) → (hidden_dim,)
                x_2 = self.W_2(x_pos)
                
                # Swish activation: x_2 * sigmoid(x_2)
                sigmoid_x2 = torch.sigmoid(x_2)
                swish_x2 = x_2 * sigmoid_x2
                
                # Element-wise multiply: x_1 ⊙ swish(x_2) → (hidden_dim,)
                gated = x_1 * swish_x2
                
                # Output projection: (hidden_dim,) → (dim,)
                output_pos = self.W_out(gated)
                
                output[b, s, :] = output_pos
        
        return output


def run_swiglu_test() -> float:
    """
    Verify SwiGLU against reference baseline.
    
    Returns:
        max_error: Maximum absolute error between custom and reference outputs.
    """
    torch.manual_seed(42)
    
    # Test configuration
    batch_size, seq_len, hidden_dim = 2, 8, 64
    hidden_intermediate = 256  # Typically 4x or 8/3 x hidden_dim
    
    # Generate random input
    input_x = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Custom SwiGLU implementation
    from modules.swiglu import SwiGLU
    custom_swiglu = SwiGLU(hidden_dim, hidden_intermediate)
    
    # Reference SwiGLU implementation
    ref_swiglu = ReferenceSwiGLUValidation(hidden_dim, hidden_intermediate)
    
    # Mirror parameters for fair comparison
    with torch.no_grad():
        ref_swiglu.W_1.weight.copy_(custom_swiglu.W_1.weight)
        ref_swiglu.W_2.weight.copy_(custom_swiglu.W_2.weight)
        ref_swiglu.W_out.weight.copy_(custom_swiglu.W_out.weight)
    
    # Forward pass
    custom_out = custom_swiglu(input_x)
    ref_out = ref_swiglu(input_x)
    
    # Calculate maximal discrepancy
    max_error = torch.max(torch.abs(custom_out - ref_out)).item()
    
    return max_error


if __name__ == "__main__":
    error = run_swiglu_test()
    print(f"Component 1E (SwiGLU) Max Error: {error:.2e}")
    assert error < 1e-4, f"Gate Violation! Error {error} matches or exceeds 1e-4."
    print("Verification Gate: PASSED.")