import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler alternative to LayerNorm used in modern LLMs (Llama, Mistral, etc.).
    
    Instead of: (x - mean) / sqrt(var + eps)
    RMSNorm uses: x / sqrt(mean(x^2) + eps) * gamma
    
    No bias term, only learnable scale (gamma).
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Args:
            dim: Embedding dimension.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Learnable scale parameter (initialized to 1)
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input.
        
        Args:
            x: shape (batch, seq_len, dim) or (batch, dim)
        
        Returns:
            x_norm: same shape as input
        """
        # Compute RMS: sqrt(mean(x^2))
        # Reduce over last dimension only
        x_squared = x ** 2
        rms = torch.sqrt(torch.mean(x_squared, dim=-1, keepdim=True) + self.eps)
        
        # Normalize: x / rms
        x_norm = x / rms
        
        # Scale: x_norm * gamma (weight)
        x_norm = x_norm * self.weight
        
        return x_norm