import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU) feed-forward network.
    
    Two projections with element-wise multiplication and swish activation:
    Output = (W_1(x) ⊙ swish(W_2(x)))
    
    Where swish(z) = z * sigmoid(z)
    
    This is the MLP in modern transformers (Llama, Mistral, etc.)
    """
    
    def __init__(self, dim: int, hidden_dim: int):
        """
        Args:
            dim: Input/output embedding dimension.
            hidden_dim: Intermediate hidden dimension (typically 4x or 8/3 x dim).
        """
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        
        # First projection: dim → hidden_dim
        self.W_1 = nn.Linear(dim, hidden_dim, bias=False)
        
        # Second projection: dim → hidden_dim
        self.W_2 = nn.Linear(dim, hidden_dim, bias=False)
        
        # Output projection: hidden_dim → dim
        self.W_out = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply SwiGLU feed-forward.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            output: shape (batch, seq_len, dim)
        """
        # First branch: W_1(x) → (batch, seq_len, hidden_dim)
        x_1 = self.W_1(x)
        
        # Second branch: W_2(x) → (batch, seq_len, hidden_dim)
        x_2 = self.W_2(x)
        
        # Swish activation: x_2 * sigmoid(x_2)
        swish_x2 = x_2 * torch.sigmoid(x_2)
        
        # Element-wise multiply: x_1 ⊙ swish(x_2)
        gated = x_1 * swish_x2
        
        # Output projection: hidden_dim → dim
        output = self.W_out(gated)
        
        return output