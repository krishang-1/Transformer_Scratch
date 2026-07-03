import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.rmsnorm import RMSNorm
from modules.swiglu import SwiGLU
from modules.positional import RoPEPositionalEmbedding


class TransformerBlock(nn.Module):
    """
    Single Transformer layer with pre-norm residual connections.
    
    Architecture (pre-norm):
    x → norm1 → attention (with RoPE) → + x → out1
    out1 → norm2 → swiglu → + out1 → out2
    
    Pre-norm: normalization happens BEFORE each sub-layer, skip connections bypass norm.
    This improves gradient flow compared to post-norm.
    """
    
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, rope_module: RoPEPositionalEmbedding = None):
        """
        Args:
            dim: Embedding dimension.
            num_heads: Number of attention heads.
            hidden_dim: Hidden dimension for SwiGLU (typically 4x dim or 8/3 x dim).
            rope_module: RoPE module (optional, passed to attention).
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Pre-norm layers (norm applied before each sub-layer)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        
        # Sub-layers
        self.attention = MultiHeadAttention(dim, num_heads, rope_module=rope_module)
        self.swiglu = SwiGLU(dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block with pre-norm residuals.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            output: shape (batch, seq_len, dim)
        """
        # Attention branch with residual
        x_attn_norm = self.norm1(x)  # Apply norm BEFORE attention
        x_attn = self.attention(x_attn_norm)
        x = x + x_attn  # Residual connection
        
        # MLP branch with residual
        x_mlp_norm = self.norm2(x)  # Apply norm BEFORE MLP
        x_mlp = self.swiglu(x_mlp_norm)
        x = x + x_mlp  # Residual connection
        
        return x