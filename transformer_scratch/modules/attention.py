import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    Scaled Dot-Product Multi-Head Attention with causal masking.
    
    For each head:
    - Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
    - Causal mask prevents attending to future positions
    
    Multiple heads run in parallel, then concatenated and projected.
    """
    
    def __init__(self, dim: int, num_heads: int, rope_module=None):
        """
        Args:
            dim: Embedding dimension (must be divisible by num_heads).
            num_heads: Number of attention heads.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.d_k = dim // num_heads  # Dimension per head
        self.scale = 1.0 / (self.d_k ** 0.5)  # Scaling factor
        self.rope_module = rope_module

        # Linear projections for Q, K, V (no bias for simplicity/verification)
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(dim, dim, bias=False)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Split embedding into multiple heads.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            x_split: shape (batch, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch, num_heads, seq_len, d_k)
    
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Merge multiple heads back into single embedding.
        
        Args:
            x: shape (batch, num_heads, seq_len, d_k)
        
        Returns:
            x_merged: shape (batch, seq_len, dim)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2)  # (batch, seq_len, num_heads, d_k)
        return x.reshape(batch_size, seq_len, self.dim)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create causal mask: position i can attend to positions <= i.
        
        Args:
            seq_len: Sequence length.
            device: Torch device.
        
        Returns:
            mask: shape (seq_len, seq_len), True where attention is allowed.
        """
        # Upper triangular mask (prevent attending to future)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head attention with causal masking.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            attn_out: shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        device = x.device
        
        # Project to Q, K, V: (batch, seq_len, dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        if self.rope_module:
            Q = self.rope_module(Q)
            K = self.rope_module(K)
        
        # Split into heads: (batch, num_heads, seq_len, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)
        
        # Scaled dot-product: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = self._create_causal_mask(seq_len, device)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        
        # Softmax over key dimension
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Handle NaN from -inf (masked positions)
        attn_weights = torch.nan_to_num(attn_weights, 0.0)
        
        # Apply attention to values: (batch, num_heads, seq_len, d_k)
        attn_out = torch.matmul(attn_weights, V)
        
        # Merge heads: (batch, seq_len, dim)
        attn_out = self._merge_heads(attn_out)
        
        # Output projection: (batch, seq_len, dim)
        attn_out = self.W_o(attn_out)
        
        return attn_out