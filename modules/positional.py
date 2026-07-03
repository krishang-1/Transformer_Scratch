import torch
import torch.nn as nn


class RoPEPositionalEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE) — standard implementation.
    
    For each position m, applies a rotation to dimension pairs (2i, 2i+1)
    using frequency θ_i = base^(-2i/d), where base defaults to 10000.
    
    Interleaved pairs: [cos(m·θ_0), sin(m·θ_0), cos(m·θ_1), sin(m·θ_1), ...]
    applied via: x' = [x_0·cos - x_1·sin, x_0·sin + x_1·cos, ...]
    """
    
    def __init__(self, dim: int, base: float = 10000.0, device: str = "cpu"):
        """
        Args:
            dim: Embedding dimension (must be even).
            base: Frequency base (default 10000, matches transformer literature).
            device: Torch device (cpu or cuda).
        """
        super().__init__()
        assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
        
        self.dim = dim
        self.base = base
        self.device = device
        
        # Lazy cache: will store precomputed sin/cos for (max_seq_len, dim)
        self._cache = {}
    
    def _compute_frequencies(self) -> torch.Tensor:
        """
        Compute inverse frequency for each dimension pair.
        
        θ_i = base^(-2i/d) for i in [0, d/2)
        
        Returns:
            freq: shape (dim/2,), device-agnostic
        """
        # Dimension indices: 0, 2, 4, ..., dim-2
        i = torch.arange(0, self.dim, 2, dtype=torch.float32)
        
        # θ_i = base^(-2i/d)
        freq = 1.0 / (self.base ** (i / self.dim))
        
        return freq  # shape: (dim/2,)
    
    def _get_sin_cos_cache(self, seq_len: int) -> tuple:
        """
        Retrieve or compute cached sin/cos tables for this seq_len.
        
        Returns:
            (sin_table, cos_table): each shape (seq_len, dim)
        """
        cache_key = seq_len
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Compute frequency: (dim/2,)
        freq = self._compute_frequencies()
        
        # Position indices: (seq_len,)
        positions = torch.arange(seq_len, dtype=torch.float32)
        
        # Outer product: positions ⊗ freq -> (seq_len, dim/2)
        angles = torch.outer(positions, freq)
        
        # Interleave: duplicate each angle for the pair
        # (seq_len, dim/2) -> (seq_len, dim)
        angles_interleaved = torch.repeat_interleave(angles, 2, dim=1)
        
        # Precompute sin and cos
        sin_table = torch.sin(angles_interleaved)
        cos_table = torch.cos(angles_interleaved)
        
        # Cache for reuse
        self._cache[cache_key] = (sin_table, cos_table)
        
        return sin_table, cos_table
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """
        Apply RoPE to input embeddings.
        
        Args:
            x: shape (batch, seq_len, dim)
            positions: (optional) shape (seq_len,) or (batch, seq_len) for custom positions.
                       If None, uses [0, 1, ..., seq_len-1].
        
        Returns:
            x_rotated: shape (batch, seq_len, dim)
        """
        batch_size, seq_len, dim = x.shape
        assert dim == self.dim, f"Expected dim {self.dim}, got {dim}"
        
        # Default positions: [0, 1, ..., seq_len-1]
        if positions is None:
            positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        
        # Get sin/cos tables (cached)
        sin_table, cos_table = self._get_sin_cos_cache(seq_len)
        sin_table = sin_table.to(x.device).to(x.dtype)
        cos_table = cos_table.to(x.device).to(x.dtype)
        
        # Apply rotation: x' = [x_0·cos - x_1·sin, x_0·sin + x_1·cos, ...]
        # Pair consecutive dimensions
        x_0 = x[..., 0::2]  # Even indices: (batch, seq_len, dim/2)
        x_1 = x[..., 1::2]  # Odd indices: (batch, seq_len, dim/2)
        
        cos_vals = cos_table[..., 0::2]  # Even (cos for each pair)
        sin_vals = sin_table[..., 0::2]  # Even (sin for each pair)
        
        # Apply rotation matrix
        x_0_rot = x_0 * cos_vals - x_1 * sin_vals
        x_1_rot = x_0 * sin_vals + x_1 * cos_vals
        
        # Interleave back: [x_0_rot, x_1_rot, x_0_rot, x_1_rot, ...]
        x_rotated = torch.zeros_like(x)
        x_rotated[..., 0::2] = x_0_rot
        x_rotated[..., 1::2] = x_1_rot
        
        return x_rotated