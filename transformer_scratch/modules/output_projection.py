import torch
import torch.nn as nn


class OutputProjection(nn.Module):
    """
    Output projection layer: embedding space → vocabulary logits.
    
    Maps (batch, seq_len, dim) → (batch, seq_len, vocab_size)
    for language modeling loss computation (next token prediction).
    
    Single learnable linear projection without bias.
    """
    
    def __init__(self, dim: int, vocab_size: int):
        """
        Args:
            dim: Embedding dimension.
            vocab_size: Size of vocabulary.
        """
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        
        # Linear projection: dim → vocab_size (no bias)
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project embeddings to vocabulary logits.
        
        Args:
            x: shape (batch, seq_len, dim)
        
        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """
        # Matrix multiply: (batch, seq_len, dim) @ (dim, vocab_size)^T
        # = (batch, seq_len, vocab_size)
        logits = torch.matmul(x, self.weight.t())
        
        return logits