import torch
import torch.nn as nn

class CustomTokenEmbedding(nn.Module):
    """
    Isolated Token Embedding Layer.
    Maps discrete token vocabulary indices to a dense continuous vector space.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        # Internal parameter matrix tracking the dense representations
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: (Batch, SeqLen)
        # Using index lookup to extract hidden state dimensions cleanly
        return self.weight[tokens]