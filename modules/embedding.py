import torch
import torch.nn as nn

class CustomTokenEmbedding(nn.Module):
    """
    Token embedding lookup layer.

    init_std=0.02 (GPT-2 style) rather than raw torch.randn (std=1) is
    critical when this embedding matrix is weight-tied to the output
    projection. With std=1, residual connections let untrained hidden
    states stay close to their own embedding vector, making the logit
    for the "correct" token approximately ||embedding[token]||^2 - a
    large, trivial self-similarity artifact (not real learning) that
    produces deceptively near-zero loss on a completely untrained model.
    Scaling down to std=0.02 keeps this artifact negligible at init.
    """
    def __init__(self, vocab_size: int, dim: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim) * init_std)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens shape: (Batch, SeqLen)
        return self.weight[tokens]
