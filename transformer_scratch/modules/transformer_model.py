import torch
import torch.nn as nn
from modules.embedding import CustomTokenEmbedding
from modules.positional import RoPEPositionalEmbedding
from modules.transformer_block import TransformerBlock
from modules.output_projection import OutputProjection
from modules.rmsnorm import RMSNorm


class TransformerModel(nn.Module):
    """
    Full transformer model: stacked blocks with embeddings and output projection.
    Default config targets ~123M parameters, with tied input/output embeddings.
    """

    def __init__(self, vocab_size: int = 50257, dim: int = 768, num_heads: int = 12,
                 num_layers: int = 12, hidden_dim: int = None, tie_weights: bool = True):
        super().__init__()

        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"

        self.vocab_size = vocab_size
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim or int(round((8 / 3) * dim / 64)) * 64
        self.tie_weights = tie_weights

        self.embedding = CustomTokenEmbedding(vocab_size, dim)
        self.rope = RoPEPositionalEmbedding(dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, self.hidden_dim, rope_module=self.rope)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(dim)
        self.output_proj = OutputProjection(dim, vocab_size)

        if self.tie_weights:
            self.output_proj.weight = self.embedding.weight

    def num_parameters(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_proj(x)


if __name__ == "__main__":
    model = TransformerModel()
    total = model.num_parameters()
    print(f"Model config: dim={model.dim}, heads={model.num_heads}, "
          f"layers={model.num_layers}, hidden_dim={model.hidden_dim}")
    print(f"Weight tying: {model.tie_weights}")
    print(f"Total trainable parameters: {total:,} ({total / 1e6:.1f}M)")