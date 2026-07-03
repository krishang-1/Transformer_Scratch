"""
Phase 4 done-criterion: model trains to comparable converged loss
with the custom CUDA kernel swapped in for attention.
"""

import torch
import torch.nn as nn

from modules.embedding import CustomTokenEmbedding
from modules.positional import RoPEPositionalEmbedding
from modules.attention_cuda import MultiHeadAttentionCUDA
from modules.rmsnorm import RMSNorm
from modules.swiglu import SwiGLU
from modules.output_projection import OutputProjection


class TransformerBlockCUDA(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, rope_module=None):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.attention = MultiHeadAttentionCUDA(dim, num_heads, rope_module=rope_module)
        self.swiglu = SwiGLU(dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.swiglu(self.norm2(x))
        return x


class TransformerModelCUDA(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, num_layers, hidden_dim=None, tie_weights=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.hidden_dim = hidden_dim or (4 * dim)

        self.embedding = CustomTokenEmbedding(vocab_size, dim)
        self.rope = RoPEPositionalEmbedding(dim)

        self.blocks = nn.ModuleList([
            TransformerBlockCUDA(dim, num_heads, self.hidden_dim, rope_module=self.rope)
            for _ in range(num_layers)
        ])

        self.final_norm = RMSNorm(dim)
        self.output_proj = OutputProjection(dim, vocab_size)

        if tie_weights:
            self.output_proj.weight = self.embedding.weight

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        return self.output_proj(x)


def run_cuda_overfit_test(num_iterations: int = 100):
    assert torch.cuda.is_available(), "This test requires a CUDA GPU."
    device = "cuda"

    torch.manual_seed(42)

    batch_size, seq_len, vocab_size, dim, num_heads, num_layers = 4, 8, 256, 64, 8, 2

    model = TransformerModelCUDA(vocab_size, dim, num_heads, num_layers, hidden_dim=256)
    model.to(device)

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"Attention implementation: {type(model.blocks[0].attention).__name__}")
    print()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    initial_loss = None
    final_loss = None

    for step in range(num_iterations):
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 0:
            initial_loss = loss.item()
        if step == num_iterations - 1:
            final_loss = loss.item()

        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}/{num_iterations} | Loss: {loss.item():.6f}")

    return initial_loss, final_loss


if __name__ == "__main__":
    print("=" * 70)
    print("PHASE 4 GATE 4B: CUDA Kernel Swapped Into Full Model - Overfit Test")
    print("=" * 70)
    print()

    initial_loss, final_loss = run_cuda_overfit_test(num_iterations=100)

    print()
    print("=" * 70)
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss:   {final_loss:.6f}")
    print(f"Loss Reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    print("=" * 70)

    loss_ratio = final_loss / initial_loss
    if loss_ratio < 0.05:
        print("GATE 4B: PASSED")
    else:
        print("GATE 4B: FAILED")
        raise AssertionError(f"CUDA kernel overfit test failed! Loss ratio: {loss_ratio}")