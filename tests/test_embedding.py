import torch
import torch.nn as nn
from modules.embedding import CustomTokenEmbedding

class ReferenceEmbeddingValidation(nn.Module):
    """
    Ground-truth baseline using itemized lookup extraction 
    to verify custom tensor retrieval correctness.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(vocab_size, dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.shape
        dim = self.weight.shape[1]
        
        # Explicitly initialize an un-optimized target output tensor
        out = torch.zeros(batch_size, seq_len, dim, device=tokens.device)
        
        # Manually extract weights sequentially to act as our mathematical baseline
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = tokens[b, s].item()
                out[b, s] = self.weight[token_id]
                
        return out

def run_embedding_test() -> float:
    torch.manual_seed(42)
    vocab_size, hidden_dim, seq_len, batch_size = 500, 64, 8, 2
    
    # Generate discrete token indices
    input_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    custom_layer = CustomTokenEmbedding(vocab_size, hidden_dim)
    ref_layer = ReferenceEmbeddingValidation(vocab_size, hidden_dim)
    
    # Mirror internal parameters to isolate lookup behavior from random initialization
    with torch.no_grad():
        ref_layer.weight.copy_(custom_layer.weight)
        
    custom_out = custom_layer(input_tokens)
    ref_out = ref_layer(input_tokens)
    
    # Calculate maximal discrepancy
    max_error = torch.max(torch.abs(custom_out - ref_out)).item()
    return max_error

if __name__ == "__main__":
    error = run_embedding_test()
    print(f"Component 1A (Embedding) Max Error: {error:.2e}")
    assert error < 1e-4, f"Gate Violation! Error {error} matches or exceeds 1e-4."
    print("Verification Gate: PASSED.")