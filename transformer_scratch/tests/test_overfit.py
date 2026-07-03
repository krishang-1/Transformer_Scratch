import torch
import torch.nn as nn
from modules.transformer_model import TransformerModel


def run_overfit_test(num_iterations: int = 100) -> tuple:
    """
    Test that model can overfit a single tiny batch to near-zero loss.
    
    This verifies:
    1. Forward pass works
    2. Backward pass works
    3. Parameters update correctly
    4. Loss decreases (model can learn)
    
    Returns:
        (initial_loss, final_loss)
    """
    torch.manual_seed(42)
    
    # Tiny batch: 4 samples, 8 tokens
    batch_size, seq_len, vocab_size, dim, num_heads, num_layers = 4, 8, 256, 64, 8, 2
    
    # Create model
    model = TransformerModel(vocab_size, dim, num_heads, num_layers, hidden_dim=256)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Fixed tiny batch (same data, repeat iterations)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Track loss
    initial_loss = None
    final_loss = None
    
    for step in range(num_iterations):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track losses
        if step == 0:
            initial_loss = loss.item()
        if step == num_iterations - 1:
            final_loss = loss.item()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1:3d}/{num_iterations} | Loss: {loss.item():.6f}")
    
    return initial_loss, final_loss


if __name__ == "__main__":
    print("Component 2B (Overfit Test)")
    print("=" * 60)
    print(f"Running overfit on 4 samples, 8 tokens each...")
    print()
    
    initial_loss, final_loss = run_overfit_test(num_iterations=100)
    
    print()
    print("=" * 60)
    print(f"Initial Loss: {initial_loss:.6f}")
    print(f"Final Loss:   {final_loss:.6f}")
    print(f"Loss Reduction: {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
    print("=" * 60)
    
    # Check: loss should drop significantly
    loss_ratio = final_loss / initial_loss
    
    if loss_ratio < 0.05:  # Final loss < 5% of initial
        print("Gate 2B Verification Status: PASSED.")
        print(f"Model successfully overfit batch (loss reduced to {loss_ratio*100:.1f}% of initial)")
    else:
        print("Gate 2B Verification Status: FAILED.")
        print(f"Loss did not reduce enough (ratio: {loss_ratio:.3f}, need < 0.05)")
        raise AssertionError(f"Overfit test failed! Loss ratio: {loss_ratio}")