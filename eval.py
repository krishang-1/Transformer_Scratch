import torch
import math
from tqdm import tqdm


def compute_perplexity(model, dataloader, device: str = "cpu", max_batches: int = None):
    """
    Compute perplexity on a dataset.
    
    Perplexity = exp(average cross-entropy loss)
    
    Args:
        model: TransformerModel
        dataloader: DataLoader with validation data
        device: Device to run on
        max_batches: Max batches to evaluate (for speed)
    
    Returns:
        perplexity: Scalar perplexity value
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            batch = batch.to(device)
            
            # Forward pass
            logits = model(batch[:, :-1])  # Predict next token
            
            # Compute loss
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(
                logits.view(-1, model.vocab_size),
                batch[:, 1:].contiguous().view(-1)
            )
            
            total_loss += loss.item()
            num_batches += 1
    
    # Perplexity = exp(loss)
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    
    return perplexity


if __name__ == "__main__":
    print("Perplexity computation module ready.")