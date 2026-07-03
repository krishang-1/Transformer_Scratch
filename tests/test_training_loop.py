import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from modules.transformer_model import TransformerModel


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps):
    """
    Linear warmup then linear decay schedule.
    
    lr = base_lr * min(step / warmup_steps, 1 - (step - warmup) / (total - warmup))
    """
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        else:
            return max(0.0, float(num_total_steps - step) / float(max(1, num_total_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_training_loop_test(num_epochs: int = 2, batch_size: int = 4) -> dict:
    """
    Full training loop: forward, backward, update, validation.
    
    Returns:
        results: Dict with train/val losses
    """
    torch.manual_seed(42)
    
    # Setup
    vocab_size, seq_len = 256, 8
    dim, num_heads, num_layers = 64, 8, 2
    
    model = TransformerModel(vocab_size, dim, num_heads, num_layers, hidden_dim=256)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Create toy dataset
    num_samples_train = 32
    num_samples_val = 8
    
    train_inputs = torch.randint(0, vocab_size, (num_samples_train, seq_len))
    train_labels = torch.randint(0, vocab_size, (num_samples_train, seq_len))
    
    val_inputs = torch.randint(0, vocab_size, (num_samples_val, seq_len))
    val_labels = torch.randint(0, vocab_size, (num_samples_val, seq_len))
    
    # DataLoaders
    train_dataset = TensorDataset(train_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(val_inputs, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Training schedule
    num_total_steps = num_epochs * len(train_loader)
    num_warmup_steps = num_total_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps)
    
    # Training loop
    results = {
        'train_losses': [],
        'val_losses': [],
        'epochs': num_epochs
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            logits = model(inputs)
            loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        results['train_losses'].append(avg_train_loss)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                logits = model(inputs)
                loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        results['val_losses'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
    
    return results


if __name__ == "__main__":
    print("Component 2C (Training Loop)")
    print("=" * 60)
    
    results = run_training_loop_test(num_epochs=2, batch_size=4)
    
    print("=" * 60)
    print(f"Final Train Loss: {results['train_losses'][-1]:.6f}")
    print(f"Final Val Loss:   {results['val_losses'][-1]:.6f}")
    print("=" * 60)
    
    # Check: losses should decrease
    train_decreased = results['train_losses'][0] > results['train_losses'][-1]
    
    if train_decreased:
        print("Gate 2C Verification Status: PASSED.")
        print("Training loop completed successfully with decreasing loss.")
    else:
        print("Gate 2C Verification Status: FAILED.")
        raise AssertionError("Training loss did not decrease!")