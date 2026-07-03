import torch
import torch.nn as nn
from modules.transformer_model import TransformerModel


def compute_finite_difference_grad(model, loss_fn, inputs, labels, param_name, epsilon=1e-4):
    """
    Compute gradient via finite difference for a single parameter.
    
    gradient ≈ (f(x+ε) - f(x-ε)) / (2ε)
    
    Args:
        model: TransformerModel instance
        loss_fn: Loss function
        inputs: Input batch (batch, seq_len)
        labels: Target batch (batch, seq_len)
        param_name: Name of parameter to check (e.g., 'embedding.weight')
        epsilon: Finite difference step size
    
    Returns:
        grad_fd: Finite difference gradient
    """
    # Get the parameter
    param = dict(model.named_parameters())[param_name]
    
    # Store original value
    original_val = param.data.clone()
    
    # Compute f(x + ε)
    param.data = original_val + epsilon
    logits_plus = model(inputs)
    loss_plus = loss_fn(logits_plus.view(-1, model.vocab_size), labels.view(-1))
    
    # Compute f(x - ε)
    param.data = original_val - epsilon
    logits_minus = model(inputs)
    loss_minus = loss_fn(logits_minus.view(-1, model.vocab_size), labels.view(-1))
    
    # Finite difference
    grad_fd = (loss_plus - loss_minus) / (2 * epsilon)
    
    # Restore original value
    param.data = original_val
    
    return grad_fd


def run_gradient_check_test(num_checks: int = 5) -> dict:
    """
    Verify gradients via finite difference checks.
    
    Returns:
        results: Dict of parameter_name → relative_error
    """
    torch.manual_seed(42)
    
    # Setup
    batch_size, seq_len, vocab_size, dim, num_heads, num_layers = 2, 8, 256, 64, 8, 2
    
    model = TransformerModel(vocab_size, dim, num_heads, num_layers, hidden_dim=256)
    loss_fn = nn.CrossEntropyLoss()
    
    # Create tiny batch
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    logits = model(inputs)
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    
    # Backward pass (compute autograd gradients)
    loss.backward()
    
    # Select a few parameters to check
    param_names = [
        'embedding.weight',
        'blocks.0.attention.W_q.weight',
        'blocks.0.swiglu.W_1.weight',
        'output_proj.weight'
    ]
    
    results = {}
    
    for param_name in param_names:
        # Get autograd gradient
        param = dict(model.named_parameters())[param_name]
        grad_autograd = param.grad.clone()
        
        # Compute finite difference gradient (on a random element)
        grad_fd = compute_finite_difference_grad(model, loss_fn, inputs, labels, param_name)
        
        # Compute relative error
        # relative_error = ||grad_autograd - grad_fd|| / ||grad_autograd||
        grad_autograd_flat = grad_autograd.view(-1)
        
        # Use a few random elements for faster computation
        indices = torch.randperm(grad_autograd_flat.numel())[:10]
        grad_autograd_sample = grad_autograd_flat[indices]
        
        # Recompute finite diff for each sampled element (expensive but necessary)
        errors = []
        for idx in indices:
            param_orig = dict(model.named_parameters())[param_name].data.clone()
            
            # f(x+ε) at specific element
            param_dict = dict(model.named_parameters())
            param_dict[param_name].data.view(-1)[idx] += 1e-4
            logits_plus = model(inputs)
            loss_plus = loss_fn(logits_plus.view(-1, vocab_size), labels.view(-1))
            
            # f(x-ε) at specific element
            param_dict[param_name].data = param_orig.clone()
            param_dict[param_name].data.view(-1)[idx] -= 1e-4
            logits_minus = model(inputs)
            loss_minus = loss_fn(logits_minus.view(-1, vocab_size), labels.view(-1))
            
            grad_fd_elem = (loss_plus - loss_minus) / (2 * 1e-4)
            grad_autograd_elem = grad_autograd.view(-1)[idx].item()
            
            rel_error = abs(grad_fd_elem - grad_autograd_elem) / (abs(grad_autograd_elem) + 1e-8)
            errors.append(rel_error)
            
            # Restore
            param_dict[param_name].data = param_orig.clone()
        
        max_error = max(errors)
        results[param_name] = max_error
    
    return results


if __name__ == "__main__":
    results = run_gradient_check_test()
    print(f"Component 2A (Gradient Checks)")
    print("=" * 60)
    
    all_pass = True
    for param_name, error in results.items():
        status = "✅ PASS" if error < 1e-4 else "❌ FAIL"
        print(f"{param_name:40s} Error: {error:.2e} {status}")
        if error >= 1e-4:
            all_pass = False
    
    print("=" * 60)
    if all_pass:
        print("Gate 2A Verification Status: PASSED.")
    else:
        print("Gate 2A Verification Status: FAILED.")
        raise AssertionError("Gradient check failed!")