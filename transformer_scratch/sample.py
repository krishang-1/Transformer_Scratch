import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer


def generate_samples(
    model,
    num_samples: int = 3,
    max_length: int = 128,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str = "cpu"
):
    """
    Generate text samples from the model.
    
    Args:
        model: Trained TransformerModel
        num_samples: Number of samples to generate
        max_length: Maximum length of generated text
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device to run on
    
    Returns:
        samples: List of generated text strings
    """
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Start with BOS token
            input_ids = torch.tensor([[tokenizer.bos_token_id]], device=device)
            
            for step in range(max_length):
                # Forward pass
                logits = model(input_ids)
                
                # Get logits of last token
                next_logits = logits[0, -1, :] / temperature
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                next_probs = F.softmax(top_k_logits, dim=-1)
                
                # Sample
                next_token_idx = torch.multinomial(next_probs, num_samples=1)
                next_token = top_k_indices[next_token_idx]
                
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop if EOS
                if next_token.item() == tokenizer.eos_token_id:
                    break
            
            # Decode
            sample_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            samples.append(sample_text)
    
    return samples


if __name__ == "__main__":
    from modules.transformer_model import TransformerModel
    
    print("Testing sample generation...")
    
    # Create small model
    model = TransformerModel(vocab_size=50257, dim=64, num_heads=8, num_layers=2)
    
    # Generate
    samples = generate_samples(model, num_samples=2, max_length=64)
    
    print("\nGenerated Samples:")
    for i, sample in enumerate(samples, 1):
        print(f"\n[Sample {i}]")
        print(sample)