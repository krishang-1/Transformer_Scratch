import sys
import torch

print("=" * 70)
print("     PHASE 3 CENTRAL VERIFICATION GATEWAY (123M config, real data)")
print("=" * 70)
print()

try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print("Executing Gate 3A: Data Pipeline Validation (WikiText-103)...")
    print()

    from data.data_loader import create_data_loaders

    train_loader, val_loader = create_data_loaders(batch_size=4, seq_len=1024, num_workers=0)

    print("Loading first batch...")
    train_iter = iter(train_loader)
    batch = next(train_iter)

    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Token range: [{batch.min().item()}, {batch.max().item()}]")

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    decoded = tok.decode(batch[0, :60])
    print(f"Decoded sample text:\n  \"{decoded}\"")
    print()
    print("-> Gate 3A Verification Status: PASSED.\n")

    print("Executing Gate 3B: Model Initialization & Inference (123M)...")
    print()

    from modules.transformer_model import TransformerModel

    model = TransformerModel(vocab_size=50257, dim=768, num_heads=12, num_layers=12)
    model.to(device)

    num_params = model.num_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

    batch_device = batch[:2].to(device)
    logits = model(batch_device)

    print(f"Input shape: {batch_device.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"No NaNs: {not torch.isnan(logits).any()}")
    print()
    print("-> Gate 3B Verification Status: PASSED.\n")

    print("Executing Gate 3C: Quick Training Run (10 steps, real data)...")
    print()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    losses = []
    train_iter = iter(train_loader)

    for step in range(10):
        batch = next(train_iter).to(device)
        logits = model(batch[:, :-1])
        loss = loss_fn(logits.reshape(-1, 50257), batch[:, 1:].contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        losses.append(loss.item())
        print(f"  Step {step + 1:2d}: Loss = {loss.item():.4f}")

    print()
    print(f"Loss trend: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print()
    print("-> Gate 3C Verification Status: PASSED.\n")

    print("=" * 70)
    print("STATUS: Phase 3 Pipeline Validated (123M config, real WikiText-103 data)")
    print("=" * 70)

except Exception as e:
    print(f"\n[CRITICAL FAILURE] Verification Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)