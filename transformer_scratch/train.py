import torch
import torch.nn as nn
from pathlib import Path
import time
import itertools

from modules.transformer_model import TransformerModel
from data.data_loader import create_data_loaders
from utils.logging import TrainingLogger
from eval import compute_perplexity
from sample import generate_samples


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_total_steps):
    def lr_lambda(step):
        if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps))
        else:
            return max(0.0, float(num_total_steps - step) / float(max(1, num_total_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(total_steps: int = 20000, batch_size: int = 16, seq_len: int = 1024,
          learning_rate: float = 3e-4, warmup_steps: int = 1000,
          checkpoint_dir: str = "checkpoints", log_interval: int = 50,
          val_interval: int = 1000, sample_interval: int = 2000,
          device: str = "cpu", vocab_size: int = 50257,
          dim: int = 768, num_heads: int = 12, num_layers: int = 12,
          resume_from: str = None):

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = TrainingLogger()

    print("=" * 60)
    print("PHASE 3: TRAIN TO CONVERGENCE (123M config, step-bounded)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size} | Seq len: {seq_len}")
    print(f"Architecture: dim={dim}, heads={num_heads}, layers={num_layers}")
    print(f"Learning rate: {learning_rate} | Warmup: {warmup_steps}")
    print(f"Total steps (target): {total_steps}")
    print(f"Checkpoint dir: {checkpoint_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 60 + "\n")

    print("Creating model...")
    model = TransformerModel(vocab_size=vocab_size, dim=dim, num_heads=num_heads,
                              num_layers=num_layers, tie_weights=True)
    model.to(device)

    num_params = model.num_parameters()
    print(f"Model parameters: {num_params:,} ({num_params / 1e6:.1f}M)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    start_step = 0
    if resume_from:
        print(f"Loading checkpoint from {resume_from}...")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_step = checkpoint["step"]
        print(f"Resumed at step {start_step}\n")

        if start_step >= total_steps:
            print(f"[WARNING] Checkpoint step ({start_step}) >= target total_steps ({total_steps}).")
            return

    print("Creating dataloaders...")
    train_loader, val_loader = create_data_loaders(batch_size=batch_size, seq_len=seq_len, num_workers=0)
    print("Dataloaders ready\n")

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    for _ in range(start_step):
        scheduler.step()

    loss_fn = nn.CrossEntropyLoss()
    remaining_steps = total_steps - start_step

    print("=" * 60)
    print(f"TRAINING ({remaining_steps} steps remaining, target step {total_steps})")
    print("=" * 60 + "\n")

    model.train()
    start_time = time.time()
    train_iter = itertools.cycle(train_loader)
    running_loss = 0.0

    for step in range(start_step + 1, total_steps + 1):
        batch = next(train_iter).to(device)

        logits = model(batch[:, :-1])
        loss = loss_fn(logits.reshape(-1, vocab_size), batch[:, 1:].contiguous().view(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            steps_done_this_run = step - start_step
            throughput = steps_done_this_run / elapsed if elapsed > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Step {step:6d}/{total_steps} | Loss: {loss.item():.4f} | "
                  f"Avg: {avg_loss:.4f} | LR: {current_lr:.2e} | {throughput:.2f} steps/sec")
            logger.log(step, train_loss=avg_loss, learning_rate=current_lr)
            running_loss = 0.0

        if step % val_interval == 0:
            print(f"\n  Computing validation perplexity...")
            model.eval()
            val_perplexity = compute_perplexity(model, val_loader, device, max_batches=20)
            print(f"  Val Perplexity: {val_perplexity:.2f}\n")
            logger.log(step, val_perplexity=val_perplexity)
            model.train()

        if step % sample_interval == 0:
            print(f"\n  Generating samples...\n")
            model.eval()
            samples = generate_samples(model, num_samples=2, max_length=100, device=device)
            for i, sample in enumerate(samples, 1):
                print(f"  [Sample {i}] {sample[:200]}...\n")
            model.train()

        if step % val_interval == 0 or step == total_steps:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save({"step": step, "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict()}, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}\n")

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"This run duration: {elapsed / 60:.1f} minutes")
    print(f"Steps this run: {remaining_steps}")
    print(f"Total steps (cumulative): {total_steps}")
    logger.print_summary()

    final_checkpoint = checkpoint_dir / "final_model.pt"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Final model saved: {final_checkpoint}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    train(total_steps=args.total_steps, batch_size=args.batch_size, seq_len=args.seq_len,
          learning_rate=args.lr, warmup_steps=args.warmup_steps, checkpoint_dir=args.checkpoint_dir,
          device=args.device, dim=args.dim, num_heads=args.num_heads, num_layers=args.num_layers,
          resume_from=args.resume_from)