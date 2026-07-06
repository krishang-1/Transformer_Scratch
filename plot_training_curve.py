import re
import matplotlib.pyplot as plt

with open("training_log.txt", "r", encoding="utf-8", errors="ignore") as f:
    log_text = f.read()

# Parse step/loss/lr lines
step_pattern = re.compile(r"Step\s+(\d+)/\d+\s+\|\s+Loss:\s+([\d.]+)\s+\|\s+Avg:\s+([\d.]+)\s+\|\s+LR:\s+([\d.e+-]+)")
steps, avg_losses, lrs = [], [], []
for match in step_pattern.finditer(log_text):
    step, loss, avg, lr = match.groups()
    steps.append(int(step))
    avg_losses.append(float(avg))
    lrs.append(float(lr))

# Parse validation perplexity lines
val_pattern = re.compile(r"Val Perplexity:\s+([\d.]+)")
val_perplexities = [float(x) for x in val_pattern.findall(log_text)]
val_steps = list(range(1000, 1000 * (len(val_perplexities) + 1), 1000))

print(f"Parsed {len(steps)} training step entries")
print(f"Parsed {len(val_perplexities)} validation perplexity readings")
print(f"Final train avg loss: {avg_losses[-1] if avg_losses else 'N/A'}")
print(f"Final val perplexity: {val_perplexities[-1] if val_perplexities else 'N/A'}")

# Plot both curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(steps, avg_losses, linewidth=1)
axes[0].set_xlabel("Step")
axes[0].set_ylabel("Training Loss (rolling avg)")
axes[0].set_title("Training Loss — 123M Model, WikiText-103")
axes[0].grid(alpha=0.3)

axes[1].plot(val_steps[:len(val_perplexities)], val_perplexities, marker='o')
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Validation Perplexity")
axes[1].set_title("Validation Perplexity — 123M Model, WikiText-103")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("training_curve_123M_wikitext.png", dpi=150)
plt.show()
print("Saved plot to training_curve_123M_wikitext.png")