import json
from pathlib import Path
from datetime import datetime


class TrainingLogger:
    """
    Simple logger for training metrics and curves.
    """
    
    def __init__(self, log_dir: str = "logs"):
        """
        Args:
            log_dir: Directory to save logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.json"
        
        # In-memory logs
        self.logs = {
            "train_loss": [],
            "val_loss": [],
            "val_perplexity": [],
            "learning_rate": [],
            "step": []
        }
    
    def log(self, step: int, train_loss: float = None, val_loss: float = None, 
            val_perplexity: float = None, learning_rate: float = None):
        """
        Log metrics for a step.
        
        Args:
            step: Training step
            train_loss: Training loss
            val_loss: Validation loss
            val_perplexity: Validation perplexity
            learning_rate: Current learning rate
        """
        self.logs["step"].append(step)
        
        if train_loss is not None:
            self.logs["train_loss"].append(train_loss)
        if val_loss is not None:
            self.logs["val_loss"].append(val_loss)
        if val_perplexity is not None:
            self.logs["val_perplexity"].append(val_perplexity)
        if learning_rate is not None:
            self.logs["learning_rate"].append(learning_rate)
        
        self._save()
    
    def _save(self):
        """Save logs to file."""
        with open(self.log_file, "w") as f:
            json.dump(self.logs, f, indent=2)
    
    def print_summary(self):
        """Print training summary."""
        if not self.logs["train_loss"]:
            return
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Final Train Loss:       {self.logs['train_loss'][-1]:.6f}")
        print(f"Initial Train Loss:     {self.logs['train_loss'][0]:.6f}")
        print(f"Loss Reduction:         {(self.logs['train_loss'][0] - self.logs['train_loss'][-1]) / self.logs['train_loss'][0] * 100:.1f}%")
        
        if self.logs["val_loss"]:
            print(f"Final Val Loss:         {self.logs['val_loss'][-1]:.6f}")
            print(f"Final Perplexity:       {self.logs['val_perplexity'][-1]:.2f}")
        
        print(f"Log file:               {self.log_file}")
        print("=" * 60 + "\n")