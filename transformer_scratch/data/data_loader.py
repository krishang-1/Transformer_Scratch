import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from torch.utils.data import IterableDataset, DataLoader


class WikiTextDataset(IterableDataset):
    """WikiText-103: standard, public corpus appropriate for a 100-300M model."""

    def __init__(self, split: str = "train", seq_len: int = 1024, vocab_size: int = 50257):
        self.split = split
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading WikiText-103 ({split} split)...")
        self.dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, streaming=True)

        self.buffer_size = seq_len * 100

    def tokenize_chunk(self, text: str) -> list:
        if not text.strip():
            return []
        return self.tokenizer.encode(text, add_special_tokens=True)

    def pack_sequences(self, token_buffer: list):
        packed = []
        i = 0
        while i + self.seq_len <= len(token_buffer):
            chunk = token_buffer[i:i + self.seq_len]
            packed.append(torch.tensor(chunk, dtype=torch.long))
            i += self.seq_len
        remaining = token_buffer[i:]
        return packed, remaining

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            tokens = self.tokenize_chunk(example["text"])
            buffer.extend(tokens)
            if len(buffer) > self.buffer_size:
                packed, buffer = self.pack_sequences(buffer)
                for seq in packed:
                    yield seq


def create_data_loaders(batch_size: int = 16, seq_len: int = 1024, num_workers: int = 0) -> tuple:
    print("Creating WikiText-103 dataloaders...")

    train_dataset = WikiTextDataset(split="train", seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    val_dataset = WikiTextDataset(split="validation", seq_len=seq_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader


if __name__ == "__main__":
    print("Testing WikiText-103 data loader...")
    train_loader, val_loader = create_data_loaders(batch_size=4, seq_len=1024)

    train_iter = iter(train_loader)
    batch = next(train_iter)

    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    decoded = tok.decode(batch[0, :80])
    print(f"Decoded sample:\n  \"{decoded}\"")
    print("Data loader works with WikiText-103!")