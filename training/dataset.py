# training/dataset.py

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional


class HindiTextDataset(Dataset):
    """
    PyTorch Dataset for Hindi text data saved as JSONL.
    Loads from train.jsonl / val.jsonl / test.jsonl
    produced by scripts/collect_data.py
    
    Each example is tokenized and chunked into
    fixed-length sequences for language model training.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer,                    # your trained BPE tokenizer
        seq_len: int = 512,           # context length
        stride: int  = 256,           # overlap between chunks (50% overlap)
        max_examples: Optional[int] = None,   # cap for debugging
    ):
        self.tokenizer    = tokenizer
        self.seq_len      = seq_len
        self.stride       = stride
        self.chunks       = []        # list of token id tensors

        self._load_and_chunk(jsonl_path, max_examples)

    def _load_and_chunk(self, jsonl_path: str, max_examples: Optional[int]):
        """
        Load JSONL, tokenize each document, chunk into seq_len sequences.
        Uses stride to avoid losing context at document boundaries.
        """
        path = Path(jsonl_path)
        assert path.exists(), f"Data file not found: {jsonl_path}"

        print(f"[Dataset] Loading from {jsonl_path}")

        all_token_ids = []
        doc_count     = 0
        skipped       = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if max_examples and doc_count >= max_examples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    text   = record.get("text", "").strip()
                except json.JSONDecodeError:
                    skipped += 1
                    continue

                if not text:
                    skipped += 1
                    continue

                # Tokenize
                token_ids = self.tokenizer.encode(text).ids

                # Skip documents shorter than 64 tokens — not useful for LM
                if len(token_ids) < 64:
                    skipped += 1
                    continue

                # Add EOS token between documents
                token_ids.append(self.tokenizer.eos_token_id)
                all_token_ids.extend(token_ids)
                doc_count += 1

        print(f"[Dataset] Documents loaded:  {doc_count:,}")
        print(f"[Dataset] Documents skipped: {skipped:,}")
        print(f"[Dataset] Total tokens:      {len(all_token_ids):,}")

        # Chunk into fixed-length sequences with stride
        self._chunk(all_token_ids)
        print(f"[Dataset] Total chunks:      {len(self.chunks):,}")

    def _chunk(self, token_ids: list):
        """
        Slide a window of seq_len over all tokens.
        Each chunk becomes one training example.
        Input:  tokens[i : i + seq_len]
        Target: tokens[i+1 : i + seq_len + 1]  (next token prediction)
        """
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        total        = len(token_tensor)

        i = 0
        while i + self.seq_len + 1 <= total:
            chunk = token_tensor[i : i + self.seq_len + 1]  # +1 for target
            self.chunks.append(chunk)
            i += self.stride

        if len(self.chunks) == 0:
            raise ValueError(
                f"No chunks created. Total tokens ({total}) may be less than seq_len ({self.seq_len})."
            )

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        """
        Returns:
            input_ids:  seq_len tokens
            labels:     seq_len tokens shifted by 1 (next token prediction)
        """
        chunk     = self.chunks[idx]
        input_ids = chunk[:-1]   # all but last
        labels    = chunk[1:]    # all but first
        return {
            "input_ids": input_ids,
            "labels":    labels,
        }


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir:   str,
    tokenizer,
    seq_len:    int = 512,
    stride:     int = 256,
    batch_size: int = 16,
    num_workers: int = 2,
) -> tuple:
    """
    Returns train, val, test DataLoaders.
    Call this from train.py.
    """
    data_dir = Path(data_dir)

    train_dataset = HindiTextDataset(
        jsonl_path   = str(data_dir / "train.jsonl"),
        tokenizer    = tokenizer,
        seq_len      = seq_len,
        stride       = stride,
    )

    val_dataset = HindiTextDataset(
        jsonl_path   = str(data_dir / "val.jsonl"),
        tokenizer    = tokenizer,
        seq_len      = seq_len,
        stride       = seq_len,   # no overlap for val/test — cleaner eval
    )

    test_dataset = HindiTextDataset(
        jsonl_path   = str(data_dir / "test.jsonl"),
        tokenizer    = tokenizer,
        seq_len      = seq_len,
        stride       = seq_len,   # no overlap
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size * 2,   # larger batch for eval, no grad
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size  = batch_size * 2,
        shuffle     = False,
        num_workers = num_workers,
        pin_memory  = True,
    )

    print(f"\n[DataLoaders]")
    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches:   {len(val_loader):,}")
    print(f"  Test batches:  {len(test_loader):,}")

    return train_loader, val_loader, test_loader


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this directly to verify dataset loads correctly before training.
    python training/dataset.py
    """
    from tokenizers import ByteLevelBPETokenizer

    print("Loading tokenizer...")
    tokenizer = ByteLevelBPETokenizer(
        "tokenizer/hindi_bpe/vocab.json",
        "tokenizer/hindi_bpe/merges.txt",
    )
    tokenizer.add_special_tokens(["<eos>", "<pad>", "<unk>"])
    tokenizer.eos_token_id = tokenizer.token_to_id("<eos>")

    print("\nRunning sanity check on train set (first 1000 docs)...")
    dataset = HindiTextDataset(
        jsonl_path   = "data/processed/train.jsonl",
        tokenizer    = tokenizer,
        seq_len      = 512,
        max_examples = 1000,
    )

    print(f"\nFirst chunk:")
    sample = dataset[0]
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape:    {sample['labels'].shape}")
    print(f"  First 10 input tokens: {sample['input_ids'][:10].tolist()}")
    print(f"  First 10 label tokens: {sample['labels'][:10].tolist()}")

    # Verify input and labels are shifted by 1
    assert torch.equal(sample['input_ids'][1:], sample['labels'][:-1]), \
        "Input/label shift is wrong — check _chunk logic"

    print("\n✓ Sanity check passed. Dataset is ready for training.")
