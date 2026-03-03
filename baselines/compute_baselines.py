import json
import math
import os
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BertForMaskedLM,
)
from torch.utils.data import  Dataset
from tqdm import tqdm

OUTPUT_DIR   = "baselines/results"
TEST_DATA    = "data/processed/test.jsonl"
MAX_SAMPLES  = 500          
SEQ_LEN      = 512
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

DECODER_MODELS = {
    "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B",
}

ENCODER_MODELS = {
    "mBERT":     "bert-base-multilingual-cased",
    "IndicBERT": "ai4bharat/indic-bert",
}

class TestDataset(Dataset):
    def __init__(self, path: str, tokenizer, seq_len: int, max_samples: int):
        self.examples = []
        self.seq_len  = seq_len

        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                record = json.loads(line.strip())
                text   = record.get("text", "").strip()
                if not text:
                    continue

                ids = tokenizer.encode(
                    text,
                    add_special_tokens = True,
                    max_length         = seq_len,
                    truncation         = True,
                    return_tensors     = "pt",
                )

                if ids.shape[-1] < 16:   
                    continue

                self.examples.append(ids.squeeze(0))

        print(f"  Loaded {len(self.examples):,} test examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    max_len = max(x.shape[0] for x in batch)
    padded  = torch.zeros(len(batch), max_len, dtype=torch.long)
    mask    = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :x.shape[0]] = x
        mask[i, :x.shape[0]]   = 1
    return padded, mask

def compute_decoder_perplexity(
    model_name: str,
    model_id:   str,
    test_path:  str,
    seq_len:    int,
    max_samples: int,
) -> dict:
    print(f"\n[Baseline] Computing perplexity for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype    = torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map     = "auto" if DEVICE == "cuda" else None,
            load_in_8bit   = DEVICE == "cuda",    
        )
        model.eval()

        dataset = TestDataset(test_path, tokenizer, seq_len, max_samples)

        total_nll   = 0.0
        total_tokens = 0

        with torch.no_grad():
            for ids in tqdm(dataset, desc=f"  {model_name}"):
                ids = ids.unsqueeze(0).to(DEVICE)
                outputs = model(input_ids=ids, labels=ids)
                nll          = outputs.loss.item()
                n_tokens     = ids.shape[-1] - 1    
                total_nll   += nll * n_tokens
                total_tokens += n_tokens

        mean_nll    = total_nll / total_tokens
        perplexity  = math.exp(mean_nll)

        result = {
            "model":          model_name,
            "model_id":       model_id,
            "type":           "decoder",
            "perplexity":     round(perplexity, 4),
            "mean_nll":       round(mean_nll, 6),
            "samples_used":   len(dataset),
            "seq_len":        seq_len,
            "device":         DEVICE,
            "status":         "success",
        }

        print(f"  ✓ {model_name} perplexity: {perplexity:.4f}")
        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

        return result

    except Exception as e:
        print(f"  ✗ {model_name} failed: {e}")
        return {
            "model":   model_name,
            "status":  "failed",
            "error":   str(e),
        }

def compute_encoder_perplexity(
    model_name: str,
    model_id:   str,
    test_path:  str,
    seq_len:    int,
    max_samples: int,
) -> dict:
    print(f"\n[Baseline] Computing pseudo-perplexity for {model_name}...")
    print(f"  Note: encoder pseudo-PPL, not directly comparable to decoder PPL")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = BertForMaskedLM.from_pretrained(model_id)
        model     = model.to(DEVICE)
        model.eval()

        total_log_prob = 0.0
        total_tokens   = 0
        samples_done   = 0

        with open(test_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc=f"  {model_name}", total=max_samples):
                if samples_done >= max_samples:
                    break

                record = json.loads(line.strip())
                text   = record.get("text", "").strip()
                if not text:
                    continue

                ids = tokenizer.encode(
                    text,
                    add_special_tokens = True,
                    max_length         = seq_len,
                    truncation         = True,
                )

                if len(ids) < 8:
                    continue

                ids_tensor = torch.tensor([ids], dtype=torch.long).to(DEVICE)
                for i in range(1, len(ids) - 1):
                    masked     = ids_tensor.clone()
                    masked[0, i] = tokenizer.mask_token_id

                    with torch.no_grad():
                        output = model(input_ids=masked)
                        logits = output.logits[0, i]
                        log_probs = F.log_softmax(logits, dim=-1)
                        token_log_prob = log_probs[ids[i]].item()

                    total_log_prob += token_log_prob
                    total_tokens   += 1

                samples_done += 1

        mean_log_prob      = total_log_prob / total_tokens
        pseudo_perplexity  = math.exp(-mean_log_prob)

        result = {
            "model":               model_name,
            "model_id":            model_id,
            "type":                "encoder_pseudo_ppl",
            "perplexity":          round(pseudo_perplexity, 4),
            "mean_log_prob":       round(mean_log_prob, 6),
            "samples_used":        samples_done,
            "tokens_evaluated":    total_tokens,
            "seq_len":             seq_len,
            "device":              DEVICE,
            "status":              "success",
            "note":                "pseudo-perplexity via MLM masking, not directly comparable to decoder PPL",
        }

        print(f"  ✓ {model_name} pseudo-perplexity: {pseudo_perplexity:.4f}")

        del model
        torch.cuda.empty_cache() if DEVICE == "cuda" else None

        return result

    except Exception as e:
        print(f"  ✗ {model_name} failed: {e}")
        return {
            "model":  model_name,
            "status": "failed",
            "error":  str(e),
        }

def compute_random_baseline(vocab_size: int = 8000) -> dict:
    perplexity = float(vocab_size)
    result = {
        "model":      "Random (uniform)",
        "type":       "theoretical",
        "perplexity": perplexity,
        "note":       f"Theoretical ceiling: uniform distribution over {vocab_size} tokens",
        "status":     "success",
    }
    print(f"\n[Baseline] Random baseline: {perplexity:.1f} (theoretical)")
    return result

def save_and_lock(results: list):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    timestamp = datetime.now().isoformat()
    output    = {
        "locked_at":     timestamp,
        "warning":       "DO NOT MODIFY. Baselines locked before training. See README.",
        "device":        DEVICE,
        "test_data":     TEST_DATA,
        "max_samples":   MAX_SAMPLES,
        "seq_len":       SEQ_LEN,
        "baselines":     results,
    }

    path = f"{OUTPUT_DIR}/baseline_perplexity.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    os.chmod(path, 0o444)

    print(f"\n{'='*55}")
    print(f"BASELINES LOCKED")
    print(f"{'='*55}")
    print(f"Saved to:  {path}")
    print(f"Locked at: {timestamp}")
    print(f"File is now READ-ONLY.")
    print(f"\n⚠️  Do not re-run this script after training starts.")
    print(f"⚠️  These numbers are your ground truth comparison.")
    print(f"{'='*55}")

    return path

def print_summary(results: list):
    print(f"\n{'='*55}")
    print(f"BASELINE PERPLEXITY SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<22} {'PPL':>10} {'Type':>18}")
    print(f"{'-'*55}")

    for r in results:
        if r.get("status") != "success":
            print(f"{r['model']:<22} {'FAILED':>10}")
            continue
        ppl   = r.get("perplexity", 0)
        ptype = r.get("type", "")
        print(f"{r['model']:<22} {ppl:>10.2f} {ptype:>18}")

    print(f"\nYour trained model target:")
    print(f"  Beat Qwen2.5-0.5B perplexity on Hindi text.")
    print(f"  That is your primary success criterion.")

def main():
    print(f"{'='*55}")
    print(f"COMPUTING BASELINES — lock before training")
    print(f"Device: {DEVICE}")
    print(f"Test data: {TEST_DATA}")
    print(f"Samples: {MAX_SAMPLES}")
    print(f"{'='*55}")

    assert Path(TEST_DATA).exists(), (
        f"Test data not found: {TEST_DATA}\n"
        f"Run scripts/collect_data.py first."
    )

    results = []
    results.append(compute_random_baseline(vocab_size=8000))
    for name, model_id in DECODER_MODELS.items():
        result = compute_decoder_perplexity(
            model_name  = name,
            model_id    = model_id,
            test_path   = TEST_DATA,
            seq_len     = SEQ_LEN,
            max_samples = MAX_SAMPLES,
        )
        results.append(result)

    for name, model_id in ENCODER_MODELS.items():
        result = compute_encoder_perplexity(
            model_name  = name,
            model_id    = model_id,
            test_path   = TEST_DATA,
            seq_len     = SEQ_LEN,
            max_samples = MAX_SAMPLES,
        )
        results.append(result)
    save_and_lock(results)
    print_summary(results)

if __name__ == "__main__":
    main()
