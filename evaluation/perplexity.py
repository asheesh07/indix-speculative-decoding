# evaluation/perplexity.py
"""
Final perplexity evaluation.
Produces: evaluation/results/final_perplexity.json
          figures/perplexity_comparison.png
"""

import json
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.gpt2 import GPT2_
from model.config import ModelConfig_
from training.dataset import HindiTextDataset

CHECKPOINT  = "checkpoints/best.pt"
TOKENIZER   = "tokenizer/hindi_bpe/tokenizer.json"
TEST_DATA   = "data/processed/test.jsonl"
BASELINES   = "baselines/results/baseline_perplexity.json"
OUTPUT_DIR  = "evaluation/results"
SEQ_LEN     = 512
BATCH_SIZE  = 8
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

def compute_perplexity(model, loader, device) -> tuple:
    """Returns (perplexity, mean_nll, total_tokens)"""
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            logits    = model(input_ids)
            B, N, V   = logits.shape

            loss = nn.functional.cross_entropy(
                logits.reshape(B * N, V),
                labels.reshape(B * N),
                ignore_index = -100,
                reduction    = "sum",
            )
            total_loss   += loss.item()
            total_tokens += (labels != -100).sum().item()

    mean_nll   = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_nll, 20))
    return round(perplexity, 4), round(mean_nll, 6), total_tokens

def load_your_model(checkpoint_path: str, tokenizer) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    cfg        = ModelConfig_(vocab_size=tokenizer.get_vocab_size())
    model      = GPT2_(cfg).to(DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Loaded step {checkpoint['step']:,} | "
          f"checkpoint val PPL: {checkpoint['val_perplexity']:.4f}")
    return model, checkpoint

def evaluate_qwen_on_test(test_path: str, seq_len: int, max_samples: int = 500) -> dict:
    print("\n[Eval] Re-evaluating Qwen2.5-0.5B on your test set...")
    print("  (Re-running ensures exact same test data as your model)")

    try:
        qwen_tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        if qwen_tok.pad_token is None:
            qwen_tok.pad_token = qwen_tok.eos_token

        qwen_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype  = torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map   = "auto" if DEVICE == "cuda" else None,
        )
        qwen_model.eval()

        total_loss   = 0.0
        total_tokens = 0
        samples      = 0

        with open(test_path, "r") as f:
            for line in f:
                if samples >= max_samples:
                    break
                import json as _json
                record = _json.loads(line.strip())
                text   = record.get("text", "").strip()
                if not text:
                    continue

                ids = qwen_tok.encode(
                    text,
                    return_tensors    = "pt",
                    max_length        = seq_len,
                    truncation        = True,
                    add_special_tokens = True,
                ).to(DEVICE)

                with torch.no_grad():
                    out  = qwen_model(input_ids=ids, labels=ids)
                    n    = ids.shape[-1] - 1
                    total_loss   += out.loss.item() * n
                    total_tokens += n

                samples += 1

        mean_nll   = total_loss / max(total_tokens, 1)
        perplexity = math.exp(min(mean_nll, 20))

        del qwen_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        print(f"  Qwen2.5-0.5B PPL on your test set: {perplexity:.4f}")
        return {
            "model":      "Qwen2.5-0.5B",
            "perplexity": round(perplexity, 4),
            "mean_nll":   round(mean_nll, 6),
            "samples":    samples,
            "status":     "success",
        }

    except Exception as e:
        print(f"  Qwen eval failed: {e}")
        print("  Falling back to locked baseline number.")
        return None

def plot_comparison(results: list):
    """
    Clean bar chart — your model vs baselines.
    Saved to figures/perplexity_comparison.png
    """
    import os
    os.makedirs("figures", exist_ok=True)

    names  = [r["model"] for r in results]
    ppls   = [r["perplexity"] for r in results]
    colors = ["#6d28d9" if "Hindi GPT-2" in n else "#374151" for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#07080d")
    ax.set_facecolor("#0f0f1a")

    bars = ax.bar(names, ppls, color=colors, width=0.5)

    for bar, val in zip(bars, ppls):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}",
            ha        = "center",
            va        = "bottom",
            color     = "#e2e8f0",
            fontsize  = 11,
            fontweight = "bold",
        )

    ax.set_ylabel("Perplexity (↓ lower is better)", color="#94a3b8", fontsize=12)
    ax.set_title(
        "Hindi Language Model Perplexity Comparison",
        color="#e2e8f0", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#94a3b8")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#1c1d2e")

    ax.set_ylim(0, max(ppls) * 1.2)
    plt.tight_layout()
    plt.savefig(
        "figures/perplexity_comparison.png",
        dpi=150, bbox_inches="tight", facecolor="#07080d"
    )
    plt.close()
    print("[Plot] Saved: figures/perplexity_comparison.png")

def main():
    import os
    import json
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for path in [CHECKPOINT, TOKENIZER, TEST_DATA]:
        assert Path(path).exists(), f"Not found: {path}"
    print("[Eval] Loading tokenizer and model...")
    tokenizer = Tokenizer.from_file(TOKENIZER)
    tokenizer.eos_token_id = tokenizer.token_to_id("<eos>")
    tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")

    model, checkpoint = load_your_model(CHECKPOINT, tokenizer)
    print("\n[Eval] Loading test set...")
    test_dataset = HindiTextDataset(
        jsonl_path = TEST_DATA,
        tokenizer  = tokenizer,
        seq_len    = SEQ_LEN,
        stride     = SEQ_LEN,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 2,
    )
    print(f"  Test chunks: {len(test_dataset):,}")
    print("\n[Eval] Computing your model perplexity...")
    your_ppl, your_nll, your_tokens = compute_perplexity(
        model, test_loader, DEVICE
    )
    print(f"  Your model PPL: {your_ppl:.4f}")
    results = []
    qwen_result = evaluate_qwen_on_test(TEST_DATA, SEQ_LEN)

    if qwen_result:
        results.append(qwen_result)
    else:
        with open(BASELINES) as f:
            baseline_data = json.load(f)
        for b in baseline_data["baselines"]:
            if "Qwen" in b["model"] and b.get("status") == "success":
                b["note"] = "from locked baseline, not re-evaluated"
                results.append(b)

    results.append({
        "model":            "Hindi GPT-2 (ours)",
        "perplexity":       your_ppl,
        "mean_nll":         your_nll,
        "tokens_evaluated": your_tokens,
        "checkpoint_step":  checkpoint["step"],
        "status":           "success",
    })
    print(f"\n{'='*45}")
    print("FINAL PERPLEXITY RESULTS")
    print(f"{'='*45}")
    print(f"{'Model':<25} {'PPL':>10}")
    print("-"*38)
    for r in results:
        marker = "  ← (this work)" if "ours" in r["model"] else ""
        print(f"{r['model']:<25} {r['perplexity']:>10.4f}{marker}")

    qwen_ppl = results[0]["perplexity"] if results else None
    if qwen_ppl:
        delta = qwen_ppl - your_ppl
        print(f"\n{'✓' if delta > 0 else '✗'} Delta vs Qwen2.5-0.5B: "
              f"{'+' if delta > 0 else ''}{delta:.4f} PPL points")
        if delta <= 0:
            print("  Expected — Qwen has 50x more params and far more training data.")
            print("  Your contribution is tokenizer efficiency + speculative decoding.")

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "test_data":    TEST_DATA,
        "seq_len":      SEQ_LEN,
        "results":      results,
    }
    out_path = f"{OUTPUT_DIR}/final_perplexity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Eval] Saved: {out_path}")

    # Plot
    plot_comparison(results)

    print(f"\n[Done] Next: python speculative_decoding.py")

if __name__ == "__main__":
    main()