import json
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.gpt2 import GPT2_
from model.config import ModelConfig_
from training.dataset import HindiTextDataset
from evaluation.perplexity import compute_perplexity

OUTPUT_DIR  = "evaluation/results"
TOKENIZER   = "tokenizer/hindi_bpe/tokenizer.json"
TEST_DATA   = "data/processed/test.jsonl"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN     = 512
BATCH_SIZE  = 8

RUNS = [
    {"label": "100M tokens", "tokens_B": 0.1,  "checkpoint": "checkpoints/run_small/best.pt"},
    {"label": "200M tokens", "tokens_B": 0.2,  "checkpoint": "checkpoints/run_medium/best.pt"},
    {"label": "400M tokens", "tokens_B": 0.4,  "checkpoint": "checkpoints/run_large/best.pt"},
]

def load_model(checkpoint_path: str, vocab_size: int) -> nn.Module:
    state = torch.load(checkpoint_path, map_location=DEVICE)
    cfg   = ModelConfig_(vocab_size=vocab_size)
    model = GPT2_(cfg).to(DEVICE)
    model.load_state_dict(state["model_state"])
    return model

def plot_scaling_curve(results: list):
    import os
    os.makedirs("figures", exist_ok=True)

    tokens = [r["tokens_B"] for r in results]
    ppls   = [r["perplexity"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#07080d")
    ax.set_facecolor("#0f0f1a")

    ax.plot(
        tokens, ppls,
        color     = "#6d28d9",
        linewidth = 2.5,
        marker    = "o",
        markersize = 8,
        markerfacecolor = "#c4b5fd",
    )

    for t, p, r in zip(tokens, ppls, results):
        ax.annotate(
            f"{p:.1f}",
            xy         = (t, p),
            xytext     = (5, -15),
            textcoords = "offset points",
            color      = "#e2e8f0",
            fontsize   = 10,
        )

    ax.set_xlabel("Training Data (Billion tokens)", color="#94a3b8", fontsize=12)
    ax.set_ylabel("Test Perplexity (↓ lower is better)", color="#94a3b8", fontsize=12)
    ax.set_title(
        "Scaling Curve — Perplexity vs Training Data\n(Hindi GPT-2, 13.9M params)",
        color="#e2e8f0", fontsize=12, pad=12,
    )
    ax.tick_params(colors="#94a3b8")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#1c1d2e")

    plt.tight_layout()
    plt.savefig(
        "figures/scaling_curve.png",
        dpi=150, bbox_inches="tight", facecolor="#07080d"
    )
    plt.close()
    print("[Plot] Saved: figures/scaling_curve.png")

def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[Scaling] Loading tokenizer...")
    tokenizer = Tokenizer.from_file(TOKENIZER)
    tokenizer.eos_token_id = tokenizer.token_to_id("<eos>")
    vocab_size = tokenizer.get_vocab_size()

    print("[Scaling] Building test loader...")
    test_dataset = HindiTextDataset(
        TEST_DATA, tokenizer, SEQ_LEN, stride=SEQ_LEN
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2,
    )

    results = []
    print(f"\n{'='*45}")
    print("SCALING CURVE EVALUATION")
    print(f"{'='*45}")

    for run in RUNS:
        path = Path(run["checkpoint"])
        if not path.exists():
            print(f"  SKIP {run['label']} — checkpoint not found: {path}")
            continue

        print(f"\n[Scaling] Evaluating {run['label']}...")
        model = load_model(str(path), vocab_size)
        ppl, nll, tokens = compute_perplexity(model, test_loader, DEVICE)
        print(f"  PPL: {ppl:.4f}")

        results.append({
            "label":      run["label"],
            "tokens_B":   run["tokens_B"],
            "checkpoint": str(path),
            "perplexity": ppl,
            "mean_nll":   nll,
        })

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    if len(results) < 2:
        print("\n[Scaling] Need at least 2 runs for a curve.")
        print("  Run all 3 training configs first.")
        return

    print(f"\n{'Tokens':<20} {'PPL':>10}")
    print("-"*32)
    for r in results:
        print(f"{r['label']:<20} {r['perplexity']:>10.4f}")

    output = {
        "evaluated_at": datetime.now().isoformat(),
        "test_data":    TEST_DATA,
        "model":        "Hindi GPT-2 13.9M",
        "runs":         results,
    }
    out_path = f"{OUTPUT_DIR}/scaling_curve.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Scaling] Saved: {out_path}")

    plot_scaling_curve(results)

if __name__ == "__main__":
    main()
