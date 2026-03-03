import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

RESULTS_A  = "experiments/experiment_a/results"
RESULTS_B  = "experiments/experiment_b/results"
OUTPUT_DIR = "evaluation/results"

def load_result(folder: str, filename: str) -> dict:
    path = Path(folder) / filename
    assert path.exists(), f"Not found: {path}\nRun speculative_decoding.py first."
    with open(path) as f:
        return json.load(f)

def plot_comparison(a: dict, b: dict):
    import os
    os.makedirs("figures", exist_ok=True)

    metrics = {
        "Acceptance Rate (%)": (
            a["acceptance_rate"] * 100,
            b["acceptance_rate"] * 100,
        ),
        "Speedup (×)": (
            a["speedup"],
            b["speedup"],
        ),
        "Tokens/sec": (
            a["tokens_per_second"],
            b["tokens_per_second"],
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.patch.set_facecolor("#07080d")

    for ax, (metric, (val_a, val_b)) in zip(axes, metrics.items()):
        ax.set_facecolor("#0f0f1a")

        bars = ax.bar(
            ["Monolingual\n(ours)", "Multilingual\n(Qwen2.5-0.5B)"],
            [val_a, val_b],
            color = ["#6d28d9", "#374151"],
            width = 0.5,
        )
        for bar, val in zip(bars, [val_a, val_b]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.2f}",
                ha        = "center",
                va        = "bottom",
                color     = "#e2e8f0",
                fontsize  = 11,
                fontweight = "bold",
            )

        ax.set_title(metric, color="#e2e8f0", fontsize=11, pad=10)
        ax.tick_params(colors="#94a3b8")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color("#1c1d2e")
        ax.set_ylim(0, max(val_a, val_b) * 1.25)

    mono = mpatches.Patch(color="#6d28d9", label="Monolingual draft (ours)")
    mult = mpatches.Patch(color="#374151", label="Multilingual draft (Qwen2.5-0.5B)")
    fig.legend(
        handles    = [mono, mult],
        loc        = "lower center",
        ncol       = 2,
        facecolor  = "#0f0f1a",
        edgecolor  = "#1c1d2e",
        labelcolor = "#94a3b8",
        fontsize   = 10,
        bbox_to_anchor = (0.5, -0.05),
    )

    fig.suptitle(
        "Speculative Decoding: Monolingual vs Multilingual Draft Model",
        color="#e2e8f0", fontsize=13, y=1.02,
    )

    plt.tight_layout()
    plt.savefig(
        "figures/experiment_comparison.png",
        dpi=150, bbox_inches="tight", facecolor="#07080d"
    )
    plt.close()
    print("[Plot] Saved: figures/experiment_comparison.png")


def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[Compare] Loading experiment results...")
    a_accept = load_result(RESULTS_A, "acceptance_rate.json")
    b_accept = load_result(RESULTS_B, "acceptance_rate.json")
    a_speed  = load_result(RESULTS_A, "speedup.json")
    b_speed  = load_result(RESULTS_B, "speedup.json")

    a = {**a_accept, **a_speed, "draft_model": "Hindi GPT-2 13.9M (monolingual)"}
    b = {**b_accept, **b_speed, "draft_model": "Qwen2.5-0.5B (multilingual)"}
    print(f"\n{'='*55}")
    print("EXPERIMENT A vs B — FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"{'Metric':<28} {'Monolingual':>12} {'Multilingual':>12}")
    print("-"*55)

    metrics = [
        ("Acceptance Rate",    f"{a['acceptance_rate']*100:.2f}%",  f"{b['acceptance_rate']*100:.2f}%"),
        ("Speedup vs baseline", f"{a['speedup']:.3f}×",             f"{b['speedup']:.3f}×"),
        ("Tokens / second",    f"{a['tokens_per_second']:.1f}",     f"{b['tokens_per_second']:.1f}"),
        ("Draft model params", "13.9M",                              "500M"),
    ]

    for name, val_a, val_b in metrics:
        print(f"{name:<28} {val_a:>12} {val_b:>12}")

    print(f"\n{'='*55}")
    if a["acceptance_rate"] > b["acceptance_rate"]:
        delta = (a["acceptance_rate"] - b["acceptance_rate"]) * 100
        print(f"✓ Monolingual draft wins acceptance rate by {delta:.2f}pp")
        print(f"  Supports hypothesis: domain-specific draft models")
        print(f"  improve speculative decoding for Indic languages.")
    else:
        delta = (b["acceptance_rate"] - a["acceptance_rate"]) * 100
        print(f"✗ Multilingual draft wins acceptance rate by {delta:.2f}pp")
        print(f"  Suggests vocabulary alignment cost of monolingual model")
        print(f"  outweighs distribution advantage at this model scale.")
        print(f"  This is a valid and publishable negative result.")
    output = {
        "compared_at":    datetime.now().isoformat(),
        "experiment_a":   a,
        "experiment_b":   b,
        "verdict":        "monolingual_wins" if a["acceptance_rate"] > b["acceptance_rate"]
                          else "multilingual_wins",
        "acceptance_delta_pp": round(
            abs(a["acceptance_rate"] - b["acceptance_rate"]) * 100, 4
        ),
    }

    out_path = f"{OUTPUT_DIR}/experiment_comparison.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Compare] Saved: {out_path}")

    plot_comparison(a, b)
    print(f"[Done] Next: write the paper.")

if __name__ == "__main__":
    main()