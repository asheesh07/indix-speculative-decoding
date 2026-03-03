import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from tokenizers import Tokenizer

OUTPUT_DIR    = "tokenizer/results"
TOKENIZER_DIR = "tokenizer/hindi_bpe"
DATA_PATH     = "data/processed/test.jsonl"   
MAX_SENTENCES = 2000                           

BASELINES = {
    "mBERT":      "bert-base-multilingual-cased",
    "IndicBERT":  "ai4bharat/indic-bert",
    "Qwen2.5":    "Qwen/Qwen2.5-0.5B",
}

def load_test_sentences(path: str, max_sentences: int) -> list:
    sentences = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(sentences) >= max_sentences:
                break
            record = json.loads(line.strip())
            text   = record.get("text", "").strip()

            if not text:
                continue

            for sent in text.split("।"):
                sent = sent.strip()
                if len(sent.split()) >= 5:    
                    sentences.append(sent)
                    break

    print(f"[Analysis] Loaded {len(sentences):,} test sentences")
    return sentences[:max_sentences]

def compute_fertility(tokenizer, sentences: list, name: str) -> dict:
    total_tokens = 0
    total_words  = 0
    per_sentence = []

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        try:
            if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
                ids    = tokenizer.encode(sent, add_special_tokens=False)
                tokens = len(ids) if isinstance(ids, list) else len(ids.ids)
            else:
                tokens = len(tokenizer.encode(sent))
        except Exception:
            continue

        word_count    = len(words)
        fertility     = tokens / word_count
        total_tokens += tokens
        total_words  += word_count
        per_sentence.append(fertility)

    overall_fertility = total_tokens / total_words if total_words > 0 else 0

    result = {
        "tokenizer":          name,
        "overall_fertility":  round(overall_fertility, 4),
        "mean_fertility":     round(float(np.mean(per_sentence)), 4),
        "std_fertility":      round(float(np.std(per_sentence)), 4),
        "median_fertility":   round(float(np.median(per_sentence)), 4),
        "total_tokens":       total_tokens,
        "total_words":        total_words,
        "sentences_analyzed": len(per_sentence),
        "per_sentence":       per_sentence,   
    }

    print(f"  {name:15s}  fertility={overall_fertility:.4f}  "
          f"(mean={result['mean_fertility']:.4f} ± {result['std_fertility']:.4f})")

    return result

def analyze_vocabulary_overlap(your_tokenizer, sentences: list) -> dict:
    vocab = your_tokenizer.get_vocab()

    devanagari = 0
    latin      = 0
    mixed      = 0
    other      = 0

    for token in vocab.keys():
        if token.startswith("<"):
            continue

        has_deva  = any('\u0900' <= c <= '\u097F' for c in token)
        has_latin = any('a' <= c.lower() <= 'z' for c in token)

        if has_deva and has_latin:
            mixed += 1
        elif has_deva:
            devanagari += 1
        elif has_latin:
            latin += 1
        else:
            other += 1

    total = devanagari + latin + mixed + other

    result = {
        "vocab_size":        len(vocab),
        "devanagari_tokens": devanagari,
        "latin_tokens":      latin,
        "mixed_tokens":      mixed,
        "other_tokens":      other,
        "devanagari_pct":    round(devanagari / total * 100, 2),
        "latin_pct":         round(latin / total * 100, 2),
    }

    print(f"\n[Vocab Composition]")
    print(f"  Total vocab:    {len(vocab):,}")
    print(f"  Devanagari:     {devanagari:,} ({result['devanagari_pct']}%)")
    print(f"  Latin:          {latin:,} ({result['latin_pct']}%)")
    print(f"  Mixed:          {mixed:,}")
    print(f"  Other:          {other:,}")

    return result

def show_tokenization_examples(your_tokenizer, baseline_tokenizers: dict):
    examples = [
        "नमस्ते, मेरा नाम अशीष है।",
        "कृत्रिम बुद्धिमत्ता भारत के भविष्य को बदल रही है।",
        "मशीन लर्निंग मॉडल को प्रशिक्षित करने के लिए डेटा चाहिए।",
        "भाषा मॉडल हिंदी पाठ को समझने में सक्षम हैं।",
    ]

    qualitative = []

    print(f"\n{'='*60}")
    print("Qualitative Tokenization Examples")
    print(f"{'='*60}")

    for sent in examples:
        print(f"\nSentence: {sent}")
        entry = {"sentence": sent, "tokenizations": {}}
        encoded = your_tokenizer.encode(sent)
        tokens  = encoded.tokens if hasattr(encoded, "tokens") else []
        print(f"  {'YourBPE':15s} [{len(tokens):2d} tokens]: {tokens}")
        entry["tokenizations"]["YourBPE"] = {
            "tokens": tokens,
            "count":  len(tokens),
        }
        for name, tok in baseline_tokenizers.items():
            try:
                ids    = tok.encode(sent, add_special_tokens=False)
                tokens = tok.convert_ids_to_tokens(ids)
                print(f"  {name:15s} [{len(tokens):2d} tokens]: {tokens}")
                entry["tokenizations"][name] = {
                    "tokens": tokens,
                    "count":  len(tokens),
                }
            except Exception as e:
                print(f"  {name:15s} ERROR: {e}")

        qualitative.append(entry)

    return qualitative

def plot_fertility_comparison(results: list, output_path: str):
    names      = [r["tokenizer"] for r in results]
    fertilities = [r["overall_fertility"] for r in results]
    stds       = [r["std_fertility"] for r in results]

    colors = [
        "#6d28d9" if name == "YourBPE" else "#374151"
        for name in names
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#07080d")
    ax.set_facecolor("#0f0f1a")

    bars = ax.bar(
        names,
        fertilities,
        color   = colors,
        yerr    = stds,
        capsize = 5,
        error_kw = {"color": "#94a3b8", "linewidth": 1.5},
        width   = 0.5,
    )

    for bar, val in zip(bars, fertilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{val:.2f}",
            ha        = "center",
            va        = "bottom",
            color     = "#e2e8f0",
            fontsize  = 11,
            fontweight = "bold",
        )

    ax.set_xlabel("Tokenizer", color="#94a3b8", fontsize=12)
    ax.set_ylabel("Fertility Rate (tokens/word)", color="#94a3b8", fontsize=12)
    ax.set_title(
        "Tokenizer Fertility Rate on Hindi Text\n(Lower = More Efficient)",
        color    = "#e2e8f0",
        fontsize = 14,
        pad      = 15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#1c1d2e")
    ax.spines["left"].set_color("#1c1d2e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(fertilities) * 1.3)

    highlight = mpatches.Patch(color="#6d28d9", label="Your Hindi BPE (this work)")
    baseline  = mpatches.Patch(color="#374151", label="Baseline tokenizers")
    ax.legend(
        handles    = [highlight, baseline],
        facecolor  = "#0f0f1a",
        edgecolor  = "#1c1d2e",
        labelcolor = "#94a3b8",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#07080d")
    plt.close()
    print(f"\n[Plot] Saved to {output_path}")


def plot_fertility_distribution(results: list, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#07080d")
    ax.set_facecolor("#0f0f1a")

    data   = [r["per_sentence"] for r in results]
    names  = [r["tokenizer"] for r in results]
    colors = ["#6d28d9" if n == "YourBPE" else "#374151" for n in names]

    bp = ax.boxplot(
        data,
        labels   = names,
        patch_artist = True,
        medianprops  = {"color": "#e2e8f0", "linewidth": 2},
        whiskerprops = {"color": "#94a3b8"},
        capprops     = {"color": "#94a3b8"},
        flierprops   = {"markerfacecolor": "#94a3b8", "markersize": 3},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xlabel("Tokenizer", color="#94a3b8", fontsize=12)
    ax.set_ylabel("Fertility Rate (tokens/word)", color="#94a3b8", fontsize=12)
    ax.set_title(
        "Fertility Rate Distribution on Hindi Text",
        color="#e2e8f0", fontsize=14, pad=15,
    )
    ax.tick_params(colors="#94a3b8")
    ax.spines["bottom"].set_color("#1c1d2e")
    ax.spines["left"].set_color("#1c1d2e")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#07080d")
    plt.close()
    print(f"[Plot] Saved to {output_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    print("[Analysis] Loading your Hindi BPE tokenizer...")
    your_tokenizer = Tokenizer.from_file(f"{TOKENIZER_DIR}/tokenizer.json")
    print("\n[Analysis] Loading baseline tokenizers...")
    baseline_tokenizers = {}
    for name, model_id in BASELINES.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            baseline_tokenizers[name] = tok
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
    print(f"\n[Analysis] Loading test sentences from {DATA_PATH}...")
    sentences = load_test_sentences(DATA_PATH, MAX_SENTENCES)
    print(f"\n[Analysis] Computing fertility rates on {len(sentences):,} sentences...")
    results = []
    your_result = compute_fertility(your_tokenizer, sentences, "YourBPE")
    results.append(your_result)
    for name, tok in baseline_tokenizers.items():
        result = compute_fertility(tok, sentences, name)
        results.append(result)
    vocab_analysis = analyze_vocabulary_overlap(your_tokenizer, sentences)
    qualitative = show_tokenization_examples(your_tokenizer, baseline_tokenizers)
    output = {
        "fertility_results": [
            {k: v for k, v in r.items() if k != "per_sentence"}
            for r in results
        ],
        "vocabulary_analysis": vocab_analysis,
        "qualitative_examples": qualitative,
        "sentences_analyzed":  len(sentences),
        "your_tokenizer_path": TOKENIZER_DIR,
    }

    results_path = f"{OUTPUT_DIR}/fertility_analysis.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[Analysis] Results saved to {results_path}")
    
    plot_fertility_comparison(results, "figures/fertility_comparison.png")
    plot_fertility_distribution(results, "figures/fertility_distribution.png")

    print(f"\n{'='*55}")
    print(f"{'Tokenizer':<15} {'Fertility':>10} {'Std':>8} {'vs YourBPE':>12}")
    print(f"{'='*55}")

    your_fertility = results[0]["overall_fertility"]
    for r in results:
        delta = r["overall_fertility"] - your_fertility
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        marker = " ← (yours)" if r["tokenizer"] == "YourBPE" else ""
        print(
            f"{r['tokenizer']:<15} "
            f"{r['overall_fertility']:>10.4f} "
            f"{r['std_fertility']:>8.4f} "
            f"{delta_str:>12}"
            f"{marker}"
        )

    print(f"\n[Done] Copy this table into your paper.")
    print(f"[Done] Figures saved to figures/")


if __name__ == "__main__":
    main()
