from datasets import load_dataset
import json

def collect_hindi_subset(target_size_gb=1.0):
    target_bytes = int(target_size_gb * 1024 * 1024 * 1024)
    
    print(f"[INFO] Streaming IndicCorp Hindi...")
    print(f"[INFO] Target size: {target_size_gb}GB ({target_bytes:,} bytes)")
    
    dataset = load_dataset(
    "ai4bharat/IndicCorpv2",
    name="indiccorp_v2",
    split="hin_Deva",
    streaming=True
)
    
    collected       = []
    total_bytes     = 0
    total_examples  = 0
    skipped         = 0

    for example in dataset:
        text = example.get("text", "").strip()
        
        # Quality filters
        if len(text) < 100:          # skip very short lines
            skipped += 1
            continue
        if len(text) > 100_000:      # skip abnormally long documents
            skipped += 1
            continue
        if not any('\u0900' <= c <= '\u097F' for c in text):  # must contain Devanagari
            skipped += 1
            continue
            
        collected.append(text)
        total_bytes += len(text.encode("utf-8"))
        total_examples += 1
        
        # Progress
        if total_examples % 10_000 == 0:
            gb_collected = total_bytes / (1024**3)
            print(f"  Examples: {total_examples:,} | Size: {gb_collected:.3f}GB | Skipped: {skipped:,}")
        
        # Stop when target reached
        if total_bytes >= target_bytes:
            print(f"\n[DONE] Target reached.")
            break
    
    print(f"\n[SUMMARY]")
    print(f"  Total examples collected:  {total_examples:,}")
    print(f"  Total size:                {total_bytes / (1024**3):.3f}GB")
    print(f"  Total skipped:             {skipped:,}")
    
    return collected


def save_and_split(texts, output_dir="data"):
    import os
    import random
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle before split
    random.seed(42)
    random.shuffle(texts)
    
    # 90% train, 5% val, 5% test
    n          = len(texts)
    train_end  = int(n * 0.90)
    val_end    = int(n * 0.95)
    
    train = texts[:train_end]
    val   = texts[train_end:val_end]
    test  = texts[val_end:]
    
    # Save as jsonl
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = f"{output_dir}/{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for text in split_data:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        
        size_mb = os.path.getsize(path) / (1024**2)
        print(f"  {split_name:10s} {len(split_data):>8,} examples   {size_mb:.1f}MB   → {path}")
    
    print(f"\n[IMPORTANT] Test set is now locked. Do not touch it until final evaluation.")
    return train, val, test


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    texts = collect_hindi_subset(target_size_gb=1.0)
    save_and_split(texts, output_dir="data")


