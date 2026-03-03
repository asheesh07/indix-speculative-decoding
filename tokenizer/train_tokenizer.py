import os
import json
from pathlib import Path
from tokenizers import (
    Tokenizer,
    models,
    trainers,
    pre_tokenizers,
    decoders,
    processors,
)
from tokenizers.normalizers import NFC
from datasets import load_dataset

VOCAB_SIZE     = 8000      
OUTPUT_DIR     = "tokenizer/hindi_bpe"
DATA_PATH      = "data/processed/train.jsonl"   
MAX_TRAIN_MB   = 200        
                            

SPECIAL_TOKENS = [
    "<pad>",    
    "<unk>",    
    "<bos>",    
    "<eos>",    
]

def hindi_text_iterator(data_path: str, max_mb: int = 200):
    """
    Yields text strings from JSONL file up to max_mb.
    Used to feed the tokenizer trainer.
    """
    max_bytes  = max_mb * 1024 * 1024
    total      = 0
    count      = 0

    path = Path(data_path)
    assert path.exists(), f"Data file not found: {data_path}\nRun scripts/collect_data.py first."

    print(f"[Tokenizer] Reading from {data_path}")
    print(f"[Tokenizer] Training on up to {max_mb}MB of text")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            text   = record.get("text", "").strip()

            if not text:
                continue

            yield text
            total += len(text.encode("utf-8"))
            count += 1

            if count % 10_000 == 0:
                print(f"  {count:,} documents | {total / 1024**2:.1f}MB")

            if total >= max_bytes:
                print(f"[Tokenizer] Reached {max_mb}MB limit at {count:,} documents")
                break

    print(f"[Tokenizer] Total: {count:,} documents, {total / 1024**2:.1f}MB")
def train_tokenizer():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training Hindi BPE Tokenizer")
    print(f"Vocab size:    {VOCAB_SIZE}")
    print(f"Output:        {OUTPUT_DIR}")
    print(f"{'='*50}\n")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ])
    tokenizer.decoder = decoders.BPEDecoder()
    trainer = trainers.BpeTrainer(
        vocab_size         = VOCAB_SIZE,
        special_tokens     = SPECIAL_TOKENS,
        min_frequency      = 2,     
        show_progress      = True,
        initial_alphabet   = list(
            [chr(i) for i in range(0x0900, 0x0980)]
        ),
    )
    print("[Tokenizer] Starting BPE training...")
    tokenizer.train_from_iterator(
        hindi_text_iterator(DATA_PATH, MAX_TRAIN_MB),
        trainer = trainer,
    )
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")

    tokenizer.post_processor = processors.TemplateProcessing(
        single   = "<bos> $A <eos>",
        special_tokens = [
            ("<bos>", bos_id),
            ("<eos>", eos_id),
        ],
    )
    tokenizer.save(f"{OUTPUT_DIR}/tokenizer.json")
    from transformers import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = tokenizer,
        bos_token        = "<bos>",
        eos_token        = "<eos>",
        unk_token        = "<unk>",
        pad_token        = "<pad>",
    )
    hf_tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n[Tokenizer] Saved to {OUTPUT_DIR}/")
    print(f"[Tokenizer] Vocab size: {tokenizer.get_vocab_size()}")

    return tokenizer

def verify_tokenizer(tokenizer):
    test_sentences = [
        "नमस्ते, मेरा नाम अशीष है।",
        "भारत एक विविधताओं से भरा देश है।",
        "कृत्रिम बुद्धिमत्ता का भविष्य उज्जवल है।",
        "मशीन लर्निंग और डीप लर्निंग में अंतर क्या है?",
        "Hello this is an English sentence mixed in.",  
    ]

    print(f"\n{'='*50}")
    print("Tokenizer Verification")
    print(f"{'='*50}")

    for sentence in test_sentences:
        encoded = tokenizer.encode(sentence)
        decoded = tokenizer.decode(encoded.ids)
        tokens  = encoded.tokens

        print(f"\nOriginal:  {sentence}")
        print(f"Tokens:    {tokens}")
        print(f"Token IDs: {encoded.ids}")
        print(f"Decoded:   {decoded}")
        print(f"Fertility: {len(tokens) / len(sentence.split()):.2f} tokens/word")

if __name__ == "__main__":
    tokenizer = train_tokenizer()
    verify_tokenizer(tokenizer)
    print("\n✓ Tokenizer training complete.")
    print(f"  Next step: python tokenizer/analyze_tokenizer.py")
