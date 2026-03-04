import json
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import Tokenizer as HFTokenizer
import sys
sys.path.append(str(Path(__file__).parent))
from model.gpt2 import GPT2_
from model.config import ModelConfig_

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MODEL_ID = "Qwen/Qwen2.5-7B"
DRAFT_B_ID      = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_A    = "checkpoints/best.pt"
TOKENIZER_A     = "tokenizer/hindi_bpe/tokenizer.json"
PROMPTS_PATH    = "speculative_decoding/prompts/hindi_prompts.jsonl"
GAMMA           = 4        
MAX_NEW_TOKENS  = 100
NUM_PROMPTS     = 200     

OUTPUT_A        = "experiments/experiment_a/results"
OUTPUT_B        = "experiments/experiment_b/results"

def max_fn(x: torch.Tensor) -> torch.Tensor:
    x_max     = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / (x_max_sum + 1e-8)

class VocabAligner:
    def __init__(self, your_tokenizer, qwen_tokenizer):
        print("[Aligner] Building vocabulary alignment map...")
        your_vocab  = your_tokenizer.get_vocab()        
        qwen_vocab  = qwen_tokenizer.get_vocab()         
        your_size   = len(your_vocab)
        qwen_size   = len(qwen_vocab)
        self.mapping = torch.zeros(your_size, dtype=torch.long)
        unk_id       = qwen_vocab.get("<unk>", 0)
        matched      = 0

        for token_str, your_id in your_vocab.items():
            qwen_id = qwen_vocab.get(token_str, None)
            if qwen_id is not None:
                self.mapping[your_id] = qwen_id
                matched += 1
            else:
                self.mapping[your_id] = unk_id

        match_pct = matched / your_size * 100
        print(f"  Your vocab size:  {your_size:,}")
        print(f"  Qwen vocab size:  {qwen_size:,}")
        print(f"  Tokens matched:   {matched:,} ({match_pct:.1f}%)")
        print(f"  Tokens unmapped:  {your_size - matched:,} → mapped to <unk>")

        self.match_rate  = match_pct
        self.your_size   = your_size
        self.qwen_size   = qwen_size

    def align(self, your_probs: torch.Tensor) -> torch.Tensor:
        qwen_probs = torch.zeros(
            self.qwen_size,
            dtype  = your_probs.dtype,
            device = your_probs.device,
        )
        qwen_probs.scatter_add_(0, self.mapping.to(your_probs.device), your_probs)
        return qwen_probs
    
@torch.no_grad()
def autoregressive_baseline(
    target_model,
    input_ids:      torch.Tensor,
    max_new_tokens: int,
) -> Tuple[List[int], float]:
    generated = input_ids.clone()
    t_start   = time.perf_counter()

    for _ in range(max_new_tokens):
        out    = target_model(input_ids=generated)
        logits = out.logits[:, -1, :]
        next_t = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_t], dim=-1)

        if next_t.item() in [target_model.config.eos_token_id]:
            break

    elapsed    = time.perf_counter() - t_start
    new_tokens = generated.shape[-1] - input_ids.shape[-1]
    tps        = new_tokens / elapsed

    return generated[0].tolist(), tps

@torch.no_grad()
def speculative_decode(
    draft_model,
    target_model,
    input_ids:      torch.Tensor,
    gamma:          int,
    max_new_tokens: int,
    vocab_aligner   = None,   
) -> Tuple[List[int], float, float]:
    generated         = input_ids.clone()
    drafts_accepted   = 0
    drafts_speculated = 0
    t_start           = time.perf_counter()

    while generated.shape[-1] - input_ids.shape[-1] < max_new_tokens:
        seq_len         = generated.shape[-1]
        actual_gamma    = min(gamma, max_new_tokens - (seq_len - input_ids.shape[-1]))
        draft_tokens = []
        draft_probs  = []
        draft_input  = generated.clone()

        for _ in range(actual_gamma):
            out   = draft_model(input_ids=draft_input)
            logit = out.logits[:, -1, :]                        
            prob  = F.softmax(logit, dim=-1).squeeze(0)          
            if vocab_aligner is not None:
                prob_aligned = vocab_aligner.align(prob)          
            else:
                prob_aligned = prob                               

            token = torch.multinomial(prob, num_samples=1)
            draft_tokens.append(token.item())
            draft_probs.append(prob_aligned)

            draft_input = torch.cat(
                [draft_input, token.unsqueeze(0)], dim=-1
            )

        drafts_speculated += actual_gamma
        draft_sequence = draft_input                             
        out_target     = target_model(input_ids=draft_sequence)
        target_logits  = out_target.logits[0,
                         seq_len - 1 : seq_len + actual_gamma - 1,
                         :]                                       
        target_probs   = F.softmax(target_logits, dim=-1)        

        n = actual_gamma    

        for i in range(actual_gamma):
            token_id = draft_tokens[i]
            q_i = draft_probs[i][token_id].item()       
            p_i = target_probs[i, token_id].item()      

            r = torch.rand(1).item()
            if r <= p_i / (q_i + 1e-8):
                drafts_accepted += 1
            else:
                n = i
                break
        if n > 0:
            accepted_tokens = torch.tensor(
                draft_tokens[:n],
                dtype  = torch.long,
                device = generated.device,
            ).unsqueeze(0)
            generated = torch.cat([generated, accepted_tokens], dim=-1)

        if n == actual_gamma:
            bonus_logit = out_target.logits[0, seq_len + actual_gamma - 1, :]
            bonus_prob  = F.softmax(bonus_logit, dim=-1)
            next_token  = torch.multinomial(bonus_prob, num_samples=1)
        else:
            p_n = target_probs[n]
            q_n = draft_probs[n]
            adjusted = max_fn(p_n - q_n)
            next_token = torch.multinomial(adjusted, num_samples=1)

        generated = torch.cat(
            [generated, next_token.unsqueeze(0)], dim=-1
        )

        if next_token.item() == target_model.config.eos_token_id:
            break

    elapsed    = time.perf_counter() - t_start
    new_tokens = generated.shape[-1] - input_ids.shape[-1]
    tps        = new_tokens / max(elapsed, 1e-6)

    acceptance_rate = drafts_accepted / max(drafts_speculated, 1)
    return generated[0].tolist(), acceptance_rate, tps

def load_prompts(path: str, n: int) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            prompts.append(record["text"])
            if len(prompts) >= n:
                break
    print(f"[Prompts] Loaded {len(prompts)} prompts")
    return prompts

def run_experiment(
    name:          str,
    draft_model,
    target_model,
    qwen_tokenizer,
    prompts:       List[str],
    output_dir:    str,
    vocab_aligner  = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Experiment {name}")
    print(f"Draft:  {'Monolingual Hindi GPT-2' if name == 'A' else 'Qwen2.5-0.5B'}")
    print(f"Target: Qwen2.5-7B")
    print(f"Gamma:  {GAMMA}")
    print(f"{'='*50}\n")

    all_acceptance  = []
    all_spec_tps    = []
    all_base_tps    = []

    for i, prompt in enumerate(prompts):
        input_ids = qwen_tokenizer.encode(
            prompt,
            return_tensors    = "pt",
            max_length        = 128,
            truncation        = True,
            add_special_tokens = True,
        ).to(DEVICE)
        _, base_tps = autoregressive_baseline(
            target_model, input_ids, MAX_NEW_TOKENS
        )
        _, acc_rate, spec_tps = speculative_decode(
            draft_model    = draft_model,
            target_model   = target_model,
            input_ids      = input_ids,
            gamma          = GAMMA,
            max_new_tokens = MAX_NEW_TOKENS,
            vocab_aligner  = vocab_aligner,
        )
        all_acceptance.append(acc_rate)
        all_spec_tps.append(spec_tps)
        all_base_tps.append(base_tps)

        if (i + 1) % 20 == 0:
            print(
                f"  [{i+1}/{len(prompts)}] "
                f"acc={sum(all_acceptance)/len(all_acceptance):.3f} | "
                f"speedup={sum(all_spec_tps)/sum(all_base_tps):.3f}x"
            )
    mean_acceptance = sum(all_acceptance) / len(all_acceptance)
    mean_spec_tps   = sum(all_spec_tps)   / len(all_spec_tps)
    mean_base_tps   = sum(all_base_tps)   / len(all_base_tps)
    speedup         = mean_spec_tps / mean_base_tps

    print(f"\n  Acceptance Rate:   {mean_acceptance:.4f}")
    print(f"  Speedup:           {speedup:.4f}x")
    print(f"  Tokens/sec (spec): {mean_spec_tps:.1f}")
    print(f"  Tokens/sec (base): {mean_base_tps:.1f}")

    acceptance_result = {
        "experiment":       name,
        "acceptance_rate":  round(mean_acceptance, 4),
        "per_prompt":       [round(x, 4) for x in all_acceptance],
        "std":              round(
            torch.tensor(all_acceptance).std().item(), 4
        ),
        "num_prompts":      len(prompts),
        "gamma":            GAMMA,
    }

    speedup_result = {
        "experiment":        name,
        "speedup":           round(speedup, 4),
        "tokens_per_second": round(mean_spec_tps, 2),
        "baseline_tps":      round(mean_base_tps, 2),
        "max_new_tokens":    MAX_NEW_TOKENS,
        "device":            DEVICE,
    }

    with open(f"{output_dir}/acceptance_rate.json", "w") as f:
        json.dump(acceptance_result, f, indent=2)

    with open(f"{output_dir}/speedup.json", "w") as f:
        json.dump(speedup_result, f, indent=2)

    with open(f"{output_dir}/tokens_per_second.json", "w") as f:
        json.dump({
            "spec_tps":  round(mean_spec_tps, 2),
            "base_tps":  round(mean_base_tps, 2),
            "speedup":   round(speedup, 4),
        }, f, indent=2)

    print(f"  Saved to {output_dir}/")

    return {
        "acceptance_rate":  mean_acceptance,
        "speedup":          speedup,
        "tokens_per_second": mean_spec_tps,
    }

def main():
    for path in [CHECKPOINT_A, TOKENIZER_A, PROMPTS_PATH]:
        assert Path(path).exists(), f"Not found: {path}"
    prompts = load_prompts(PROMPTS_PATH, NUM_PROMPTS)
    print("\n[Setup] Loading target model (Qwen2.5-7B)...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    target_model   = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        torch_dtype  = torch.float16,
        device_map   = "auto",
    )
    target_model.eval()
    print("  Target model loaded.")
    print("\n[Setup] Loading your Hindi GPT-2 (draft model A)...")
    your_tokenizer = HFTokenizer.from_file(TOKENIZER_A)
    your_tokenizer.eos_token_id = your_tokenizer.token_to_id("<eos>")

    checkpoint     = torch.load(CHECKPOINT_A, map_location=DEVICE)
    model_cfg      = ModelConfig_(vocab_size=your_tokenizer.get_vocab_size())
    draft_model_a  = GPT2_(model_cfg).to(DEVICE)
    draft_model_a.load_state_dict(checkpoint["model_state"])
    draft_model_a.eval()
    print(f"  Loaded from step {checkpoint['step']:,}")
    vocab_aligner = VocabAligner(your_tokenizer, qwen_tokenizer)
    results_a = run_experiment(
        name           = "A",
        draft_model    = draft_model_a,
        target_model   = target_model,
        qwen_tokenizer = qwen_tokenizer,
        prompts        = prompts,
        output_dir     = OUTPUT_A,
        vocab_aligner  = vocab_aligner,
    )
    del draft_model_a
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print("\n[Setup] Loading Qwen2.5-0.5B (draft model B)...")
    draft_model_b = AutoModelForCausalLM.from_pretrained(
        DRAFT_B_ID,
        torch_dtype  = torch.float16,
        device_map   = "auto",
    )
    draft_model_b.eval()
    print("  Draft B loaded.")

    results_b = run_experiment(
        name           = "B",
        draft_model    = draft_model_b,
        target_model   = target_model,
        qwen_tokenizer = qwen_tokenizer,
        prompts        = prompts,
        output_dir     = OUTPUT_B,
        vocab_aligner  = None,    
    )
    print(f"\n{'='*50}")
    print("FINAL COMPARISON")
    print(f"{'='*50}")
    print(f"{'Metric':<28} {'Exp A':>10} {'Exp B':>10}")
    print("-"*50)
    print(f"{'Acceptance Rate':<28} "
          f"{results_a['acceptance_rate']:>10.4f} "
          f"{results_b['acceptance_rate']:>10.4f}")
    print(f"{'Speedup':<28} "
          f"{results_a['speedup']:>10.4f}x "
          f"{results_b['speedup']:>10.4f}x")
    print(f"{'Tokens/sec':<28} "
          f"{results_a['tokens_per_second']:>10.1f} "
          f"{results_b['tokens_per_second']:>10.1f}")
    print(f"{'Draft params':<28} {'13.9M':>10} {'500M':>10}")
    print(f"{'Vocab alignment needed':<28} {'Yes':>10} {'No':>10}")

    winner = "A (monolingual)" \
        if results_a["acceptance_rate"] > results_b["acceptance_rate"] \
        else "B (multilingual)"
    print(f"\nHigher acceptance rate: Experiment {winner}")
    print(f"\n[Done] Run: python evaluation/compare_experiments.py")


if __name__ == "__main__":
    main()
