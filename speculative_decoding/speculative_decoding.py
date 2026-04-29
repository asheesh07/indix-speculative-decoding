import json
import os
import time
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tokenizers import Tokenizer as HFTokenizer
import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.gpt2 import GPT2_
from model.config import ModelConfig_

DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_MODEL_ID = "Qwen/Qwen2.5-1.5B"
DRAFT_B_ID      = "Qwen/Qwen2.5-0.5B"
CHECKPOINT_A    = "checkpoints/best.pt"
TOKENIZER_A     = "tokenizer/hindi_bpe/tokenizer.json"
PROMPTS_PATH    = "speculative_decoding/prompts/hindi_prompts.jsonl"
GAMMA           = 4
MAX_NEW_TOKENS  = 100
NUM_PROMPTS     = 136

OUTPUT_A        = "experiments/experiment_a/results"
OUTPUT_B        = "experiments/experiment_b/results"

def max_fn(x: torch.Tensor) -> torch.Tensor:
    x_max     = torch.where(x > 0, x, torch.zeros_like(x))
    x_max_sum = torch.sum(x_max, dim=-1, keepdim=True)
    return x_max / (x_max_sum + 1e-8)

class VocabAligner:
    def __init__(self, draft_tokenizer, qwen_tokenizer):
        print("[Aligner] Building byte-level vocabulary alignment map...")
        draft_vocab = draft_tokenizer.get_vocab()
        qwen_vocab  = qwen_tokenizer.get_vocab()
        draft_size  = len(draft_vocab)
        qwen_size   = len(qwen_vocab)

        self.draft_size  = draft_size
        self.qwen_size   = qwen_size
        self.draft_unk_id = draft_vocab.get("<unk>", draft_vocab.get("[UNK]", 1))

        # For each draft token, store the list of Qwen token IDs it decomposes into
        # self.mapping[draft_id] = [qwen_id1, qwen_id2, ...]
        self.mapping      = {}   # draft_id → list of qwen_ids
        self.single_map   = torch.zeros(draft_size, dtype=torch.long)  # for draft_id_to_qwen_id

        exact_matched = 0
        byte_matched  = 0
        unmatched     = 0
        unk_id        = qwen_vocab.get("<unk>", qwen_vocab.get("[UNK]", 1))

        for token_str, draft_id in draft_vocab.items():
            # Step 1: try exact string match first
            qwen_id = qwen_vocab.get(token_str, None)
            if qwen_id is not None:
                self.mapping[draft_id]    = [qwen_id]
                self.single_map[draft_id] = qwen_id
                exact_matched += 1
                continue

            # Step 2: decompose token into UTF-8 bytes
            # Qwen stores bytes as hex strings like "<0xe0>" or as raw byte chars
            # Try encoding the token and looking up each byte in Qwen vocab
            try:
                token_bytes = token_str.encode("utf-8")
                qwen_ids = []

                for byte in token_bytes:
                    # Qwen byte token format: "Ġ" prefix for space, raw byte otherwise
                    byte_str  = bytes([byte]).decode("latin-1")
                    byte_hex  = f"<0x{byte:02X}>"

                    # Try multiple formats Qwen uses for byte tokens
                    bid = (qwen_vocab.get(byte_str)
                        or qwen_vocab.get(byte_hex)
                        or qwen_vocab.get(f"▁{byte_str}"))

                    if bid is not None:
                        qwen_ids.append(bid)
                    else:
                        qwen_ids = []
                        break

                if qwen_ids:
                    self.mapping[draft_id]    = qwen_ids
                    self.single_map[draft_id] = qwen_ids[0]  # first byte as primary
                    byte_matched += 1
                else:
                    self.mapping[draft_id]    = [unk_id]
                    self.single_map[draft_id] = unk_id
                    unmatched += 1

            except Exception:
                self.mapping[draft_id]    = [unk_id]
                self.single_map[draft_id] = unk_id
                unmatched += 1

        total_matched = exact_matched + byte_matched
        match_pct     = total_matched / draft_size * 100

        print(f"  Draft vocab size:  {draft_size:,}")
        print(f"  Qwen vocab size:   {qwen_size:,}")
        print(f"  Exact matched:     {exact_matched:,}")
        print(f"  Byte matched:      {byte_matched:,}")
        print(f"  Unmatched:         {unmatched:,}")
        print(f"  Total match rate:  {match_pct:.1f}%")

        self.match_rate = match_pct

    def align(self, draft_probs: torch.Tensor) -> torch.Tensor:
        """
        Map draft vocab probs → Qwen vocab probs.
        For byte-decomposed tokens, probability is split equally across
        the constituent byte tokens.
        """
        qwen_probs = torch.zeros(
            self.qwen_size,
            dtype  = draft_probs.dtype,
            device = draft_probs.device,
        )

        for draft_id, qwen_ids in self.mapping.items():
            prob  = draft_probs[draft_id]
            share = prob / len(qwen_ids)       # split equally across byte tokens
            for qid in qwen_ids:
                qwen_probs[qid] += share

        return qwen_probs

    def draft_id_to_qwen_id(self, draft_token_id: int) -> int:
        return self.single_map[draft_token_id].item()

    def qwen_id_to_draft_id(self, qwen_id: int) -> int:
        matches = (self.single_map == qwen_id).nonzero(as_tuple=True)[0]
        if matches.numel() > 0:
            return matches[0].item()
        return self.draft_unk_id


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
        probs  = F.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_t], dim=-1)

        if next_t.item() == target_model.config.eos_token_id:
            break

    elapsed    = time.perf_counter() - t_start
    new_tokens = generated.shape[-1] - input_ids.shape[-1]
    tps        = new_tokens / max(elapsed, 1e-6)

    return generated[0].tolist(), tps


@torch.no_grad()
def speculative_decode(
    draft_model,
    target_model,
    input_ids:       torch.Tensor,
    gamma:           int,
    max_new_tokens:  int,
    vocab_aligner:   Optional[object] = None,
    draft_input_ids: Optional[torch.Tensor] = None,
) -> Tuple[List[int], float, float]:

    generated      = input_ids.clone()
    draft_context  = draft_input_ids.clone() if draft_input_ids is not None \
                     else input_ids.clone()

    drafts_accepted   = 0
    drafts_speculated = 0
    t_start           = time.perf_counter()

    while generated.shape[-1] - input_ids.shape[-1] < max_new_tokens:
        seq_len      = generated.shape[-1]
        actual_gamma = min(gamma, max_new_tokens - (seq_len - input_ids.shape[-1]))

        draft_tokens    = []
        draft_probs_raw = []
        draft_probs_aln = []
        draft_input     = draft_context.clone()

        for _ in range(actual_gamma):
            out   = draft_model(input_ids=draft_input)
            logit = out.logits[:, -1, :] if hasattr(out, "logits") else out[:, -1, :]
            prob  = F.softmax(logit, dim=-1).squeeze(0)

            if vocab_aligner is not None:
                prob_aligned = vocab_aligner.align(prob)
            else:
                prob_aligned = prob

            token = torch.multinomial(prob, num_samples=1)
            draft_tokens.append(token.item())
            draft_probs_raw.append(prob)
            draft_probs_aln.append(prob_aligned)

            draft_input = torch.cat(
                [draft_input, token.unsqueeze(0)], dim=-1
            )

        drafts_speculated += actual_gamma

        if vocab_aligner is not None:
            draft_qwen_ids = torch.tensor(
                [vocab_aligner.draft_id_to_qwen_id(t) for t in draft_tokens],
                dtype  = torch.long,
                device = generated.device,
            ).unsqueeze(0)
            draft_sequence = torch.cat([generated, draft_qwen_ids], dim=-1)
        else:
            draft_sequence = draft_input

        out_target    = target_model(input_ids=draft_sequence)
        target_logits = out_target.logits[0,
                        seq_len - 1 : seq_len + actual_gamma - 1,
                        :]
        target_probs  = F.softmax(target_logits, dim=-1)

        n = actual_gamma

        for i in range(actual_gamma):
            token_id = draft_tokens[i]

            if vocab_aligner is not None:
                q_i     = draft_probs_raw[i][token_id].item()
                qwen_id = vocab_aligner.draft_id_to_qwen_id(token_id)
                p_i     = target_probs[i, qwen_id].item()
            else:
                q_i = draft_probs_raw[i][token_id].item()
                p_i = target_probs[i, token_id].item()

            r = torch.rand(1).item()
            if r <= p_i / (q_i + 1e-8):
                drafts_accepted += 1
            else:
                n = i
                break

        if n > 0:
            if vocab_aligner is not None:
                accepted_qwen = torch.tensor(
                    [vocab_aligner.draft_id_to_qwen_id(t) for t in draft_tokens[:n]],
                    dtype  = torch.long,
                    device = generated.device,
                ).unsqueeze(0)
                generated = torch.cat([generated, accepted_qwen], dim=-1)
            else:
                accepted_tokens = torch.tensor(
                    draft_tokens[:n],
                    dtype  = torch.long,
                    device = generated.device,
                ).unsqueeze(0)
                generated = torch.cat([generated, accepted_tokens], dim=-1)
            accepted_draft = torch.tensor(
                draft_tokens[:n],
                dtype  = torch.long,
                device = draft_context.device,
            ).unsqueeze(0)
            draft_context = torch.cat([draft_context, accepted_draft], dim=-1)

        if n == actual_gamma:
            bonus_logit     = out_target.logits[0, seq_len + actual_gamma - 1, :]
            bonus_prob      = F.softmax(bonus_logit, dim=-1)
            next_token_qwen = torch.multinomial(bonus_prob, num_samples=1)
        else:
            p_n             = target_probs[n]
            q_n             = draft_probs_aln[n]
            adjusted        = max_fn(p_n - q_n)
            next_token_qwen = torch.multinomial(adjusted, num_samples=1)

        generated = torch.cat(
            [generated, next_token_qwen.unsqueeze(0)], dim=-1
        )

        if vocab_aligner is not None:
            next_qwen_id = next_token_qwen.item()
            reverse      = (vocab_aligner.mapping == next_qwen_id).nonzero(as_tuple=True)[0]
            if reverse.numel() > 0:
                next_draft_id = reverse[0].unsqueeze(0).unsqueeze(0).to(draft_context.device)
            else:
                # fix 3 continued: use stored draft_unk_id instead of missing attribute
                next_draft_id = torch.tensor(
                    [[vocab_aligner.draft_unk_id]],
                    dtype=torch.long, device=draft_context.device,
                )
        else:
            next_draft_id = next_token_qwen.unsqueeze(0)

        draft_context = torch.cat([draft_context, next_draft_id], dim=-1)

        if next_token_qwen.item() == target_model.config.eos_token_id:
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
    draft_tokenizer = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Experiment {name}")
    print(f"Draft:  {'Monolingual Hindi GPT-2' if name == 'A' else 'Qwen2.5-0.5B'}")
    print(f"Target: {TARGET_MODEL_ID}")
    print(f"Gamma:  {GAMMA}")
    print(f"{'='*50}\n")

    if vocab_aligner is not None and draft_tokenizer is None:
        raise ValueError("draft_tokenizer must be provided for Experiment A (cross-vocab)")

    all_acceptance = []
    all_spec_tps   = []
    all_base_tps   = []

    for i, prompt in enumerate(prompts):

        input_ids_qwen = qwen_tokenizer.encode(
            prompt,
            return_tensors     = "pt",
            max_length         = 128,
            truncation         = True,
            add_special_tokens = True,
        ).to(DEVICE)

        if vocab_aligner is not None:
            draft_ids = torch.tensor(
                [draft_tokenizer.encode(prompt, add_special_tokens=True).ids],
                dtype  = torch.long,
                device = DEVICE,
            )
        else:
            draft_ids = None

        _, base_tps = autoregressive_baseline(
            target_model, input_ids_qwen, MAX_NEW_TOKENS
        )
        _, acc_rate, spec_tps = speculative_decode(
            draft_model      = draft_model,
            target_model     = target_model,
            input_ids        = input_ids_qwen,
            gamma            = GAMMA,
            max_new_tokens   = MAX_NEW_TOKENS,
            vocab_aligner    = vocab_aligner,
            draft_input_ids  = draft_ids,
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
    speedup         = sum(all_spec_tps)   / sum(all_base_tps)

    print(f"\n  Acceptance Rate:   {mean_acceptance:.4f}")
    print(f"  Speedup:           {speedup:.4f}x")
    print(f"  Tokens/sec (spec): {mean_spec_tps:.1f}")
    print(f"  Tokens/sec (base): {mean_base_tps:.1f}")

    acceptance_result = {
        "experiment":      name,
        "acceptance_rate": round(mean_acceptance, 4),
        "per_prompt":      [round(x, 4) for x in all_acceptance],
        "std":             round(torch.tensor(all_acceptance).std().item(), 4),
        "num_prompts":     len(prompts),
        "gamma":           GAMMA,
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
            "spec_tps": round(mean_spec_tps, 2),
            "base_tps": round(mean_base_tps, 2),
            "speedup":  round(speedup, 4),
        }, f, indent=2)

    print(f"  Saved to {output_dir}/")

    return {
        "acceptance_rate":   mean_acceptance,
        "speedup":           speedup,
        "tokens_per_second": mean_spec_tps,
    }


def main():
    for path in [CHECKPOINT_A, TOKENIZER_A, PROMPTS_PATH]:
        assert Path(path).exists(), f"Not found: {path}"

    prompts = load_prompts(PROMPTS_PATH, NUM_PROMPTS)

    print(f"\n[Setup] Loading target model ({TARGET_MODEL_ID})...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    target_model   = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_ID,
        dtype      = torch.float16,
        device_map = "auto",
    )
    target_model.eval()
    print("  Target model loaded.")

    print("\n[Setup] Loading your Hindi GPT-2 (draft model A)...")
    draft_tokenizer = HFTokenizer.from_file(TOKENIZER_A)
    draft_tokenizer.eos_token_id = draft_tokenizer.token_to_id("<eos>")

    checkpoint    = torch.load(CHECKPOINT_A, map_location=DEVICE)
    model_cfg     = ModelConfig_(vocab_size=draft_tokenizer.get_vocab_size())
    draft_model_a = GPT2_(model_cfg).to(DEVICE)
    draft_model_a.load_state_dict(checkpoint["model_state"])
    draft_model_a.eval()
    print(f"  Loaded from step {checkpoint['step']:,}")

    vocab_aligner = VocabAligner(draft_tokenizer, qwen_tokenizer)

    results_a = run_experiment(
        name           = "A",
        draft_model    = draft_model_a,
        target_model   = target_model,
        qwen_tokenizer = qwen_tokenizer,
        prompts        = prompts,
        output_dir     = OUTPUT_A,
        vocab_aligner  = vocab_aligner,
        draft_tokenizer = draft_tokenizer,
    )

    del draft_model_a
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print("\n[Setup] Loading Qwen2.5-0.5B (draft model B)...")
    draft_model_b = AutoModelForCausalLM.from_pretrained(
        DRAFT_B_ID,
        dtype      = torch.float16,
        device_map = "auto",
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
        draft_tokenizer = None,
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