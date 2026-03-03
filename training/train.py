import os
import math
import time
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Optional
import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tokenizers import Tokenizer

import sys
sys.path.append(str(Path(__file__).parent.parent))
from model.gpt2 import GPT2_
from model.config import ModelConfig_
from training.dataset import get_dataloaders

@dataclass
class TrainConfig:

    data_dir:          str   = "data/processed"
    tokenizer_path:    str   = "tokenizer/hindi_bpe/tokenizer.json"
    output_dir:        str   = "checkpoints"


    max_steps:         int   = 50_000      
    batch_size:        int   = 8           
    grad_accum_steps:  int   = 8           
    seq_len:           int   = 512
    stride:            int   = 256


    learning_rate:     float = 3e-4
    weight_decay:      float = 0.1
    beta1:             float = 0.9
    beta2:             float = 0.95
    grad_clip:         float = 1.0


    warmup_steps:      int   = 2_000        
    lr_decay_steps:    int   = 50_000       
    min_lr:            float = 3e-5         


    eval_every:        int   = 500         
    eval_steps:        int   = 50           
    save_every:        int   = 2_000        
    log_every:         int   = 50           

    use_wandb:         bool  = False       
    wandb_project:     str   = "indic-speculative-decoding"
    run_name:          str   = "hindi-gpt2-13.9M"

    resume_from:       Optional[str] = None    

    @property
    def effective_batch_size(self):
        return self.batch_size * self.grad_accum_steps

def get_lr_scheduler(optimizer, cfg: TrainConfig) -> LambdaLR:
    
    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)

        if step > cfg.lr_decay_steps:
            return cfg.min_lr / cfg.learning_rate

        progress = (step - cfg.warmup_steps) / max(
            1, cfg.lr_decay_steps - cfg.warmup_steps
        )
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr / cfg.learning_rate + cosine * (
            1.0 - cfg.min_lr / cfg.learning_rate
        )

    return LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    val_loader,
    device:     str,
    max_steps:  int = 50,
) -> dict:
    model.eval()

    total_loss   = 0.0
    total_tokens = 0
    steps        = 0

    for batch in val_loader:
        if steps >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        logits = model(input_ids)

        B, N, V = logits.shape
        loss    = nn.functional.cross_entropy(
            logits.reshape(B * N, V),
            labels.reshape(B * N),
            ignore_index = -100,
            reduction    = "sum",
        )

        n_tokens      = (labels != -100).sum().item()
        total_loss   += loss.item()
        total_tokens += n_tokens
        steps        += 1

    mean_loss  = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20))   

    model.train()
    return {
        "val_loss":        round(mean_loss, 6),
        "val_perplexity":  round(perplexity, 4),
        "steps_evaluated": steps,
    }

def save_checkpoint(
    model:      nn.Module,
    optimizer,
    scheduler,
    step:       int,
    val_ppl:    float,
    cfg:        TrainConfig,
    is_best:    bool = False,
):
    os.makedirs(cfg.output_dir, exist_ok=True)

    state = {
        "step":            step,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "val_perplexity":  val_ppl,
        "train_config":    asdict(cfg),
        "saved_at":        datetime.now().isoformat(),
    }

    path = f"{cfg.output_dir}/step_{step:06d}.pt"
    torch.save(state, path)
    print(f"  [Checkpoint] Saved: {path}")

    if is_best:
        best_path = f"{cfg.output_dir}/best.pt"
        torch.save(state, best_path)
        print(f"  [Checkpoint] New best: {best_path} (PPL={val_ppl:.4f})")

    return path


def load_checkpoint(path: str, model: nn.Module, optimizer, scheduler) -> int:
    print(f"[Resume] Loading checkpoint: {path}")
    state     = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    step      = state["step"]
    val_ppl   = state.get("val_perplexity", float("inf"))
    print(f"[Resume] Resuming from step {step:,} (val PPL={val_ppl:.4f})")
    return step

class MetricsLogger:
    
    def __init__(self, cfg: TrainConfig):
        self.cfg      = cfg
        self.log_path = f"{cfg.output_dir}/metrics.jsonl"
        self.use_wandb = False
        os.makedirs(cfg.output_dir, exist_ok=True)

        if cfg.use_wandb:
            try:
                wandb.init(
                    project = cfg.wandb_project,
                    name    = cfg.run_name,
                    config  = asdict(cfg),
                )
                self.use_wandb = True
                print("[WandB] Logging enabled")
            except ImportError:
                print("[WandB] Not installed, falling back to file logging")

    def log(self, metrics: dict):
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        if self.use_wandb:
            wandb.log(metrics)

    def close(self):
        if self.use_wandb:
            wandb.finish()

def train(cfg: TrainConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*55}")
    print(f"Hindi GPT-2 Training")
    print(f"{'='*55}")
    print(f"Device:           {device}")
    print(f"Max steps:        {cfg.max_steps:,}")
    print(f"Batch size:       {cfg.batch_size} × {cfg.grad_accum_steps} = {cfg.effective_batch_size}")
    print(f"Seq length:       {cfg.seq_len}")
    print(f"Learning rate:    {cfg.learning_rate}")
    print(f"Warmup steps:     {cfg.warmup_steps:,}")
    print(f"{'='*55}\n")

    print("[Setup] Loading tokenizer...")
    assert Path(cfg.tokenizer_path).exists(), (
        f"Tokenizer not found: {cfg.tokenizer_path}\n"
        f"Run tokenizer/train_tokenizer.py first."
    )
    tokenizer = Tokenizer.from_file(cfg.tokenizer_path)
    tokenizer.eos_token_id = tokenizer.token_to_id("<eos>")
    tokenizer.pad_token_id = tokenizer.token_to_id("<pad>")
    vocab_size = tokenizer.get_vocab_size()
    print(f"  Vocab size: {vocab_size:,}")

    print("\n[Setup] Building dataloaders...")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = cfg.data_dir,
        tokenizer   = tokenizer,
        seq_len     = cfg.seq_len,
        stride      = cfg.stride,
        batch_size  = cfg.batch_size,
        num_workers = 2,
    )

    print("\n[Setup] Initializing model...")
    model_cfg = ModelConfig_(vocab_size=vocab_size)
    model     = GPT2_(model_cfg).to(device)
    n_params  = model.count_parameters()
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    decay_params     = []
    no_decay_params  = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr    = cfg.learning_rate,
        betas = (cfg.beta1, cfg.beta2),
        eps   = 1e-8,
    )

    scheduler = get_lr_scheduler(optimizer, cfg)
    logger    = MetricsLogger(cfg)

    start_step = 0
    best_ppl   = float("inf")

    if cfg.resume_from and Path(cfg.resume_from).exists():
        start_step = load_checkpoint(
            cfg.resume_from, model, optimizer, scheduler
        )

    print(f"\n[Training] Starting from step {start_step:,}...")
    print(f"[Training] Target: {cfg.max_steps:,} steps\n")

    model.train()
    step            = start_step
    accum_loss      = 0.0
    accum_tokens    = 0
    t_start         = time.time()
    t_step          = time.time()

    def infinite_loader(loader):
        while True:
            for batch in loader:
                yield batch

    data_iter = infinite_loader(train_loader)

    optimizer.zero_grad()

    while step < cfg.max_steps:
        for micro_step in range(cfg.grad_accum_steps):
            batch     = next(data_iter)
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            logits    = model(input_ids)
            B, N, V   = logits.shape

            loss = nn.functional.cross_entropy(
                logits.reshape(B * N, V),
                labels.reshape(B * N),
                ignore_index = -100,
            )

            loss = loss / cfg.grad_accum_steps
            loss.backward()

            accum_loss   += loss.item()
            accum_tokens += (labels != -100).sum().item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        current_lr = scheduler.get_last_lr()[0]

        if step % cfg.log_every == 0:
            elapsed      = time.time() - t_start
            step_time    = (time.time() - t_step) / cfg.log_every
            steps_left   = cfg.max_steps - step
            eta_seconds  = steps_left * step_time
            eta          = str(timedelta(seconds=int(eta_seconds)))
            train_ppl    = math.exp(min(accum_loss * cfg.grad_accum_steps, 20))
            t_step       = time.time()

            print(
                f"Step {step:>6,}/{cfg.max_steps:,} | "
                f"Loss: {accum_loss * cfg.grad_accum_steps:.4f} | "
                f"PPL: {train_ppl:.2f} | "
                f"LR: {current_lr:.2e} | "
                f"Grad: {grad_norm:.3f} | "
                f"ETA: {eta}"
            )

            logger.log({
                "step":       step,
                "train_loss": round(accum_loss * cfg.grad_accum_steps, 6),
                "train_ppl":  round(train_ppl, 4),
                "lr":         current_lr,
                "grad_norm":  round(grad_norm.item(), 4),
                "elapsed":    round(elapsed, 1),
            })

            accum_loss   = 0.0
            accum_tokens = 0

        if step % cfg.eval_every == 0:
            print(f"\n[Eval] Step {step:,}...")
            eval_metrics = evaluate(
                model, val_loader, device, cfg.eval_steps
            )
            val_ppl = eval_metrics["val_perplexity"]

            print(
                f"  Val Loss:        {eval_metrics['val_loss']:.4f}\n"
                f"  Val Perplexity:  {val_ppl:.4f}\n"
            )

            logger.log({"step": step, **eval_metrics})

            is_best = val_ppl < best_ppl
            if is_best:
                best_ppl = val_ppl

        if step % cfg.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler,
                step, best_ppl, cfg,
                is_best = (step % cfg.eval_every == 0 and best_ppl == val_ppl
                           if step % cfg.eval_every == 0 else False),
            )
    print(f"\n[Training] Complete. Running final evaluation...")
    final_metrics = evaluate(model, val_loader, device, max_steps=200)
    print(f"  Final Val PPL: {final_metrics['val_perplexity']:.4f}")
    print(f"  Best Val PPL:  {best_ppl:.4f}")

    save_checkpoint(
        model, optimizer, scheduler,
        step, final_metrics["val_perplexity"], cfg,
        is_best = final_metrics["val_perplexity"] < best_ppl,
    )

    summary = {
        "completed_at":    datetime.now().isoformat(),
        "total_steps":     step,
        "best_val_ppl":    best_ppl,
        "final_val_ppl":   final_metrics["val_perplexity"],
        "total_time_hrs":  round((time.time() - t_start) / 3600, 2),
        "model_params":    n_params,
        "train_config":    asdict(cfg),
    }

    with open(f"{cfg.output_dir}/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print(f"Training Complete")
    print(f"Best Val PPL:  {best_ppl:.4f}")
    print(f"Time:          {summary['total_time_hrs']:.2f} hours")
    print(f"Checkpoints:   {cfg.output_dir}/")
    print(f"{'='*55}")
    print(f"\nNext step: python evaluation/perplexity.py")

    logger.close()
    return best_ppl

def load_config(yaml_path: str) -> TrainConfig:
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type    = str,
        default = None,
        help    = "Path to YAML config file. Uses defaults if not provided.",
    )
    parser.add_argument(
        "--resume",
        type    = str,
        default = None,
        help    = "Path to checkpoint to resume from.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else TrainConfig()
    if args.resume:
        cfg.resume_from = args.resume
    train(cfg)