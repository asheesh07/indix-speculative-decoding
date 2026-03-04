from dataclasses import dataclass
import torch

@dataclass
class ModelConfig_:
    vocab_size:int = 8000
    embed_size:int = 384
    num_layers:int = 6
    heads:int = 6
    forward_expansion: int = 4
    max_length:int = 768
    dropout:float = 0.1
    
    @property
    def ffn_dim(self) -> int:
        return self.forward_expansion * self.embed_size

if __name__ == "__main__":
    from gpt2 import GPT2
    cfg   = ModelConfig_()
    model = GPT2(cfg)
    params = model.count_parameters()
    print(f"Parameters: {params:,}")
    print(f"Approx size: {params * 4 / 1024**2:.1f}MB (float32)")
    x = torch.randint(0, cfg.vocab_size, (2, 64))
    logits = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (2, 64, cfg.vocab_size)
    print("✓ Forward pass correct")