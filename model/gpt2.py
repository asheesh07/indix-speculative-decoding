import torch
import torch.nn as nn
import torch.nn.functional as F
from model.config import ModelConfig_

def causal_mask(seq_len: int, device: str = "cpu") -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0).bool()   

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_length: int):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding  = nn.Embedding(max_length, embed_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        return self.word_embedding(x) + self.pos_embedding(positions)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_size % heads == 0, "embed_size must be divisible by heads"

        self.heads    = heads
        self.head_dim = embed_size // heads

        self.qkv = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.out  = nn.Linear(embed_size, embed_size)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = x.shape
        H, HD   = self.heads, self.head_dim

        qkv = self.qkv(x).reshape(B, N, 3, H, HD).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   
        scale  = HD ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale   

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        scores = F.softmax(scores, dim=-1)
        scores = self.attn_dropout(scores)

        out = torch.matmul(scores, v)                            
        out = out.transpose(1, 2).reshape(B, N, D)               
        return self.out(out)

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size:int,
        heads:int,
        dropout:float,
        forward_expansion:int,
    ):
        super().__init__()
        self.norm1= nn.LayerNorm(embed_size)
        self.norm2= nn.LayerNorm(embed_size)
        self.attention = MultiHeadAttention(embed_size, heads, dropout)
        self.ff= nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.GELU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        x = x + self.ff(self.norm2(x))
        return x

class GPT2_(nn.Module):
    def __init__(self, cfg: ModelConfig_):
        super().__init__()
        self.embedding = Embedding(cfg.vocab_size, cfg.embed_size, cfg.max_length)
        self.blocks    = nn.ModuleList([
            TransformerBlock(
                cfg.embed_size,
                cfg.heads,
                cfg.dropout,
                cfg.forward_expansion,
            )
            for _ in range(cfg.num_layers)
        ])
        self.ln_f    = nn.LayerNorm(cfg.embed_size)
        self.lm_head = nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.word_embedding.weight
        self.max_length = cfg.max_length
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask:      torch.Tensor = None,
    ) -> torch.Tensor:
        B, N = input_ids.shape

        if mask is None:
            mask = causal_mask(N, device=input_ids.device)

        x = self.embedding(input_ids)
        for block in self.blocks:
            x = block(x, mask)
        x      = self.ln_f(x)
        logits = self.lm_head(x)    
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)