#!/usr/bin/env python3
import os, math
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np

from tinygrad import Tensor, Variable, nn
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, get_state_dict, safe_save, safe_load
from tinygrad.dtype import dtypes
from tinygrad.ops import UOp

# Configuration
MAX_CONTEXT = 128
VOCAB_SIZE = 50257


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = VOCAB_SIZE
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    inference_mode: bool = False


class CausalSelfAttention:
    def __init__(self, config: GPTConfig):
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.inference_mode = config.inference_mode

        # for both inference and training
        self.bias = Tensor.ones(1, 1, config.block_size, config.block_size).tril()
        self.bias.requires_grad = False

    def __call__(
        self,
        x: Tensor,
        start_pos: Optional[Variable] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        B, T, C = x.shape

        # Regular attention for training
        if not self.inference_mode:
            qkv = self.c_attn(x)
            q, k, v = qkv.split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = att.softmax()
            y = att @ v
            y = y.transpose(1, 2).view(B, T, C)
            return self.c_proj(y)

        # KV cache attention for inference
        else:
            if mask is not None or start_pos.val == 0:
                start_pos = start_pos.val

            xqkv = self.c_attn(x)
            xq, xk, xv = [
                xqkv.shrink(
                    (None, None, (i * self.n_embd, (i + 1) * self.n_embd))
                ).reshape(None, None, self.n_head, self.head_dim)
                for i in range(3)
            ]

            if not hasattr(self, "cache_kv"):
                self.cache_kv = (
                    Tensor.zeros(
                        2, B, MAX_CONTEXT, self.n_head, self.head_dim, dtype=x.dtype
                    )
                    .contiguous()
                    .realize()
                )

            self.cache_kv.shrink(
                (None, None, (start_pos, start_pos + T), None, None)
            ).assign(Tensor.stack(xk, xv)).realize()

            if start_pos > 0:
                keys = self.cache_kv[0].shrink((None, (0, start_pos + T), None, None))
                values = self.cache_kv[1].shrink((None, (0, start_pos + T), None, None))
            else:
                keys, values = xk, xv

            xq, keys, values = (
                xq.transpose(1, 2),
                keys.transpose(1, 2),
                values.transpose(1, 2),
            )
            return self.c_proj(
                xq.scaled_dot_product_attention(keys, values, mask)
                .transpose(1, 2)
                .reshape(B, T, self.n_embd)
            )


class MLP:
    def __init__(self, config: GPTConfig):
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class Block:
    def __init__(self, config: GPTConfig):
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(
        self,
        x: Tensor,
        start_pos: Optional[Variable] = None,
        mask: Optional[Tensor] = None,
    ):
        h = x + self.attn(self.ln_1(x), start_pos, mask)
        return h + self.mlp(self.ln_2(h))


class GPT:
    def __init__(self, config: GPTConfig):
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.h = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight  # weight tying

    def forward(
        self,
        idx: Tensor,
        targets: Optional[Tensor] = None,
        start_pos: Optional[Variable] = None,
        temperature: float = 0.0,
    ):
        b, t = idx.shape
        pos = (
            Tensor.arange(0, t)
            if not self.config.inference_mode
            else Tensor.arange(0, MAX_CONTEXT).reshape(1, -1)
        )

        if self.config.inference_mode:
            if isinstance(idx, UOp):
                tok_emb = self.wte.weight.shrink(((idx, idx + 1), None))
            else:
                tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos.shrink((None, (start_pos, start_pos + t))))
        else:
            tok_emb = self.wte(idx)
            pos_emb = self.wpe(pos)

        x = tok_emb + pos_emb

        if self.config.inference_mode:
            mask = (
                Tensor.full(
                    (1, 1, t, start_pos.val + t), float("-inf"), dtype=x.dtype
                ).triu(start_pos.val + 1)
                if t > 1
                else None
            )
            for layer in self.h:
                x = layer(x, start_pos, mask)
            logits = self.lm_head(self.ln_f(x))
            if logits.shape[1] == 0:
                logits = Tensor.ones(
                    (logits.shape[0], self.config.vocab_size),
                    dtype=logits.dtype,
                    device=logits.device,
                )
            else:
                logits = logits[:, -1, :]
            if temperature < 1e-6:
                ret = logits.argmax(-1)
            else:
                ret = (logits / temperature).softmax().multinomial()
            return ret.flatten().realize()
        else:
            for layer in self.h:
                x = layer(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)
            if targets is not None:
                loss = self.sparse_categorical_crossentropy(logits, targets)
                return logits, loss
            return logits, None

    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: Optional[int] = None,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.shape[1] <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            idx_next = self.forward(
                idx_cond,
                start_pos=Variable(
                    "start_pos", 1 if idx.shape[1] > 1 else 0, MAX_CONTEXT
                ).bind(idx.shape[1] - 1),
                temperature=temperature,
            )
            idx = Tensor.cat(idx, idx_next.reshape(1, 1), dim=1)
        return idx

    def sparse_categorical_crossentropy(
        self,
        logits: Tensor,
        Y: Tensor,
        ignore_index: int = -1,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ) -> Tensor:
        assert 0.0 <= label_smoothing <= 1.0
        assert reduction in ("mean", "sum", "none")

        log_probs, loss_mask = logits.log_softmax(), (
            (Y != ignore_index)
            if ignore_index != -1
            else Y.ones_like(dtype=dtypes.bool)
        )
        y_counter = (
            Tensor.arange(logits.shape[-1], requires_grad=False, device=logits.device)
            .unsqueeze(0)
            .expand(Y.numel(), logits.shape[-1])
        )
        y = (
            (y_counter == Y.flatten().reshape(-1, 1)) * loss_mask.reshape(-1, 1)
        ).reshape(*Y.shape, logits.shape[-1])

        smoothing = label_smoothing * (log_probs.mean(-1) * loss_mask)
        unreduced = (1 - label_smoothing) * (log_probs * y).sum(-1) + smoothing

        return -(
            unreduced.sum() / loss_mask.sum()
            if reduction == "mean"
            else (unreduced.sum() if reduction == "sum" else unreduced)
        )

    def save_model(self, filename: str):
        state_dict = get_state_dict(self)
        safe_save(state_dict, filename)

    @staticmethod
    def load_model(config: GPTConfig, filename: str):
        model = GPT(config)
        weights = safe_load(filename)
        load_state_dict(model, weights)
        return model

    def load_pretrained(self, model_size: str = "gpt2"):
        weights = nn.state.torch_load(
            fetch(f"https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin")
        )
        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )
        for k in weights:
            if k.endswith(transposed):
                weights[k] = weights[k].T
        weights["lm_head.weight"] = weights["wte.weight"]
        load_state_dict(self, weights)
