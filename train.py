#!/usr/bin/env python3
import os, time
import numpy as np
from tinygrad import Tensor, nn, fetch, Device, TinyJit, GlobalCounters
from tinygrad.helpers import getenv
from model import GPT, GPTConfig
import tiktoken
import argparse
from typing import List
from dataclasses import dataclass

os.environ["GPU"] = "1"


# Optimizer Implementation
class Optimizer:
    def __init__(self, params: List[Tensor], lr: float):
        for x in params:
            if x.requires_grad is None:
                x.requires_grad = True
        self.params = [x for x in params if x.requires_grad]
        assert len(self.params) != 0, "optimizer must have at least one param"
        self.device = self.params[0].device
        self.buffers = [x for x in params if not x.requires_grad]
        self.lr = Tensor([lr], requires_grad=False, device=self.device)

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def _step(self) -> List[Tensor]:
        raise NotImplementedError


def AdamW(
    params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-8, weight_decay=0.01
):
    return LAMB(params, lr, b1, b2, eps, weight_decay, adam=True)


class LAMB(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr=0.001,
        b1=0.9,
        b2=0.999,
        eps=1e-6,
        weight_decay=0.0,
        adam=False,
    ):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps, self.wd, self.adam = b1, b2, eps, weight_decay, adam
        self.b1_t, self.b2_t = (
            Tensor.ones((1,), requires_grad=False, device=self.device).contiguous()
            for _ in [b1, b2]
        )
        self.m = [
            Tensor.zeros(*t.shape, requires_grad=False, device=t.device).contiguous()
            for t in self.params
        ]
        self.v = [
            Tensor.zeros(*t.shape, requires_grad=False, device=t.device).contiguous()
            for t in self.params
        ]

    def _step(self) -> List[Tensor]:
        self.b1_t *= self.b1
        self.b2_t *= self.b2
        for i, t in enumerate(self.params):
            assert t.grad is not None
            self.m[i].assign(self.b1 * self.m[i] + (1.0 - self.b1) * t.grad)
            self.v[i].assign(self.b2 * self.v[i] + (1.0 - self.b2) * (t.grad * t.grad))
            m_hat = self.m[i] / (1.0 - self.b1_t)
            v_hat = self.v[i] / (1.0 - self.b2_t)
            up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
            if not self.adam:
                r1 = t.detach().square().sum().sqrt()
                r2 = up.square().sum().sqrt()
                r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
            else:
                r = 1.0
            t.assign((t.detach() - self.lr * r * up))
        return [self.b1_t, self.b2_t] + self.m + self.v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_iterations", type=int, default=10, help="number of iterations to run"
    )
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--sequence_length", type=int, default=64, help="sequence length"
    )
    parser.add_argument("--seed", type=int, help="random seed")
    args = parser.parse_args()
    B, T = args.batch_size, args.sequence_length
    assert 1 <= T <= 1024

    if args.seed is not None:
        Tensor.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize model
    config = GPTConfig(inference_mode=False)  # Training mode
    model = GPT(config)
    model.load_pretrained("gpt2")  # Load pretrained weights

    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")

    # Load dataset
    tokens_bin = fetch(
        "https://huggingface.co/datasets/karpathy/llmc-starter-pack/resolve/main/tiny_shakespeare_val.bin"
    )
    assert os.path.isfile(tokens_bin)
    print(f"loading cached tokens in {tokens_bin}")
    with open(tokens_bin, "rb") as f:
        f.seek(0x400)
        tokens = np.frombuffer(f.read(), dtype=np.uint16).astype(np.int32)
    tokens = Tensor(tokens)

    # Data loader
    def get_batch():
        assert B * T + 1 <= len(tokens), "not enough tokens"
        i = 0
        while True:
            x = tokens[i : i + B * T].view(B, T)
            y = tokens[i + 1 : i + B * T + 1].view(B, T)
            yield x, y
            i += B * T
            if i + B * T + 1 >= len(tokens):
                i = 0

    # Create checkpoints directory
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    # Training loop setup
    data_iter = iter(get_batch())
    x, y = next(data_iter)
    optimizer = AdamW(nn.state.get_parameters(model), lr=1e-4, weight_decay=0)

    @TinyJit
    def step(x, y):
        _, loss = model.forward(x, y)
        optimizer.zero_grad()
        loss.backward()
        return loss.realize()

    # Training loop
    with Tensor.train():
        for i in range(args.num_iterations):
            GlobalCounters.reset()
            t0 = time.time()
            loss = step(x.contiguous(), y.contiguous())
            Device[Device.DEFAULT].synchronize()
            t1 = time.time()
            print(
                f"iteration {i}, loss: {loss.item():.6f}, time: {(t1-t0)*1000:.3f}ms, {int(B*T/(t1-t0))} tok/s"
            )

            # Save checkpoint every 10 iterations
            if i > 0 and i % 10 == 0:
                checkpoint_path = f"checkpoints/gpt2_iter_{i}.safetensors"
                model.save_model(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

        # Save final model
        model.save_model("checkpoints/gpt2_final.safetensors")
        print("Saved final model")
