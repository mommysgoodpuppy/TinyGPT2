#!/usr/bin/env python3
import os

os.environ["GPU"] = "1"
import argparse
import numpy as np
import tiktoken
from tinygrad import Tensor, Device, Variable, GlobalCounters
from tinygrad.helpers import Timing, DEBUG, getenv
from model import GPT, GPTConfig, MAX_CONTEXT

if __name__ == "__main__":
    Tensor.no_grad = True
    print(f"using {Device.DEFAULT} backend")
    default_prompt = "What is the answer to life, the universe, and everything?"

    parser = argparse.ArgumentParser(description="Run GPT2 inference in tinygrad")
    parser.add_argument(
        "--prompt", type=str, default=default_prompt, help="Phrase to start with"
    )
    parser.add_argument(
        "--count", type=int, default=100, help="Max number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Temperature in the softmax"
    )
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--seed", type=int, help="Set the random seed")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Set the input batch size"
    )
    parser.add_argument(
        "--use_pretrained",
        action="store_true",
        help="Use original pretrained weights instead of finetuned",
    )
    parser.add_argument("--half", action="store_true", help="Use float16")
    args = parser.parse_args()

    if args.seed is not None:
        Tensor.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Set up tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Initialize model in inference mode
    config = GPTConfig(inference_mode=True)
    if args.half:
        Tensor.default_type = np.float16

    # Load model
    if not args.use_pretrained and os.path.exists("checkpoints/gpt2_final.safetensors"):
        print("Loading finetuned model")
        model = GPT.load_model(config, "checkpoints/gpt2_final.safetensors")
    else:
        print("Loading pretrained model")
        model = GPT(config)
        model.load_pretrained("gpt2")

    def generate_text(
    prompt: str,
    max_tokens: int,
    temperature: float,
    timing: bool = False,
    batch_size: int = 1,
    ):
        # Encode prompt
        prompt_tokens = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        x = Tensor([prompt_tokens for _ in range(batch_size)])

        # Generate
        with Timing("generation took", enabled=timing):
            # Generate tokens
            total_tokens = []
            start_pos = 0
            for _ in range(max_tokens):
                if timing:
                    GlobalCounters.reset()
                    st = GlobalCounters.time_sum_s

                # Get next token
                if batch_size == 1 and len(prompt_tokens[start_pos:]) == 1:
                    # Replace Variable with Tensor
                    tokens = Tensor([[prompt_tokens[start_pos]]])
                else:
                    tokens = x

                # Forward pass
                with Timing(
                    "ran model in ",
                    on_exit=(
                        (
                            lambda et: (
                                f", {(GlobalCounters.time_sum_s-st)*1e3:.2f} ms on GPU"
                                if DEBUG >= 2
                                else ""
                            )
                            + f", {GlobalCounters.global_ops*1e-9:.2f} GOPS, {GlobalCounters.global_mem*1e-9:.2f} GB"
                            + (
                                f", {GlobalCounters.global_mem*1e-9/(GlobalCounters.time_sum_s-st):.2f} GB/s"
                                if DEBUG >= 2
                                else ""
                            )
                        )
                        if DEBUG
                        else None
                    ),
                    enabled=timing,
                ):
                    next_token = model.forward(
                        tokens,
                        start_pos=Variable(
                            "start_pos", 1 if start_pos else 0, MAX_CONTEXT
                        ).bind(start_pos),
                        temperature=temperature,
                    )

                # Update generated sequence
                x = Tensor.cat(x, next_token.reshape(batch_size, 1), dim=1)
                start_pos = len(prompt_tokens)
                prompt_tokens.append(next_token.numpy().item())

        # Decode results
        return [tokenizer.decode(x[i].numpy().tolist()) for i in range(batch_size)]

    # Generate text
    texts = generate_text(
        args.prompt,
        args.count,
        args.temperature,
        timing=args.timing,
        batch_size=args.batch_size,
    )

    # Print results
    print("\nGenerated text:")
    if len(texts) == 1:
        print(texts[0])
    else:
        for i, text in enumerate(texts):
            print(f"\nResponse {i}:")
            print(text)
