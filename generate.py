"""
Generate responses to the supplied prompts
"""

import tiktoken
import jax.numpy as jnp
import flax.nnx as nn
import jax
from typing import List
from jax_gpt2 import GPT

enc = tiktoken.get_encoding("gpt2")


def prepare_tokens(prompts: List[str], repeat: int = 0):
    if len(prompts) == 1:
        tokens = enc.encode(prompts[0])
        tokens = jnp.expand_dims(jnp.array(tokens), axis=0)
        if repeat > 0:
            tokens = jnp.repeat(tokens, repeat, axis=0)
        each_len = None
    else:
        prompts = [enc.encode(prompt) for prompt in prompts]
        each_len = [len(encoded_prompt) for encoded_prompt in prompts]
        max_len = max(len(encoded_prompt) for encoded_prompt in prompts)
        number_of_prompts = len(prompts)
        tokens = jnp.zeros((number_of_prompts, max_len), dtype=jnp.int32)
        for i, encoded_prompt in enumerate(prompts):
            tokens = tokens.at[i, : len(encoded_prompt)].set(
                jnp.array(encoded_prompt, dtype=jnp.int32)
            )
    # print(tokens.shape, tokens.dtype)
    return tokens, each_len


def generate(tokens: jax.Array, max_tokens: int):
    step_key = jax.random.key(0)

    while tokens.shape[1] < max_tokens:
        # forward the model to get the logits
        logits = model(tokens)  # (B, T, vocab_size)
        # take the logits at the last position
        logits = logits[:, -1, :]  # (B, vocab_size)
        # do top-k sampling of 50 (huggingface pipeline default)
        # topk_probs here becomes (5, 50), topk_indices is (5, 50)
        top_logits, top_tokens = jax.lax.top_k(logits, min(50, logits.shape[-1]))
        step_key, subkey = jax.random.split(step_key)
        token_idx = jax.random.categorical(subkey, top_logits, axis=-1)
        next_token = jnp.take_along_axis(
            top_tokens, token_idx[:, None], axis=-1
        ).squeeze(-1)
        tokens = jnp.concatenate((tokens, jnp.vstack(next_token)), axis=1)
        # print(f"Updated value of tokens.shape[1]: {tokens.shape[1]}")

    for i in range(tokens.shape[0]):
        x = tokens[i, :max_tokens].tolist()
        decoded = enc.decode(x)
        print(">", decoded)


def generate_multiple_prompts(
    tokens: jax.Array, max_tokens: int, each_prompt_lengths: List[int]
):
    step_key = jax.random.key(0)

    last_token = jnp.array(each_prompt_lengths) - 1

    while tokens.shape[1] < max_tokens:
        # forward the model to get the logits
        logits = model(tokens)  # (B, T, vocab_size)
        # take the logits at the prompt endings initially, then the recent response token
        logits = jnp.take_along_axis(
            logits, (last_token)[:, None, None], axis=1
        ).squeeze(
            1
        )  # (B, vocab_size)
        top_logits, top_tokens = jax.lax.top_k(logits, min(50, logits.shape[-1]))
        step_key, subkey = jax.random.split(step_key)
        token_idx = jax.random.categorical(subkey, top_logits, axis=-1)
        next_token = jnp.take_along_axis(
            top_tokens, token_idx[:, None], axis=-1
        ).squeeze(-1)

        # Keep stretching
        # Not setting dtype=jnp.int32 caused a problematic datatype coercion to float32
        tokens = jnp.concatenate(
            (tokens, jnp.zeros((tokens.shape[0]), dtype=jnp.int32)[:, None]), axis=1
        )
        last_token = last_token + 1  # Set generated token at last_token + 1
        # Pulling out last token indices from indentity
        # generates the mask positions to place generated tokens
        place_mask = jnp.eye(tokens.shape[1])[last_token]
        tokens = jnp.place(tokens, place_mask, next_token, inplace=False)

    for token, last in zip(tokens, last_token):
        x = token[:last].tolist()
        decoded = enc.decode(x)
        print(">", decoded)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", type=str, default="gpt2", help="The model type to use"
    )
    parser.add_argument(
        "-p", "--prompt", type=str, action="append", help="Prompt for the model"
    )
    parser.add_argument(
        "-r",
        "--repeat_prompt",
        type=int,
        default=0,
        help="Generate responses for n repetitions of same prompt",
    )
    parser.add_argument(
        "-n",
        "--max_tokens",
        type=int,
        default=30,
        help="Max. tokens: prompt + response",
    )
    args = parser.parse_args()

    model = GPT.from_pretrained(args.model_type)

    if len(args.prompt) == 1:
        prepared_tokens, _ = prepare_tokens(args.prompt, args.repeat_prompt)
        generate(prepared_tokens, args.max_tokens)
    else:
        prepared_tokens, each_prompt_lengths = prepare_tokens(args.prompt)
        generate_multiple_prompts(prepared_tokens, args.max_tokens, each_prompt_lengths)
