"""
Downloads and evaluates HellaSwag in Python.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

import tiktoken
import jax.numpy as jnp
import optax

# from transformers import FlaxGPT2LMHeadModel
from jax_gpt2 import GPT
from datasets import load_dataset


dataset = load_dataset("Rowan/hellaswag", split="validation")
enc = tiktoken.get_encoding("gpt2")


def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    data = {
        "label": label,
        "ctx_tokens": None,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []

    # print(f"Context tokens: {ctx}")
    # print(f"Endings: {endings}")

    for end in endings:
        end_tokens = enc.encode(
            " " + end
        )  # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data["ending_tokens"].append(end_tokens)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) for row in tok_rows)
    tokens = jnp.zeros((4, max_len), dtype=jnp.int32)
    mask = jnp.zeros((4, max_len))
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens = tokens.at[i, : len(tok_row)].set(jnp.array(tok_row))
        mask = mask.at[i, : len(mask_row)].set(jnp.array(mask_row))

    return data, tokens, mask, label


def evaluate(model_type: str, number_of_samples: int):
    # model = FlaxGPT2LMHeadModel.from_pretrained('gpt2')
    model = GPT.from_pretrained(model_type)
    num_correct_norm = 0
    num_correct = 0
    num_total = 0
    for example in dataset:
        data, tokens, mask, label = render_example(example)
        # get the logits
        # logits = model(tokens).logits
        logits = model(tokens)
        # evaluate the autoregressive loss at all positions
        shift_logits = logits[..., :-1, :]
        shift_tokens = tokens[..., 1:]
        flat_shift_logits = shift_logits.reshape([-1, shift_logits.shape[-1]])
        flat_shift_tokens = shift_tokens.reshape([-1])
        shift_losses = optax.softmax_cross_entropy_with_integer_labels(
            flat_shift_logits, flat_shift_tokens
        )
        shift_losses = shift_losses.reshape(tokens.shape[0], -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = mask[
            ..., 1:
        ]  # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        sum_loss = masked_shift_losses.sum(axis=1)
        avg_loss = sum_loss / shift_mask.sum(axis=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        num_total += 1
        num_correct += int(pred == int(label))
        num_correct_norm += int(pred_norm == int(label))
        print(
            f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}"
        )

        # debug: pretty print a few examples, and the losses in each case
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

        if num_total > number_of_samples:
            break


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model_type", type=str, default="gpt2", help="The model type to use"
    )
    parser.add_argument(
        "-n",
        "--number_of_samples",
        type=int,
        default=50,
        help="Number of samples to evaluate",
    )
    args = parser.parse_args()
    evaluate(args.model_type, args.number_of_samples)
