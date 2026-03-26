"""Generate a sharded activation dataset for canonicalization experiment v2.

Each shard stores only endpoint hidden states for many individual masked-LM
examples under a fixed seed set. The training script assembles minibatches on
the fly and handles any full-model perplexity evaluation separately.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

from quineformer.bias_absorption import load_multibert_model
from quineformer.experiment_utils import (
    load_serialized_models,
)
from quineformer.serialization import deserialize


REPO_ROOT = Path(__file__).resolve().parents[1]
SERIALIZED_CACHE = REPO_ROOT / "data" / "multiberts" / "serialized"
DEFAULT_OUTPUT_DIR = Path("/data/bill/datasets/quineformer/canonicalization_v2_activations")

TRAIN_SEEDS = list(range(20))
TEST_SEEDS = list(range(20, 25))

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
}


class StreamingTextExampleSource:
    """Generate masked MLM examples from a corpus without materializing all tokens."""

    def __init__(
        self,
        tokenizer: BertTokenizer,
        split: str,
        max_length: int,
        seed: int,
    ):
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError(
                "generate_canonicalization_v2_activation_dataset.py requires the `datasets` package."
            ) from error

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inner_len = max_length - 2
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        self.generator = torch.Generator().manual_seed(seed)
        self.row_order: list[int] = []
        self.row_ptr = 0
        self.buffer: list[int] = []
        self._reshuffle_rows()

    def _reshuffle_rows(self) -> None:
        self.row_order = torch.randperm(len(self.dataset), generator=self.generator).tolist()
        self.row_ptr = 0

    def _next_sequence(self) -> torch.Tensor:
        while len(self.buffer) < self.inner_len:
            if self.row_ptr >= len(self.row_order):
                self._reshuffle_rows()
            text = self.dataset[self.row_order[self.row_ptr]]["text"].strip()
            self.row_ptr += 1
            if not text:
                continue
            self.buffer.extend(self.tokenizer.encode(text, add_special_tokens=False))

        token_ids = [
            self.tokenizer.cls_token_id,
            *self.buffer[: self.inner_len],
            self.tokenizer.sep_token_id,
        ]
        self.buffer = self.buffer[self.inner_len :]
        return torch.tensor(token_ids, dtype=torch.long)

    def sample_example(self) -> dict[str, torch.Tensor]:
        input_ids = self._next_sequence()
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()

        mask_draw = torch.rand(input_ids.shape, generator=self.generator)
        mask_draw[0] = 1.0
        mask_draw[-1] = 1.0
        masked_positions = mask_draw < 0.15

        masked_input = input_ids.clone()
        masked_input[masked_positions] = self.tokenizer.mask_token_id
        labels[~masked_positions] = -100
        return {
            "input_ids": masked_input,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--validation-examples", type=int, default=512)
    parser.add_argument("--examples-per-shard", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dtype", choices=sorted(DTYPE_MAP), default="float16")
    parser.add_argument("--train-corpus-split", default="train")
    parser.add_argument("--validation-corpus-split", default="validation")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def sample_example(
    source: StreamingTextExampleSource,
) -> dict[str, torch.Tensor]:
    return source.sample_example()


def compute_hidden_state_stack(
    shell_model: BertForMaskedLM,
    endpoint_params: dict[int, dict[str, torch.Tensor]],
    seeds: list[int],
    examples: list[dict[str, torch.Tensor]],
    device: torch.device,
    hidden_dtype: torch.dtype,
) -> torch.Tensor:
    """Return a tensor shaped [seed, example, layer, token, hidden]."""
    hidden_by_seed = []

    for seed in tqdm(seeds, desc="  seeds", unit="seed", leave=False):
        params = {
            name: value.to(device)
            for name, value in endpoint_params[seed].items()
        }
        example_hidden_states = []
        for example in examples:
            with torch.no_grad():
                output = torch.func.functional_call(
                    shell_model.bert,
                    params,
                    kwargs={
                        "input_ids": example["input_ids"].unsqueeze(0).to(device),
                        "attention_mask": example["attention_mask"].unsqueeze(0).to(device),
                        "output_hidden_states": True,
                    },
                )
            example_hidden_states.append(
                torch.stack(
                    [
                        hidden.detach().cpu().to(hidden_dtype).squeeze(0)
                        for hidden in output.hidden_states
                    ],
                    dim=0,
                )
            )
        hidden_by_seed.append(torch.stack(example_hidden_states, dim=0))

    return torch.stack(hidden_by_seed, dim=0)


def write_split(
    output_dir: Path,
    split_name: str,
    corpus_split: str,
    seeds: list[int],
    num_examples: int,
    examples_per_shard: int,
    tokenizer: BertTokenizer,
    shell_model: BertForMaskedLM,
    endpoint_params: dict[int, dict[str, torch.Tensor]],
    max_length: int,
    seed_offset: int,
    hidden_dtype: torch.dtype,
    device: torch.device,
) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for stale_shard in split_dir.glob("shard_*.pt"):
        stale_shard.unlink()

    print(f"Streaming WikiText examples for split '{corpus_split}'...")
    example_source = StreamingTextExampleSource(
        tokenizer=tokenizer,
        split=corpus_split,
        max_length=max_length,
        seed=seed_offset,
    )

    num_shards = math.ceil(num_examples / examples_per_shard)
    shard_bar = tqdm(range(num_shards), desc=f"{split_name}: bottling activations", unit="shard")

    for shard_idx in shard_bar:
        example_start = shard_idx * examples_per_shard
        example_stop = min(example_start + examples_per_shard, num_examples)
        shard_examples = [
            sample_example(
                example_source,
            )
            for example_idx in range(example_start, example_stop)
        ]
        hidden_states = compute_hidden_state_stack(
            shell_model,
            endpoint_params,
            seeds,
            shard_examples,
            device,
            hidden_dtype,
        )
        payload = {
            "hidden_states": hidden_states,
        }
        shard_path = split_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(payload, shard_path)
        shard_bar.set_postfix(examples=f"{example_stop - example_start}", path=shard_path.name)

        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        args.train_examples = 8
        args.validation_examples = 4
        args.examples_per_shard = 4
        args.max_length = 32

    output_dir = args.output_dir
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{metadata_path} already exists. Pass --overwrite to regenerate the dataset."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("google/multiberts-seed_0")

    all_seeds = sorted(set(TRAIN_SEEDS + TEST_SEEDS))
    print("Loading serialized checkpoints...")
    serialized, config = load_serialized_models(all_seeds, SERIALIZED_CACHE)
    print("Pre-deserializing endpoint BERT parameters...")
    endpoint_params = {
        seed: deserialize(serialized[seed], config)
        for seed in all_seeds
    }

    shell_model = load_multibert_model(TRAIN_SEEDS[0]).eval().to(device)
    hidden_dtype = DTYPE_MAP[args.hidden_dtype]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "version": 1,
        "hidden_dtype": args.hidden_dtype,
        "max_length": args.max_length,
        "examples_per_shard": args.examples_per_shard,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "splits": {
            "train": {
                "seed_order": TRAIN_SEEDS,
                "num_examples": args.train_examples,
                "corpus_split": args.train_corpus_split,
            },
            "validation": {
                "seed_order": TEST_SEEDS,
                "num_examples": args.validation_examples,
                "corpus_split": args.validation_corpus_split,
            },
        },
    }
    with open(metadata_path, "w", encoding="ascii") as handle:
        json.dump(metadata, handle, indent=2)

    write_split(
        output_dir=output_dir,
        split_name="train",
        corpus_split=args.train_corpus_split,
        seeds=TRAIN_SEEDS,
        num_examples=args.train_examples,
        examples_per_shard=args.examples_per_shard,
        tokenizer=tokenizer,
        shell_model=shell_model,
        endpoint_params=endpoint_params,
        max_length=args.max_length,
        seed_offset=args.seed,
        hidden_dtype=hidden_dtype,
        device=device,
    )
    write_split(
        output_dir=output_dir,
        split_name="validation",
        corpus_split=args.validation_corpus_split,
        seeds=TEST_SEEDS,
        num_examples=args.validation_examples,
        examples_per_shard=args.examples_per_shard,
        tokenizer=tokenizer,
        shell_model=shell_model,
        endpoint_params=endpoint_params,
        max_length=args.max_length,
        seed_offset=args.seed + args.train_examples + 10_000,
        hidden_dtype=hidden_dtype,
        device=device,
    )


if __name__ == "__main__":
    main()