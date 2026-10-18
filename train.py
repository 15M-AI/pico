#!/usr/bin/env python3
"""Train the Pico 15M model and write OMF checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_mind import Trainer


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    training = config["training"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="UTF-8 plain-text training corpus")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "pico-0.1.0.omf")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=training["device"])
    parser.add_argument("--steps", type=int, default=training["steps"])
    parser.add_argument("--checkpoint-every", type=int, default=training["checkpoint_every"])
    parser.add_argument("--batch-size", type=int, default=training["batch_size"])
    parser.add_argument("--seq-len", type=int, default=training["seq_len"])
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=training["gradient_accumulation_steps"],
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", choices=("adamw", "sgd8bit"))
    parser.add_argument("--seed", type=int, default=training["seed"])
    parser.add_argument("--eval-interval", type=int, default=training["eval_interval"])
    parser.add_argument("--eval-batches", type=int, default=training["eval_batches"])
    parser.add_argument("--warmup-steps", type=int, default=training["warmup_steps"])
    parser.add_argument(
        "--min-learning-rate-ratio",
        type=float,
        default=training["min_learning_rate_ratio"],
    )
    parser.add_argument("--weight-decay", type=float, default=training["weight_decay"])
    parser.add_argument("--gradient-clip", type=float, default=training["gradient_clip"])
    parser.add_argument("--resume-from", type=Path, help="resume model weights from an OMF checkpoint")
    args = parser.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            parser.error("CUDA was requested, but this environment has no available CUDA device")

    optimizer = args.optimizer or (
        training["cpu_optimizer"] if args.device == "cpu" else training["optimizer"]
    )
    learning_rate = args.learning_rate if args.learning_rate is not None else (
        training["cpu_learning_rate"]
        if args.device == "cpu"
        else training["learning_rate"]
    )

    Trainer(
        model="Pico",
        device=args.device,
        dataset=args.dataset,
        output=args.output,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=learning_rate,
        dtype=training["dtype"],
        seed=args.seed,
        optimizer=optimizer,
        weight_decay=args.weight_decay,
        validation_fraction=training["validation_fraction"],
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        warmup_steps=args.warmup_steps,
        min_learning_rate_ratio=args.min_learning_rate_ratio,
        gradient_clip=args.gradient_clip,
        resume_from=args.resume_from,
    ).train()


if __name__ == "__main__":
    main()
