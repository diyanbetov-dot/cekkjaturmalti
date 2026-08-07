from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from neural_corrector.dataset.analyze_pairs import read_jsonl
from neural_corrector.models.alignment import COPY_ACTION, chunk_aligned, derive_actions
from neural_corrector.models.char_edit_tagger import CharEditTagger
from neural_corrector.models.vocab import (
    PAD_ACTION,
    PAD_CHAR,
    UNK_CHAR,
    Vocabularies,
    build_vocabularies,
)

IGNORE_INDEX = -100


@dataclass(frozen=True)
class SequenceExample:
    source: str
    actions: list[str]
    source_id: str


class EditDataset(Dataset):
    def __init__(
        self,
        examples: list[SequenceExample],
        vocabularies: Vocabularies,
        training: bool,
    ) -> None:
        self.examples = examples
        self.vocabularies = vocabularies
        self.training = training

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        example = self.examples[index]
        chars = [
            self.vocabularies.characters.get(char, self.vocabularies.characters[UNK_CHAR])
            for char in example.source
        ]
        actions = [
            self.vocabularies.actions.get(
                action,
                self.vocabularies.actions[COPY_ACTION] if self.training else IGNORE_INDEX,
            )
            for action in example.actions
        ]
        return chars, actions


def collate(batch: list[tuple[list[int], list[int]]]) -> tuple[torch.Tensor, ...]:
    max_length = max(len(chars) for chars, _ in batch)
    inputs = torch.zeros((len(batch), max_length), dtype=torch.long)
    targets = torch.full(
        (len(batch), max_length), IGNORE_INDEX, dtype=torch.long
    )
    lengths = torch.tensor([len(chars) for chars, _ in batch], dtype=torch.long)
    for row, (chars, actions) in enumerate(batch):
        inputs[row, : len(chars)] = torch.tensor(chars)
        targets[row, : len(actions)] = torch.tensor(actions)
    return inputs, targets, lengths


def make_examples(
    rows: list[dict], allowed_ids: set[str] | None, max_length: int
) -> list[SequenceExample]:
    result: list[SequenceExample] = []
    for row in rows:
        if allowed_ids is not None and row["id"] not in allowed_ids:
            continue
        source = row["noisy"]
        target = row["clean"]
        if "*" in target or re.search(r"\b\w+/\w+\b", target):
            continue
        if not source:
            continue
        actions = derive_actions(source, target)
        for chunk_index, (source_chunk, action_chunk) in enumerate(
            chunk_aligned(source, actions, max_length)
        ):
            result.append(
                SequenceExample(
                    source_chunk, action_chunk, f"{row['id']}:{chunk_index}"
                )
            )
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def class_weights(examples: list[SequenceExample], vocab: Vocabularies) -> torch.Tensor:
    counts = collections.Counter(action for row in examples for action in row.actions)
    weights = torch.ones(len(vocab.actions), dtype=torch.float32)
    total = sum(counts.values())
    for action, index in vocab.actions.items():
        if action == PAD_ACTION:
            weights[index] = 0.0
        elif counts[action]:
            weights[index] = min(8.0, (total / counts[action]) ** 0.35)
    weights[vocab.actions[COPY_ACTION]] = 0.35
    return weights


def run_epoch(
    model: CharEditTagger,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    observed = 0
    for inputs, targets, lengths in loader:
        inputs, targets, lengths = (
            inputs.to(device),
            targets.to(device),
            lengths.to(device),
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, lengths)
        loss = loss_function(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if training:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        mask = targets != IGNORE_INDEX
        predictions = logits.argmax(dim=-1)
        correct += int(((predictions == targets) & mask).sum().item())
        observed += int(mask.sum().item())
        total_loss += float(loss.item())
    return {
        "loss": total_loss / max(1, len(loader)),
        "action_accuracy": correct / max(1, observed),
    }


def train(
    config_path: Path,
    pairs_path: Path,
    split_path: Path,
    synthetic_path: Path,
    artifact_dir: Path,
) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    set_seed(config["seed"])
    rows = read_jsonl(pairs_path)
    synthetic = read_jsonl(synthetic_path) if synthetic_path.exists() else []
    splits = json.loads(split_path.read_text(encoding="utf-8"))["splits"]
    train_examples = make_examples(
        rows, set(splits["train"]), config["max_sequence_length"]
    )
    train_examples.extend(
        make_examples(synthetic, None, config["max_sequence_length"])
    )
    validation_examples = make_examples(
        rows, set(splits["validation"]), config["max_sequence_length"]
    )
    vocab = build_vocabularies(
        [example.source for example in train_examples],
        [example.actions for example in train_examples],
    )
    train_dataset = EditDataset(train_examples, vocab, training=True)
    validation_dataset = EditDataset(validation_examples, vocab, training=False)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharEditTagger(
        len(vocab.characters),
        len(vocab.actions),
        config["embedding_dim"],
        config["hidden_dim"],
        config["layers"],
        config["dropout"],
    ).to(device)
    loss_function = nn.CrossEntropyLoss(
        weight=class_weights(train_examples, vocab).to(device),
        ignore_index=IGNORE_INDEX,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    vocab.save(artifact_dir / "vocab.json")
    history: list[dict] = []
    best_loss = float("inf")
    epochs_without_improvement = 0
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        train_metrics = run_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                model, validation_loader, loss_function, None, device
            )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record))
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": config,
                    "model_version": "char-edit-bigru-0.1.0",
                    "best_validation_loss": best_loss,
                },
                artifact_dir / "model.pt",
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.get(
            "early_stopping_patience", config["epochs"]
        ):
            print(
                json.dumps(
                    {
                        "early_stopping": True,
                        "epoch": epoch,
                        "best_validation_loss": best_loss,
                    }
                )
            )
            break
    elapsed = time.perf_counter() - started
    model_path = artifact_dir / "model.pt"
    report = {
        "model_version": "char-edit-bigru-0.1.0",
        "architecture": "bidirectional GRU character edit tagger",
        "custom_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "training_seconds": round(elapsed, 3),
        "training_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "character_vocab": len(vocab.characters),
        "action_vocab": len(vocab.actions),
        "best_validation_loss": best_loss,
        "model_bytes": model_path.stat().st_size,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "config": config,
        "history": history,
    }
    (artifact_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("neural_corrector/configs/baseline.json"),
    )
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("neural_corrector/data/processed/all_pairs.jsonl"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("neural_corrector/data/splits/LOCKED_SPLITS.json"),
    )
    parser.add_argument(
        "--synthetic",
        type=Path,
        default=Path("neural_corrector/data/processed/synthetic_train.jsonl"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("neural_corrector/artifacts/char_edit_bigru_v1"),
    )
    args = parser.parse_args()
    report = train(
        args.config, args.pairs, args.splits, args.synthetic, args.artifact_dir
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
