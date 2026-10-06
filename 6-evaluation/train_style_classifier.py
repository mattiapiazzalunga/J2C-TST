"""Train one binary XLM-RoBERTa-base style classifier for IT or EN."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


LABELS = {"journalistic": 0, "conversational": 1}
CHECKPOINT_MODEL_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def find_latest_resumable_checkpoint(output_dir: Path) -> Path | None:
    """Return the newest complete Trainer checkpoint, skipping partial saves."""

    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        if not child.is_dir() or not child.name.startswith("checkpoint-"):
            continue
        try:
            step = int(child.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        candidates.append((step, child))
    for _, checkpoint in sorted(candidates, reverse=True):
        missing = [
            filename
            for filename in ("trainer_state.json", "optimizer.pt", "scheduler.pt")
            if not (checkpoint / filename).is_file()
        ]
        if not any((checkpoint / filename).is_file() for filename in CHECKPOINT_MODEL_FILES):
            missing.append("model weights")
        state_path = checkpoint / "trainer_state.json"
        if not missing:
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if int(state.get("global_step", -1)) < 0:
                    raise ValueError("missing global_step")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                missing.append(f"valid trainer_state.json ({error})")
        if missing:
            print(
                f"Ignoring incomplete Trainer checkpoint {checkpoint}: "
                + ", ".join(missing),
                flush=True,
            )
            continue
        return checkpoint
    return None


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_corpus(language: str) -> Path:
    return project_root() / "4-official_corpora" / language / f"J2C_TST_{language}.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["IT", "EN", "it", "en"])
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--model-id", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-pairs", type=int, default=None)
    parser.add_argument("--max-validation-pairs", type=int, default=None)
    parser.add_argument("--max-test-pairs", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a completed run or resume the latest Trainer checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_pairs(path: Path, split: str, limit: int | None = None) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    pairs = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"journalistic_style", "conversational_style", "split"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            if row["split"].strip().lower() != split:
                continue
            source = row["journalistic_style"].strip()
            target = row["conversational_style"].strip()
            if not source or not target:
                raise ValueError(f"Empty source/target encountered in {path}")
            examples.append({"text": source, "labels": LABELS["journalistic"]})
            examples.append({"text": target, "labels": LABELS["conversational"]})
            pairs += 1
            if limit is not None and pairs >= limit:
                break
    if not examples:
        raise ValueError(f"No examples for split={split!r} in {path}")
    return examples


def classification_metrics(eval_prediction: object) -> dict[str, float]:
    import numpy as np

    logits, labels = eval_prediction
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    accuracy = float((predictions == labels).mean())
    tp = int(((predictions == 1) & (labels == 1)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"accuracy": accuracy, "precision_conversational": precision, "recall_conversational": recall, "f1_conversational": f1}


def main() -> None:
    args = parse_args()
    if (
        args.max_length < 2
        or args.epochs <= 0
        or args.early_stopping_patience < 1
        or args.early_stopping_threshold < 0
        or args.learning_rate <= 0
        or args.train_batch_size < 1
        or args.eval_batch_size < 1
        or args.gradient_accumulation_steps < 1
        or any(
            limit is not None and limit < 1
            for limit in (
                args.max_train_pairs,
                args.max_validation_pairs,
                args.max_test_pairs,
            )
        )
    ):
        raise ValueError("Invalid token, optimization, batch, or pair-limit value")
    language = args.language.upper()
    corpus = args.corpus or default_corpus(language)
    train_examples = load_pairs(corpus, "train", args.max_train_pairs)
    validation_examples = load_pairs(corpus, "validation", args.max_validation_pairs)
    test_examples = load_pairs(corpus, "test", args.max_test_pairs)
    configuration = {
        "language": language,
        "corpus": str(corpus.resolve()),
        "model_id": args.model_id,
        "train_pairs": len(train_examples) // 2,
        "validation_pairs": len(validation_examples) // 2,
        "test_pairs": len(test_examples) // 2,
        "labels": LABELS,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "restore_callback_states_from_checkpoint": True,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
        "output_dir": str(args.output_dir.resolve()),
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        return

    run_identity = {
        "language": language,
        "corpus": str(corpus.resolve()),
        "model_id": args.model_id,
        "train_pairs": len(train_examples) // 2,
        "validation_pairs": len(validation_examples) // 2,
        "test_pairs": len(test_examples) // 2,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "restore_callback_states_from_checkpoint": True,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "seed": args.seed,
    }
    metadata_path = args.output_dir / "classifier_metadata.json"
    model_config_path = args.output_dir / "config.json"
    if args.resume and metadata_path.is_file() and model_config_path.is_file():
        completed = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (completed.get(key), value)
            for key, value in run_identity.items()
            if completed.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Completed classifier in {args.output_dir} has incompatible metadata: {mismatches}"
            )
        print(f"Completed classifier already exists; skipping: {args.output_dir}")
        return
    run_config_path = args.output_dir / "run_configuration.json"
    if args.resume and run_config_path.is_file():
        previous_run = json.loads(run_config_path.read_text(encoding="utf-8"))
        if previous_run != run_identity:
            raise ValueError(
                f"Cannot resume {args.output_dir}: run configuration has changed"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config_temporary = run_config_path.with_name(run_config_path.name + ".tmp")
    run_config_temporary.write_text(
        json.dumps(run_identity, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run_config_temporary.replace(run_config_path)

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=2,
        id2label={0: "journalistic", 1: "conversational"},
        label2id=LABELS,
    )

    def prepare(examples: list[dict[str, object]]) -> Dataset:
        dataset = Dataset.from_list(examples)
        return dataset.map(
            lambda batch: tokenizer(
                batch["text"], truncation=True, max_length=args.max_length
            ),
            batched=True,
            remove_columns=["text"],
        )

    train_dataset = prepare(train_examples)
    validation_dataset = prepare(validation_examples)
    test_dataset = prepare(test_examples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    use_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=use_cuda and not use_bf16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=True,
        save_total_limit=2,
        restore_callback_states_from_checkpoint=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=classification_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )
    resume_checkpoint = None
    if args.resume and args.output_dir.is_dir():
        selected_checkpoint = find_latest_resumable_checkpoint(args.output_dir)
        if selected_checkpoint is not None:
            resume_checkpoint = str(selected_checkpoint)
            print(f"Resuming classifier training from {resume_checkpoint}")
        else:
            print(
                "No complete Trainer checkpoint found; "
                "starting classifier training from the beginning"
            )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    configuration["train_metrics"] = train_result.metrics
    configuration["test_metrics"] = test_metrics
    configuration["full_determinism"] = True
    configuration["resume_requested"] = args.resume
    configuration["resumed_from_checkpoint"] = resume_checkpoint
    configuration["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    configuration["stopped_early"] = (
        float(train_result.metrics.get("epoch", args.epochs)) + 1e-8 < args.epochs
    )
    (args.output_dir / "classifier_metadata.json").write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
