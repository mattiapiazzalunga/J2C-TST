"""Evaluate style, content, fluency, and ANSA editorial constraints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

from io_utils import (
    default_corpus_path,
    file_sha256,
    load_predictions,
    load_test_references,
    write_rows,
)


EUROLLM_ID = "utter-project/EuroLLM-1.7B"
DEFAULT_BERTSCORE_MODEL = "xlm-roberta-large"
HASHTAG_PATTERN = re.compile(r"(?<!\w)#[\w]+", re.UNICODE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--language", required=True, choices=["IT", "EN", "it", "en"])
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow a subset of test row_ids for debugging; never use for final scores",
    )
    parser.add_argument("--style-classifier", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bert-model", default=DEFAULT_BERTSCORE_MODEL)
    parser.add_argument("--perplexity-model", default=EUROLLM_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--perplexity-max-length", type=int, default=4096)
    parser.add_argument(
        "--attention-implementation",
        choices=["model_default", "eager", "sdpa", "flash_attention_2"],
        default="model_default",
        help="Attention backend used only by the EuroLLM perplexity model",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Require CUDA bfloat16 for the EuroLLM perplexity model",
    )
    parser.add_argument("--skip-style", action="store_true")
    parser.add_argument("--skip-bertscore", action="store_true")
    parser.add_argument("--skip-bleu", action="store_true")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a completed evaluation and resume from completed metric stages",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def count_emojis(text: str) -> int:
    try:
        import emoji
    except ImportError as error:
        raise RuntimeError("Install the 'emoji' package to evaluate editorial constraints") from error
    return len(emoji.emoji_list(text))


def editorial_features(text: str) -> dict[str, object]:
    characters = len(text)
    hashtags = len(HASHTAG_PATTERN.findall(text))
    emojis = count_emojis(text)
    length_ok = 100 <= characters <= 500
    hashtags_ok = hashtags <= 3
    emojis_ok = emojis <= 3
    return {
        "characters": characters,
        "hashtags": hashtags,
        "emojis": emojis,
        "length_ok": length_ok,
        "hashtags_ok": hashtags_ok,
        "emojis_ok": emojis_ok,
        "all_editorial_constraints_ok": length_ok and hashtags_ok and emojis_ok,
    }


def add_style_scores(rows: list[dict[str, Any]], classifier_path: Path, batch_size: int) -> float:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(classifier_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(classifier_path)).to(device).eval()
    label2id = {str(label).lower(): int(label_id) for label, label_id in model.config.label2id.items()}
    conversational_id = label2id.get("conversational", 1)
    if conversational_id >= model.config.num_labels:
        raise ValueError(
            f"Invalid conversational label id {conversational_id} for {classifier_path}"
        )
    target_predictions = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer(
            [str(row["generated_style"]) for row in batch],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            probabilities = model(**encoded).logits.softmax(dim=-1)
        predicted = probabilities.argmax(dim=-1).cpu().tolist()
        target_probabilities = probabilities[:, conversational_id].cpu().tolist()
        for row, label, probability in zip(batch, predicted, target_probabilities):
            row["style_predicted_label"] = int(label)
            row["style_conversational_probability"] = float(probability)
            row["style_target_correct"] = label == conversational_id
            target_predictions += int(label == conversational_id)
    return target_predictions / len(rows)


def add_bertscore(rows: list[dict[str, Any]], model_type: str, batch_size: int) -> dict[str, float]:
    from bert_score import BERTScorer

    scorer = BERTScorer(
        model_type=model_type,
        batch_size=batch_size,
        rescale_with_baseline=False,
        use_fast_tokenizer=True,
    )
    precision, recall, f1 = scorer.score(
        [str(row["generated_style"]) for row in rows],
        [str(row["journalistic_style"]) for row in rows],
    )
    p_values = precision.cpu().tolist()
    r_values = recall.cpu().tolist()
    f_values = f1.cpu().tolist()
    for row, p_value, r_value, f_value in zip(rows, p_values, r_values, f_values):
        row["bertscore_precision_source"] = float(p_value)
        row["bertscore_recall_source"] = float(r_value)
        row["bertscore_f1_source"] = float(f_value)
    return {
        "precision": sum(p_values) / len(p_values),
        "recall": sum(r_values) / len(r_values),
        "f1": sum(f_values) / len(f_values),
    }


def add_bleu(rows: list[dict[str, Any]]) -> tuple[float, str]:
    import sacrebleu

    hypotheses = [str(row["generated_style"]) for row in rows]
    references = [str(row["conversational_style"]) for row in rows]
    for row, hypothesis, reference in zip(rows, hypotheses, references):
        row["sentence_bleu_reference"] = float(
            sacrebleu.sentence_bleu(hypothesis, [reference]).score
        )
    metric = sacrebleu.metrics.BLEU()
    score = metric.corpus_score(hypotheses, [references])
    return float(score.score), str(metric.get_signature())


def add_perplexity(
    rows: list[dict[str, Any]],
    model_id: str,
    batch_size: int,
    max_length: int,
    attention_implementation: str,
    require_bf16: bool,
) -> dict[str, object]:
    import torch
    import torch.nn.functional as functional
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_bf16 and (
        device.type != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "--bf16 requires a CUDA GPU with bfloat16 support (Ampere or newer)"
        )
    dtype = (
        torch.bfloat16
        if require_bf16
        or (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else (torch.float16 if device.type == "cuda" else torch.float32)
    )
    resolved_attention = (
        None if attention_implementation == "model_default" else attention_implementation
    )
    if (
        resolved_attention == "flash_attention_2"
        and importlib.util.find_spec("flash_attn") is None
    ):
        raise RuntimeError(
            "FlashAttention 2 was requested but package 'flash-attn' is unavailable"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_kwargs: dict[str, Any] = {"dtype": dtype}
    if resolved_attention is not None:
        model_kwargs["attn_implementation"] = resolved_attention
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs).to(device).eval()
    corpus_nll = 0.0
    corpus_tokens = 0
    finite_example_ppl: list[float] = []
    nonempty_rows: list[dict[str, Any]] = []
    for row in rows:
        if str(row["generated_style"]).strip():
            nonempty_rows.append(row)
        else:
            row["perplexity_token_count"] = 0
            row["perplexity"] = None
    for start in range(0, len(nonempty_rows), batch_size):
        batch = nonempty_rows[start : start + batch_size]
        encoded = tokenizer(
            [str(row["generated_style"]) for row in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        if encoded["input_ids"].shape[1] < 2:
            for row in batch:
                row["perplexity_token_count"] = 0
                row["perplexity"] = None
            continue
        with torch.inference_mode():
            logits = model(**encoded).logits[:, :-1, :]
        labels = encoded["input_ids"][:, 1:]
        token_mask = encoded["attention_mask"][:, 1:].bool()
        token_losses = functional.cross_entropy(
            logits.transpose(1, 2).float(), labels, reduction="none"
        )
        for index, row in enumerate(batch):
            count = int(token_mask[index].sum().item())
            nll = float((token_losses[index] * token_mask[index]).sum().item())
            row["perplexity_token_count"] = count
            if count:
                mean_nll = nll / count
                value = math.exp(mean_nll) if mean_nll < 700 else None
                row["perplexity"] = value
                if value is not None and math.isfinite(value):
                    finite_example_ppl.append(value)
                corpus_nll += nll
                corpus_tokens += count
            else:
                row["perplexity"] = None
        del logits, token_losses
    if corpus_tokens == 0:
        corpus_perplexity = None
    else:
        mean_nll = corpus_nll / corpus_tokens
        corpus_perplexity = math.exp(mean_nll) if mean_nll < 700 else None
    return {
        "model": model_id,
        "compute_dtype": str(dtype),
        "attention_implementation": resolved_attention or "model_default",
        "max_length": max_length,
        "perplexity": corpus_perplexity,
        "mean_example_perplexity": (
            sum(finite_example_ppl) / len(finite_example_ppl) if finite_example_ppl else None
        ),
        "evaluated_tokens": corpus_tokens,
        "empty_outputs": len(rows) - len(nonempty_rows),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.perplexity_max_length < 2:
        raise ValueError("--batch-size must be positive and perplexity length at least 2")
    language = args.language.upper()
    corpus = args.corpus or default_corpus_path(language)
    test_references = load_test_references(corpus)
    rows: list[dict[str, Any]] = load_predictions(
        args.predictions,
        corpus=corpus,
        language=language,
        require_complete=not args.allow_partial,
    )
    declared_languages = {row.get("language", "").upper() for row in rows if row.get("language")}
    if declared_languages and declared_languages != {language}:
        raise ValueError(
            f"Language mismatch: --language={language}, CSV languages={sorted(declared_languages)}"
        )
    if not args.skip_style and args.style_classifier is None:
        raise ValueError("--style-classifier is required unless --skip-style is set")
    if (
        not args.skip_style
        and not args.dry_run
        and not args.style_classifier.is_dir()
    ):
        raise FileNotFoundError(f"Style classifier directory not found: {args.style_classifier}")
    submission = {
        "system_id": args.predictions.stem,
        "sha256": file_sha256(args.predictions),
        "corpus": str(corpus.resolve()),
        "evaluated_rows": len(rows),
        "expected_test_rows": len(test_references),
        "complete": len(rows) == len(test_references),
        "allow_partial": args.allow_partial,
    }
    print(json.dumps({"language": language, "predictions": str(args.predictions.resolve()), **submission}, indent=2))
    if args.dry_run:
        return

    evaluation_config = {
        "bert_model": args.bert_model,
        "perplexity_model": args.perplexity_model,
        "batch_size": args.batch_size,
        "perplexity_max_length": args.perplexity_max_length,
        "attention_implementation": args.attention_implementation,
        "bf16_required": args.bf16,
        "style_classifier": (
            str(args.style_classifier.resolve()) if args.style_classifier else None
        ),
        "skip_style": args.skip_style,
        "skip_bertscore": args.skip_bertscore,
        "skip_bleu": args.skip_bleu,
        "skip_perplexity": args.skip_perplexity,
    }
    final_summary_path = args.output_dir / "metrics_summary.json"
    progress_path = args.output_dir / "metrics_progress.json"
    if args.resume and final_summary_path.is_file():
        completed_summary = json.loads(final_summary_path.read_text(encoding="utf-8"))
        if (
            completed_summary.get("submission", {}).get("sha256") == submission["sha256"]
            and completed_summary.get("evaluation_config") == evaluation_config
        ):
            print(f"Completed modern evaluation already exists; skipping: {args.output_dir}")
            return

    summary: dict[str, Any] = {
        "language": language,
        "system_id": args.predictions.stem,
        "predictions": str(args.predictions.resolve()),
        "examples": len(rows),
        "model_id": rows[0].get("model_id"),
        "strategy": rows[0].get("strategy"),
        "submission": submission,
        "evaluation_config": evaluation_config,
    }

    base_row_keys = set().union(*(row.keys() for row in rows))
    completed_stages: set[str] = set()
    if args.resume and progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("submission_sha256") != submission["sha256"]:
            raise ValueError(f"Evaluation progress belongs to a different submission: {progress_path}")
        if progress.get("evaluation_config") != evaluation_config:
            raise ValueError(f"Evaluation progress uses a different configuration: {progress_path}")
        saved_metrics = progress.get("per_example_metrics", {})
        if set(saved_metrics) != {str(row["row_id"]) for row in rows}:
            raise ValueError(f"Evaluation progress row_id mismatch: {progress_path}")
        for row in rows:
            row.update(saved_metrics[str(row["row_id"])])
        summary = progress["summary"]
        completed_stages = set(progress.get("completed_stages", []))
        print(f"Resuming evaluation after stages: {sorted(completed_stages)}")

    def save_stage(stage: str) -> None:
        completed_stages.add(stage)
        metric_values = {
            str(row["row_id"]): {
                key: value for key, value in row.items() if key not in base_row_keys
            }
            for row in rows
        }
        atomic_write_json(
            progress_path,
            {
                "submission_sha256": submission["sha256"],
                "evaluation_config": evaluation_config,
                "completed_stages": sorted(completed_stages),
                "summary": summary,
                "per_example_metrics": metric_values,
            },
        )

    if "editorial" not in completed_stages:
        editorial = [editorial_features(str(row["generated_style"])) for row in rows]
        for row, features in zip(rows, editorial):
            row.update(features)
        summary["editorial_constraints"] = {
            "length_100_500_rate": sum(bool(item["length_ok"]) for item in editorial) / len(rows),
            "max_three_hashtags_rate": sum(bool(item["hashtags_ok"]) for item in editorial) / len(rows),
            "max_three_emojis_rate": sum(bool(item["emojis_ok"]) for item in editorial) / len(rows),
            "all_constraints_rate": sum(bool(item["all_editorial_constraints_ok"]) for item in editorial) / len(rows),
            "mean_characters": sum(int(item["characters"]) for item in editorial) / len(rows),
        }
        save_stage("editorial")
    if not args.skip_style and "style" not in completed_stages:
        target_style_rate = add_style_scores(rows, args.style_classifier, args.batch_size)
        summary["target_style_accuracy"] = target_style_rate
        summary["target_style_accuracy_percent"] = 100.0 * target_style_rate
        summary["style_classifier"] = str(args.style_classifier.resolve())
        save_stage("style")
    if not args.skip_bertscore and "bertscore" not in completed_stages:
        summary["bertscore_source"] = add_bertscore(rows, args.bert_model, args.batch_size)
        summary["bertscore_config"] = {
            "model": args.bert_model,
            "idf": False,
            "rescale_with_baseline": False,
            "use_fast_tokenizer": True,
        }
        save_stage("bertscore")
    if not args.skip_bleu and "bleu" not in completed_stages:
        bleu_score, bleu_signature = add_bleu(rows)
        summary["bleu_reference"] = bleu_score
        summary["bleu_signature"] = bleu_signature
        save_stage("bleu")
    if not args.skip_perplexity and "perplexity" not in completed_stages:
        summary["fluency"] = add_perplexity(
            rows,
            args.perplexity_model,
            max(1, args.batch_size // 2),
            args.perplexity_max_length,
            args.attention_implementation,
            args.bf16,
        )
        save_stage("perplexity")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "per_example_metrics.csv", rows)
    atomic_write_json(final_summary_path, summary)
    if progress_path.exists():
        progress_path.unlink()
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
