"""Run BELA and compute entity-faithfulness metrics for J2C-TST outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from io_utils import (
    default_corpus_path,
    file_sha256,
    load_predictions,
    load_test_references,
    write_rows,
)


TEXT_FIELDS = ("journalistic_style", "conversational_style", "generated_style")


def sanitize_legacy_torch_environment() -> None:
    """Remove allocator options unsupported by BELA's pinned PyTorch 1.13."""

    allocator_conf = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    if allocator_conf:
        print(
            "Ignoring PYTORCH_CUDA_ALLOC_CONF for the legacy BELA/PyTorch 1.13 runtime: "
            f"{allocator_conf}",
            flush=True,
        )


sanitize_legacy_torch_environment()


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--linked-entities", type=Path, default=None, help="Reuse JSONL produced by a previous BELA run")
    parser.add_argument("--bela-repo", type=Path, default=None, help="Clone of facebookresearch/BELA")
    parser.add_argument("--checkpoint", type=Path, default=None, help="BELA model_e2e.ckpt")
    parser.add_argument("--embeddings", type=Path, default=None, help="BELA embeddings.pt")
    parser.add_argument("--entity-index", type=Path, default=None, help="BELA index.txt")
    parser.add_argument("--config-name", default="joint_el_mel_new")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-length", type=int, default=254)
    parser.add_argument("--window-overlap", type=int, default=10)
    parser.add_argument("--mention-threshold", type=float, default=0.2)
    parser.add_argument(
        "--entity-threshold",
        type=float,
        default=0.05,
        help="BELA entity-linking threshold (official ModelEval evaluation default: 0.05)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed EF or continue BELA linking from saved text batches",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_file(path: Path | None, label: str) -> Path:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.resolve()


def load_bela_progress(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    raw_lines = path.read_bytes().splitlines(keepends=True)
    linked: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(raw_lines) - 1:
                raise ValueError(f"Corrupt BELA progress before final line: {path}")
            path.write_bytes(b"".join(raw_lines[:index]))
            print(f"Discarded a truncated final BELA progress record from {path}")
            break
        key = (str(item.get("row_id", "")), str(item.get("field", "")))
        if not key[0] or key[1] not in TEXT_FIELDS or key in linked:
            raise ValueError(f"Invalid or duplicate BELA progress record in {path}: {key}")
        linked[key] = item.get("mentions", [])
    return linked


def run_bela(
    rows: list[dict[str, str]], args: argparse.Namespace, progress_path: Path
) -> list[dict[str, Any]]:
    if args.bela_repo is None or not (args.bela_repo / "bela").is_dir():
        raise FileNotFoundError("--bela-repo must point to a clone of facebookresearch/BELA")
    checkpoint = _require_file(args.checkpoint, "BELA checkpoint")
    embeddings = _require_file(args.embeddings, "BELA embeddings")
    entity_index = _require_file(args.entity_index, "BELA entity index")
    sys.path.insert(0, str(args.bela_repo.resolve()))

    flat_examples: list[dict[str, Any]] = []
    keys: list[tuple[str, str]] = []
    for row in rows:
        for field in TEXT_FIELDS:
            text = row[field]
            if not text.strip():
                continue
            document_id = f"{row['row_id']}::{field}"
            flat_examples.append(
                {"document_id": document_id, "original_text": text, "gt_entities": []}
            )
            keys.append((row["row_id"], field))
    expected_keys = set(keys)
    linked = load_bela_progress(progress_path) if args.resume else {}
    unknown = set(linked).difference(expected_keys)
    if unknown:
        raise ValueError(f"BELA progress contains unknown text keys: {sorted(unknown)[:10]}")
    pending = [
        (example, key)
        for example, key in zip(flat_examples, keys)
        if key not in linked
    ]
    if linked:
        print(f"Resuming BELA linking with {len(linked)}/{len(keys)} texts complete")

    if pending:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("The official BELA ModelEval implementation requires a CUDA GPU")
        from bela.evaluation.model_eval import ModelEval
        from bela.utils.prediction_utils import get_predictions_using_windows

        model_eval = ModelEval(
            checkpoint_path=str(checkpoint),
            config_name=args.config_name,
            embeddings_path=str(embeddings),
            ent_catalogue_idx_path=str(entity_index),
        )
        model_eval.task.md_threshold = args.mention_threshold
        model_eval.task.el_threshold = args.entity_threshold
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8", newline="\n") as progress_handle:
            for start in range(0, len(pending), args.batch_size):
                batch_pairs = pending[start : start + args.batch_size]
                batch_examples = [pair[0] for pair in batch_pairs]
                batch_keys = [pair[1] for pair in batch_pairs]
                raw_predictions = get_predictions_using_windows(
                    model_eval,
                    batch_examples,
                    batch_size=args.batch_size,
                    window_length=args.window_length,
                    window_overlap=args.window_overlap,
                    do_merge_predictions=True,
                )
                if len(raw_predictions) != len(batch_examples):
                    raise RuntimeError(
                        "BELA returned a different number of predictions than input texts: "
                        f"{len(raw_predictions)} != {len(batch_examples)}"
                    )
                for example, key, prediction in zip(
                    batch_examples, batch_keys, raw_predictions
                ):
                    mentions: list[dict[str, Any]] = []
                    for offset, length, entity, md_score, el_score in zip(
                        prediction["offsets"],
                        prediction["lengths"],
                        prediction["entities"],
                        prediction["md_scores"],
                        prediction["el_scores"],
                    ):
                        if md_score < args.mention_threshold or el_score < args.entity_threshold:
                            continue
                        mentions.append(
                            {
                                "entity_id": str(entity),
                                "mention": example["original_text"][offset : offset + length],
                                "offset": int(offset),
                                "length": int(length),
                                "mention_score": float(md_score),
                                "entity_score": float(el_score),
                            }
                        )
                    linked[key] = mentions
                    progress_handle.write(
                        json.dumps(
                            {"row_id": key[0], "field": key[1], "mentions": mentions},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                progress_handle.flush()
                os.fsync(progress_handle.fileno())

    results: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {"row_id": row["row_id"]}
        for field in TEXT_FIELDS:
            item[field] = linked.get((row["row_id"], field), [])
        results.append(item)
    return results


def save_linked_entities(path: Path, linked: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for item in linked:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_linked_entities(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Linked-entities JSONL not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        linked = [json.loads(line) for line in handle if line.strip()]
    row_ids = [str(item.get("row_id", "")) for item in linked]
    if any(not row_id for row_id in row_ids):
        raise ValueError(f"Missing row_id in {path}")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"Duplicate row_id values in {path}")
    return linked


def entity_ids(item: dict[str, Any], field: str) -> set[str]:
    return {str(mention["entity_id"]) for mention in item.get(field, [])}


def PRF(predicted: set[str], expected: set[str]) -> tuple[float, float, float, int]:
    true_positive = len(predicted & expected)
    precision = true_positive / len(predicted) if predicted else 1.0
    recall = true_positive / len(expected) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, true_positive


def compute_entity_metrics(
    rows: list[dict[str, str]], linked: list[dict[str, Any]]
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    by_id = {str(item["row_id"]): item for item in linked}
    if set(by_id) != {str(row["row_id"]) for row in rows}:
        raise ValueError("row_id mismatch between predictions CSV and linked-entities JSONL")
    per_example: list[dict[str, object]] = []
    total_generated = total_reference = total_source = 0
    tp_reference = tp_source = 0
    macro_ref_p: list[float] = []
    macro_ref_r: list[float] = []
    macro_ref_f1: list[float] = []
    macro_source_p: list[float] = []
    source_nonempty = reference_nonempty = generated_nonempty = 0
    for row in rows:
        item = by_id[str(row["row_id"])]
        source = entity_ids(item, "journalistic_style")
        reference = entity_ids(item, "conversational_style")
        generated = entity_ids(item, "generated_style")
        source_nonempty += bool(source)
        reference_nonempty += bool(reference)
        generated_nonempty += bool(generated)
        ref_p, ref_r, ref_f1, ref_tp = PRF(generated, reference)
        source_p, _, _, source_tp = PRF(generated, source)
        macro_ref_p.append(ref_p)
        macro_ref_r.append(ref_r)
        macro_ref_f1.append(ref_f1)
        macro_source_p.append(source_p)
        total_generated += len(generated)
        total_reference += len(reference)
        total_source += len(source)
        tp_reference += ref_tp
        tp_source += source_tp
        per_example.append(
            {
                "row_id": row["row_id"],
                "source_entity_count": len(source),
                "reference_entity_count": len(reference),
                "generated_entity_count": len(generated),
                "reference_precision": ref_p,
                "reference_recall": ref_r,
                "reference_f1": ref_f1,
                "source_supported_precision": source_p,
            }
        )
    micro_ref_p = tp_reference / total_generated if total_generated else 1.0
    micro_ref_r = tp_reference / total_reference if total_reference else 1.0
    micro_ref_f1 = (
        2 * micro_ref_p * micro_ref_r / (micro_ref_p + micro_ref_r)
        if micro_ref_p + micro_ref_r
        else 0.0
    )
    summary = {
        "examples": len(rows),
        "reference_comparison": {
            "precision_micro": micro_ref_p,
            "recall_micro": micro_ref_r,
            "f1_micro": micro_ref_f1,
            "precision_macro": sum(macro_ref_p) / len(rows),
            "recall_macro": sum(macro_ref_r) / len(rows),
            "f1_macro": sum(macro_ref_f1) / len(rows),
            "true_positive_entities": tp_reference,
            "generated_entities": total_generated,
            "reference_entities": total_reference,
        },
        "source_support": {
            "precision_micro": tp_source / total_generated if total_generated else 1.0,
            "precision_macro": sum(macro_source_p) / len(rows),
            "supported_generated_entities": tp_source,
            "generated_entities": total_generated,
            "source_entities": total_source,
        },
        "entity_coverage": {
            "source_nonempty_examples": source_nonempty,
            "reference_nonempty_examples": reference_nonempty,
            "generated_nonempty_examples": generated_nonempty,
            "source_nonempty_rate": source_nonempty / len(rows),
            "reference_nonempty_rate": reference_nonempty / len(rows),
            "generated_nonempty_rate": generated_nonempty / len(rows),
        },
        "empty_set_convention": "precision=1 when generated set is empty; recall=1 when expected set is empty",
    }
    return per_example, summary


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.window_length < 2
        or args.window_overlap < 0
        or args.window_overlap >= args.window_length
        or not 0.0 <= args.mention_threshold <= 1.0
        or not 0.0 <= args.entity_threshold <= 1.0
    ):
        raise ValueError("Invalid batch size, BELA window, overlap, or threshold configuration")
    language = args.language.upper()
    corpus = args.corpus or default_corpus_path(language)
    test_references = load_test_references(corpus)
    rows = load_predictions(
        args.predictions,
        corpus=corpus,
        language=language,
        require_complete=not args.allow_partial,
    )
    configuration = {
        "language": language,
        "system_id": args.predictions.stem,
        "submission_sha256": file_sha256(args.predictions),
        "corpus": str(corpus.resolve()),
        "predictions": str(args.predictions.resolve()),
        "rows": len(rows),
        "expected_test_rows": len(test_references),
        "complete": len(rows) == len(test_references),
        "allow_partial": args.allow_partial,
        "reuse_linked_entities": str(args.linked_entities.resolve()) if args.linked_entities else None,
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entity_evaluation_config = {
        "linked_entities_source": (
            str(args.linked_entities.resolve()) if args.linked_entities else None
        ),
        "bela_repo": str(args.bela_repo.resolve()) if args.bela_repo else None,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "embeddings": str(args.embeddings.resolve()) if args.embeddings else None,
        "entity_index": str(args.entity_index.resolve()) if args.entity_index else None,
        "config_name": args.config_name,
        "batch_size": args.batch_size,
        "window_length": args.window_length,
        "window_overlap": args.window_overlap,
        "mention_threshold": args.mention_threshold,
        "entity_threshold": args.entity_threshold,
    }
    summary_path = args.output_dir / "entity_metrics_summary.json"
    if args.resume and summary_path.is_file():
        completed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            completed_summary.get("submission", {}).get("sha256")
            == configuration["submission_sha256"]
            and completed_summary.get("entity_evaluation_config")
            == entity_evaluation_config
        ):
            print(f"Completed entity-faithfulness evaluation already exists; skipping: {args.output_dir}")
            return
    if args.linked_entities:
        linked = load_linked_entities(args.linked_entities)
    else:
        cache_key = hashlib.sha256(
            json.dumps(entity_evaluation_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        linked_path = args.output_dir / f"linked_entities.{cache_key}.jsonl"
        progress_path = args.output_dir / f"linked_entities.{cache_key}.progress.jsonl"
        progress_config_path = args.output_dir / f"linked_entities.{cache_key}.progress.json"
        progress_identity = {
            "submission_sha256": configuration["submission_sha256"],
            "entity_evaluation_config": entity_evaluation_config,
        }
        if args.resume and progress_config_path.is_file():
            previous_progress = json.loads(
                progress_config_path.read_text(encoding="utf-8")
            )
            if previous_progress != progress_identity:
                raise ValueError(
                    f"Cannot resume BELA with a different configuration: {progress_config_path}"
                )
        atomic_write_json(progress_config_path, progress_identity)
        if args.resume and linked_path.is_file():
            linked = load_linked_entities(linked_path)
        else:
            if not args.resume and progress_path.exists():
                progress_path.unlink()
            linked = run_bela(rows, args, progress_path)
            save_linked_entities(linked_path, linked)
        if progress_path.exists():
            progress_path.unlink()
    per_example, summary = compute_entity_metrics(rows, linked)
    summary["predictions"] = str(args.predictions.resolve())
    summary["entity_evaluation_config"] = entity_evaluation_config
    summary["submission"] = {
        "language": language,
        "system_id": args.predictions.stem,
        "sha256": file_sha256(args.predictions),
        "corpus": str(corpus.resolve()),
        "evaluated_rows": len(rows),
        "expected_test_rows": len(test_references),
        "complete": len(rows) == len(test_references),
        "allow_partial": args.allow_partial,
    }
    if args.linked_entities:
        summary["linked_entities_source"] = str(args.linked_entities.resolve())
    else:
        summary["bela_config"] = {
            "repository": str(args.bela_repo.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "embeddings": str(args.embeddings.resolve()),
            "entity_index": str(args.entity_index.resolve()),
            "config_name": args.config_name,
            "mention_threshold": args.mention_threshold,
            "entity_threshold": args.entity_threshold,
            "window_length": args.window_length,
            "window_overlap": args.window_overlap,
        }
    write_rows(args.output_dir / "entity_metrics_per_example.csv", per_example)
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
