"""Generate J2C-TST test outputs for few-shot, LoRA, or prefix baselines."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from data_utils import (
    default_corpus_path,
    encode_prompt_preserving_structure,
    load_corpus,
    normalize_language,
    select_demonstrations,
)
from train_adapter import (
    MODEL_ALIASES,
    install_gemma3_prefix_cache_compat,
    installed_version,
    resolve_attention_implementation,
    resolve_model_id,
)


OUTPUT_COLUMNS = [
    "row_id",
    "language",
    "model_id",
    "strategy",
    "journalistic_style",
    "conversational_style",
    "generated_style",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["IT", "EN", "it", "en"])
    parser.add_argument("--strategy", required=True, choices=["few_shot", "lora", "prefix"])
    parser.add_argument("--model-id", default=None, help=f"HF id or alias: {sorted(MODEL_ALIASES)}")
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--demo-source-max-chars",
        type=int,
        default=None,
        help="Optional memory-saving cap; default keeps complete demonstrations",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--attention-implementation",
        choices=["model_default", "auto", "eager", "sdpa", "flash_attention_2"],
        default="model_default",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Require CUDA bfloat16 instead of silently falling back to fp16/fp32",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from saved generated rows or skip an already complete output",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _load_progress(path: Path) -> dict[str, str]:
    """Load complete JSONL records and repair only a truncated final record."""
    if not path.is_file():
        return {}
    raw_lines = path.read_bytes().splitlines(keepends=True)
    generated: dict[str, str] = {}
    for index, raw_line in enumerate(raw_lines):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index != len(raw_lines) - 1:
                raise ValueError(f"Corrupt generation progress before final line: {path}")
            path.write_bytes(b"".join(raw_lines[:index]))
            print(f"Discarded a truncated final progress record from {path}")
            break
        row_id = str(item.get("row_id", "")).strip()
        if not row_id or "generated_style" not in item:
            raise ValueError(f"Invalid generation progress record in {path}")
        if row_id in generated:
            raise ValueError(f"Duplicate row_id={row_id} in {path}")
        generated[row_id] = str(item["generated_style"])
    return generated


def _validate_existing_output(
    path: Path,
    test_by_id: dict[str, dict[str, str]],
    language: str,
    strategy: str,
    expected_model_id: str | None,
) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(OUTPUT_COLUMNS).difference(reader.fieldnames or []):
            return False
        rows = list(reader)
    if len(rows) != len(test_by_id) or {row["row_id"] for row in rows} != set(test_by_id):
        return False
    for row in rows:
        official = test_by_id[row["row_id"]]
        if (
            row["language"] != language
            or row["strategy"] != strategy
            or (expected_model_id is not None and row["model_id"] != expected_model_id)
            or row.get("generated_style") is None
            or row["journalistic_style"] != official["journalistic_style"]
            or row["conversational_style"] != official["conversational_style"]
        ):
            raise ValueError(f"Existing output is incompatible with this run: {path}")
    return True


def _load_model(args: argparse.Namespace) -> tuple[Any, Any, str, str, str]:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_id = resolve_model_id(args.model_id) if args.model_id else None
    if args.strategy != "few_shot":
        if args.adapter_path is None:
            raise ValueError("--adapter-path is required for LoRA/prefix generation")
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(str(args.adapter_path))
        adapter_base_model = peft_config.base_model_name_or_path
        if model_id and model_id != adapter_base_model:
            raise ValueError(
                f"--model-id={model_id!r} does not match adapter base model "
                f"{adapter_base_model!r}"
            )
        model_id = model_id or adapter_base_model
        peft_type = getattr(peft_config.peft_type, "value", str(peft_config.peft_type))
        expected_peft_type = "LORA" if args.strategy == "lora" else "PREFIX_TUNING"
        if str(peft_type).upper() != expected_peft_type:
            raise ValueError(
                f"--strategy={args.strategy!r} does not match adapter type {peft_type!r}"
            )
    if not model_id:
        raise ValueError("--model-id is required for few-shot generation")

    tokenizer_path = args.adapter_path if args.adapter_path and (args.adapter_path / "tokenizer_config.json").exists() else model_id
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    use_cuda = torch.cuda.is_available()
    if args.bf16 and (not use_cuda or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            "--bf16 requires a CUDA GPU with bfloat16 support (Ampere or newer)"
        )
    dtype = (
        torch.bfloat16
        if args.bf16 or (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )
    model_config = AutoConfig.from_pretrained(model_id)
    attention_implementation = resolve_attention_implementation(
        str(getattr(model_config, "model_type", "")),
        args.attention_implementation,
    )
    if (
        attention_implementation == "flash_attention_2"
        and importlib.util.find_spec("flash_attn") is None
    ):
        raise RuntimeError(
            "FlashAttention 2 was requested but package 'flash-attn' is unavailable. "
            "Install it after PyTorch with: pip install flash-attn --no-build-isolation"
        )
    model_kwargs: dict[str, Any] = {"config": model_config, "dtype": dtype}
    if attention_implementation is not None:
        model_kwargs["attn_implementation"] = attention_implementation
    if args.load_in_4bit:
        if not use_cuda:
            raise RuntimeError("4-bit loading requires CUDA")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if args.strategy != "few_shot":
        model = PeftModel.from_pretrained(model, str(args.adapter_path))
        if args.strategy == "prefix":
            install_gemma3_prefix_cache_compat(model)
    if use_cuda and not args.load_in_4bit:
        model = model.cuda()
    model.eval()
    return (
        model,
        tokenizer,
        model_id,
        attention_implementation or "model_default",
        str(dtype),
    )


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.max_input_tokens < 1
        or args.max_new_tokens < 1
        or (args.limit is not None and args.limit < 1)
        or (
            args.demo_source_max_chars is not None
            and args.demo_source_max_chars < 1
        )
        or (args.do_sample and args.temperature <= 0)
    ):
        raise ValueError("Invalid batch, token, sample-limit, demonstration, or decoding value")
    language = normalize_language(args.language)
    corpus = args.corpus or default_corpus_path(language)
    test_rows = load_corpus(corpus, "test", args.limit)
    demonstrations: list[dict[str, str]] = []
    if args.strategy == "few_shot":
        demonstrations = select_demonstrations(load_corpus(corpus, "train"), 4, args.seed)

    configuration = {
        "language": language,
        "strategy": args.strategy,
        "model_id": resolve_model_id(args.model_id) if args.model_id else None,
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "corpus": str(corpus.resolve()),
        "test_rows": len(test_rows),
        "demonstration_row_ids": [row["row_id"] for row in demonstrations],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "demo_source_max_chars": args.demo_source_max_chars,
        "decoding": "sampling" if args.do_sample else "greedy",
        "temperature": args.temperature if args.do_sample else None,
        "load_in_4bit": args.load_in_4bit,
        "attention_implementation_requested": args.attention_implementation,
        "bf16_required": args.bf16,
        "resume_requested": args.resume,
        "output": str(args.output.resolve()),
    }
    print(json.dumps(configuration, indent=2))
    if args.dry_run:
        return

    test_by_id = {row["row_id"]: row for row in test_rows}
    metadata_path = args.output.with_suffix(".metadata.json")
    progress_path = args.output.with_name(args.output.name + ".progress.jsonl")
    expected_model_id = resolve_model_id(args.model_id) if args.model_id else None
    if args.resume and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        resume_keys = (
            "language",
            "strategy",
            "adapter_path",
            "corpus",
            "seed",
            "batch_size",
            "max_input_tokens",
            "max_new_tokens",
            "demo_source_max_chars",
            "decoding",
            "temperature",
            "load_in_4bit",
            "attention_implementation_requested",
            "bf16_required",
        )
        mismatches = {
            key: (previous.get(key), configuration.get(key))
            for key in resume_keys
            if previous.get(key) != configuration.get(key)
        }
        if expected_model_id is not None and previous.get("model_id") != expected_model_id:
            mismatches["model_id"] = (previous.get("model_id"), expected_model_id)
        if mismatches:
            raise ValueError(f"Incompatible generation progress metadata: {mismatches}")
    if args.resume and _validate_existing_output(
        args.output, test_by_id, language, args.strategy, expected_model_id
    ):
        completed_configuration = previous if metadata_path.is_file() else configuration
        completed_configuration["completed_rows"] = len(test_rows)
        completed_configuration["complete"] = True
        _atomic_json(metadata_path, completed_configuration)
        if progress_path.exists():
            progress_path.unlink()
        print(f"Complete prediction file already exists; skipping: {args.output}")
        return
    if args.resume:
        generated_by_id = _load_progress(progress_path)
        unknown = set(generated_by_id).difference(test_by_id)
        if unknown:
            raise ValueError(f"Generation progress has unknown row_ids: {sorted(unknown)[:10]}")
    else:
        generated_by_id = {}
        if progress_path.exists():
            progress_path.unlink()
    configuration["completed_rows"] = len(generated_by_id)
    configuration["complete"] = False
    _atomic_json(metadata_path, configuration)
    remaining_rows = [row for row in test_rows if row["row_id"] not in generated_by_id]
    if generated_by_id:
        print(f"Resuming generation with {len(generated_by_id)}/{len(test_rows)} rows complete")

    import torch
    from tqdm.auto import tqdm
    from transformers import set_seed

    set_seed(args.seed)
    model, tokenizer, model_id, attention_implementation, compute_dtype = _load_model(args)
    configuration["attention_implementation"] = attention_implementation
    configuration["runtime_environment"] = {
        "torch": torch.__version__,
        "transformers": installed_version("transformers"),
        "peft": installed_version("peft"),
        "flash_attn": installed_version("flash-attn"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None
        ),
        "compute_dtype": compute_dtype,
        "attention_implementation": attention_implementation,
    }
    eos_token_id = model.generation_config.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    configuration["generation_eos_token_id"] = eos_token_id
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_handle = progress_path.open("a", encoding="utf-8", newline="\n")
    for start in tqdm(range(0, len(remaining_rows), args.batch_size), desc="Generating"):
        batch_rows = remaining_rows[start : start + args.batch_size]
        prompt_ids = [
            encode_prompt_preserving_structure(
                tokenizer,
                language,
                row["journalistic_style"],
                max_tokens=args.max_input_tokens,
                demonstrations=demonstrations,
                demo_source_max_chars=args.demo_source_max_chars,
            )
            for row in batch_rows
        ]
        encoded = tokenizer.pad(
            {
                "input_ids": prompt_ids,
                "attention_mask": [[1] * len(ids) for ids in prompt_ids],
            },
            return_tensors="pt",
            padding=True,
        )
        device = model.get_input_embeddings().weight.device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if args.do_sample:
            generation_kwargs["temperature"] = args.temperature
        with torch.inference_mode():
            outputs = model.generate(**encoded, **generation_kwargs)
        new_tokens = outputs[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, generated in zip(batch_rows, texts):
            generated_text = generated.strip()
            generated_by_id[row["row_id"]] = generated_text
            progress_handle.write(
                json.dumps(
                    {"row_id": row["row_id"], "generated_style": generated_text},
                    ensure_ascii=False,
                )
                + "\n"
            )
        progress_handle.flush()
        os.fsync(progress_handle.fileno())
        configuration["completed_rows"] = len(generated_by_id)
        configuration["model_id"] = model_id
        _atomic_json(metadata_path, configuration)
    progress_handle.close()

    if len(generated_by_id) != len(test_rows):
        raise RuntimeError(
            f"Generation ended with {len(generated_by_id)}/{len(test_rows)} completed rows"
        )
    generated_rows = [
        {
            "row_id": row["row_id"],
            "language": language,
            "model_id": model_id,
            "strategy": args.strategy,
            "journalistic_style": row["journalistic_style"],
            "conversational_style": row["conversational_style"],
            "generated_style": generated_by_id[row["row_id"]],
        }
        for row in test_rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(args.output.name + ".tmp")
    with temporary_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(generated_rows)
    temporary_output.replace(args.output)
    configuration["model_id"] = model_id
    configuration["completed_rows"] = len(generated_rows)
    configuration["complete"] = True
    _atomic_json(metadata_path, configuration)
    if progress_path.exists():
        progress_path.unlink()
    print(f"Wrote {len(generated_rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
