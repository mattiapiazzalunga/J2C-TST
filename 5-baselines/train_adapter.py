"""Train a LoRA or prefix-tuning adapter for a J2C-TST corpus."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from types import MethodType
from typing import Any

from data_utils import (
    default_corpus_path,
    encode_chat_training_example,
    encode_prompt_preserving_structure,
    load_corpus,
    normalize_language,
)


MODEL_ALIASES = {
    "llama": "meta-llama/Llama-3.2-1B",
    "gemma": "google/gemma-3-1b-it",
}
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
            missing.append("model/adapter weights")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", required=True, choices=["IT", "EN", "it", "en"])
    parser.add_argument("--model-id", required=True, help="HF model id or alias: llama, gemma")
    parser.add_argument("--method", required=True, choices=["lora", "prefix"])
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-target-tokens", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--attention-implementation",
        choices=["model_default", "auto", "eager", "sdpa", "flash_attention_2"],
        default="model_default",
        help=(
            "Attention backend. 'auto' selects FlashAttention 2 for Llama and "
            "the officially supported SDPA backend for Gemma 3."
        ),
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Require CUDA bfloat16 instead of silently falling back to fp16/fp32",
    )
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 CUDA matmuls")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--group-by-length", action="store_true")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument(
        "--full-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request fully deterministic PyTorch algorithms (slower)",
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-virtual-tokens", type=int, default=30)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a completed run or resume the latest Trainer checkpoint",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate data/config without loading a model")
    return parser.parse_args()


def resolve_model_id(value: str) -> str:
    return MODEL_ALIASES.get(value.lower(), value)


def resolve_attention_implementation(model_type: str, requested: str) -> str | None:
    if requested == "model_default":
        return None
    if requested != "auto":
        return requested
    if model_type == "llama":
        return "flash_attention_2"
    # Gemma 3's official Transformers example uses SDPA. On supported CUDA
    # hardware PyTorch dispatches SDPA to its fused flash-attention kernel.
    return "sdpa"


def installed_version(distribution: str) -> str | None:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return None


def install_gemma3_prefix_cache_compat(peft_model: Any) -> bool:
    """Correct the Gemma 3 per-layer prefix-cache conversion in PEFT 0.19.x."""

    peft_config = peft_model.active_peft_config
    peft_type = getattr(peft_config.peft_type, "value", str(peft_config.peft_type))
    base_config = getattr(peft_model.get_base_model(), "config", None)
    if hasattr(base_config, "get_text_config"):
        base_config = base_config.get_text_config()
    model_type = str(getattr(base_config, "model_type", ""))
    if str(peft_type).upper() != "PREFIX_TUNING" or "gemma3_text" not in model_type:
        return False
    if getattr(peft_model, "_j2c_gemma3_prefix_cache_compat", False):
        return True
    if int(peft_config.num_transformer_submodules) != 1:
        raise ValueError(
            "Gemma 3 prefix compatibility requires one transformer submodule"
        )

    def get_prompt_compat(
        self: Any,
        batch_size: int,
        task_ids: Any = None,
        max_cache_len: int | None = None,
    ) -> Any:
        del task_ids, max_cache_len
        import torch
        from transformers import DynamicCache

        config = self.active_peft_config
        adapter_name = getattr(self, "active_adapter", "default")
        if not isinstance(adapter_name, str):
            adapter_name = list(adapter_name)[0]
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        prompt_tokens = prompt_tokens[:, : config.num_virtual_tokens]
        if config.inference_mode:
            raw_cache = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
        else:
            raw_cache = prompt_encoder(prompt_tokens)
        if self.base_model_torch_dtype is not None:
            raw_cache = raw_cache.to(self.base_model_torch_dtype)
        raw_cache = raw_cache.view(
            batch_size,
            config.num_virtual_tokens,
            config.num_layers * 2,
            config.num_attention_heads,
            config.token_dim // config.num_attention_heads,
        )
        layer_caches = raw_cache.permute(2, 0, 3, 1, 4).split(2)
        if len(layer_caches) != int(config.num_layers):
            raise RuntimeError(
                "Invalid Gemma 3 prefix cache: "
                f"expected {config.num_layers} layers, found {len(layer_caches)}"
            )
        text_config = self.get_base_model().config
        if hasattr(text_config, "get_text_config"):
            text_config = text_config.get_text_config()
        dynamic_cache = DynamicCache(config=text_config)
        cache_position = torch.arange(
            config.num_virtual_tokens, device=raw_cache.device
        )
        for layer_idx, layer_cache in enumerate(layer_caches):
            key_states, value_states = layer_cache
            dynamic_cache.update(
                key_states,
                value_states,
                layer_idx,
                cache_kwargs={"cache_position": cache_position},
            )
        return dynamic_cache

    peft_model.get_prompt = MethodType(get_prompt_compat, peft_model)
    peft_model._j2c_gemma3_prefix_cache_compat = True
    print("Enabled Gemma 3 prefix-cache compatibility for PEFT")
    return True


def encode_example(
    row: dict[str, str],
    tokenizer: Any,
    language: str,
    max_length: int,
    max_target_tokens: int,
) -> dict[str, list[int]]:
    if getattr(tokenizer, "chat_template", None):
        return encode_chat_training_example(
            tokenizer,
            language,
            row["journalistic_style"],
            row["conversational_style"],
            max_length,
            max_target_tokens,
        )
    target_ids = tokenizer(
        row["conversational_style"],
        add_special_tokens=False,
        truncation=True,
        max_length=min(max_target_tokens, max_length - 2),
    )["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
        target_ids.append(eos_id)
    target_ids = target_ids[: max_length - 1]
    prompt_budget = max(1, max_length - len(target_ids))
    prompt_ids = encode_prompt_preserving_structure(
        tokenizer,
        language,
        row["journalistic_style"],
        max_tokens=prompt_budget,
    )
    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


class CausalCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_len - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    if (
        args.max_length < 3
        or args.max_target_tokens < 1
        or args.epochs <= 0
        or args.early_stopping_patience < 1
        or args.early_stopping_threshold < 0
        or args.learning_rate <= 0
        or args.train_batch_size < 1
        or args.eval_batch_size < 1
        or args.gradient_accumulation_steps < 1
        or args.logging_steps < 1
        or args.dataloader_num_workers < 0
        or args.lora_r < 1
        or args.lora_alpha < 1
        or not 0.0 <= args.lora_dropout < 1.0
        or args.num_virtual_tokens < 1
        or (args.max_train_samples is not None and args.max_train_samples < 1)
        or (
            args.max_validation_samples is not None
            and args.max_validation_samples < 1
        )
    ):
        raise ValueError("Invalid token, optimization, adapter, batch, or sample-limit value")
    if args.method == "prefix" and args.gradient_checkpointing:
        raise ValueError(
            "PEFT prefix-tuning is incompatible with gradient checkpointing; "
            "remove --gradient-checkpointing"
        )
    language = normalize_language(args.language)
    corpus = args.corpus or default_corpus_path(language)
    model_id = resolve_model_id(args.model_id)
    train_rows = load_corpus(corpus, "train", args.max_train_samples)
    validation_rows = load_corpus(corpus, "validation", args.max_validation_samples)
    print(
        json.dumps(
            {
                "language": language,
                "corpus": str(corpus.resolve()),
                "model_id": model_id,
                "method": args.method,
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "train_batch_size": args.train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size_single_gpu": (
                    args.train_batch_size * args.gradient_accumulation_steps
                ),
                "bf16_required": args.bf16,
                "tf32": args.tf32,
                "attention_implementation_requested": args.attention_implementation,
                "gradient_checkpointing": args.gradient_checkpointing,
                "full_determinism": args.full_determinism,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_threshold": args.early_stopping_threshold,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )
    if args.dry_run:
        return

    run_identity = {
        "language": language,
        "corpus": str(corpus.resolve()),
        "base_model": model_id,
        "method": args.method,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_target_tokens": args.max_target_tokens,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "restore_callback_states_from_checkpoint": True,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_implementation_requested": args.attention_implementation,
        "bf16_required": args.bf16,
        "tf32": args.tf32,
        "dataloader_num_workers": args.dataloader_num_workers,
        "group_by_length": args.group_by_length,
        "optim": args.optim,
        "full_determinism": args.full_determinism,
        "load_in_4bit": args.load_in_4bit,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "adapter_hyperparameters": (
            {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": "all-linear",
                "bias": "none",
            }
            if args.method == "lora"
            else {
                "num_virtual_tokens": args.num_virtual_tokens,
                "prefix_projection": True,
            }
        ),
    }
    metadata_path = args.output_dir / "experiment_metadata.json"
    adapter_path = args.output_dir / "adapter_config.json"
    if args.resume and metadata_path.is_file() and adapter_path.is_file():
        completed = json.loads(metadata_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (completed.get(key), value)
            for key, value in run_identity.items()
            if completed.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Completed adapter in {args.output_dir} has incompatible metadata: {mismatches}"
            )
        print(f"Completed adapter already exists; skipping: {args.output_dir}")
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
    from peft import (
        LoraConfig,
        PeftModel,
        PrefixTuningConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    prefix_trainable_state_name = "prefix_encoder_trainable.pt"

    def active_adapter_name(peft_model: Any) -> str:
        active = getattr(peft_model, "active_adapter", "default")
        if isinstance(active, str):
            return active
        return list(active)[0]

    class PrefixSafeTrainer(Trainer):
        """Preserve trainable prefix-projection weights across checkpoints."""

        def _save_checkpoint(self, model: Any, trial: Any) -> None:
            super()._save_checkpoint(model, trial)
            if not self.args.should_save:
                return
            checkpoint_dir = (
                Path(self._get_output_dir(trial))
                / f"checkpoint-{self.state.global_step}"
            )
            if not checkpoint_dir.is_dir():
                return
            adapter_name = active_adapter_name(self.model)
            state = {
                key: value.detach().cpu()
                for key, value in self.model.prompt_encoder[
                    adapter_name
                ].state_dict().items()
            }
            state_path = checkpoint_dir / prefix_trainable_state_name
            temporary = state_path.with_name(state_path.name + ".tmp")
            torch.save(state, temporary)
            temporary.replace(state_path)

        def _load_from_checkpoint(
            self, resume_from_checkpoint: str, model: Any = None
        ) -> None:
            state_path = Path(resume_from_checkpoint) / prefix_trainable_state_name
            if not state_path.is_file():
                raise RuntimeError(
                    "This interrupted prefix checkpoint predates safe prefix-state "
                    f"saving and cannot be resumed exactly: {state_path}"
                )
            target = model if model is not None else self.model
            adapter_name = active_adapter_name(target)
            trainable_state = torch.load(
                state_path, map_location="cpu", weights_only=True
            )
            target.prompt_encoder[adapter_name].load_state_dict(trainable_state)
            print(f"Restored trainable prefix encoder from {state_path}")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    use_cuda = torch.cuda.is_available()
    if args.bf16 and (not use_cuda or not torch.cuda.is_bf16_supported()):
        raise RuntimeError(
            "--bf16 requires a CUDA GPU with bfloat16 support (Ampere or newer)"
        )
    compute_dtype = (
        torch.bfloat16
        if args.bf16 or (use_cuda and torch.cuda.is_bf16_supported())
        else (torch.float16 if use_cuda else torch.float32)
    )
    if args.tf32:
        if not use_cuda:
            raise RuntimeError("--tf32 requires a CUDA GPU")
        torch.backends.cuda.matmul.allow_tf32 = True
    model_config = AutoConfig.from_pretrained(model_id)
    model_type = str(getattr(model_config, "model_type", ""))
    attention_implementation = resolve_attention_implementation(
        model_type, args.attention_implementation
    )
    if (
        attention_implementation == "flash_attention_2"
        and importlib.util.find_spec("flash_attn") is None
    ):
        raise RuntimeError(
            "FlashAttention 2 was requested but package 'flash-attn' is unavailable. "
            "Install it after PyTorch with: pip install flash-attn --no-build-isolation"
        )
    model_kwargs: dict[str, Any] = {
        "config": model_config,
        "dtype": compute_dtype,
    }
    if attention_implementation is not None:
        model_kwargs["attn_implementation"] = attention_implementation
    if args.load_in_4bit:
        if not use_cuda:
            raise RuntimeError("4-bit adapter training requires a CUDA GPU")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    runtime_environment = {
        "torch": torch.__version__,
        "transformers": installed_version("transformers"),
        "peft": installed_version("peft"),
        "datasets": installed_version("datasets"),
        "flash_attn": installed_version("flash-attn"),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
        "gpu_total_memory_bytes": (
            torch.cuda.get_device_properties(0).total_memory if use_cuda else None
        ),
        "compute_dtype": str(compute_dtype),
        "attention_implementation": attention_implementation or "model_default",
    }
    runtime_path = args.output_dir / "runtime_environment.json"
    runtime_temporary = runtime_path.with_name(runtime_path.name + ".tmp")
    runtime_temporary.write_text(
        json.dumps(runtime_environment, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    runtime_temporary.replace(runtime_path)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    elif args.gradient_checkpointing:
        # PEFT freezes the embeddings; checkpointing still needs an input that
        # requires gradients in order to backpropagate into adapter parameters.
        model.enable_input_require_grads()

    if args.method == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
        )
    else:
        peft_config = PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=args.num_virtual_tokens,
            prefix_projection=True,
        )
    model = get_peft_model(model, peft_config)
    if args.method == "prefix":
        install_gemma3_prefix_cache_compat(model)
    model.config.use_cache = False
    model.print_trainable_parameters()

    def make_dataset(rows: list[dict[str, str]]) -> Dataset:
        raw = Dataset.from_list(
            [
                {
                    "journalistic_style": row["journalistic_style"],
                    "conversational_style": row["conversational_style"],
                }
                for row in rows
            ]
        )
        return raw.map(
            lambda row: encode_example(
                row, tokenizer, language, args.max_length, args.max_target_tokens
            ),
            remove_columns=raw.column_names,
            desc=f"Tokenizing {language}",
        )

    train_dataset = make_dataset(train_rows)
    validation_dataset = make_dataset(validation_rows)
    train_row_count = len(train_rows)
    validation_row_count = len(validation_rows)
    del train_rows, validation_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        # Transformers/PEFT cannot reload projected prefix checkpoints into
        # the live trainable prompt encoder. Prefix reload is handled below
        # with PeftModel.from_pretrained, as recommended by Transformers.
        load_best_model_at_end=args.method != "prefix",
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_cuda and compute_dtype == torch.bfloat16,
        fp16=use_cuda and compute_dtype == torch.float16,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        tf32=args.tf32,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_persistent_workers=args.dataloader_num_workers > 0,
        group_by_length=args.group_by_length,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        save_total_limit=2,
        restore_callback_states_from_checkpoint=True,
    )
    trainer_class = PrefixSafeTrainer if args.method == "prefix" else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=CausalCollator(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )
    resume_checkpoint = None
    recovered_completed_prefix = False
    recovered_state: dict[str, Any] = {}
    if args.resume and args.output_dir.is_dir():
        selected_checkpoint = find_latest_resumable_checkpoint(args.output_dir)
        if selected_checkpoint is not None:
            resume_checkpoint = str(selected_checkpoint)
            print(f"Resuming adapter training from {resume_checkpoint}")
            state_path = Path(resume_checkpoint) / "trainer_state.json"
            if args.method == "prefix" and state_path.is_file():
                recovered_state = json.loads(state_path.read_text(encoding="utf-8"))
                global_step = int(recovered_state.get("global_step", 0))
                max_steps = int(recovered_state.get("max_steps", 0))
                checkpoint_epoch = float(recovered_state.get("epoch") or 0.0)
                recovered_completed_prefix = (
                    (max_steps > 0 and global_step >= max_steps)
                    or checkpoint_epoch + 1e-8 >= args.epochs
                )
                if recovered_completed_prefix:
                    print(
                        "Prefix training already reached its configured maximum; "
                        "recovering the best adapter without rerunning optimization"
                    )
        else:
            print(
                "No complete Trainer checkpoint found; "
                "starting adapter training from the beginning"
            )

    if recovered_completed_prefix:
        train_metrics: dict[str, Any] = {
            "epoch": recovered_state.get("epoch", args.epochs),
            "global_step": recovered_state.get("global_step"),
            "recovered_from_completed_checkpoint": True,
        }
        best_model_checkpoint = recovered_state.get("best_model_checkpoint")
        best_metric = recovered_state.get("best_metric")
    else:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        train_metrics = dict(train_result.metrics)
        best_model_checkpoint = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric

    if args.method == "prefix":
        checkpoint_path = (
            Path(best_model_checkpoint)
            if best_model_checkpoint
            else (
                Path(resume_checkpoint)
                if resume_checkpoint
                else args.output_dir / "__missing_best_checkpoint__"
            )
        )
        if not checkpoint_path.is_dir() and checkpoint_path.name:
            alternate = args.output_dir / checkpoint_path.name
            if alternate.is_dir():
                checkpoint_path = alternate
        if not checkpoint_path.is_dir():
            raise FileNotFoundError(
                f"Cannot locate the best prefix checkpoint: {best_model_checkpoint!r}"
            )
        del trainer, model
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        reload_kwargs: dict[str, Any] = {
            "config": AutoConfig.from_pretrained(model_id),
            "dtype": compute_dtype,
        }
        if args.load_in_4bit:
            reload_kwargs["quantization_config"] = model_kwargs["quantization_config"]
            reload_kwargs["device_map"] = {"": 0}
        base_for_reload = AutoModelForCausalLM.from_pretrained(
            model_id, **reload_kwargs
        )
        best_prefix_model = PeftModel.from_pretrained(
            base_for_reload, str(checkpoint_path)
        )
        best_prefix_model.save_pretrained(str(args.output_dir))
        del best_prefix_model, base_for_reload
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()
        best_model_loading = "manual_peft_reload"
    else:
        trainer.save_model(str(args.output_dir))
        best_model_loading = "trainer"
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "language": language,
        "corpus": str(corpus.resolve()),
        "base_model": model_id,
        "method": args.method,
        "seed": args.seed,
        "max_length": args.max_length,
        "max_target_tokens": args.max_target_tokens,
        "epochs": args.epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "restore_callback_states_from_checkpoint": True,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "logging_steps": args.logging_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_implementation_requested": args.attention_implementation,
        "bf16_required": args.bf16,
        "tf32": args.tf32,
        "dataloader_num_workers": args.dataloader_num_workers,
        "group_by_length": args.group_by_length,
        "optim": args.optim,
        "full_determinism": args.full_determinism,
        "load_in_4bit": args.load_in_4bit,
        "resume_requested": args.resume,
        "resumed_from_checkpoint": resume_checkpoint,
        "compute_dtype": str(compute_dtype),
        "attention_implementation": attention_implementation or "model_default",
        "runtime_environment": runtime_environment,
        "adapter_hyperparameters": (
            {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": "all-linear",
                "bias": "none",
            }
            if args.method == "lora"
            else {
                "num_virtual_tokens": args.num_virtual_tokens,
                "prefix_projection": True,
            }
        ),
        "train_rows": train_row_count,
        "validation_rows": validation_row_count,
        "train_metrics": train_metrics,
        "best_model_checkpoint": (
            str(best_model_checkpoint) if best_model_checkpoint else None
        ),
        "best_metric": best_metric,
        "best_model_loading": best_model_loading,
        "recovered_completed_prefix": recovered_completed_prefix,
        "stopped_early": (
            float(train_metrics.get("epoch", args.epochs)) + 1e-8 < args.epochs
        ),
    }
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
