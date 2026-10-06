# J2C-TST baselines

This directory implements the three adaptation strategies required by the
experimental protocol for both official corpora:

- four-shot prompting with four deterministic training demonstrations;
- LoRA adaptation;
- prefix-tuning.

All strategies are run separately with `meta-llama/Llama-3.2-1B` and
`google/gemma-3-1b-it`. Italian and English use independent training,
validation, and test splits from `../4-official_corpora`.

## Complete notebook runs

Use one notebook per language:

- `J2C-TST-IT/run_baselines_IT.ipynb`;
- `J2C-TST-EN/run_baselines_EN.ipynb`.

Each notebook installs `requirements.txt`, installs FlashAttention 2 on Linux,
checks CUDA/BF16 support, trains LoRA and prefix adapters, and generates all
four-shot, LoRA, and prefix test outputs. `INSTALL_DEPENDENCIES = True` and
`EXECUTE_EXPENSIVE = True` are enabled for a real end-to-end run.

Before starting any training, the dependency cell authenticates with Hugging
Face when no token is available and downloads each model's `config.json` as an
access preflight. Llama 3.2 and Gemma access conditions must first be accepted
on their respective model pages by the same personal account used for the
token. The login is stored by `huggingface_hub`, never embedded in notebook
source or command-line arguments. Set `FORCE_HF_LOGIN = True` only when the
cached token belongs to the wrong account, rerun the dependency/authentication
cell, and then restore it to `False`. The cell detects whether the installed
Hub client exposes the newer `skip_if_logged_in` parameter or the legacy
`new_session` parameter, so authentication works with both API generations.

The notebooks use seed 42, BF16, TF32, fused AdamW, length grouping, two data
loader workers, and model-specific memory profiles for a nominal 48 GB GPU and
32 GB host RAM:

| Model | Train batch | Accumulation | Effective batch | Eval batch | Generation batch | Attention |
|---|---:|---:|---:|---:|---:|---|
| Llama 3.2 1B | 4 | 4 | 16 | 8 | 4 | FlashAttention 2 |
| Gemma 3 1B IT | 2 | 8 | 16 | 4 | 2 | PyTorch SDPA |

Llama artifacts use `gpu48_bf16_fa2_es1_b4`; Gemma artifacts use
`gpu48_bf16_fa2_es1_gemma_b2`. The separate Gemma directory prevents the
failed batch-4 attempt from conflicting with the corrected run.

## Training and early stopping

`train_adapter.py` defaults to at most three epochs and learning rate `2e-4`.
Early stopping evaluates validation loss once per epoch with patience 1 and
threshold 0.0. LoRA uses Trainer's automatic best-model reload. Projected
prefix adapters cannot use that PEFT reload path safely, so prefix training
reloads the recorded best checkpoint manually with `PeftModel.from_pretrained`
and saves it as the final adapter.

Gemma 3 requires a narrow compatibility hook for the per-layer prefix-cache
layout emitted by PEFT 0.19.x. The hook follows PEFT's corrected
`DynamicCache` conversion and is enabled in both training and generation. It
does not change the prefix-tuning parameters, objective, or saved adapter
format.

## Resume behavior

Every notebook command includes `--resume`:

- a completed adapter with compatible metadata is skipped;
- an interrupted adapter run resumes its newest complete `checkpoint-*`;
  partially written checkpoints are reported and skipped in favor of the
  preceding intact one;
- new prefix checkpoints include `prefix_encoder_trainable.pt`, allowing the
  trainable projection, optimizer, scheduler, Trainer state, and early-stopping
  state to resume together;
- a completed legacy prefix run that failed only during automatic best-model
  loading is recovered from its recorded best checkpoint without optimization;
- generation durably records every completed `row_id` in a progress JSONL and
  atomically writes the final CSV only when all test rows are present.

Do not delete completed output directories before rerunning. With the existing
artifacts in place, completed Llama work is preserved and only missing Gemma or
generation stages execute.

## Outputs and evaluation hand-off

Predictions are written to each language's `predictions/` directory. Every CSV
contains `row_id`, source text, reference rewrite, and `generated_style`.
The notebooks in `../6-evaluation/J2C-TST-IT` and
`../6-evaluation/J2C-TST-EN` automatically discover all final CSV files in the
corresponding predictions directory.

Generated instructions request 100-500 Unicode characters, at most three
hashtags, and at most three emojis. Compliance is measured rather than forced
after generation.

## Reproducibility notes

Run metadata records corpus path, model id, adapter hyperparameters, batch
configuration, seed, installed library versions, CUDA runtime, resolved
attention backend, and GPU. Fixed seeds make independent runs comparable, but
`--no-full-determinism` and fused attention mean a clean retraining is not
guaranteed to be bit-identical. Rerunning with completed outputs preserved does
not retrain or overwrite them.
