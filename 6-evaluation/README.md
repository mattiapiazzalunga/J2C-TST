# J2C-TST evaluation

The scripts implement the four evaluation dimensions and the ANSA editorial
checks described in the paper text.

The modern evaluator uses the same recommended Python 3.10/3.11 and
Transformers 4.56+ family as the baselines. Install `requirements.txt` in its
own environment; BELA remains in the separate legacy environment described
below. For the 48 GB GPU / 32 GB host-RAM profile used by the notebooks:

```bash
pip install -r requirements.txt
MAX_JOBS=2 NVCC_THREADS=2 pip install --no-build-isolation -r requirements-flash-attn.txt
```

## 1. Style classifiers

Train two independent binary `FacebookAI/xlm-roberta-base` classifiers. Each
source/target pair becomes one journalistic example (label 0) and one
conversational example (label 1); only the official training split is optimized.

```bash
python train_style_classifier.py --language IT --output-dir J2C-TST-IT/classifier/early_stopping_p1 --early-stopping-patience 1 --early-stopping-threshold 0.0 --resume
python train_style_classifier.py --language EN --output-dir J2C-TST-EN/classifier/early_stopping_p1 --early-stopping-patience 1 --early-stopping-threshold 0.0 --resume
```

The validation split selects the best checkpoint. Held-out accuracy on both
human styles is saved for auditing, but target-style accuracy is computed on
generated test outputs by `evaluate_metrics.py`. Training uses seed 42 and
Trainer full determinism. Early stopping monitors validation accuracy once per
epoch with patience 1 and restores the best checkpoint. The notebooks use the
separate `classifier/early_stopping_p1` directory so classifiers trained before
this policy are not mistaken for early-stopped runs.

## 2. Submission contract for any new system

A new model or technique does not need to reproduce baseline-specific columns.
Its UTF-8 CSV only needs:

```csv
row_id,generated_style
42,"Generated conversational rewrite"
```

`row_id` is the stable zero-based row number in the complete official corpus
CSV. A final submission must contain every `row_id` belonging to the official
test split exactly once: 7,432 rows for IT or 278 for EN. The evaluator joins
the official source and reference itself and rejects duplicates, unknown IDs,
wrong-language rows, altered references, and incomplete submissions. Use
`--allow-partial` only for debugging; partial scores are explicitly marked and
must not be reported as final results. Optional `model_id`, `strategy`, and
`language` columns are retained in the output metadata. Every summary also
records the submission filename stem as `system_id` and the input SHA-256, so a
reported result can be tied back to the exact CSV that produced it.

## 3. Style, content, fluency, and constraints

```bash
python evaluate_metrics.py \
  --language IT \
  --predictions ../5-baselines/J2C-TST-IT/predictions/IT_llama_lora_gpu48_bf16_fa2_es1_b4.csv \
  --style-classifier J2C-TST-IT/classifier/early_stopping_p1 \
  --output-dir J2C-TST-IT/results/IT_llama_lora_gpu48_bf16_fa2_es1_b4 \
  --batch-size 32 --bf16 --attention-implementation flash_attention_2 \
  --resume
```

This produces per-example CSV data and a JSON summary containing:

- conversational target-style accuracy as both a 0-1 rate and a percentage;
- BERTScore P/R/F1 between generated output and source;
- corpus BLEU between generated output and the human reference;
- token-weighted perplexity under `utter-project/EuroLLM-1.7B` (lower is better);
- compliance rates for 100-500 Unicode characters, at most three hashtags, and at
  most three emoji sequences.

The paper does not prescribe a BERTScore backbone. The default is multilingual
`xlm-roberta-large` for both languages and no baseline rescaling; `--bert-model`
can reproduce another declared choice. BLEU uses sacreBLEU's case-sensitive
default 13a tokenization. The notebooks use batch 32 for modern metrics and
batch 16 internally for EuroLLM perplexity; EuroLLM is loaded in BF16 with
FlashAttention 2. These batching/backend choices do not change the metric
definitions and are recorded in the summary.

## 4. Entity Faithfulness with BELA

BELA's official repository is archived and its inference code hard-codes CUDA,
so it should be isolated from the modern evaluation environment. This step also
requires Linux x86-64, a CUDA-capable NVIDIA GPU/driver, and Git;
these system prerequisites are not installed by `pip`:

The notebooks enable `expandable_segments` for the modern PyTorch allocator,
but remove `PYTORCH_CUDA_ALLOC_CONF` from BELA subprocesses because the pinned
PyTorch 1.13 runtime predates that allocator option. `entity_faithfulness.py`
also removes it defensively when invoked directly.

The IT and EN evaluation notebooks run `setup_bela.py` automatically before
training the style classifier. The setup installs `uv`, obtains a managed Python
3.9 interpreter, creates `.venv-bela`, installs the pinned legacy dependencies,
clones BELA, and downloads the three required assets. Interrupted asset downloads
use `.part` files and resume when the cell is run again. The commands below are
the equivalent manual alternative.

The setup intentionally executes `.venv-bela/bin/python` without resolving its
`uv` symlink. Resolving that path would bypass `pyvenv.cfg`, target the
externally managed base interpreter, and trigger PEP 668. Before installing
anything, the script verifies that `sys.prefix` is exactly `.venv-bela`.
The legacy requirements pin `setuptools==80.9.0` and `six==1.16.0`: PyTorch
Lightning 1.7.7 still imports `pkg_resources`, while PyTorch 1.13's TensorBoard
bridge imports `six` without declaring it. Every setup run repairs an existing
BELA environment if necessary and then smoke-tests the actual BELA datamodule
and evaluator imports before starting any large download.

Because the three assets require about 21 GiB, the evaluation notebooks mirror
the cache-base selection used by
`llama-3-2-atas-prompt-classifier-baseline-bert.ipynb`: `ATAS_CACHE_BASE` when
set, then the first writable location among `/tmp`, `/tmp/$USER`,
`/scratch/$USER`, `SCRATCH`, `LOCAL_SCRATCH`, `TMPDIR`, and `~/.cache`. BELA
assets are stored in the `BELA-models` child of that base (normally
`/tmp/BELA-models`, beside `/tmp/hf_cache/datasets`). Set
`J2CTST_BELA_ASSETS_DIR=/large/filesystem/path` to override this selection.
`setup_bela.py --assets-dir PATH` exposes the same choice on the command line,
and both setup and entity evaluation use that exact directory.
When the selected directory differs from `external/BELA/models`, setup moves
any existing complete asset or `.part` file to the selected filesystem before
downloading. A quota failure can therefore continue from the already downloaded
bytes instead of restarting, while also releasing the corresponding home-space
usage after the move completes.

```bash
git clone https://github.com/facebookresearch/BELA.git external/BELA
python3.9 -m venv .venv-bela
.venv-bela/bin/python -m pip install "pip<24.1"
.venv-bela/bin/pip install -r requirements-bela.txt
.venv-bela/bin/pip install -e external/BELA
mkdir -p external/BELA/models
wget -nc -P external/BELA/models http://dl.fbaipublicfiles.com/bela/models/model_e2e.ckpt
wget -nc -P external/BELA/models http://dl.fbaipublicfiles.com/bela/embeddings/index.txt
wget -nc -P external/BELA/models http://dl.fbaipublicfiles.com/bela/embeddings/embeddings.pt
```

These three required BELA assets occupy about 21 GiB in total. The upstream
`download_models.sh` also downloads three checkpoints unused by this evaluator,
so the commands above intentionally fetch only the files referenced below.

Then run (Linux/CUDA is required by the official `faiss-gpu`/BELA stack):

```bash
python entity_faithfulness.py \
  --language IT \
  --predictions ../5-baselines/J2C-TST-IT/predictions/IT_llama_lora_gpu48_bf16_fa2_es1_b4.csv \
  --output-dir J2C-TST-IT/results/IT_llama_lora_gpu48_bf16_fa2_es1_b4/entities \
  --bela-repo external/BELA \
  --checkpoint external/BELA/models/model_e2e.ckpt \
  --embeddings external/BELA/models/embeddings.pt \
  --entity-index external/BELA/models/index.txt \
  --resume
```

Entity identifiers are deduplicated per text before comparison. The JSON summary
reports micro and macro precision/recall/F1 against the reference, plus generated
entity precision against the source. Micro scores are the headline corpus
scores. An empty generated entity set has precision 1 (no unsupported entity);
an empty expected set has recall 1. Raw linked mentions and scores are retained
in `linked_entities.jsonl`. They can be rescored without BELA using
`--linked-entities`.

The language-specific notebooks are:

- `J2C-TST-IT/run_evaluation_IT.ipynb`;
- `J2C-TST-EN/run_evaluation_EN.ipynb`.

Each scans the corresponding baseline prediction directory and also accepts
arbitrary paths through `EXTRA_SUBMISSIONS`. `BELA_PYTHON` already points to the
dedicated `.venv-bela` interpreter created by the setup cell, so BELA does not
share incompatible dependencies with the modern metrics kernel. Results remain separated by
language and submission filename; `all_results.json` aggregates only the files
selected in that notebook run, avoiding stale results from older experiments.

For a first complete run of one language:

1. the notebooks already enable dependency installation, classifier training,
   modern metrics, and BELA;
2. add external CSV paths to `EXTRA_SUBMISSIONS`, if any;
3. keep `ALLOW_PARTIAL = False` for reportable results;
4. allow the BELA setup cell to finish on the first run; it may require more than
   21 GiB of free disk space and is reused on subsequent runs.

All notebook commands use `--resume`. Classifier training resumes the newest
complete Trainer checkpoint; a partially written checkpoint is reported and
skipped in favor of the preceding intact one. Completed metric stages are
checkpointed independently;
BELA stores each completed text-linking batch; final results are skipped when
their submission hash and configuration already match. Re-running the notebook
therefore continues unfinished work rather than repeating completed stages.

On later submissions the saved language-specific classifier is reused
automatically: `--resume` detects the completed classifier even while
`RUN_CLASSIFIER_TRAINING = True`. Submission filenames must have unique stems
because the stem is the system identifier and result-directory name.

## Declared assumptions

Several implementation details are not fixed by the supplied text. The style
classifiers therefore use at most 3 epochs, early-stopping patience 1 on
validation accuracy, learning rate `2e-5`, balanced paired labels,
and a 512-token maximum. BERTScore uses one multilingual backbone for IT and EN
without baseline rescaling; sources beyond that backbone's context are handled
with the standard BERTScore truncation. BLEU is a corpus score. EuroLLM perplexity is the
token-weighted corpus perplexity over generated text (not conditional on the
source). Entity identifiers are treated as per-document sets; corpus micro EF is
primary and macro EF is also retained. Character limits are inclusive and count
Python Unicode code points after generation whitespace is stripped by the
baseline generator. BELA uses mention threshold 0.2 and entity threshold 0.05,
matching the defaults of the official `ModelEval.compute_scores` evaluation
helper. Both can be changed from the CLI. Linked mentions that pass the
configured thresholds retain their mention and entity scores. Linking caches
include a hash of the complete configuration, so changing a threshold creates a
new resumable cache instead of silently reusing mentions filtered under an
older setting.
