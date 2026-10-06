"""Shared data and prompt utilities for the J2C-TST baselines."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Iterable


LANGUAGES = ("IT", "EN")
REQUIRED_COLUMNS = {"journalistic_style", "conversational_style", "split"}

INSTRUCTIONS = {
    "IT": (
        "Riscrivi il testo giornalistico in italiano con uno stile conversazionale, "
        "naturale e adatto ai social media. Mantieni corretti fatti ed entita. "
        "Produci soltanto la riscrittura, tra 100 e 500 caratteri, con al massimo "
        "tre hashtag e tre emoji."
    ),
    "EN": (
        "Rewrite the journalistic text in English in a natural, conversational "
        "social-media style. Preserve facts and entities. Output only the rewrite, "
        "between 100 and 500 characters, with at most three hashtags and three emojis."
    ),
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_corpus_path(language: str) -> Path:
    language = normalize_language(language)
    return project_root() / "4-official_corpora" / language / f"J2C_TST_{language}.csv"


def normalize_language(language: str) -> str:
    language = language.upper()
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported language {language!r}; expected one of {LANGUAGES}")
    return language


def iter_corpus(path: str | Path, split: str | None = None) -> Iterable[dict[str, str]]:
    """Yield rows while retaining a stable zero-based row id from the source CSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row_id, row in enumerate(reader):
            if split is not None and row["split"].strip().lower() != split.lower():
                continue
            source = row["journalistic_style"].strip()
            target = row["conversational_style"].strip()
            if not source or not target:
                raise ValueError(f"Empty source/target in {path}, row_id={row_id}")
            row["row_id"] = str(row_id)
            yield row


def load_corpus(
    path: str | Path,
    split: str | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    rows: list[dict[str, str]] = []
    for row in iter_corpus(path, split=split):
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No rows found in {path} for split={split!r}")
    return rows


def select_demonstrations(
    rows: list[dict[str, str]], count: int = 4, seed: int = 42
) -> list[dict[str, str]]:
    if count < 1:
        raise ValueError("Demonstration count must be positive")
    if len(rows) < count:
        raise ValueError(f"Need {count} demonstrations, but only {len(rows)} rows are available")
    return random.Random(seed).sample(rows, count)


def _request(language: str, source: str) -> str:
    label = "Testo giornalistico" if language == "IT" else "Journalistic text"
    return f"{INSTRUCTIONS[language]}\n\n{label}:\n{source}"


def build_messages(
    language: str,
    source: str,
    demonstrations: list[dict[str, str]] | None = None,
    demo_source_max_chars: int | None = None,
    demo_target_max_chars: int | None = None,
) -> list[dict[str, str]]:
    """Build alternating user/assistant messages for a chat-template model."""
    language = normalize_language(language)
    demonstrations = demonstrations or []

    def maybe_truncate(text: str, limit: int | None) -> str:
        return text if limit is None or limit <= 0 else text[:limit]

    messages: list[dict[str, str]] = []
    for demo in demonstrations:
        messages.extend(
            [
                {
                    "role": "user",
                    "content": _request(
                        language,
                        maybe_truncate(
                            demo["journalistic_style"], demo_source_max_chars
                        ),
                    ),
                },
                {
                    "role": "assistant",
                    "content": maybe_truncate(
                        demo["conversational_style"], demo_target_max_chars
                    ),
                },
            ]
        )
    messages.append({"role": "user", "content": _request(language, source)})
    return messages


def build_prompt(
    tokenizer: Any,
    language: str,
    source: str,
    demonstrations: list[dict[str, str]] | None = None,
    demo_source_max_chars: int | None = None,
    demo_target_max_chars: int | None = None,
) -> str:
    """Build the same prompt representation for training and inference.

    Chat templates are used when the tokenizer exposes one (e.g. Gemma-IT).
    Base checkpoints without a template receive an equivalent plain-text prompt.
    Optional character caps are exposed for memory-constrained experiments; the
    faithful default keeps all four source/target demonstration pairs intact.
    """
    language = normalize_language(language)
    demonstrations = demonstrations or []
    has_chat_template = bool(getattr(tokenizer, "chat_template", None))

    if has_chat_template:
        messages = build_messages(
            language,
            source,
            demonstrations,
            demo_source_max_chars,
            demo_target_max_chars,
        )
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def maybe_truncate(text: str, limit: int | None) -> str:
        return text if limit is None or limit <= 0 else text[:limit]

    source_label = "Testo giornalistico" if language == "IT" else "Journalistic text"
    target_label = "Riscrittura conversazionale" if language == "IT" else "Conversational rewrite"
    sections = [INSTRUCTIONS[language]]
    for index, demo in enumerate(demonstrations, start=1):
        sections.append(
            f"Example {index}\n{source_label}:\n"
            f"{maybe_truncate(demo['journalistic_style'], demo_source_max_chars)}\n"
            f"{target_label}:\n"
            f"{maybe_truncate(demo['conversational_style'], demo_target_max_chars)}"
        )
    # Repeat the instruction next to the query so the current request remains
    # explicit after the four demonstration pairs.
    sections.append(
        f"{INSTRUCTIONS[language]}\n\n{source_label}:\n{source}\n\n{target_label}:\n"
    )
    return "\n\n".join(sections)


def prompt_uses_chat_template(tokenizer: Any) -> bool:
    return bool(getattr(tokenizer, "chat_template", None))


def _common_prefix_length(first: list[int], second: list[int]) -> int:
    length = 0
    for first_token, second_token in zip(first, second):
        if first_token != second_token:
            break
        length += 1
    return length


def _fit_source_span(
    full_ids: list[int], empty_source_ids: list[int], max_tokens: int
) -> list[int]:
    """Fit token ids by trimming only the span that replaced an empty source."""
    if len(full_ids) <= max_tokens:
        return full_ids
    prefix_length = _common_prefix_length(full_ids, empty_source_ids)
    max_suffix = min(
        len(full_ids) - prefix_length,
        len(empty_source_ids) - prefix_length,
    )
    suffix_length = 0
    while (
        suffix_length < max_suffix
        and full_ids[-(suffix_length + 1)] == empty_source_ids[-(suffix_length + 1)]
    ):
        suffix_length += 1
    if suffix_length == 0:
        # A tokenizer may merge the last source character with the following
        # separator. Preserve a small tail explicitly so the response marker is
        # never lost even when no exact common token suffix can be detected.
        suffix_length = min(16, len(full_ids) - prefix_length)

    scaffold_length = prefix_length + suffix_length
    if scaffold_length >= max_tokens:
        raise ValueError(
            "The instruction, demonstrations, and response markers exceed the "
            "token limit; increase it or shorten the demonstrations"
        )
    source_budget = max_tokens - scaffold_length
    source_end = len(full_ids) - suffix_length if suffix_length else len(full_ids)
    source_tokens = full_ids[prefix_length:source_end]
    suffix = full_ids[-suffix_length:] if suffix_length else []
    fitted = full_ids[:prefix_length] + source_tokens[:source_budget] + suffix
    if len(fitted) > max_tokens:
        raise AssertionError("Internal prompt truncation error")
    return fitted


def encode_prompt_preserving_structure(
    tokenizer: Any,
    language: str,
    source: str,
    max_tokens: int,
    demonstrations: list[dict[str, str]] | None = None,
    demo_source_max_chars: int | None = None,
    demo_target_max_chars: int | None = None,
) -> list[int]:
    """Tokenize a prompt while truncating only the current source article.

    Comparing the prompt with an otherwise identical empty-source prompt locates
    the source span at token level. This keeps the instruction, demonstrations,
    and final assistant-generation marker intact for both plain and chat prompts.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")
    demonstrations = demonstrations or []
    add_special_tokens = not prompt_uses_chat_template(tokenizer)

    def token_ids(text: str) -> list[int]:
        return tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            verbose=False,
        )["input_ids"]

    full_prompt = build_prompt(
        tokenizer,
        language,
        source,
        demonstrations=demonstrations,
        demo_source_max_chars=demo_source_max_chars,
        demo_target_max_chars=demo_target_max_chars,
    )
    full_ids = token_ids(full_prompt)
    if len(full_ids) <= max_tokens:
        return full_ids

    empty_prompt = build_prompt(
        tokenizer,
        language,
        "",
        demonstrations=demonstrations,
        demo_source_max_chars=demo_source_max_chars,
        demo_target_max_chars=demo_target_max_chars,
    )
    empty_ids = token_ids(empty_prompt)

    return _fit_source_span(full_ids, empty_ids, max_tokens)


def encode_chat_training_example(
    tokenizer: Any,
    language: str,
    source: str,
    target: str,
    max_length: int,
    max_target_tokens: int,
) -> dict[str, list[int]]:
    """Serialize a supervised chat turn, including its native turn terminator."""
    if not prompt_uses_chat_template(tokenizer):
        raise ValueError("A chat template is required")
    if max_length < 3 or max_target_tokens < 1:
        raise ValueError("max_length must be at least 3 and max_target_tokens positive")
    max_target_tokens = min(max_target_tokens, max_length - 2)
    raw_target_ids = tokenizer(
        target,
        add_special_tokens=False,
        truncation=True,
        max_length=max_target_tokens,
    )["input_ids"]
    fitted_target = tokenizer.decode(raw_target_ids, skip_special_tokens=True)

    def render(current_source: str, completed: bool) -> str:
        messages = build_messages(language, current_source)
        if completed:
            messages.append({"role": "assistant", "content": fitted_target})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not completed,
        )

    def token_ids(text: str) -> list[int]:
        return tokenizer(
            text,
            add_special_tokens=False,
            verbose=False,
        )["input_ids"]

    full_completed_ids = token_ids(render(source, completed=True))
    empty_completed_ids = token_ids(render("", completed=True))
    input_ids = _fit_source_span(full_completed_ids, empty_completed_ids, max_length)

    empty_prompt_ids = token_ids(render("", completed=False))
    response_start = _common_prefix_length(empty_prompt_ids, empty_completed_ids)
    response_token_count = len(empty_completed_ids) - response_start
    if response_token_count < 1 or response_token_count >= len(input_ids):
        raise ValueError("Could not identify the assistant response span in the chat template")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * (len(input_ids) - response_token_count)
        + input_ids[-response_token_count:],
    }
