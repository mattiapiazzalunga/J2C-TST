"""Complete preprocessing for the J2C-TST corpora.

Dataset usage:
    python preprocess_corpora.py IT
    python preprocess_corpora.py EN

Single-text usage (pre-tokenization normalization):
    python preprocess_corpora.py IT --text "Text to normalize"

Required dependencies:
    pip install beautifulsoup4 ftfy regex phonenumbers emoji \
        datasketch spylls sentence-transformers torch numpy
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from html.entities import html5 as HTML5_ENTITIES
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence
from urllib.request import urlopen

import emoji
import ftfy
import numpy as np
import phonenumbers
import regex
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, NavigableString
from datasketch import MinHash, MinHashLSH
from spylls.hunspell import Dictionary as HunspellDictionary


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RESOURCE_DIR = SCRIPT_DIR / ".resources"

INPUT_PATHS = {
    "IT": PROJECT_DIR / "2-raw_data" / "IT" / "raw_dataset_IT.csv",
    "EN": PROJECT_DIR / "2-raw_data" / "EN" / "raw_dataset_EN.csv",
}
OUTPUT_PATHS = {
    "IT": PROJECT_DIR / "4-official_corpora" / "IT" / "J2C_TST_IT.csv",
    "EN": PROJECT_DIR / "4-official_corpora" / "EN" / "J2C_TST_EN.csv",
}

TEXT_COLUMNS = ("journalistic_style", "conversational_style")
MORPH_IT_URL = (
    "https://raw.githubusercontent.com/giodegas/morphit-lemmatizer/"
    "master/master/morph-it_048.txt"
)
MORPH_IT_PATH = RESOURCE_DIR / "morph-it_048.txt"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_THRESHOLD = 0.7
EMBEDDING_BATCH_SIZE = 16
MINHASH_PERMUTATIONS = 256
JACCARD_THRESHOLD = 0.8
RANDOM_SEED = 1729

STRUCTURAL_TAGS = {
    "br",
    "p",
    "div",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "section",
    "article",
}
REMOVED_TAGS = {
    "script",
    "style",
    "template",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "object",
    "embed",
}

APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "‚": "'",
        "‛": "'",
        "′": "'",
        "‵": "'",
        "`": "'",
        "´": "'",
        "ʼ": "'",
        "＇": "'",
    }
)
QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "«": '"',
        "»": '"',
        "″": '"',
        "‶": '"',
        "＂": '"',
        "〝": '"',
        "〞": '"',
        "〟": '"',
    }
)
EXPLICIT_DASHES = {
    "‐",
    "‑",
    "‒",
    "–",
    "—",
    "―",
    "−",
    "﹘",
    "﹣",
    "－",
}
REMOVED_INVISIBLES = {
    "\u00ad",
    "\u2060",
    "\u200c",
    "\ufeff",
    "\u180e",
    "\u034f",
    "\u200e",
    "\u200f",
    *map(chr, range(0x202A, 0x202F)),
    *map(chr, range(0x2061, 0x2065)),
    *map(chr, range(0x2066, 0x206A)),
}
VALID_UNACCENTED_APOSTROPHE_FORMS = {"po'", "fa'", "va'", "sta'", "da'", "di'"}
ACCENT_VARIANTS = {
    "a": ("à",),
    "e": ("è", "é"),
    "i": ("ì",),
    "o": ("ò",),
    "u": ("ù",),
}
PLACEHOLDER_SENTINELS = {
    "<EMAIL>": "\ue000J2CEMAIL\ue001",
    "<URL>": "\ue000J2CURL\ue001",
    "<PHONE>": "\ue000J2CPHONE\ue001",
}
LINE_SEPARATOR_SENTINELS = {
    "\r\n": "\ue010J2CCRLF\ue011",
    "\r": "\ue010J2CCR\ue011",
    "\u0085": "\ue010J2CNEXTLINE\ue011",
    "\u2028": "\ue010J2CLINESEPARATOR\ue011",
    "\u2029": "\ue010J2CPARAGRAPHSEPARATOR\ue011",
    "\v": "\ue010J2CVERTICALTAB\ue011",
    "\f": "\ue010J2CFORMFEED\ue011",
}
DOUBLE_ANGLE_SENTINELS = {
    "<<": "\ue020J2CDOUBLELEFTANGLE\ue021",
    ">>": "\ue020J2CDOUBLERIGHTANGLE\ue021",
}
LITERAL_AMPERSAND_SENTINEL = "\ue000J2CAMPERSAND\ue001"
HTML_ENTITY_RE = regex.compile(
    r"&(?:#[xX][0-9A-Fa-f]+;?|#[0-9]+;?|[A-Za-z][A-Za-z0-9]+;?)"
)
EMAIL_RE = regex.compile(
    r"(?<![\p{L}\p{N}._%+-])[\p{L}\p{N}._%+-]+@"
    r"(?:[\p{L}\p{N}-]+\.)+[\p{L}]{2,}(?![\p{L}\p{N}._%+-])",
    regex.IGNORECASE,
)
URL_RE = regex.compile(
    r"(?i)(?:https?://|ftp://|www\.)"
    r"[^\s<>\"']*[^\s<>\"'.,;:!?\)\]\}]"
)
ITALIAN_APOSTROPHE_WORD_RE = regex.compile(
    r"(?<!\p{L})(?P<word>\p{L}*[AEIOUaeiou])'(?!\p{L})"
)
LETTER_WORD_PATTERN = r"\p{L}(?:[\p{L}']*\p{L})?"
HYPHENATED_WORD_PATTERN = rf"{LETTER_WORD_PATTERN}(?:-{LETTER_WORD_PATTERN})*"
LINE_END_HYPHEN_RE = regex.compile(
    r"(?<![\p{L}\p{N}_-])"
    rf"(?P<left>{HYPHENATED_WORD_PATTERN})-\n"
    rf"(?P<right>{HYPHENATED_WORD_PATTERN})"
    r"(?![\p{L}\p{N}_-])"
)
HASHTAG_RE = regex.compile(r"(?<![\p{L}\p{N}_])#[\p{L}\p{N}_]+")
WORD_WITH_LETTERS_RE = regex.compile(r"\p{L}+(?:'\p{L}+)?")
WHITESPACE_WORD_RE = regex.compile(r"\S+")
ENGLISH_COMMON_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
BULLET_PREFIXES = ("-", "*", "•", "‣", "◦", "▪", "▫")


@dataclass(frozen=True)
class MorphItLexicon:
    forms: frozenset[str]
    canonical_forms: frozenset[str]


def _download_resource(url: str, destination: Path) -> Path:
    """Atomically download a required resource if it is not already cached."""
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urlopen(url) as response, NamedTemporaryFile(
            mode="wb", delete=False, dir=destination.parent
        ) as temporary:
            temporary_path = Path(temporary.name)
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


@lru_cache(maxsize=1)
def _load_morph_it_forms() -> MorphItLexicon:
    path = _download_resource(MORPH_IT_URL, MORPH_IT_PATH)
    forms: set[str] = set()
    canonical_forms: set[str] = set()
    with path.open("r", encoding="latin-1", newline="") as lexicon:
        for line in lexicon:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 2:
                form = unicodedata.normalize("NFC", fields[0]).casefold()
                lemma = unicodedata.normalize("NFC", fields[1]).casefold()
                forms.add(form)
                if form == lemma:
                    canonical_forms.add(form)
    if not forms:
        raise RuntimeError(f"The Morph-it lexicon is empty: {path}")
    return MorphItLexicon(frozenset(forms), frozenset(canonical_forms))


@lru_cache(maxsize=1)
def _load_english_hunspell() -> HunspellDictionary:
    return HunspellDictionary.from_files("en_US")


def _protect_placeholders(text: str) -> str:
    for placeholder, sentinel in PLACEHOLDER_SENTINELS.items():
        text = text.replace(placeholder, sentinel)
    return text


def _restore_placeholders(text: str) -> str:
    for placeholder, sentinel in PLACEHOLDER_SENTINELS.items():
        text = text.replace(sentinel, placeholder)
    return text


def _protect_line_separators(text: str) -> str:
    for separator, sentinel in LINE_SEPARATOR_SENTINELS.items():
        text = text.replace(separator, sentinel)
    return text


def _restore_line_separators(text: str) -> str:
    for separator, sentinel in LINE_SEPARATOR_SENTINELS.items():
        text = text.replace(sentinel, separator)
    return text


def _html_to_text_once(text: str) -> str:
    for sequence, sentinel in DOUBLE_ANGLE_SENTINELS.items():
        text = text.replace(sequence, sentinel)

    # BeautifulSoup also accepts very permissive references without a semicolon:
    # protecting literal ampersands prevents a second pass from losing normal text.
    pieces: list[str] = []
    position = 0
    while position < len(text):
        if text[position] != "&":
            pieces.append(text[position])
            position += 1
            continue
        entity = HTML_ENTITY_RE.match(text, position)
        if entity is not None:
            token = entity.group(0)
            name = token[1:]
            if name.startswith("#") or name in HTML5_ENTITIES:
                pieces.append(token)
                position = entity.end()
                continue
        pieces.append(LITERAL_AMPERSAND_SENTINEL)
        position += 1
    text = "".join(pieces)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(text, "html.parser")

    for tag in soup.find_all(REMOVED_TAGS):
        tag.decompose()

    for tag in soup.find_all(STRUCTURAL_TAGS):
        if tag.name == "br":
            tag.replace_with(NavigableString("\n"))
        else:
            tag.insert_before(NavigableString("\n"))
            tag.insert_after(NavigableString("\n"))

    pieces = []
    last_character = ""
    for node in soup.strings:
        piece = str(node)
        if not piece:
            continue
        if last_character.isalnum() and piece[0].isalnum():
            pieces.append(" ")
        pieces.append(piece)
        last_character = piece[-1]
    text = "".join(pieces).replace(LITERAL_AMPERSAND_SENTINEL, "&")
    for sequence, sentinel in DOUBLE_ANGLE_SENTINELS.items():
        text = text.replace(sentinel, sequence)
    return text


def _html_to_text(text: str) -> str:
    # A single decoding layer is not idempotent for nested entities
    # (for example, &amp;amp;). The HTML block is therefore decoded to a fixed point.
    while True:
        normalized = _html_to_text_once(text)
        if normalized == text:
            return normalized
        text = normalized


def _normalize_unicode_spaces_and_controls(text: str) -> str:
    output: list[str] = []
    for index, character in enumerate(text):
        if character == "\t" or unicodedata.category(character) == "Zs":
            output.append(" ")
            continue

        if character == "\u200b":
            previous_is_alphanumeric = index > 0 and text[index - 1].isalnum()
            next_is_alphanumeric = index + 1 < len(text) and text[index + 1].isalnum()
            if previous_is_alphanumeric and next_is_alphanumeric:
                output.append(" ")
            continue

        codepoint = ord(character)
        if character in REMOVED_INVISIBLES:
            continue
        if character != "\n" and (codepoint < 0x20 or 0x7F <= codepoint <= 0x9F):
            continue
        output.append(character)
    return "".join(output)


def _match_case(source: str, replacement: str) -> str:
    result: list[str] = []
    for source_character, replacement_character in zip(source, replacement):
        result.append(
            replacement_character.upper()
            if source_character.isupper()
            else replacement_character.lower()
        )
    return "".join(result)


def _repair_italian_apostrophe_accents(text: str, lexicon: MorphItLexicon) -> str:
    def replace(match: regex.Match[str]) -> str:
        original_word = match.group("word")
        original_form = f"{original_word.casefold()}'"
        if original_form in lexicon.forms or original_form in VALID_UNACCENTED_APOSTROPHE_FORMS:
            return match.group(0)

        base = original_word.casefold()
        candidates = [
            f"{base[:-1]}{accent}"
            for accent in ACCENT_VARIANTS[base[-1]]
            if f"{base[:-1]}{accent}" in lexicon.forms
        ]
        if len(candidates) == 1:
            replacement = candidates[0]
        else:
            canonical_candidates = [
                candidate
                for candidate in candidates
                if candidate in lexicon.canonical_forms
            ]
            if len(canonical_candidates) == 1:
                replacement = canonical_candidates[0]
            elif base == "e" and "è" in candidates:
                replacement = "è"
            else:
                return match.group(0)
        return _match_case(original_word, replacement)

    return ITALIAN_APOSTROPHE_WORD_RE.sub(replace, text)


def _normalize_dashes(text: str) -> str:
    return "".join(
        "-"
        if character in EXPLICIT_DASHES or unicodedata.category(character) == "Pd"
        else character
        for character in text
    )


def _repair_line_end_hyphenation(
    text: str,
    language: str,
    morph_it: MorphItLexicon | None,
    hunspell: HunspellDictionary | None,
) -> str:
    def valid(word: str) -> bool:
        if language == "IT":
            assert morph_it is not None
            return word.casefold() in morph_it.forms
        assert hunspell is not None
        return hunspell.lookup(word) or hunspell.lookup(word.casefold())

    def replace(match: regex.Match[str]) -> str:
        left = match.group("left")
        right = match.group("right")
        joined = f"{left}{right}"
        hyphenated = f"{left}-{right}"
        joined_is_valid = valid(joined)
        hyphenated_is_valid = valid(hyphenated)
        if joined_is_valid and not hyphenated_is_valid:
            return joined
        if hyphenated_is_valid and not joined_is_valid:
            return hyphenated
        return match.group(0)

    return LINE_END_HYPHEN_RE.sub(replace, text)


def _normalize_long_punctuation_sequences(text: str) -> str:
    text = regex.sub(
        r"[^\S\n]*\.(?:[^\S\n]*\.){3,}[^\S\n]*(?=[\p{L}\p{N}_])",
        "... ",
        text,
    )
    text = regex.sub(r"[^\S\n]*\.(?:[^\S\n]*\.){3,}", "...", text)
    text = regex.sub(r"!{4,}", "!!!", text)
    return regex.sub(r"\?{4,}", "???", text)


def _replace_urls(text: str) -> str:
    return URL_RE.sub("<URL>", text)


def _replace_phone_numbers(text: str, language: str) -> str:
    region = "IT" if language == "IT" else "US"
    matches = list(
        phonenumbers.PhoneNumberMatcher(
            text,
            region,
            leniency=phonenumbers.Leniency.VALID,
        )
    )
    for match in reversed(matches):
        text = f"{text[:match.start]}<PHONE>{text[match.end:]}"
    return text


class TextNormalizer:
    """Idempotent normalizer that can also be used before HF tokenization."""

    def __init__(self, language: str) -> None:
        self.language = language.upper()
        if self.language not in INPUT_PATHS:
            raise ValueError("language deve essere IT oppure EN")
        self.morph_it = _load_morph_it_forms() if self.language == "IT" else None
        self.hunspell = _load_english_hunspell() if self.language == "EN" else None

    def __call__(self, value: Any) -> str:
        text = "" if value is None else str(value)
        text = _protect_placeholders(text)

        # 1. Remove HTML, discard non-text tags, preserve blocks with \n, and
        # stabilize HTML decoding to ensure idempotence even with nested entities.
        text = _html_to_text(text)

        # 2. Repair Unicode/mojibake without decoding HTML entities again, and
        # normalize to NFC.
        text = _protect_line_separators(text)
        text = ftfy.fix_text(text, unescape_html=False)
        text = unicodedata.normalize("NFC", text)
        text = _restore_line_separators(text)

        # 3. Convert all required line-separator variants to \n.
        text = regex.sub(r"\r\n|[\r\u0085\u2028\u2029\v\f]", "\n", text)

        # 4. Normalize Unicode spaces and remove invisible characters and C0/C1
        # controls, then normalize to NFC again.
        text = _normalize_unicode_spaces_and_controls(text)
        text = unicodedata.normalize("NFC", text)

        # 5. Collapse horizontal whitespace and remove spaces around \n.
        text = regex.sub(r" +", " ", text)
        text = regex.sub(r" *\n *", "\n", text)

        # 6. Reduce sequences of more than two \n characters to two.
        text = regex.sub(r"\n{3,}", "\n\n", text)

        # 7. Convert all required apostrophe variants to the ASCII apostrophe.
        text = text.translate(APOSTROPHE_TRANSLATION)

        # 8. In Italian, correct a'/e'/i'/o'/u' to the only form valid in
        # Morph-it; example ambiguities are resolved using the canonical lemma
        # form (and the copula è), while preserving capitalization.
        if self.language == "IT":
            assert self.morph_it is not None
            text = _repair_italian_apostrophe_accents(text, self.morph_it)

        # 9. Normalize the required quotation-mark variants to ASCII quotes.
        text = text.replace("<<", '"').replace(">>", '"')
        text = text.translate(QUOTE_TRANSLATION)

        # 10. Convert Unicode dashes to the ASCII hyphen.
        text = _normalize_dashes(text)

        # 11. Repair only unambiguous end-of-line hyphenation, using Morph-it for
        # IT and a Hunspell dictionary for EN.
        text = _repair_line_end_hyphenation(
            text, self.language, self.morph_it, self.hunspell
        )

        # 12. Normalize only long sequences of periods, exclamation marks, and
        # question marks according to the required thresholds.
        text = _normalize_long_punctuation_sequences(text)

        # 13. Apply the final strip to the text block.
        text = text.strip()

        # 14. Replace email addresses, URLs, and phone numbers with the required placeholders.
        text = _replace_urls(text)
        text = EMAIL_RE.sub("<EMAIL>", text)
        text = _replace_phone_numbers(text, self.language)

        return _restore_placeholders(text)


@lru_cache(maxsize=2)
def _normalizer(language: str) -> TextNormalizer:
    return TextNormalizer(language.upper())


def normalize_text(text: str, language: str) -> str:
    """Normalize a single text, including at inference time."""
    return _normalizer(language)(text)


def normalize_huggingface_example(
    example: Mapping[str, Any], language: str
) -> dict[str, str]:
    """Callback compatible with Hugging Face ``Dataset.map``."""
    normalizer = _normalizer(language.upper())
    return {column: normalizer(example[column]) for column in TEXT_COLUMNS}


def normalize_huggingface_batch(
    batch: Mapping[str, Sequence[Any]], language: str
) -> dict[str, list[str]]:
    """Callback compatible with ``Dataset.map(..., batched=True)``."""
    normalizer = _normalizer(language.upper())
    return {
        column: [normalizer(value) for value in batch[column]]
        for column in TEXT_COLUMNS
    }


def normalize_huggingface_dataset(dataset: Any, language: str) -> Any:
    """Normalize an HF dataset and discard rows containing empty texts."""
    normalized = dataset.map(
        normalize_huggingface_batch,
        batched=True,
        fn_kwargs={"language": language.upper()},
    )
    # 15. Discard the entire row when at least one of the two texts is empty.
    return normalized.filter(
        lambda row: bool(row["journalistic_style"].strip())
        and bool(row["conversational_style"].strip())
    )


def _words(text: str) -> list[str]:
    return WHITESPACE_WORD_RE.findall(text)


def _nonempty_lines(text: str) -> list[str]:
    return [line for line in text.split("\n") if line]


def _paragraphs(text: str) -> list[str]:
    return [paragraph for paragraph in regex.split(r"\n{2,}", text) if paragraph]


def _duplicate_statistics(units: Sequence[str], document_length: int) -> tuple[float, float]:
    if not units or document_length == 0:
        return 0.0, 0.0
    seen: set[str] = set()
    duplicated_units = 0
    duplicated_characters = 0
    for unit in units:
        if unit in seen:
            duplicated_units += 1
            duplicated_characters += len(unit)
        else:
            seen.add(unit)
    return duplicated_units / len(units), duplicated_characters / document_length


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in WHITESPACE_WORD_RE.finditer(text)]


def _covered_character_fraction(
    text: str,
    n: int,
    all_duplicated_ngrams: bool,
    tokens: Sequence[tuple[str, int, int]] | None = None,
) -> float:
    if not text:
        return 0.0
    if tokens is None:
        tokens = _token_spans(text)
    if len(tokens) < n:
        return 0.0

    occurrences: dict[tuple[str, ...], list[tuple[int, int]]] = {}
    for index in range(len(tokens) - n + 1):
        ngram = tuple(token for token, _, _ in tokens[index : index + n])
        span = (tokens[index][1], tokens[index + n - 1][2])
        occurrences.setdefault(ngram, []).append(span)

    if all_duplicated_ngrams:
        # The first occurrence is not a duplicate: only subsequent repetitions
        # cover the text, without double-counting overlapping intervals.
        intervals = sorted(
            span
            for spans in occurrences.values()
            if len(spans) > 1
            for span in spans[1:]
        )
    else:
        maximum = max(len(spans) for spans in occurrences.values())
        selected = {
            next(ngram for ngram, spans in occurrences.items() if len(spans) == maximum)
        }
        intervals = sorted(
            span
            for ngram in selected
            for span in occurrences[ngram]
        )
    if not intervals:
        return 0.0

    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end
    covered += current_end - current_start
    return covered / len(text)


def _fails_generic_text_rules(text: str) -> str | None:
    token_spans = _token_spans(text)
    words = [token for token, _, _ in token_spans]
    lines = _nonempty_lines(text)
    paragraphs = _paragraphs(text)

    # Remove a row if more than 90% of its lines begin with a bullet point.
    if lines and sum(line.lstrip().startswith(BULLET_PREFIXES) for line in lines) / len(lines) > 0.90:
        return "bullet_lines"

    # Remove a row if fewer than 80% of its words contain an alphabetic character.
    if not words or sum(any(character.isalpha() for character in word) for word in words) / len(words) < 0.80:
        return "alphabetic_words"

    line_duplicates, line_duplicate_characters = _duplicate_statistics(lines, len(text))
    paragraph_duplicates, paragraph_duplicate_characters = _duplicate_statistics(
        paragraphs, len(text)
    )

    # Remove a row if more than 30% of its lines are duplicated.
    if line_duplicates > 0.30:
        return "duplicate_lines"

    # Remove a row if more than 30% of its paragraphs are duplicated.
    if paragraph_duplicates > 0.30:
        return "duplicate_paragraphs"

    # Remove a row if more than 20% of its characters belong to duplicated lines.
    if line_duplicate_characters > 0.20:
        return "duplicate_line_characters"

    # Remove a row if more than 20% of its characters belong to duplicated paragraphs.
    if paragraph_duplicate_characters > 0.20:
        return "duplicate_paragraph_characters"

    # In Gopher, n-gram filters are applied after the minimum requirement of 50
    # document words. Shorter texts are not eligible for these measures based on
    # the fraction of covered characters.
    if len(words) < 50:
        return None

    # Remove a row if its most frequent 2-gram covers more than 20% of the characters.
    if _covered_character_fraction(text, 2, False, token_spans) > 0.20:
        return "top_2gram"

    # Remove a row if its most frequent 3-gram covers more than 18% of the characters.
    if _covered_character_fraction(text, 3, False, token_spans) > 0.18:
        return "top_3gram"

    # Remove a row if its most frequent 4-gram covers more than 16% of the characters.
    if _covered_character_fraction(text, 4, False, token_spans) > 0.16:
        return "top_4gram"

    # Remove rows when duplicated 5- to 10-grams cover more than 15%, 14%, 13%,
    # 12%, 11%, and 10% of the characters, respectively.
    duplicate_ngram_thresholds = {
        5: 0.15,
        6: 0.14,
        7: 0.13,
        8: 0.12,
        9: 0.11,
        10: 0.10,
    }
    for n, threshold in duplicate_ngram_thresholds.items():
        if _covered_character_fraction(text, n, True, token_spans) > threshold:
            return f"duplicate_{n}grams"

    return None


def _corpus_rule_failure(row: Mapping[str, str], language: str) -> str | None:
    journalistic = row["journalistic_style"]
    conversational = row["conversational_style"]
    journalistic_words = _words(journalistic)

    # Inspired by DeepMind's Gopher training preprocessing.

    # Remove a row if the journalistic text has fewer than 50 or more than 100,000 words.
    if len(journalistic_words) < 50 or len(journalistic_words) > 100_000:
        return "journalistic_word_count"

    # Remove a row if the average journalistic word length is below 3 or above 10.
    average_word_length = sum(map(len, journalistic_words)) / len(journalistic_words)
    if average_word_length < 3 or average_word_length > 10:
        return "journalistic_average_word_length"

    # Remove a row if either the hash/word or ellipsis/word ratio exceeds 0.1.
    if journalistic.count("#") / len(journalistic_words) > 0.1:
        return "journalistic_hash_ratio"
    if journalistic.count("...") / len(journalistic_words) > 0.1:
        return "journalistic_ellipsis_ratio"

    # Apply all rules stated generically as applying "in the text" to both texts.
    for column in TEXT_COLUMNS:
        failure = _fails_generic_text_rules(row[column])
        if failure is not None:
            return f"{column}:{failure}"

    # For English, remove a row unless it contains at least two distinct common words.
    if language == "EN":
        present = {
            word.casefold()
            for word in WORD_WITH_LETTERS_RE.findall(journalistic)
            if word.casefold() in ENGLISH_COMMON_WORDS
        }
        if len(present) < 2:
            return "english_common_words"

    # Remove a row unless the conversational text is 100 to 500 characters long, inclusive.
    if not 100 <= len(conversational) <= 500:
        return "conversational_character_count"

    # Remove a row with more than three hashtags or more than three emoji in the conversational text.
    if len(HASHTAG_RE.findall(conversational)) > 3:
        return "conversational_hashtags"
    if len(emoji.emoji_list(conversational)) > 3:
        return "conversational_emoji"

    return None


def _filter_by_embeddings(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    # Remove a row unless the cosine similarity of its E5 embeddings exceeds 0.7.
    if not rows:
        return [], 0

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    kept: list[dict[str, str]] = []
    removed = 0
    for start in range(0, len(rows), EMBEDDING_BATCH_SIZE):
        batch = rows[start : start + EMBEDDING_BATCH_SIZE]
        journalistic_embeddings = model.encode(
            [row["journalistic_style"] for row in batch],
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        conversational_embeddings = model.encode(
            [row["conversational_style"] for row in batch],
            batch_size=EMBEDDING_BATCH_SIZE,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        similarities = np.einsum(
            "ij,ij->i", journalistic_embeddings, conversational_embeddings
        )
        for row, similarity in zip(batch, similarities):
            if float(similarity) > EMBEDDING_THRESHOLD:
                kept.append(row)
            else:
                removed += 1
    return kept, removed


def _remove_exact_duplicates(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    # Remove exact journalistic duplicates only within the same social network.
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, str]] = []
    for row in rows:
        key = (row["social_network"].strip().casefold(), row["journalistic_style"])
        if key not in seen:
            seen.add(key)
            kept.append(row)
    return kept, len(rows) - len(kept)


def _jaccard_words(text: str) -> list[str]:
    without_punctuation = "".join(
        "" if unicodedata.category(character).startswith("P") else character
        for character in text
    )
    return without_punctuation.split()


def _thirteen_grams(text: str) -> set[bytes]:
    words = _jaccard_words(text)
    return {
        " ".join(words[index : index + 13]).encode("utf-8")
        for index in range(len(words) - 12)
    }


def _minhash(text: str) -> MinHash:
    signature = MinHash(num_perm=MINHASH_PERMUTATIONS)
    for ngram in _thirteen_grams(text):
        signature.update(ngram)
    return signature


def _remove_near_duplicates(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    # Keep a separate MinHash index for each social network so cross-social
    # variants of the same article are preserved.
    lsh_by_social: dict[str, MinHashLSH] = {}
    signatures: dict[int, MinHash] = {}
    alive: set[int] = set()
    rng = random.Random(RANDOM_SEED)

    for index, row in enumerate(rows):
        social_network = row["social_network"].strip().casefold()
        lsh = lsh_by_social.get(social_network)
        if lsh is None:
            lsh = MinHashLSH(
                threshold=JACCARD_THRESHOLD,
                num_perm=MINHASH_PERMUTATIONS,
            )
            lsh_by_social[social_network] = lsh
        signature = _minhash(row["journalistic_style"])
        candidates = sorted(int(key) for key in lsh.query(signature))
        current_is_alive = True
        for candidate in candidates:
            if candidate not in alive:
                continue
            if signature.jaccard(signatures[candidate]) <= JACCARD_THRESHOLD:
                continue
            if rng.randrange(2) == 0:
                lsh.remove(str(candidate))
                alive.remove(candidate)
                del signatures[candidate]
            else:
                current_is_alive = False
                break
        if current_is_alive:
            lsh.insert(str(index), signature)
            signatures[index] = signature
            alive.add(index)

    kept = [row for index, row in enumerate(rows) if index in alive]
    return kept, len(rows) - len(kept)


def _assign_splits(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    # Group identical journalistic texts before splitting so every social variant
    # of the same article is assigned to the same dataset split.
    grouped_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped_indices.setdefault(row["journalistic_style"], []).append(index)

    groups = list(grouped_indices.values())
    random.Random(RANDOM_SEED).shuffle(groups)
    train_target = len(rows) * 0.70
    validation_target = len(rows) * 0.10
    assigned_rows = 0
    split_by_index: dict[int, str] = {}
    for group in groups:
        if assigned_rows < train_target:
            split = "train"
        elif assigned_rows < train_target + validation_target:
            split = "validation"
        else:
            split = "test"
        for index in group:
            split_by_index[index] = split
        assigned_rows += len(group)

    return [dict(row, split=split_by_index[index]) for index, row in enumerate(rows)]


def _remove_train_test_overlap(
    rows: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    # Remove training journalistic texts with MinHash Jaccard > 0.8 relative to the test set.
    test_lsh = MinHashLSH(
        threshold=JACCARD_THRESHOLD, num_perm=MINHASH_PERMUTATIONS
    )
    test_signatures: dict[int, MinHash] = {}
    for index, row in enumerate(rows):
        if row["split"] != "test":
            continue
        signature = _minhash(row["journalistic_style"])
        test_lsh.insert(str(index), signature)
        test_signatures[index] = signature

    kept: list[dict[str, str]] = []
    removed = 0
    for row in rows:
        if row["split"] != "train":
            kept.append(row)
            continue
        signature = _minhash(row["journalistic_style"])
        overlaps_test = any(
            signature.jaccard(test_signatures[int(candidate)]) > JACCARD_THRESHOLD
            for candidate in test_lsh.query(signature)
        )
        if overlaps_test:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = set(TEXT_COLUMNS).difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def process_corpus(language: str) -> Path:
    language = language.upper()
    input_path = INPUT_PATHS[language]
    output_path = OUTPUT_PATHS[language]
    fieldnames, raw_rows = _read_csv(input_path)
    normalizer = _normalizer(language)

    normalized_rows: list[dict[str, str]] = []
    empty_rows = 0
    for row in raw_rows:
        for column in TEXT_COLUMNS:
            row[column] = normalizer(row[column])
        # 15. Discard the entire row when at least one of the two texts is empty.
        if any(not row[column] for column in TEXT_COLUMNS):
            empty_rows += 1
        else:
            normalized_rows.append(row)

    # Apply all corpus-cleaning rules in the stated order.
    rule_removals: Counter[str] = Counter()
    quality_rows: list[dict[str, str]] = []
    for row in normalized_rows:
        failure = _corpus_rule_failure(row, language)
        if failure is None:
            quality_rows.append(row)
        else:
            rule_removals[failure] += 1

    quality_rows, embedding_removals = _filter_by_embeddings(quality_rows)
    quality_rows, exact_duplicate_removals = _remove_exact_duplicates(quality_rows)
    quality_rows, near_duplicate_removals = _remove_near_duplicates(quality_rows)
    split_rows = _assign_splits(quality_rows)
    final_rows, train_test_removals = _remove_train_test_overlap(split_rows)

    _write_csv(output_path, [*fieldnames, "split"], final_rows)

    print(f"Input: {len(raw_rows)} rows")
    print(f"Removed because of empty text: {empty_rows}")
    for reason, count in rule_removals.items():
        print(f"Removed because of {reason}: {count}")
    print(f"Removed because of embedding similarity: {embedding_removals}")
    print(f"Removed as exact duplicates: {exact_duplicate_removals}")
    print(f"Removed as near-duplicates: {near_duplicate_removals}")
    print(f"Removed from training because of test overlap: {train_test_removals}")
    print(f"Output: {len(final_rows)} rows in {output_path}")
    return output_path


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("language", type=str.upper, choices=("IT", "EN"))
    parser.add_argument(
        "--text",
        help="Normalize a single text instead of processing the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()
    if arguments.text is not None:
        print(normalize_text(arguments.text, arguments.language))
    else:
        process_corpus(arguments.language)


if __name__ == "__main__":
    main()
