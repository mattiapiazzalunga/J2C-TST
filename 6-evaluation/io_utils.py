"""I/O and validation helpers shared by J2C-TST evaluation scripts."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


SUBMISSION_COLUMNS = {"row_id", "generated_style"}
REFERENCE_COLUMNS = {"journalistic_style", "conversational_style"}
CORPUS_COLUMNS = REFERENCE_COLUMNS | {"split"}
SYSTEM_COLUMNS = ("model_id", "strategy")


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def normalize_language(language: str) -> str:
    normalized = language.upper()
    if normalized not in {"IT", "EN"}:
        raise ValueError(f"Unsupported language {language!r}; expected IT or EN")
    return normalized


def default_corpus_path(language: str) -> Path:
    language = normalize_language(language)
    return project_root() / "4-official_corpora" / language / f"J2C_TST_{language}.csv"


def load_test_references(path: str | Path) -> dict[str, dict[str, str]]:
    """Index official test rows by their stable zero-based CSV row id."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus not found: {path}")
    references: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = CORPUS_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row_id, row in enumerate(reader):
            if row["split"].strip().lower() != "test":
                continue
            source = row["journalistic_style"].strip()
            reference = row["conversational_style"].strip()
            if not source or not reference:
                raise ValueError(f"Empty source/reference in {path}, row_id={row_id}")
            references[str(row_id)] = {
                "journalistic_style": source,
                "conversational_style": reference,
            }
    if not references:
        raise ValueError(f"No test rows found in {path}")
    return references


def load_predictions(
    path: str | Path,
    *,
    corpus: str | Path | None = None,
    language: str | None = None,
    require_complete: bool = True,
) -> list[dict[str, str]]:
    """Load a submission and, when a corpus is supplied, attach gold texts.

    External systems only need ``row_id`` and ``generated_style``. Optional
    source/reference columns are checked against the official test corpus rather
    than trusted, preventing accidental evaluation on misaligned examples.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Predictions not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = SUBMISSION_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No prediction rows in {path}")

    normalized_language = normalize_language(language) if language else None
    for csv_row, row in enumerate(rows, start=2):
        if row.get("row_id") is None or row.get("generated_style") is None:
            raise ValueError(f"{path}, CSV row {csv_row} has a missing required value")
        row["row_id"] = row["row_id"].strip()
        if not row["row_id"]:
            raise ValueError(f"{path}, CSV row {csv_row} has an empty row_id")
        declared_language = str(row.get("language") or "").strip().upper()
        if normalized_language and declared_language and declared_language != normalized_language:
            raise ValueError(
                f"{path}, row_id={row['row_id']} declares {declared_language}, "
                f"expected {normalized_language}"
            )
        if normalized_language:
            row["language"] = normalized_language

    row_ids = [row["row_id"] for row in rows]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError(f"Duplicate row_id values in {path}")
    for field in SYSTEM_COLUMNS:
        if field not in fieldnames:
            continue
        for row in rows:
            row[field] = str(row.get(field) or "").strip()
        values = {row[field] for row in rows}
        if len(values) > 1:
            raise ValueError(
                f"A submission must contain one consistent {field}; found {sorted(values)}"
            )

    if corpus is None:
        missing = REFERENCE_COLUMNS.difference(rows[0])
        if missing:
            raise ValueError(
                f"A corpus is required for minimal submissions; missing {sorted(missing)}"
            )
        for csv_row, row in enumerate(rows, start=2):
            if not row["journalistic_style"].strip() or not row["conversational_style"].strip():
                raise ValueError(f"{path}, CSV row {csv_row} has an empty source/reference")
        return rows

    references = load_test_references(corpus)
    submitted = set(row_ids)
    unknown = submitted.difference(references)
    if unknown:
        preview = sorted(unknown)[:10]
        raise ValueError(f"Unknown/non-test row_id values in {path}: {preview}")
    missing_ids = set(references).difference(submitted)
    if require_complete and missing_ids:
        preview = sorted(missing_ids, key=int)[:10]
        raise ValueError(
            f"Incomplete test submission: {len(missing_ids)} row_ids are missing; "
            f"first missing ids: {preview}. Use --allow-partial only for debugging."
        )

    for row in rows:
        official = references[row["row_id"]]
        for field in REFERENCE_COLUMNS:
            supplied = row.get(field)
            if supplied is not None and supplied.strip() and supplied.strip() != official[field]:
                raise ValueError(
                    f"{path}, row_id={row['row_id']} has a {field} value that "
                    "does not match the official test corpus"
                )
            row[field] = official[field]
    return rows


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_rows(path: str | Path, rows: list[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
