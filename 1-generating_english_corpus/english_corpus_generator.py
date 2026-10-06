"""High-precision, resumable article/social-post alignment.

The pipeline is intentionally conservative:

* social-to-article alignment is based only on structured URLs and HTTP
  redirects;
* publisher identities are pinned to a Bluesky DID or Mastodon actor URI;
* article text and written-author names are selected from publisher-specific
  DOM containers;
* licenses are assigned from the configured publisher;
* unsupported or ambiguous structures are rejected;
* no linguistic cleaning, Unicode normalization, or semantic matching is
  performed.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import datetime as dt
import email.utils
import ipaddress
import json
import logging
import os
import re
import sqlite3
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
from bs4 import BeautifulSoup, Comment, NavigableString, Tag


MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent

CSV_COLUMNS = (
    "title",
    "author",
    "journalistic_style",
    "link_to_journalistic",
    "conversational_style",
    "link_to_conversational",
    "social_network",
    "license",
)

BSKY_PUBLIC_API = "https://public.api.bsky.app/xrpc"
DEFAULT_TIMEOUT = (20.0, 90.0)
TRANSIENT_SERVER_CODES = {
    403,
    408,
    409,
    423,
    425,
    500,
    502,
    503,
    504,
}
SHORTENER_HOSTS = {
    "bit.ly",
    "buff.ly",
    "europa.eu",
    "ift.tt",
    "link.europa.eu",
    "lnkd.in",
    "t.co",
    "tinyurl.com",
}
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "pk_campaign",
    "pk_content",
    "pk_keyword",
    "pk_medium",
    "pk_source",
}
MEDIA_TAGS = {
    "audio",
    "canvas",
    "iframe",
    "img",
    "picture",
    "script",
    "source",
    "style",
    "svg",
    "video",
}
TEXT_BLOCK_TAGS = {"p", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre"}
LIST_BLOCK_TAGS = {"ul", "ol"}
GLOBAL_VOICES_PARTNER_NOTICE = re.compile(
    r"^(?:"
    r"(?:this|the)\s+"
    r"(?:article|story|post|piece|essay|editorial|report)\b.{0,1200}"
    r"|an edited version\b.{0,600}"
    r"|(?:originally|first)\s+(?:published|appeared)\b.{0,1000}"
    r")"
    r"(?:originally published|first appeared|first published|co-published|"
    r"republished|content-sharing|content partner|content partnership|"
    r"partnership agreement|in partnership with|"
    r"published here through an agreement|agreement between|agreement with)",
    re.IGNORECASE | re.DOTALL,
)
GLOBAL_VOICES_UNWRAPPED_PROMOTION = re.compile(
    r"^This\s+(?:post|article)\s+is\s+(?:also\s+)?part\s+of\s+"
    r"Global\s+Voices[’']\s+.{0,1500}"
    r"\bsupport\s+this\s+coverage\s+by\s+donating\b",
    re.IGNORECASE | re.DOTALL,
)
GLOBAL_VOICES_WRITTEN_CREDIT = re.compile(
    r"^Written(?: \([^)]+\))? by$"
)


class PermanentReject(Exception):
    """A candidate cannot meet the precision contract."""


class TemporarySourceError(Exception):
    """A source failed after bounded server-side retries."""


class SourceDisabled(Exception):
    """A configured social identity is deactivated or no longer verifiable."""


@dataclasses.dataclass(frozen=True)
class AgencySpec:
    key: str
    label: str
    host: str
    path_pattern: re.Pattern[str]
    title_selector: str
    body_selector: str
    exclusion_selectors: tuple[str, ...]
    license_text: str


@dataclasses.dataclass(frozen=True)
class BlueskySource:
    key: str
    agency_key: str
    expected_handle: str
    expected_did: str
    allow_missing_language: bool = False
    page_size: int = 100

    @property
    def social_network(self) -> str:
        return "Bluesky"


@dataclasses.dataclass(frozen=True)
class MastodonSource:
    key: str
    agency_key: str
    base_url: str
    account_name: str
    expected_actor_uri: str
    expected_profile_url: str
    expected_website_url: str
    page_size: int = 40

    @property
    def social_network(self) -> str:
        return "Mastodon"


SocialSource = BlueskySource | MastodonSource


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    selected_agencies: tuple[str, ...] = ("global_voices",)
    checkpoint_db: Path = (
        MODULE_DIR / "state" / "english_corpus_checkpoint.sqlite3"
    )
    output_csv: Path = (
        PROJECT_ROOT / "2-raw_data" / "EN" / "raw_dataset_EN.csv"
    )
    max_pages_per_source: int | None = 25
    max_redirects: int = 10
    server_error_retries: int = 8
    network_wait_initial_seconds: float = 5.0
    network_wait_max_seconds: float = 300.0
    verbose: bool = True


@dataclasses.dataclass(frozen=True)
class SocialPost:
    source_key: str
    agency_key: str
    network: str
    post_uri: str
    post_url: str
    text: str
    links: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Article:
    agency_key: str
    canonical_url: str
    title: str
    author: str
    body: str
    license: str


AGENCIES: dict[str, AgencySpec] = {
    "global_voices": AgencySpec(
        key="global_voices",
        label="Global Voices",
        host="globalvoices.org",
        path_pattern=re.compile(
            r"^/[0-9]{4}/[0-9]{2}/[0-9]{2}/[^/]+/?$"
        ),
        title_selector=(
            "body.single-post .post-header "
            "h2.screen-title.post-title > a[rel='bookmark']"
        ),
        body_selector=(
            "body.single-post div.post.type-post.status-publish "
            "> div.entry-container > div.entry"
        ),
        exclusion_selectors=(
            ".wp-caption",
            ".captioned-image-container",
            "figure",
            "figcaption",
            ".factbox",
            ".notes",
            "blockquote.twitter-tweet",
            "blockquote.tiktok-embed",
            "blockquote.instagram-media",
            "blockquote.bluesky-embed",
            (
                "p:has(> em > "
                "a[href^='https://globalvoices.org/special/'])"
            ),
            "p:has(> em > a[href^='/special/'])",
        ),
        license_text="CC BY 3.0",
    ),
}


SOCIAL_SOURCES: tuple[SocialSource, ...] = (
    BlueskySource(
        key="global_voices_bluesky",
        agency_key="global_voices",
        expected_handle="globalvoices.org",
        expected_did="did:plc:5x3a6jke5p3ni7mnpd57ijhw",
        allow_missing_language=True,
    ),
    MastodonSource(
        key="global_voices_mastodon",
        agency_key="global_voices",
        base_url="https://mstdn.social",
        account_name="globalvoices",
        expected_actor_uri="https://mstdn.social/users/globalvoices",
        expected_profile_url="https://mstdn.social/@globalvoices",
        expected_website_url="https://globalvoices.org/",
    ),
)


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _element_text(element: Tag) -> str:
    """Project one selected element to text without linguistic normalization."""

    clone = copy.deepcopy(element)
    for media in clone.find_all(MEDIA_TAGS):
        media.decompose()
    for br in clone.find_all("br"):
        br.replace_with("\n")
    return clone.get_text(separator="", strip=False).strip()


def _serialize_body(
    root: Tag,
    exclusion_selectors: Sequence[str],
) -> str:
    """Serialize only supported direct body blocks.

    Blank lines represent DOM block boundaries. No case folding, whitespace
    collapsing, punctuation editing, or Unicode normalization is applied.
    """

    clone = copy.deepcopy(root)
    for selector in exclusion_selectors:
        for node in clone.select(selector):
            node.decompose()
    for media in clone.find_all(MEDIA_TAGS):
        media.decompose()

    blocks: list[str] = []
    for child in clone.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if str(child).strip():
                raise PermanentReject(
                    "The article body contains unsupported direct text."
                )
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name.lower()
        if name in TEXT_BLOCK_TAGS:
            text = _element_text(child)
            if text:
                blocks.append(text)
            continue
        if name in LIST_BLOCK_TAGS:
            items = child.find_all("li", recursive=False)
            if not items:
                raise PermanentReject("A list has no direct list items.")
            item_texts = [_element_text(item) for item in items]
            if any(not item for item in item_texts):
                raise PermanentReject("A list contains an empty item.")
            blocks.append("\n".join(item_texts))
            continue
        if _element_text(child):
            raise PermanentReject(
                f"Unsupported visible body block: <{child.name}>."
            )

    body = "\n\n".join(blocks)
    if len(body) < 100:
        raise PermanentReject("The structurally selected article body is empty or too short.")
    if "\x00" in body:
        raise PermanentReject("The article body contains a NUL character.")
    return body


def _serialize_mastodon_content(content_html: str) -> str:
    """Project Mastodon HTML to its exact visible plain-text structure."""

    soup = BeautifulSoup(content_html, "html.parser")
    if soup.select("img.emoji"):
        raise PermanentReject(
            "A Mastodon custom emoji cannot be recovered as exact plain text."
        )
    paragraphs: list[str] = []
    for child in soup.children:
        if isinstance(child, Comment):
            continue
        if isinstance(child, NavigableString):
            if str(child).strip():
                raise PermanentReject(
                    "The Mastodon post has unsupported direct text."
                )
            continue
        if not isinstance(child, Tag):
            continue
        if child.name != "p":
            if _element_text(child):
                raise PermanentReject(
                    f"Unsupported Mastodon content block: <{child.name}>."
                )
            continue
        text = _element_text(child)
        if text:
            paragraphs.append(text)
    result = "\n\n".join(paragraphs)
    if not result or "\x00" in result:
        raise PermanentReject(
            "The Mastodon post does not contain exact usable text."
        )
    return result


def _is_tracking_query(query: str) -> bool:
    if not query:
        return True
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    if not pairs:
        return False
    return all(
        key.lower().startswith("utm_") or key.lower() in TRACKING_QUERY_KEYS
        for key, _ in pairs
    )


def _path_identity(url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    return scheme, host, path


def _validate_public_http_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise PermanentReject("Only HTTP(S) links are accepted.")
    if not parsed.hostname or parsed.username or parsed.password:
        raise PermanentReject("The URL authority is not acceptable.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise PermanentReject("The URL port is malformed.") from exc
    if port not in {None, 80, 443}:
        raise PermanentReject("Non-standard URL ports are not accepted.")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise PermanentReject("Local URLs are not accepted.")
    try:
        literal_ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return
    if not literal_ip.is_global:
        raise PermanentReject("Non-public IP addresses are not accepted.")


def classify_article_url(url: str) -> AgencySpec | None:
    """Classify an exact English article path; do not inspect its text."""

    _validate_public_http_url(url)
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme.lower() != "https":
        return None
    if not _is_tracking_query(parsed.query):
        return None
    host = (parsed.hostname or "").lower().rstrip(".")
    for agency in AGENCIES.values():
        if host == agency.host and agency.path_pattern.fullmatch(parsed.path):
            return agency
    return None


class PrecisionHTTP:
    """HTTP client with host pacing and connection-loss waiting."""

    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "precision-news-pair-dataset/1.0 "
                    "(research notebook; contact configured by operator)"
                ),
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._last_request_by_host: dict[str, float] = {}
        self._minimum_interval = {
            "globalvoices.org": 0.75,
            "mstdn.social": 0.35,
            "projects.research-and-innovation.ec.europa.eu": 1.0,
            "public.api.bsky.app": 0.15,
            "ift.tt": 0.25,
            "link.europa.eu": 0.25,
        }

    def _pace(self, url: str) -> None:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
        interval = self._minimum_interval.get(host, 0.1)
        last = self._last_request_by_host.get(host)
        if last is not None:
            remaining = interval - (time.monotonic() - last)
            if remaining > 0:
                time.sleep(remaining)
        self._last_request_by_host[host] = time.monotonic()

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
        stream: bool = False,
        accept: str | None = None,
    ) -> requests.Response:
        connection_delay = self.config.network_wait_initial_seconds
        server_attempt = 0

        while True:
            self._pace(url)
            headers = {"Accept": accept} if accept else None
            try:
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    headers=headers,
                    timeout=DEFAULT_TIMEOUT,
                    allow_redirects=False,
                    stream=stream,
                )
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.ContentDecodingError,
            ) as exc:
                self.logger.warning(
                    "Connection unavailable for %s (%s). Waiting %.1f seconds; "
                    "Ctrl+C is safe.",
                    url,
                    type(exc).__name__,
                    connection_delay,
                )
                time.sleep(connection_delay)
                connection_delay = min(
                    connection_delay * 2,
                    self.config.network_wait_max_seconds,
                )
                continue

            if response.status_code == 429:
                delay = self._retry_after_seconds(response)
                response.close()
                self.logger.warning(
                    "Rate limited by %s. Waiting %.1f seconds.",
                    urllib.parse.urlsplit(url).hostname,
                    delay,
                )
                time.sleep(delay)
                continue

            if response.status_code in TRANSIENT_SERVER_CODES:
                server_attempt += 1
                if server_attempt > self.config.server_error_retries:
                    status = response.status_code
                    response.close()
                    raise TemporarySourceError(
                        f"{url} repeatedly returned HTTP {status}."
                    )
                delay = min(2 ** (server_attempt - 1), 60)
                response.close()
                self.logger.warning(
                    "%s returned a temporary server error. Retry %d/%d in %d seconds.",
                    url,
                    server_attempt,
                    self.config.server_error_retries,
                    delay,
                )
                time.sleep(delay)
                continue

            return response

    def _retry_after_seconds(self, response: requests.Response) -> float:
        value = response.headers.get("Retry-After")
        if value:
            try:
                return max(1.0, min(float(value), 3600.0))
            except ValueError:
                try:
                    retry_at = email.utils.parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=dt.timezone.utc)
                    seconds = (
                        retry_at - dt.datetime.now(dt.timezone.utc)
                    ).total_seconds()
                    return max(1.0, min(seconds, 3600.0))
                except (TypeError, ValueError):
                    pass
        return min(self.config.network_wait_max_seconds, 60.0)

    def resolve_short_url(
        self,
        url: str,
        *,
        max_redirects: int,
    ) -> str:
        current = url
        visited: set[str] = set()
        for _ in range(max_redirects + 1):
            _validate_public_http_url(current)
            if current in visited:
                raise PermanentReject("The redirect chain contains a loop.")
            visited.add(current)
            response = self.request(
                "GET",
                current,
                stream=True,
                accept="text/html,application/xhtml+xml,*/*;q=0.1",
            )
            try:
                if response.is_redirect or response.is_permanent_redirect:
                    location = response.headers.get("Location")
                    if not location:
                        raise PermanentReject(
                            "A redirect response has no Location header."
                        )
                    current = urllib.parse.urljoin(current, location)
                    continue
                if response.status_code != 200:
                    raise PermanentReject(
                        f"The structured link returned HTTP {response.status_code}."
                    )
                return response.url
            finally:
                response.close()
        raise PermanentReject("The redirect chain exceeds the configured limit.")

    def fetch_html(self, url: str, *, max_redirects: int) -> tuple[str, bytes, str]:
        current = url
        visited: set[str] = set()
        for _ in range(max_redirects + 1):
            _validate_public_http_url(current)
            if current in visited:
                raise PermanentReject("The article redirect chain contains a loop.")
            visited.add(current)
            response = self.request(
                "GET",
                current,
                accept="text/html,application/xhtml+xml",
            )
            if response.is_redirect or response.is_permanent_redirect:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise PermanentReject(
                        "An article redirect has no Location header."
                    )
                current = urllib.parse.urljoin(current, location)
                continue
            try:
                if response.status_code != 200:
                    raise PermanentReject(
                        f"The article returned HTTP {response.status_code}."
                    )
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type.lower():
                    raise PermanentReject(
                        f"The article response is not HTML: {content_type!r}."
                    )
                return response.url, response.content, response.encoding or "utf-8"
            finally:
                response.close()
        raise PermanentReject("The article redirect chain exceeds the configured limit.")


class CheckpointStore:
    """SQLite checkpoint and accepted-row store."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS processed_posts (
                source_key TEXT NOT NULL,
                post_uri TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reason TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (source_key, post_uri)
            );

            CREATE TABLE IF NOT EXISTS crawl_state (
                source_key TEXT PRIMARY KEY,
                cursor TEXT,
                status TEXT NOT NULL,
                detail TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS accepted_rows (
                news_agency TEXT NOT NULL,
                title TEXT NOT NULL,
                author TEXT NOT NULL,
                journalistic_style TEXT NOT NULL,
                link_to_journalistic TEXT NOT NULL,
                conversational_style TEXT NOT NULL,
                link_to_conversational TEXT NOT NULL,
                social_network TEXT NOT NULL,
                license TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                PRIMARY KEY (
                    link_to_journalistic,
                    link_to_conversational
                )
            );
            """
        )
        accepted_columns = {
            row["name"]
            for row in self.connection.execute(
                "PRAGMA table_info(accepted_rows)"
            )
        }
        if "author" not in accepted_columns:
            self.connection.execute(
                "ALTER TABLE accepted_rows "
                "ADD COLUMN author TEXT NOT NULL DEFAULT ''"
            )
        if "license" not in accepted_columns:
            self.connection.execute(
                "ALTER TABLE accepted_rows "
                "ADD COLUMN license TEXT NOT NULL DEFAULT ''"
            )
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def is_processed(self, source_key: str, post_uri: str) -> bool:
        row = self.connection.execute(
            """
            SELECT 1 FROM processed_posts
            WHERE source_key = ? AND post_uri = ?
            """,
            (source_key, post_uri),
        ).fetchone()
        return row is not None

    def mark_processed(
        self,
        source_key: str,
        post_uri: str,
        outcome: str,
        reason: str | None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO processed_posts
                (source_key, post_uri, outcome, reason, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_key, post_uri) DO UPDATE SET
                outcome = excluded.outcome,
                reason = excluded.reason,
                updated_at = excluded.updated_at
            """,
            (source_key, post_uri, outcome, reason, _utc_now()),
        )
        self.connection.commit()

    def insert_row(self, article: Article, post: SocialPost) -> None:
        agency = AGENCIES[article.agency_key]
        self.connection.execute(
            """
            INSERT OR IGNORE INTO accepted_rows (
                news_agency,
                title,
                author,
                journalistic_style,
                link_to_journalistic,
                conversational_style,
                link_to_conversational,
                social_network,
                license,
                inserted_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agency.label,
                article.title,
                article.author,
                article.body,
                article.canonical_url,
                post.text,
                post.post_url,
                post.network,
                article.license,
                _utc_now(),
            ),
        )
        self.connection.commit()

    def articles_missing_metadata(self) -> list[sqlite3.Row]:
        return list(
            self.connection.execute(
                """
                SELECT DISTINCT news_agency, link_to_journalistic
                FROM accepted_rows
                WHERE author = '' OR license = ''
                ORDER BY news_agency, link_to_journalistic
                """
            )
        )

    def update_article_metadata(
        self,
        existing_url: str,
        article: Article,
    ) -> None:
        agency = AGENCIES[article.agency_key]
        self.connection.execute(
            """
            UPDATE accepted_rows
            SET
                news_agency = ?,
                title = ?,
                author = ?,
                journalistic_style = ?,
                license = ?
            WHERE link_to_journalistic = ?
            """,
            (
                agency.label,
                article.title,
                article.author,
                article.body,
                article.license,
                existing_url,
            ),
        )
        self.connection.commit()

    def delete_article_rows(self, canonical_url: str) -> None:
        self.connection.execute(
            "DELETE FROM accepted_rows WHERE link_to_journalistic = ?",
            (canonical_url,),
        )
        self.connection.commit()

    def get_state(self, source_key: str) -> tuple[str | None, str | None]:
        row = self.connection.execute(
            "SELECT cursor, status FROM crawl_state WHERE source_key = ?",
            (source_key,),
        ).fetchone()
        if row is None:
            return None, None
        return row["cursor"], row["status"]

    def set_state(
        self,
        source_key: str,
        cursor: str | None,
        status: str,
        detail: str | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO crawl_state
                (source_key, cursor, status, detail, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source_key) DO UPDATE SET
                cursor = excluded.cursor,
                status = excluded.status,
                detail = excluded.detail,
                updated_at = excluded.updated_at
            """,
            (source_key, cursor, status, detail, _utc_now()),
        )
        self.connection.commit()

    def retain_active_scope(self) -> None:
        """Remove checkpoint data belonging to retired collection sources."""

        active_source_keys = tuple(source.key for source in SOCIAL_SOURCES)
        placeholders = ",".join("?" for _ in active_source_keys)
        self.connection.execute(
            f"DELETE FROM processed_posts WHERE source_key NOT IN ({placeholders})",
            active_source_keys,
        )
        self.connection.execute(
            f"DELETE FROM crawl_state WHERE source_key NOT IN ({placeholders})",
            active_source_keys,
        )
        self.connection.execute(
            """
            DELETE FROM accepted_rows
            WHERE news_agency != ?
               OR social_network NOT IN (?, ?)
            """,
            (
                AGENCIES["global_voices"].label,
                "Bluesky",
                "Mastodon",
            ),
        )
        self.connection.commit()

    def export_csv(
        self,
        path: Path,
        agency_keys: Sequence[str] | None = None,
    ) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        parameters: tuple[str, ...] = ()
        agency_filter = ""
        if agency_keys is not None:
            labels = tuple(AGENCIES[key].label for key in agency_keys)
            if not labels:
                raise ValueError("At least one agency is required for export.")
            placeholders = ",".join("?" for _ in labels)
            agency_filter = f" AND news_agency IN ({placeholders})"
            parameters = labels
        rows = self.connection.execute(
            f"""
            SELECT
                title,
                author,
                journalistic_style,
                link_to_journalistic,
                conversational_style,
                link_to_conversational,
                social_network,
                license
            FROM accepted_rows
            WHERE author <> '' AND license <> ''
            {agency_filter}
            ORDER BY
                news_agency,
                link_to_journalistic,
                social_network,
                link_to_conversational
            """,
            parameters,
        )
        count = 0
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=CSV_COLUMNS,
                quoting=csv.QUOTE_ALL,
                lineterminator="\n",
            )
            writer.writeheader()
            for row in rows:
                writer.writerow({column: row[column] for column in CSV_COLUMNS})
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return count

    def summary(self) -> dict[str, Any]:
        accepted = self.connection.execute(
            """
            SELECT COUNT(*) AS n
            FROM accepted_rows
            WHERE author <> '' AND license <> ''
            """
        ).fetchone()["n"]
        outcomes = {
            row["outcome"]: row["n"]
            for row in self.connection.execute(
                """
                SELECT outcome, COUNT(*) AS n
                FROM processed_posts
                GROUP BY outcome
                """
            )
        }
        states = [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT source_key, cursor, status, detail, updated_at
                FROM crawl_state
                ORDER BY source_key
                """
            )
        ]
        return {
            "accepted_rows": accepted,
            "processed_outcomes": outcomes,
            "source_states": states,
        }


def _require_single(soup: BeautifulSoup, selector: str, name: str) -> Tag:
    nodes = soup.select(selector)
    if len(nodes) != 1:
        raise PermanentReject(
            f"Expected exactly one {name}; found {len(nodes)}."
        )
    return nodes[0]


def _extract_author(
    soup: BeautifulSoup,
    agency: AgencySpec,
) -> str:
    """Select exact publisher-displayed author names by credit structure."""

    credit = _require_single(
        soup,
        (
            "div.postfooter div.postfooter-credits "
            "div.avatar-credits-container"
        ),
        "Global Voices author credit",
    )
    contributors = credit.select(":scope > div.contributor")
    if not contributors:
        raise PermanentReject(
            "The Global Voices author credit has no contributors."
        )
    authors: list[str] = []
    for contributor in contributors:
        labels = contributor.select(
            ":scope > div.contributor-name > span.credit-label"
        )
        if len(labels) != 1:
            raise PermanentReject(
                "A Global Voices contributor has an ambiguous role."
            )
        role = _element_text(labels[0])
        if not GLOBAL_VOICES_WRITTEN_CREDIT.fullmatch(role):
            continue
        links = contributor.select(
            ":scope > div.contributor-name "
            "> a.user-link[href*='/author/']"
        )
        if len(links) != 1:
            raise PermanentReject(
                "A Global Voices written credit has an ambiguous author."
            )
        name = _element_text(links[0])
        if not name or "\x00" in name:
            raise PermanentReject(
                "A Global Voices written author name is unusable."
            )
        authors.append(name)
    if not authors:
        raise PermanentReject(
            "The Global Voices page has no written-author credit."
        )
    return "\n".join(authors)


def _validate_owner(
    soup: BeautifulSoup,
    agency: AgencySpec,
    effective_url: str,
) -> None:
    _require_single(
        soup,
        (
            "body.single-post.wp-theme-gv-theme."
            "wp-child-theme-gv-news-child-theme"
        ),
        "Global Voices page template",
    )
    site_name = _require_single(
        soup,
        'meta[property="og:site_name"][content]',
        "Global Voices publisher marker",
    )
    if site_name.get("content") != "Global Voices":
        raise PermanentReject(
            "The Global Voices publisher marker is absent."
        )
    post = _require_single(
        soup,
        "body.single-post div.post.type-post.status-publish",
        "Global Voices post container",
    )
    if "category-announcements" in (post.get("class") or []):
        raise PermanentReject(
            "Global Voices announcement pages are outside the news scope."
        )
    _require_single(
        soup,
        ".post-header .post-header-meta.postmeta-container",
        "Global Voices header attribution",
    )
    credit = _require_single(
        soup,
        (
            "div.postfooter div.postfooter-credits "
            "div.avatar-credits-container"
        ),
        "Global Voices author credit",
    )
    author_links = [
        anchor.get("href")
        for anchor in credit.select('a[href*="/author/"]')
    ]
    if not author_links:
        raise PermanentReject(
            "The Global Voices author profile marker is absent."
        )
    for link in author_links:
        parsed = urllib.parse.urlsplit(
            urllib.parse.urljoin(effective_url, str(link))
        )
        host = (parsed.hostname or "").lower().rstrip(".")
        if host != "globalvoices.org" and not host.endswith(
            ".globalvoices.org"
        ):
            raise PermanentReject(
                "A Global Voices author marker leaves the publisher domain."
            )
        if parsed.path.rstrip("/") == "/author/guest-contributor":
            raise PermanentReject(
                "Global Voices identifies the page as a guest "
                "contribution."
            )

    title_link = _require_single(
        soup,
        agency.title_selector,
        "Global Voices title link",
    )
    title_href = urllib.parse.urljoin(
        effective_url,
        str(title_link.get("href") or ""),
    )
    if _path_identity(title_href) != _path_identity(effective_url):
        raise PermanentReject(
            "The Global Voices title link identifies a different page."
        )

    body = _require_single(
        soup,
        agency.body_selector,
        "Global Voices article body",
    )
    ownership_texts: list[str] = []
    for child in body.find_all(recursive=False):
        name = child.name.lower()
        if name in TEXT_BLOCK_TAGS:
            ownership_texts.append(_element_text(child))
        elif name in LIST_BLOCK_TAGS:
            ownership_texts.extend(
                _element_text(item)
                for item in child.find_all("li", recursive=False)
            )
    for text in ownership_texts:
        if GLOBAL_VOICES_PARTNER_NOTICE.search(text):
            raise PermanentReject(
                "The Global Voices page explicitly identifies partner "
                "or republished content."
            )
        if GLOBAL_VOICES_UNWRAPPED_PROMOTION.search(text):
            raise PermanentReject(
                "The Global Voices body contains an unwrapped "
                "promotional call to action."
            )


def extract_article(
    agency: AgencySpec,
    effective_url: str,
    payload: bytes,
) -> Article:
    soup = BeautifulSoup(payload, "html.parser")
    if soup.html is None or soup.html.get("lang", "").lower() != "en":
        raise PermanentReject("The article page is not structurally marked as English.")

    canonical_nodes = soup.select('link[rel~="canonical"][href]')
    if len(canonical_nodes) != 1:
        raise PermanentReject(
            f"Expected exactly one canonical URL; found {len(canonical_nodes)}."
        )
    canonical = urllib.parse.urljoin(
        effective_url,
        canonical_nodes[0]["href"],
    )
    canonical_agency = classify_article_url(canonical)
    if canonical_agency is None or canonical_agency.key != agency.key:
        raise PermanentReject(
            "The canonical URL is not an accepted English article path."
        )
    if _path_identity(canonical) != _path_identity(effective_url):
        raise PermanentReject(
            "The effective URL and canonical URL do not identify the same page."
        )

    _validate_owner(soup, agency, effective_url)
    title_node = _require_single(soup, agency.title_selector, "article title")
    body_node = _require_single(soup, agency.body_selector, "article body")
    title = _element_text(title_node)
    if not title or "\x00" in title:
        raise PermanentReject("The structurally selected title is unusable.")
    author = _extract_author(soup, agency)
    body = _serialize_body(
        body_node,
        agency.exclusion_selectors,
    )
    if body == title:
        raise PermanentReject("The selected article body duplicates the title.")
    return Article(
        agency_key=agency.key,
        canonical_url=canonical,
        title=title,
        author=author,
        body=body,
        license=agency.license_text,
    )


def _json_response(response: requests.Response, context: str) -> Any:
    try:
        return response.json()
    except (requests.JSONDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TemporarySourceError(
            f"{context} did not return valid JSON."
        ) from exc


def _bluesky_links(record: Mapping[str, Any]) -> tuple[str, ...]:
    text = record.get("text")
    if not isinstance(text, str):
        raise PermanentReject("The Bluesky record has no plain text.")
    encoded = text.encode("utf-8")
    links: list[str] = []
    for facet in record.get("facets") or []:
        index = facet.get("index") or {}
        start = index.get("byteStart")
        end = index.get("byteEnd")
        if not isinstance(start, int) or not isinstance(end, int):
            raise PermanentReject("A Bluesky facet has invalid byte indices.")
        try:
            label = encoded[start:end].decode("utf-8")
        except (UnicodeDecodeError, IndexError) as exc:
            raise PermanentReject(
                "A Bluesky facet does not align to UTF-8 boundaries."
            ) from exc
        if not label:
            raise PermanentReject("A Bluesky facet has an empty label.")
        for feature in facet.get("features") or []:
            if feature.get("$type") == "app.bsky.richtext.facet#link":
                uri = feature.get("uri")
                if not isinstance(uri, str):
                    raise PermanentReject("A Bluesky link facet has no URI.")
                links.append(uri)

    embed = record.get("embed") or {}
    embed_type = embed.get("$type")
    if embed_type in {
        "app.bsky.embed.record",
        "app.bsky.embed.recordWithMedia",
    }:
        raise PermanentReject("Quoted Bluesky posts are excluded.")
    if embed_type == "app.bsky.embed.external":
        uri = (embed.get("external") or {}).get("uri")
        if isinstance(uri, str):
            links.append(uri)
    return _unique_in_order(links)


def extract_bluesky_post(
    source: BlueskySource,
    feed_item: Mapping[str, Any],
    current_handle: str,
) -> SocialPost:
    if feed_item.get("reason"):
        raise PermanentReject("Bluesky reposts are excluded.")
    post = feed_item.get("post") or {}
    author = post.get("author") or {}
    if author.get("did") != source.expected_did:
        raise PermanentReject("The Bluesky post author DID does not match.")
    record = post.get("record") or {}
    if record.get("$type") != "app.bsky.feed.post":
        raise PermanentReject("The Bluesky record type is not a post.")
    if record.get("reply"):
        raise PermanentReject("Bluesky replies are excluded.")
    languages = record.get("langs") or []
    if "en" not in languages and not (
        source.allow_missing_language and not languages
    ):
        raise PermanentReject("The Bluesky post is not explicitly marked as English.")
    text = record.get("text")
    if not isinstance(text, str) or not text or "\x00" in text:
        raise PermanentReject("The Bluesky post has no exact usable text.")
    post_uri = post.get("uri")
    prefix = f"at://{source.expected_did}/app.bsky.feed.post/"
    if not isinstance(post_uri, str) or not post_uri.startswith(prefix):
        raise PermanentReject("The Bluesky post URI is malformed.")
    rkey = post_uri[len(prefix) :]
    if not re.fullmatch(r"[a-z2-7]+", rkey):
        raise PermanentReject("The Bluesky record key is malformed.")
    post_url = (
        "https://bsky.app/profile/"
        + urllib.parse.quote(current_handle, safe=".")
        + "/post/"
        + rkey
    )
    return SocialPost(
        source_key=source.key,
        agency_key=source.agency_key,
        network=source.social_network,
        post_uri=post_uri,
        post_url=post_url,
        text=text,
        links=_bluesky_links(record),
    )


def _mastodon_links(status: Mapping[str, Any]) -> tuple[str, ...]:
    soup = BeautifulSoup(status.get("content") or "", "html.parser")
    links: list[str] = []
    for anchor in soup.select("a[href]"):
        classes = set(anchor.get("class") or [])
        if "mention" in classes or "hashtag" in classes:
            continue
        href = anchor.get("href")
        if isinstance(href, str):
            links.append(href)
    card_url = (status.get("card") or {}).get("url")
    if isinstance(card_url, str):
        links.append(card_url)
    return _unique_in_order(links)


def extract_mastodon_post(
    source: MastodonSource,
    status: Mapping[str, Any],
) -> SocialPost:
    account = status.get("account") or {}
    if account.get("uri") != source.expected_actor_uri:
        raise PermanentReject("The Mastodon actor URI does not match.")
    if status.get("reblog") is not None:
        raise PermanentReject("Mastodon boosts are excluded.")
    if status.get("in_reply_to_id") is not None:
        raise PermanentReject("Mastodon replies are excluded.")
    if status.get("quote") is not None or status.get("quote_id") is not None:
        raise PermanentReject("Quoted Mastodon posts are excluded.")
    if status.get("spoiler_text"):
        raise PermanentReject("Mastodon content-warning posts are excluded.")
    if status.get("language") != "en":
        raise PermanentReject(
            "The Mastodon post is not explicitly marked as English."
        )
    post_uri = status.get("uri")
    post_url = status.get("url")
    if not isinstance(post_uri, str) or not post_uri.startswith(
        source.expected_actor_uri + "/statuses/"
    ):
        raise PermanentReject("The Mastodon status URI is malformed.")
    if not isinstance(post_url, str) or not post_url.startswith(
        source.expected_profile_url + "/"
    ):
        raise PermanentReject("The Mastodon public status URL is malformed.")
    return SocialPost(
        source_key=source.key,
        agency_key=source.agency_key,
        network=source.social_network,
        post_uri=post_uri,
        post_url=post_url,
        text=_serialize_mastodon_content(status.get("content") or ""),
        links=_mastodon_links(status),
    )


class PrecisionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger("english_corpus_generator")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if config.verbose else logging.WARNING)
        self.http = PrecisionHTTP(config, self.logger)
        self.store = CheckpointStore(config.checkpoint_db)
        self.store.retain_active_scope()

    def close(self) -> None:
        self.store.close()

    def run(self) -> dict[str, Any]:
        unknown = set(self.config.selected_agencies) - set(AGENCIES)
        if unknown:
            raise ValueError(f"Unknown agency keys: {sorted(unknown)}")
        selected_sources = [
            source
            for source in SOCIAL_SOURCES
            if source.agency_key in self.config.selected_agencies
        ]
        try:
            self._backfill_article_metadata()
            for source in selected_sources:
                if isinstance(source, BlueskySource):
                    self._crawl_bluesky(source)
                else:
                    self._crawl_mastodon(source)
        except KeyboardInterrupt:
            self.logger.warning(
                "Interrupted by the operator. The current page cursor is checkpointed."
            )
            raise
        finally:
            count = self.store.export_csv(
                self.config.output_csv,
                self.config.selected_agencies,
            )
            self.logger.info(
                "Exported %d accepted rows to %s.",
                count,
                self.config.output_csv,
            )
        return self.store.summary()

    def _backfill_article_metadata(self) -> None:
        """Upgrade legacy accepted rows through current live extraction."""

        by_label = {agency.label: agency for agency in AGENCIES.values()}
        for row in self.store.articles_missing_metadata():
            existing_url = row["link_to_journalistic"]
            agency = by_label.get(row["news_agency"])
            if agency is None:
                self.store.delete_article_rows(existing_url)
                continue
            try:
                effective_url, payload, _ = self.http.fetch_html(
                    existing_url,
                    max_redirects=self.config.max_redirects,
                )
                effective_agency = classify_article_url(effective_url)
                if (
                    effective_agency is None
                    or effective_agency.key != agency.key
                ):
                    raise PermanentReject(
                        "The legacy article left its configured publisher."
                    )
                article = extract_article(
                    agency,
                    effective_url,
                    payload,
                )
            except PermanentReject as exc:
                self.store.delete_article_rows(existing_url)
                self.logger.warning(
                    "Removed legacy rows for %s during metadata upgrade: %s",
                    existing_url,
                    exc,
                )
                continue
            self.store.update_article_metadata(existing_url, article)

    def _verify_bluesky(self, source: BlueskySource) -> str:
        response = self.http.request(
            "GET",
            f"{BSKY_PUBLIC_API}/app.bsky.actor.getProfile",
            params={"actor": source.expected_did},
            accept="application/json",
        )
        data = _json_response(response, "Bluesky profile")
        response.close()
        if response.status_code == 400 and data.get("error") in {
            "AccountDeactivated",
            "AccountTakedown",
        }:
            raise SourceDisabled(
                f"Bluesky identity {source.expected_handle} is {data.get('error')}."
            )
        if response.status_code != 200:
            raise SourceDisabled(
                f"Bluesky profile verification returned HTTP {response.status_code}."
            )
        if data.get("did") != source.expected_did:
            raise SourceDisabled("The pinned Bluesky DID no longer matches.")
        if data.get("handle") != source.expected_handle:
            raise SourceDisabled(
                "The Bluesky handle changed; manual ownership review is required."
            )
        return data["handle"]

    def _crawl_bluesky(self, source: BlueskySource) -> None:
        self.logger.info("Verifying Bluesky source %s.", source.expected_handle)
        try:
            current_handle = self._verify_bluesky(source)
        except SourceDisabled as exc:
            self.store.set_state(source.key, None, "unavailable", str(exc))
            self.logger.warning("%s", exc)
            return
        except TemporarySourceError as exc:
            saved_cursor, _ = self.store.get_state(source.key)
            self.store.set_state(
                source.key,
                saved_cursor,
                "paused",
                str(exc),
            )
            self.logger.warning(
                "Paused %s during identity verification: %s",
                source.key,
                exc,
            )
            return

        saved_cursor, saved_status = self.store.get_state(source.key)
        cursor = (
            saved_cursor
            if saved_status in {"running", "paused"}
            else None
        )
        incremental_scan = saved_status == "complete"
        pages = 0
        while True:
            page_cursor = cursor
            params: dict[str, Any] = {
                "actor": source.expected_did,
                "limit": source.page_size,
                "filter": "posts_no_replies",
                "includePins": "false",
            }
            if cursor:
                params["cursor"] = cursor
            self.store.set_state(source.key, page_cursor, "running")
            try:
                response = self.http.request(
                    "GET",
                    f"{BSKY_PUBLIC_API}/app.bsky.feed.getAuthorFeed",
                    params=params,
                    accept="application/json",
                )
                data = _json_response(response, "Bluesky author feed")
                response.close()
                if response.status_code != 200:
                    error = data.get("error")
                    if error in {"AccountDeactivated", "AccountTakedown"}:
                        raise SourceDisabled(
                            f"Bluesky source became unavailable: {error}."
                        )
                    raise TemporarySourceError(
                        f"Bluesky author feed returned HTTP {response.status_code}."
                    )
                feed = data.get("feed")
                if not isinstance(feed, list):
                    raise TemporarySourceError("Bluesky feed is not a list.")
                new_posts = 0
                for item in feed:
                    post = (item.get("post") or {})
                    post_uri = post.get("uri")
                    if not isinstance(post_uri, str):
                        continue
                    if self.store.is_processed(source.key, post_uri):
                        continue
                    new_posts += 1
                    self._process_social_item(
                        source,
                        post_uri,
                        lambda item=item: extract_bluesky_post(
                            source,
                            item,
                            current_handle,
                        ),
                    )
                cursor = data.get("cursor")
                pages += 1
                if not cursor or not feed:
                    self.store.set_state(source.key, None, "complete")
                    break
                if incremental_scan and new_posts == 0:
                    self.store.set_state(source.key, None, "complete")
                    break
                self.store.set_state(source.key, cursor, "paused")
                if (
                    self.config.max_pages_per_source is not None
                    and pages >= self.config.max_pages_per_source
                ):
                    break
            except KeyboardInterrupt:
                self.store.set_state(source.key, page_cursor, "paused")
                raise
            except SourceDisabled as exc:
                self.store.set_state(source.key, page_cursor, "unavailable", str(exc))
                self.logger.warning("%s", exc)
                break
            except TemporarySourceError as exc:
                self.store.set_state(source.key, page_cursor, "paused", str(exc))
                self.logger.warning(
                    "Paused %s at its current page: %s",
                    source.key,
                    exc,
                )
                break

    def _verify_mastodon(self, source: MastodonSource) -> str:
        response = self.http.request(
            "GET",
            f"{source.base_url}/api/v1/accounts/lookup",
            params={"acct": source.account_name},
            accept="application/json",
        )
        data = _json_response(response, "Mastodon account lookup")
        response.close()
        if response.status_code != 200:
            raise SourceDisabled(
                "Mastodon account verification returned "
                f"HTTP {response.status_code}."
            )
        if data.get("uri") != source.expected_actor_uri:
            raise SourceDisabled("The pinned Mastodon actor URI no longer matches.")
        if data.get("url") != source.expected_profile_url:
            raise SourceDisabled(
                "The pinned Mastodon profile URL no longer matches."
            )
        account_id = data.get("id")
        if not isinstance(account_id, str):
            raise SourceDisabled("The Mastodon account ID is missing.")

        verified_website = False
        for field in data.get("fields") or []:
            if not field.get("verified_at"):
                continue
            soup = BeautifulSoup(field.get("value") or "", "html.parser")
            links = soup.select("a[href]")
            if len(links) != 1:
                continue
            website = urllib.parse.urljoin(
                source.expected_website_url,
                links[0].get("href") or "",
            )
            if _path_identity(website) == _path_identity(
                source.expected_website_url
            ):
                verified_website = True
                break
        if not verified_website:
            raise SourceDisabled(
                "The Mastodon profile no longer has the verified publisher "
                "website field."
            )
        return account_id

    def _crawl_mastodon(self, source: MastodonSource) -> None:
        self.logger.info(
            "Verifying Mastodon source %s.",
            source.expected_actor_uri,
        )
        try:
            account_id = self._verify_mastodon(source)
        except SourceDisabled as exc:
            self.store.set_state(source.key, None, "unavailable", str(exc))
            self.logger.warning("%s", exc)
            return
        except TemporarySourceError as exc:
            saved_cursor, _ = self.store.get_state(source.key)
            self.store.set_state(
                source.key,
                saved_cursor,
                "paused",
                str(exc),
            )
            self.logger.warning(
                "Paused %s during identity verification: %s",
                source.key,
                exc,
            )
            return

        saved_cursor, saved_status = self.store.get_state(source.key)
        cursor = (
            saved_cursor
            if saved_status in {"running", "paused"}
            else None
        )
        incremental_scan = saved_status == "complete"
        pages = 0
        while True:
            page_cursor = cursor
            params: dict[str, Any] = {
                "limit": source.page_size,
                "exclude_reblogs": "true",
                "exclude_replies": "true",
            }
            if cursor:
                params["max_id"] = cursor
            self.store.set_state(source.key, page_cursor, "running")
            try:
                response = self.http.request(
                    "GET",
                    (
                        f"{source.base_url}/api/v1/accounts/"
                        f"{account_id}/statuses"
                    ),
                    params=params,
                    accept="application/json",
                )
                data = _json_response(response, "Mastodon status feed")
                response.close()
                if response.status_code != 200:
                    raise TemporarySourceError(
                        "Mastodon status feed returned "
                        f"HTTP {response.status_code}."
                    )
                if not isinstance(data, list):
                    raise TemporarySourceError(
                        "Mastodon status feed is not a list."
                    )
                new_posts = 0
                for status in data:
                    post_uri = status.get("uri")
                    if not isinstance(post_uri, str):
                        continue
                    if self.store.is_processed(source.key, post_uri):
                        continue
                    new_posts += 1
                    self._process_social_item(
                        source,
                        post_uri,
                        lambda status=status: extract_mastodon_post(
                            source,
                            status,
                        ),
                    )
                pages += 1
                if not data or len(data) < source.page_size:
                    self.store.set_state(source.key, None, "complete")
                    break
                next_cursor = data[-1].get("id")
                if not isinstance(next_cursor, str):
                    raise TemporarySourceError(
                        "The Mastodon page has no continuation ID."
                    )
                cursor = next_cursor
                if incremental_scan and new_posts == 0:
                    self.store.set_state(source.key, None, "complete")
                    break
                self.store.set_state(source.key, cursor, "paused")
                if (
                    self.config.max_pages_per_source is not None
                    and pages >= self.config.max_pages_per_source
                ):
                    break
            except KeyboardInterrupt:
                self.store.set_state(source.key, page_cursor, "paused")
                raise
            except TemporarySourceError as exc:
                self.store.set_state(
                    source.key,
                    page_cursor,
                    "paused",
                    str(exc),
                )
                self.logger.warning(
                    "Paused %s at its current page: %s",
                    source.key,
                    exc,
                )
                break

    def _process_social_item(
        self,
        source: SocialSource,
        post_uri: str,
        extractor: Any,
    ) -> None:
        try:
            post = extractor()
            article = self._article_for_post(post)
            self.store.insert_row(article, post)
            self.store.mark_processed(
                source.key,
                post_uri,
                "accepted",
                None,
            )
            self.logger.info(
                "Accepted %s -> %s",
                post.post_url,
                article.canonical_url,
            )
        except PermanentReject as exc:
            self.store.mark_processed(
                source.key,
                post_uri,
                "rejected",
                str(exc),
            )
        except TemporarySourceError:
            raise

    def _article_for_post(self, post: SocialPost) -> Article:
        target_by_identity: dict[tuple[str, str, str], tuple[AgencySpec, str]] = {}
        for link in post.links:
            agency, target = self._deterministic_target(link)
            if agency is None or target is None:
                continue
            if agency.key != post.agency_key:
                raise PermanentReject(
                    "The post links to a different configured publisher."
                )
            target_by_identity[_path_identity(target)] = (agency, target)
        if not target_by_identity:
            raise PermanentReject(
                "No structured URL deterministically reaches an accepted English article."
            )
        if len(target_by_identity) != 1:
            raise PermanentReject(
                "The post deterministically links to more than one article."
            )
        agency, target = next(iter(target_by_identity.values()))
        effective_url, payload, _ = self.http.fetch_html(
            target,
            max_redirects=self.config.max_redirects,
        )
        effective_agency = classify_article_url(effective_url)
        if effective_agency is None or effective_agency.key != agency.key:
            raise PermanentReject(
                "The article request left the accepted publisher path."
            )
        return extract_article(agency, effective_url, payload)

    def _deterministic_target(
        self,
        link: str,
    ) -> tuple[AgencySpec | None, str | None]:
        try:
            direct = classify_article_url(link)
        except PermanentReject:
            return None, None
        if direct is not None:
            return direct, link
        parsed = urllib.parse.urlsplit(link)
        host = (parsed.hostname or "").lower().rstrip(".")
        if host not in SHORTENER_HOSTS:
            return None, None
        resolved = self.http.resolve_short_url(
            link,
            max_redirects=self.config.max_redirects,
        )
        agency = classify_article_url(resolved)
        if agency is None:
            return None, None
        return agency, resolved


def run_pipeline(config: PipelineConfig | None = None) -> dict[str, Any]:
    """Run configured publisher sources and export accepted rows."""

    pipeline = PrecisionPipeline(config or PipelineConfig())
    try:
        return pipeline.run()
    finally:
        pipeline.close()


def export_checkpoint(
    checkpoint_db: Path | str,
    output_csv: Path | str,
) -> int:
    """Restrict and export active-scope rows without contacting the network."""

    store = CheckpointStore(Path(checkpoint_db))
    try:
        store.retain_active_scope()
        return store.export_csv(Path(output_csv))
    finally:
        store.close()


__all__ = [
    "AGENCIES",
    "CSV_COLUMNS",
    "PipelineConfig",
    "SOCIAL_SOURCES",
    "export_checkpoint",
    "run_pipeline",
]
