"""Read-only benchmark environments with explicit, versioned resource identities."""

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote, urlencode, urlparse

import requests
from bs4 import BeautifulSoup


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_assets(config):
    manifest = Path(config.retrieval.asset_manifest)
    if file_digest(manifest) != config.retrieval.corpus_revision:
        raise ValueError("Asset manifest changed; regenerate config and reuse registry")
    assets = json.loads(manifest.read_text())["assets"]
    for asset in assets:
        if file_digest(asset["path"]) != asset["sha256"]:
            raise ValueError(f"Corpus changed: {asset['path']}; regenerate config and registry")
    return assets


class ResearchEnvironment:
    def __init__(self, config):
        self.config = config
        self.timeout = config.runtime.tool_timeout
        self.last_http_attempts = []
        self.documents = {}
        self.finished_data = None
        self.session = requests.Session()
        self.session.trust_env = False
        if config.retrieval.web_proxy:
            self.session.proxies = {
                "http": config.retrieval.web_proxy,
                "https": config.retrieval.web_proxy,
            }
        self.session.headers["User-Agent"] = (
            "FlowPilotBenchmark/1.0 (https://github.com/ShishirPatil/gorilla; research)"
        )

    def prepare(self):
        verify_assets(self.config)
        if self.config.dataset.kind == "qasper":
            self.documents = {
                row["paper_id"]: row
                for row in (
                    json.loads(line)
                    for line in Path(self.config.retrieval.index_path).read_text().splitlines()
                )
            }
        return {
            "backend": self.config.retrieval.backend,
            "corpus_revision": self.config.retrieval.corpus_revision,
            "read_only": True,
            "search_backend": self.config.retrieval.web_search_backend,
            "network_proxy": self.config.retrieval.web_proxy,
        }

    @contextmanager
    def operation_timeout(self, seconds):
        previous, self.timeout = self.timeout, seconds
        try:
            yield
        finally:
            self.timeout = previous

    def call_tool(self, name, arguments):
        self.last_http_attempts = []
        return getattr(self, name)(**arguments)

    def connect(self, db_id):
        if not re.fullmatch(r"[a-z0-9_]+", db_id):
            raise ValueError("Invalid db_id")
        path = Path(self.config.retrieval.index_path) / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            raise ValueError("Unknown database")
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.set_authorizer(
            lambda code, *_: sqlite3.SQLITE_DENY
            if code in {sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH, sqlite3.SQLITE_PRAGMA}
            else sqlite3.SQLITE_OK
        )
        deadline = time.monotonic() + self.timeout
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        return connection

    def get_schema(self, db_id):
        connection = self.connect(db_id)
        try:
            schema = "\n".join(
                row[0]
                for row in connection.execute(
                    "SELECT sql FROM sqlite_master WHERE type IN ('table','view') AND sql IS NOT NULL ORDER BY name"
                )
            )
            return {"db_id": db_id, "schema": schema}
        finally:
            connection.close()

    def execute_sql(self, db_id, sql, max_rows=50):
        connection = self.connect(db_id)
        try:
            cursor = connection.execute(sql)
            if cursor.description is None:
                raise ValueError("Expected a read-only query with returned columns")
            rows = cursor.fetchmany(max_rows + 1)
            return {
                "db_id": db_id,
                "columns": [col[0] for col in cursor.description],
                "rows": rows[:max_rows],
                "truncated": len(rows) > max_rows,
            }
        finally:
            connection.close()

    def read_paper(self, paper_id):
        document = self.documents[paper_id]
        paragraphs, size = [], 0
        for paragraph in document["paragraphs"]:
            length = len(paragraph["text"])
            if size + length > self.config.runtime.max_output_chars:
                break
            paragraphs.append(paragraph)
            size += length
        return {
            "paper_id": paper_id,
            "title": document["title"],
            "abstract": document["abstract"],
            "paragraphs": paragraphs,
            "truncated": len(paragraphs) != len(document["paragraphs"]),
        }

    def read_paragraph(self, paper_id, paragraph_id):
        paragraphs = self.documents[paper_id]["paragraphs"]
        if not 0 <= paragraph_id < len(paragraphs):
            raise ValueError("paragraph_id out of range")
        return {"paper_id": paper_id, **paragraphs[paragraph_id]}

    def search_paper(self, paper_id, query, top_k=5):
        paragraphs = self.documents[paper_id]["paragraphs"]
        tokens = [re.findall(r"\w+", p["text"].lower()) for p in paragraphs]
        counts = [Counter(t) for t in tokens]
        average = sum(map(len, tokens)) / max(len(tokens), 1)
        scores = [0.0] * len(tokens)
        for term in set(re.findall(r"\w+", query.lower())):
            frequency = sum(term in counter for counter in counts)
            idf = math.log(1 + (len(tokens) - frequency + 0.5) / (frequency + 0.5))
            for index, counter in enumerate(counts):
                tf = counter[term]
                scores[index] += (
                    idf
                    * tf
                    * 2.5
                    / (tf + 1.5 * (0.25 + 0.75 * len(tokens[index]) / max(average, 1)))
                )
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:top_k]
        return [dict(paper_id=paper_id, **paragraphs[i], score=scores[i]) for i in ranked]

    def request(self, url):
        deadline = time.monotonic() + self.timeout
        for attempt in range(3):
            start = time.monotonic()
            row = {
                "url": url.split("api_key=", 1)[0],
                "proxy": self.config.retrieval.web_proxy,
                "http_attempt": attempt + 1,
            }
            wait = None
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("HTTP tool deadline exceeded")
                with self.session.get(url, timeout=remaining, stream=True) as response:
                    row["status_code"] = response.status_code
                    row["ttfb_ms"] = response.elapsed.total_seconds() * 1000
                    if response.status_code in {429, 503} and attempt < 2:
                        retry_after = response.headers.get("Retry-After", "1")
                        row["retry_after"] = retry_after
                        try:
                            wait = max(float(retry_after), 0)
                        except ValueError:
                            from datetime import UTC, datetime
                            from email.utils import parsedate_to_datetime

                            wait = max(
                                (
                                    parsedate_to_datetime(retry_after) - datetime.now(UTC)
                                ).total_seconds(),
                                0,
                            )
                        if wait >= deadline - time.monotonic():
                            raise TimeoutError("Rate limit Retry-After exceeds tool deadline")
                    else:
                        if response.status_code >= 400:
                            raise RuntimeError(
                                f"HTTP {response.status_code} from {urlparse(url).hostname}"
                            )
                        chunks, size = [], 0
                        for chunk in response.iter_content(65536):
                            if time.monotonic() > deadline:
                                raise TimeoutError("HTTP body exceeded tool deadline")
                            size += len(chunk)
                            if size > self.config.retrieval.web_max_response_bytes:
                                raise ValueError("HTTP response exceeds configured byte limit")
                            chunks.append(chunk)
                        return b"".join(chunks).decode(
                            response.encoding or "utf-8", errors="replace"
                        )
                if wait is not None:
                    sleep_started = time.monotonic()
                    time.sleep(wait)
                    row["retry_wait_ms"] = (time.monotonic() - sleep_started) * 1000
            except Exception as exc:
                row["error_type"] = type(exc).__name__
                if isinstance(exc, requests.RequestException):
                    raise RuntimeError(
                        f"HTTP transport failed: {type(exc).__name__} at {urlparse(url).hostname}"
                    ) from None
                raise
            finally:
                row["duration_ms"] = (time.monotonic() - start) * 1000
                self.last_http_attempts.append(row)
        raise TimeoutError("HTTP retry budget exhausted")

    def wiki_api(self, parameters):
        # Percent-encoded spaces also work with the user's caching HTTP proxy.
        return json.loads(
            self.request(
                "https://en.wikipedia.org/w/api.php?" + urlencode(parameters, quote_via=quote)
            )
        )

    def search_engine_query(self, keywords, max_results=10, region="wt-wt"):
        if self.config.retrieval.web_search_backend == "wikipedia":
            data = self.wiki_api(
                {
                    "action": "query",
                    "list": "search",
                    "srsearch": keywords,
                    "format": "json",
                    "srlimit": max_results,
                }
            )
            if "error" in data:
                raise ValueError(data["error"])
            return [
                {
                    "title": row["title"],
                    "url": "https://en.wikipedia.org/wiki/" + quote(row["title"].replace(" ", "_")),
                    "snippet": BeautifulSoup(row["snippet"], "html.parser").get_text(
                        " ", strip=True
                    ),
                }
                for row in data["query"]["search"]
            ]
        key = os.environ.get(self.config.retrieval.web_api_key_env)
        if not key:
            raise ValueError("SerpAPI backend requires SERPAPI_API_KEY")
        parameters = {"engine": "duckduckgo", "q": keywords, "kl": region, "api_key": key}
        data = json.loads(self.request("https://serpapi.com/search.json?" + urlencode(parameters)))
        if "error" in data:
            raise ValueError(data["error"])
        return [
            {"title": row.get("title", ""), "url": row["link"], "snippet": row.get("snippet", "")}
            for row in data.get("organic_results", [])[:max_results]
        ]

    def fetch_url_content(self, url, mode="raw"):
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Expected HTTP(S) URL")
        content = self.request(url)
        if mode == "markdown":
            import html2text

            content = html2text.html2text(content)
        elif mode == "truncate":
            soup = BeautifulSoup(content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            main = soup.select_one("#mw-content-text") or soup
            content = main.get_text(" ", strip=True)
        cap = self.config.runtime.max_output_chars
        return {"url": url, "content": content[:cap], "truncated": len(content) > cap}

    def quiesce(self):
        pass

    def close(self):
        self.session.close()
        return {"cleanup": "completed"}
