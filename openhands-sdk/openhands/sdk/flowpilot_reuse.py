"""Wire canonicalization shared by Runtime inputs and the isolated URL executor."""

import hashlib
import ipaddress
import json
import re
import shlex
from typing import Any
from urllib.parse import urlsplit, urlunsplit


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def secret_dependent(value: Any) -> bool:
    if isinstance(value, dict):
        return any(secret_dependent(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(secret_dependent(v) for v in value)
    return isinstance(value, str) and bool(re.search(r"\$(?:\{|[A-Za-z_])", value))


def normalize_url(value: str, *, public_only: bool = False) -> str:
    if not value or re.search(r"[\s\x00-\x1f\x7f\\]", value):
        raise ValueError("URL contains whitespace or control characters")
    if re.search(r"%(?![0-9a-fA-F]{2})", value):
        raise ValueError("invalid URL percent encoding")
    parsed = urlsplit(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must be HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("authenticated URLs are not supported")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    port = parsed.port
    if public_only:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if address is not None and (not address.is_global or address.is_multicast):
            raise ValueError("URL target must be public")
        if host.rstrip(".") == "localhost" or host.endswith(".localhost"):
            raise ValueError("URL target must be public")
        if address is None and re.fullmatch(r"[0-9xXa-fA-F.]+", host):
            raise ValueError("nonstandard numeric URL host")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None and (scheme, port) not in {("http", 80), ("https", 443)}:
        netloc += f":{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))


def curl_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if set(arguments) - {"command", "is_input", "reset", "timeout"}:
        raise ValueError("unsupported curl input")
    if (
        arguments.get("is_input", False) is not False
        or arguments.get("reset", False) is not False
    ):
        raise ValueError("stateful execution is not supported")
    command = arguments.get("command")
    if not isinstance(command, str) or re.search(
        r"[\x00-\x1f\x7f\\$\x60;|<>*{}~]", command
    ):
        raise ValueError("only literal curl argv is supported")
    if "&&" in command or re.search(r"\s&(?:\s|$)", command):
        raise ValueError("shell operators are not supported")
    argv = shlex.split(command)
    if not argv or argv[0] != "curl":
        raise ValueError("only curl is supported")
    allowed = {
        "-4",
        "-6",
        "--compressed",
        "--fail",
        "--fail-with-body",
        "-s",
        "-S",
        "--silent",
        "--show-error",
    }
    flags: set[str] = set()
    urls: list[str] = []
    for value in argv[1:]:
        if value in allowed:
            flags.add(value)
        elif value.startswith("-"):
            raise ValueError("unsupported curl option")
        else:
            urls.append(value)
    if len(urls) != 1 or {"-4", "-6"} <= flags:
        raise ValueError("curl requires one URL and a consistent address family")
    return {
        "command_line_adapter": "curl_url_exact_v1",
        "url": normalize_url(urls[0], public_only=True),
        "flags": sorted(flags),
    }


def canonical_input(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if secret_dependent(arguments):
        raise ValueError("secret_dependent_input")
    result = dict(arguments)
    if tool_name == "tavily-search":
        for key in ("days", "max_results"):
            if key in result:
                result[key] = float(result[key])
        for key in ("include_domains", "exclude_domains"):
            result[key] = sorted({v.casefold() for v in result.get(key, [])})
    elif tool_name == "tavily-extract":
        result["urls"] = [normalize_url(v) for v in result["urls"]]
    elif tool_name == "url_fetch":
        result = curl_arguments(result)
    return result
