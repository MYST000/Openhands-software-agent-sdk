"""Check a BrowseComp MCP endpoint with generic queries; no SDK, LLM, or tasks."""

import argparse
import asyncio
from contextlib import suppress
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import time
from urllib.parse import urlparse


def json_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def parse_result(result, *, expect_list):
    """Accept raw MCP, newer FastMCP result wrappers, and 2.9.2 content lists."""
    from mcp.types import TextContent

    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        raise ValueError("MCP server marked the tool result as an error")
    data = getattr(result, "data", None)
    if data is not None:
        return data
    contents = result if isinstance(result, list) else getattr(result, "content", None)
    if not isinstance(contents, list) or not contents:
        raise ValueError("Expected a nonempty MCP content list")
    if any(not isinstance(item, TextContent) for item in contents):
        raise ValueError("Expected only TextContent from the text retrieval tool")
    try:
        decoded = [json.loads(item.text) for item in contents]
    except (ValueError, TypeError) as exc:
        raise ValueError("Retrieval tool content is not valid JSON") from exc
    if expect_list:
        if len(decoded) == 1 and isinstance(decoded[0], list):
            return decoded[0]
        if all(isinstance(item, dict) for item in decoded):
            return decoded
        raise ValueError("Expected search JSON array or one JSON object per content block")
    if len(decoded) != 1:
        raise ValueError("Expected one JSON document result")
    return decoded[0]


async def timed_call(client, name, arguments, timeout, origin_ns, operations):
    start = time.monotonic_ns()
    operation = {"tool": name, "started_ms": (start - origin_ns) / 1e6}
    operations.append(operation)
    try:
        # FastMCP 2.9.2 call_tool() returns a content list and raises ToolError.
        # Its raw MCP method preserves the protocol isError flag for validation.
        result = await client.call_tool_mcp(name, arguments, timeout=timeout)
        operation["rpc_returned"] = True
        return result
    finally:
        end = time.monotonic_ns()
        operation["ended_ms"] = (end - origin_ns) / 1e6
        operation["round_trip_ms"] = (end - start) / 1e6


async def run_client(slot, args, barrier, origin_ns):
    from fastmcp import Client

    report = {"slot": slot, "passed": False, "operations": []}
    try:
        async with Client(args.url, timeout=args.timeout, init_timeout=args.timeout) as client:
            names = sorted(tool.name for tool in await client.list_tools())
            report["tool_names"] = names
            if not {"search", "get_document"}.issubset(names):
                raise ValueError("Endpoint must advertise both search and get_document")
            report["required_tools_present"] = True
            await asyncio.wait_for(barrier.wait(), timeout=args.timeout)

            result = await timed_call(
                client, "search", {"query": args.query}, args.timeout,
                origin_ns, report["operations"],
            )
            hits = parse_result(result, expect_list=True)
            if not isinstance(hits, list) or not hits:
                raise ValueError("Search returned no documents")
            if any(not isinstance(hit, dict) or not isinstance(hit.get("docid"), str)
                   or not hit["docid"].strip() for hit in hits):
                raise ValueError("Search hits must contain nonempty string docids")
            report["search"] = {
                "query": args.query,
                "docids": [hit["docid"] for hit in hits],
                "hit_count": len(hits),
                "result_sha256": json_hash(hits),
            }

            docid = hits[0]["docid"]
            result = await timed_call(
                client, "get_document", {"docid": docid}, args.timeout,
                origin_ns, report["operations"],
            )
            document = parse_result(result, expect_list=False)
            if not isinstance(document, dict) or document.get("docid") != docid:
                raise ValueError("Read result must identify the requested document")
            text = document.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Read result has no nonempty document text")
            raw = text.encode("utf-8")
            report["document"] = {
                "docid": docid,
                "characters": len(text),
                "utf8_bytes": len(raw),
                "text_sha256": hashlib.sha256(raw).hexdigest(),
                "result_sha256": json_hash(document),
                "client_truncation": False,
            }
            report["passed"] = True
    except Exception as exc:
        # Avoid echoing arbitrary remote error bodies or full document contents.
        report["error"] = {"type": type(exc).__name__,
                           "message": "Connection, protocol, tool-schema, or response validation failed"}
        with suppress(Exception):
            await barrier.abort()
    return report


async def run(args):
    origin_ns = time.monotonic_ns()
    started = datetime.now(timezone.utc).isoformat()
    barrier = asyncio.Barrier(args.clients)
    clients = await asyncio.gather(*(
        run_client(slot, args, barrier, origin_ns) for slot in range(args.clients)
    ))
    search_intervals = [op for client in clients for op in client["operations"]
                        if op["tool"] == "search"]
    points = sorted([(op["started_ms"], 1) for op in search_intervals]
                    + [(op["ended_ms"], -1) for op in search_intervals])
    active = peak = 0
    for _, delta in points:
        active += delta
        peak = max(peak, active)
    operations = [op for client in clients for op in client["operations"]]
    wall_range = (max(op["ended_ms"] for op in operations)
                  - min(op["started_ms"] for op in operations)) if operations else None
    return {
        "schema_version": 1,
        "passed": all(client["passed"] for client in clients),
        "scope": "Generic BrowseComp search/read connectivity; no benchmark questions or LLM",
        "url": args.url,
        "started_utc": started,
        "elapsed_including_connections_ms": (time.monotonic_ns() - origin_ns) / 1e6,
        "tool_request_wall_range_ms": wall_range,
        "independent_clients": args.clients,
        "client_search_intervals_peak_overlap": peak,
        "server_executor_parallelism_measured": False,
        "server_execution_ms": None,
        "server_queue_ms": None,
        "result_hash_format": "UTF-8 JSON, sorted keys, compact separators, ensure_ascii=False",
        "versions": {name: importlib.metadata.version(name) for name in ("fastmcp", "mcp")},
        "clients": clients,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="HTTP(S) MCP endpoint, e.g. http://127.0.0.1:8123/mcp")
    parser.add_argument("--clients", type=int, choices=(1, 4), default=1)
    parser.add_argument("--query", default="London museum", help="Generic smoke query")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-call and connection timeout, seconds")
    args = parser.parse_args()
    parsed = urlparse(args.url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        parser.error("--url must be HTTP(S), with a hostname and no embedded credentials")
    if not args.query.strip():
        parser.error("--query must be nonempty")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    report = asyncio.run(run(args))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
