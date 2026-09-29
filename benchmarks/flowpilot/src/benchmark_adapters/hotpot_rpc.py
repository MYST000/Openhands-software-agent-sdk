"""Read-only Hotpot RPC preserving the existing SQLite search/read semantics."""

import argparse
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import httpx

from benchmark_adapters.config import Config, load_config
from benchmark_adapters.retrieval import RetrievalEnvironment, index_identity


class HotpotRPCEnvironment:
    def __init__(self, config, verified_index=None):
        self.config = config
        self.verified_index = verified_index
        self._timeout = config.runtime.tool_timeout
        self.client = httpx.Client(base_url=config.retrieval.mcp_url.rstrip("/"), trust_env=False)
        self.last_timing = {}

    def _request(self, path, arguments):
        self.last_timing = {}
        try:
            response = self.client.post(
                path,
                json={**arguments, "timeout_seconds": self._timeout},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise TimeoutError("Hotpot RPC exceeded its time budget") from exc
        data = response.json()
        if response.status_code != 200:
            if data.get("error_type") == "TimeoutError":
                raise TimeoutError(data.get("error", "Remote retrieval timed out"))
            raise ValueError(data.get("error", "Hotpot RPC failed"))
        if data.get("protocol_version") != 1:
            raise ValueError("Unsupported Hotpot RPC protocol")
        self.last_timing = data["timing"]
        return data["result"]

    def prepare(self):
        identity = self._request("/info", {})
        if identity["corpus_revision"] != self.config.retrieval.corpus_revision:
            raise ValueError("Remote corpus revision does not match the profile")
        manifest = Path(self.config.retrieval.index_path + ".manifest.json")
        if manifest.is_file():
            expected = json.loads(manifest.read_text())["index_sha256"]
            if identity["index_sha256"] != expected:
                raise ValueError("Remote Hotpot index differs from the local fixed index")
        if self.verified_index is not None:
            for key in ("index_identity", "index_sha256", "corpus_revision"):
                if identity[key] != self.verified_index[key]:
                    raise ValueError("Remote verified index identity changed")
        return {**identity, "backend": "hotpot_rpc", "url": self.config.retrieval.mcp_url}

    @contextmanager
    def operation_timeout(self, seconds):
        previous, self._timeout = self._timeout, seconds
        try:
            yield
        finally:
            self._timeout = previous

    def search(self, query, top_k):
        return self._request("/search", {"query": query, "top_k": top_k})

    def read(self, docid, *, start_sentence=0, max_sentences=20):
        return self._request(
            "/read",
            {"docid": docid, "start_sentence": start_sentence, "max_sentences": max_sentences},
        )

    def quiesce(self):
        pass

    def close(self):
        self.client.close()


class HotpotServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, config: Config):
        self.environment = RetrievalEnvironment(config)
        self.identity = self.environment.prepare()
        self.gate = threading.Lock()
        self.clock_domain = "hotpot-rpc-server-monotonic"
        super().__init__(address, HotpotHandler)

    def server_close(self):
        super().server_close()
        self.environment.close()


class HotpotHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True

    def do_POST(self):
        server = cast(HotpotServer, self.server)
        received = time.monotonic_ns()
        acquired = False
        status = 200
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("RPC body must be between 1 and 65536 bytes")
            request = json.loads(self.rfile.read(size))
            timeout = min(float(request.pop("timeout_seconds", 30)), 30)
            if timeout <= 0:
                raise TimeoutError("No retrieval time budget remains")
            acquired = server.gate.acquire(timeout=timeout)
            if not acquired:
                raise TimeoutError("Hotpot RPC queue wait exceeded its time budget")
            started = time.monotonic_ns()
            remaining = timeout - (started - received) / 1e9
            if remaining <= 0:
                raise TimeoutError("Hotpot RPC time budget expired in queue")
            env = server.environment
            path = Path(env.config.retrieval.index_path)
            manifest = Path(str(path) + ".manifest.json")
            if index_identity(path, manifest.read_bytes()) != server.identity["index_identity"]:
                raise ValueError("Hotpot index changed after server validation")
            with env.operation_timeout(remaining):
                if self.path == "/info" and not request:
                    result = server.identity
                elif self.path == "/search":
                    result = env.search(**request)
                elif self.path == "/read":
                    result = env.read(**request)
                else:
                    raise ValueError("Unknown Hotpot RPC operation")
            payload = {
                "protocol_version": 1,
                "result": result,
                "timing": {
                    "executor_duration_ms": (time.monotonic_ns() - started) / 1e6,
                    "queue_wait_ms": (started - received) / 1e6,
                    "executor_clock_domain": server.clock_domain,
                },
            }
        except (ValueError, TypeError, TimeoutError) as exc:
            status = 504 if isinstance(exc, TimeoutError) else 400
            payload = {"error_type": type(exc).__name__, "error": str(exc)}
        finally:
            if acquired:
                server.gate.release()
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.dataset.kind != "hotpot" or config.retrieval.backend != "sqlite":
        raise ValueError("Serve a local fixed Hotpot SQLite profile")
    server = HotpotServer(("127.0.0.1", args.port), config)
    print(json.dumps({"ready": True, "port": args.port, "index": server.identity}), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
