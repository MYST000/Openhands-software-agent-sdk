"""HTTP bridge for the same read-only SQL and paper tool contracts."""

from pathlib import Path

from .hotpot_rpc import HotpotRPCEnvironment


class ResearchRPCEnvironment(HotpotRPCEnvironment):
    def __init__(self, config):
        super().__init__(config)
        self.last_http_attempts = []
        self.finished_data = None

    def prepare(self):
        import hashlib

        expected = hashlib.sha256(
            Path(self.config.retrieval.asset_manifest).read_bytes()
        ).hexdigest()
        if expected != self.config.retrieval.corpus_revision:
            raise ValueError("Research RPC asset manifest changed")
        identity = self._request("/info", {"benchmark": self.config.dataset.kind})
        if identity["corpus_revision"] != expected:
            raise ValueError("Remote research corpus differs from selected assets")
        if identity["server_policy_revision"] != self.config.retrieval.server_policy_revision:
            raise ValueError("Remote research tool policy differs from the profile")
        return {**identity, "backend": "research_rpc", "url": self.config.retrieval.mcp_url}

    def call_tool(self, name, arguments):
        self.last_http_attempts = []
        return self._request(
            "/tool", {"benchmark": self.config.dataset.kind, "name": name, "arguments": arguments}
        )

    def read_paragraph(self, paper_id, paragraph_id):
        return self.call_tool(
            "read_paragraph", {"paper_id": paper_id, "paragraph_id": paragraph_id}
        )

    def read_paper(self, paper_id):
        return self.call_tool("read_paper", {"paper_id": paper_id})
