"""One-shot public GET executor with no Shell, inherited environment or files."""

import ipaddress
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import Field

from openhands.sdk.flowpilot_reuse import curl_arguments, normalize_url, payload_digest
from openhands.sdk.llm import TextContent
from openhands.sdk.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
    ToolExecutor,
    register_tool,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


class UrlFetchAction(Action):
    command: str = Field(description="A literal curl GET with one public HTTP(S) URL.")
    timeout: float = Field(default=30, gt=0)


class UrlFetchObservation(Observation):
    exit_code: int
    timeout: bool = False
    status_code: int = 0
    complete: bool = False
    executor_kind: str = "isolated_curl_argv"
    network_policy_id: str = "public-pinned-get-v1"
    network_policy_validated: bool = False
    final_url_digest: str | None = None


class UrlFetchExecutor(ToolExecutor[UrlFetchAction, UrlFetchObservation]):
    def __init__(self) -> None:
        executable = shutil.which("curl")
        if executable is None:
            raise RuntimeError("UrlFetchTool requires curl")
        self.executable = executable

    def __call__(
        self, action: UrlFetchAction, conversation: "LocalConversation | None" = None
    ) -> UrlFetchObservation:
        canonical = curl_arguments(action.model_dump(exclude={"kind", "security_risk"}))
        url = canonical["url"]
        parsed = urlsplit(url)
        assert parsed.hostname is not None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        flags = canonical["flags"]
        family = (
            socket.AF_INET
            if "-4" in flags
            else socket.AF_INET6
            if "-6" in flags
            else socket.AF_UNSPEC
        )
        addresses = list(
            dict.fromkeys(
                str(info[4][0])
                for info in socket.getaddrinfo(
                    parsed.hostname, port, family, socket.SOCK_STREAM
                )
            )
        )
        if not addresses or any(
            not ipaddress.ip_address(ip).is_global
            or ipaddress.ip_address(ip).is_multicast
            for ip in addresses
        ):
            raise ValueError("URL DNS target violates public-pinned-get-v1")
        target = addresses[0]
        resolve_address = f"[{target}]" if ":" in target else target
        host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
        argv = [
            self.executable,
            "-q",
            "--proxy",
            "",
            "--noproxy",
            "*",
            "--globoff",
            "--path-as-is",
            "--proto",
            "=http,https",
            "--resolve",
            f"{host}:{port}:{resolve_address}",
            "--max-time",
            str(action.timeout),
            *flags,
            "--write-out",
            "\n%{http_code}\t%{remote_ip}\t%{url_effective}",
            "--url",
            url,
        ]
        try:
            with tempfile.TemporaryDirectory(
                prefix="openhands-url-fetch-"
            ) as directory:
                process = subprocess.run(
                    argv,
                    shell=False,
                    env={},
                    cwd=directory,
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    timeout=action.timeout,
                )
        except subprocess.TimeoutExpired:
            return UrlFetchObservation.from_text(
                "URL fetch timed out", is_error=True, exit_code=124, timeout=True
            )
        if process.returncode != 0:
            return UrlFetchObservation.from_text(
                process.stderr.decode("utf-8", errors="replace"),
                is_error=True,
                exit_code=process.returncode,
            )
        body, separator, metadata = process.stdout.rpartition(b"\n")
        if not separator:
            raise ValueError("curl did not return response metadata")
        status_text, remote_ip, effective_url = metadata.decode("utf-8").split("\t")
        status = int(status_text)
        final_url = normalize_url(effective_url, public_only=True)
        valid = remote_ip == target and final_url == url and 200 <= status < 300
        try:
            text = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return UrlFetchObservation.from_text(
                "URL response is not UTF-8 text",
                is_error=True,
                exit_code=0,
                status_code=status,
            )
        if "\x00" in text:
            return UrlFetchObservation.from_text(
                "URL response contains binary data",
                is_error=True,
                exit_code=0,
                status_code=status,
            )
        if conversation is not None:
            text = conversation.state.secret_registry.mask_secrets_in_output(text)
        return UrlFetchObservation(
            content=[TextContent(text=text)],
            is_error=not valid,
            exit_code=0,
            status_code=status,
            complete=True,
            network_policy_validated=valid,
            final_url_digest=payload_digest(final_url),
        )


class UrlFetchTool(ToolDefinition[UrlFetchAction, UrlFetchObservation]):
    @classmethod
    def create(
        cls, _conv_state: Any = None, **_params: Any
    ) -> Sequence["UrlFetchTool"]:
        return [
            cls(
                description="Fetch one public URL using an isolated curl GET. "
                "No redirects, authentication, Shell expressions or file output.",
                action_type=UrlFetchAction,
                observation_type=UrlFetchObservation,
                annotations=ToolAnnotations(
                    readOnlyHint=True,
                    destructiveHint=False,
                    idempotentHint=True,
                    openWorldHint=True,
                ),
                executor=UrlFetchExecutor(),
            )
        ]


register_tool(UrlFetchTool.name, UrlFetchTool)
