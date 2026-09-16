import socket
import subprocess

import pytest

from openhands.sdk.flowpilot_reuse import curl_arguments, payload_digest
from openhands.tools.url_fetch import UrlFetchAction, UrlFetchExecutor


@pytest.mark.parametrize(
    "target", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "224.0.0.1"]
)
def test_url_dns_policy_rejects_nonpublic_targets(monkeypatch, target):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a: [(2, 1, 6, "", (target, 443))]
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("must not execute")
    )
    with pytest.raises(ValueError, match="DNS target"):
        UrlFetchExecutor()(UrlFetchAction(command="curl https://example.com"))


def test_url_execution_is_pinned_and_has_no_inherited_state(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a: [(2, 1, 6, "", ("93.184.215.14", 443))]
    )
    captured = []

    def execute(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=b"Title: opaque text\n200\t93.184.215.14\thttps://example.com/a/../b?q=1&q=2",
            stderr=b"",
        )

    monkeypatch.setattr(subprocess, "run", execute)
    observation = UrlFetchExecutor()(
        UrlFetchAction(command="curl https://example.com/a/../b?q=1&q=2")
    )
    argv, kwargs = captured[0]
    assert argv[1] == "-q" and "--path-as-is" in argv and "--globoff" in argv
    assert "example.com:443:93.184.215.14" in argv
    assert "--location" not in argv and "-L" not in argv
    assert kwargs["env"] == {} and kwargs["shell"] is False
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["cwd"].split("/")[-1].startswith("openhands-url-fetch-")
    assert observation.text == "Title: opaque text"
    assert observation.network_policy_validated and observation.complete
    assert observation.final_url_digest == payload_digest(
        "https://example.com/a/../b?q=1&q=2"
    )


@pytest.mark.parametrize(
    "status,remote,body",
    [
        (302, "93.184.215.14", b"redirect"),
        (200, "10.0.0.1", b"rebound"),
        (200, "93.184.215.14", b"\xff"),
    ],
)
def test_redirect_rebinding_and_binary_results_are_not_reusable(
    monkeypatch, status, remote, body
):
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *a: [(2, 1, 6, "", ("93.184.215.14", 443))]
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **k: subprocess.CompletedProcess(
            argv,
            0,
            stdout=body + b"\n" + f"{status}\t{remote}\thttps://example.com/".encode(),
            stderr=b"",
        ),
    )
    observation = UrlFetchExecutor()(UrlFetchAction(command="curl https://example.com"))
    assert observation.is_error and not observation.network_policy_validated


@pytest.mark.parametrize(
    "command",
    [
        "curl -L https://example.com",
        "curl -o /tmp/x https://example.com",
        "curl https://example.com/$TOKEN",
        "env curl https://example.com",
        "curl https://example.com | cat",
        "curl http://[::1]",
        "curl -H 'Cookie:x' https://example.com",
    ],
)
def test_unsafe_commands_never_enter_executor(command):
    with pytest.raises(ValueError):
        curl_arguments({"command": command})
