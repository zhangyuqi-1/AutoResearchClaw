from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from researchclaw.pipeline import _helpers


def test_ensure_sandbox_deps_disables_proxy_only_for_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:7890")
    monkeypatch.setenv("ALL_PROXY", "socks5://proxy.example:7890")
    monkeypatch.setenv("http_proxy", "http://proxy.example:7890")
    monkeypatch.setenv("https_proxy", "http://proxy.example:7890")
    monkeypatch.setenv("all_proxy", "socks5://proxy.example:7890")

    calls: list[dict[str, object]] = []

    def _fake_run(*args, **kwargs):
        cmd = list(args[0])
        calls.append({"cmd": cmd, "env": kwargs.get("env")})
        result = MagicMock(spec=subprocess.CompletedProcess)
        if len(cmd) >= 3 and cmd[1] == "-c":
            result.returncode = 1
        else:
            result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    monkeypatch.setattr(subprocess, "run", _fake_run)

    installed = _helpers._ensure_sandbox_deps("import sklearn\n", ".venv/bin/python")

    assert installed == ["scikit-learn"]
    assert len(calls) == 2

    import_call = calls[0]
    pip_call = calls[1]

    assert import_call["env"] is None

    pip_env = pip_call["env"]
    assert isinstance(pip_env, dict)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        assert key not in pip_env
    assert "PIP_CONFIG_FILE" not in pip_env
