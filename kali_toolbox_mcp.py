#!/usr/bin/env python3
"""Kali Toolbox MCP server.

Exposes the whole Kali toolset to an MCP client (Claude Desktop, Claude Code /
zim-claude CLI) as four tools:

    list_kali_tools   - enumerate what is installed / missing
    run_kali_tool     - run any installed Kali tool with arguments
    install_kali_tool - apt-install a missing tool, then run it
    run_shell_command - arbitrary shell command (fallback for anything else)

Design notes
------------
* Tools are discovered dynamically from the Kali apt package lists plus the
  binaries already on PATH, so the catalogue is "all community tools present on
  the machine" without hard-coding a list that goes stale.
* Installation shells out to `apt-get install -y <pkg>` (Kali's archive). It is
  gated behind the caller passing allow_install=true, because installing
  packages mutates the machine; the MCP client is expected to confirm with the
  human first.
* Every subprocess is run with a timeout, its stdout/stderr captured, and the
  child's environment sanitised (no inherited ANTHROPIC_* secrets).

This server intentionally runs as a local stdio MCP server.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - startup guard
    sys.stderr.write(
        "kali-toolbox-mcp: the 'mcp' package is required in this interpreter "
        f"({exc}). Install it with: pip install 'mcp>=1.0,<2'\n"
    )
    raise

COMMAND_TIMEOUT = int(os.environ.get("KALI_MCP_TIMEOUT", "600"))
INSTALL_TIMEOUT = int(os.environ.get("KALI_MCP_INSTALL_TIMEOUT", "900"))

SAFETY_INSTRUCTIONS = """\
You can run security tools that are installed on this Kali machine.

Operational rules:
1. Only use these tools against systems you are explicitly authorised to test.
   Prefer passive/read-only modes first.
2. `run_kali_tool` needs the tool to already be installed. If it is missing,
   call `install_kali_tool` (which apt-installs it) -- but only after the human
   has agreed, because it changes the machine.
3. Long-running tools (nmap full-port, sqlmap, hydra, metasploit) can take
   minutes; pass a sensible `timeout` and prefer narrower scans.
4. Report the exact command, its exit code, and the raw output. Do not invent
   findings.
"""

mcp = FastMCP("kali_toolbox", instructions=SAFETY_INSTRUCTIONS)

# --------------------------------------------------------------------------- #
# environment handling
# --------------------------------------------------------------------------- #

# Variables that must never leak into a tool subprocess (API keys, tokens).
_SECRET_PREFIXES = ("ANTHROPIC_", "OPENAI_", "LITELLM_", "AWS_", "GCP_", "AZURE_")
_SECRET_NAMES = {"TOKENJUICE_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"}


def _clean_env() -> dict[str, str]:
    env = {}
    for k, v in os.environ.items():
        if k.startswith(_SECRET_PREFIXES) or k in _SECRET_NAMES:
            continue
        env[k] = v
    # Make sure tools that read these still work even if they were stripped.
    env.setdefault("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
    env.setdefault("HOME", os.path.expanduser("~"))
    env["DEBIAN_FRONTEND"] = "noninteractive"
    return env


def _run(cmd: list[str], timeout: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_clean_env(),
            cwd="/root",
        )
    except FileNotFoundError:
        return {"ok": False, "error": f"not found: {cmd[0]}", "exit_code": 127}
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "error": f"timed out after {timeout}s",
            "exit_code": 124,
            "stdout": (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
            "stderr": (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
        }
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-8000:],
    }


# --------------------------------------------------------------------------- #
# tool catalogue
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def _apt_tool_map() -> dict[str, str]:
    """Map tool-name -> apt package name from Kali's package lists.

    Uses `apt-cache pkgnames` (all known package names) and treats a package as
    a "tool" if it also ships a same-named binary. That covers the Kali menu
    (nmap, gobuster, ffuf, sqlmap, hydra, metasploit-framework, ...).
    """
    mapping: dict[str, str] = {}
    try:
        out = subprocess.run(
            ["apt-cache", "pkgnames"],
            capture_output=True, text=True, timeout=120, env=_clean_env(),
        ).stdout
    except Exception:
        return mapping
    for pkg in out.splitlines():
        pkg = pkg.strip()
        if not pkg or pkg.startswith("lib") or pkg.endswith(("-dev", "-doc", "-dbg", "-data")):
            continue
        mapping.setdefault(pkg, pkg)
    return mapping


@lru_cache(maxsize=1)
def _installed_bins() -> dict[str, str]:
    """Map every executable on PATH -> absolute path."""
    bins: dict[str, str] = {}
    for d in _clean_env()["PATH"].split(":"):
        p = Path(d)
        if not p.is_dir():
            continue
        try:
            for entry in p.iterdir():
                if entry.is_file() and os.access(entry, os.X_OK) and entry.name not in bins:
                    bins[entry.name] = str(entry)
        except OSError:
            continue
    return bins


@lru_cache(maxsize=1)
def _dpkg_installed() -> set[str]:
    try:
        out = subprocess.run(
            ["dpkg-query", "-W", "-f", "${Package}\n"],
            capture_output=True, text=True, timeout=60, env=_clean_env(),
        ).stdout
        return {l.strip() for l in out.splitlines() if l.strip()}
    except Exception:
        return set()


def _resolve(tool: str) -> tuple[str, str | None]:
    """Return (resolved_command, apt_package_if_missing)."""
    tool = tool.strip()
    if not tool or "/" in tool or ".." in tool:
        return tool, None  # explicit path or invalid; let the caller's exec decide
    path = shutil.which(tool, path=_clean_env()["PATH"])
    if path:
        return path, None
    return tool, tool


# --------------------------------------------------------------------------- #
# MCP tools
# --------------------------------------------------------------------------- #

@mcp.tool(name="list_kali_tools")
def list_kali_tools(query: str = "", limit: int = 200) -> dict[str, Any]:
    """List Kali tools available on this machine.

    Args:
        query: optional case-insensitive substring filter (e.g. "sql", "recon").
        limit: max rows to return (default 200).
    """
    installed = sorted(_installed_bins())
    known = set(_apt_tool_map())
    q = query.lower().strip()

    rows = []
    for name in installed:
        if q and q not in name.lower():
            continue
        rows.append({"tool": name, "installed": True, "apt_package": name if name in known else None})
    if not q:
        # Also surface a few notable known-but-missing packages so the caller
        # can offer to install them.
        missing = sorted(n for n in known if n not in _installed_bins())
        for name in missing[:50]:
            rows.append({"tool": name, "installed": False, "apt_package": name})

    rows.sort(key=lambda r: (not r["installed"], r["tool"]))
    return {
        "count": len(rows),
        "returned": rows[: max(1, limit)],
        "installed_binaries_on_path": len(installed),
        "known_kali_packages": len(known),
        "note": "Run install_kali_tool(package=...) to add a missing tool.",
    }


@mcp.tool(name="run_kali_tool")
def run_kali_tool(tool: str, args: str = "", timeout: int = COMMAND_TIMEOUT) -> dict[str, Any]:
    """Run an installed Kali tool.

    Args:
        tool: tool name (e.g. "nmap") or absolute path.
        args: argument string, split on shell-quote boundaries (e.g. "-sV -p 80 host").
        timeout: seconds before the run is killed (default 600).
    """
    import shlex

    cmd_path, missing_pkg = _resolve(tool)
    if missing_pkg is not None:
        return {
            "ok": False,
            "error": f"tool '{tool}' is not installed",
            "hint": f"call install_kali_tool(package='{missing_pkg}') first",
        }
    try:
        argv = [cmd_path, *shlex.split(args)]
    except ValueError as exc:
        return {"ok": False, "error": f"could not parse args: {exc}"}

    result = _run(argv, timeout)
    result["command"] = " ".join(argv)
    return result


@mcp.tool(name="install_kali_tool")
def install_kali_tool(package: str, allow_install: bool = False,
                      timeout: int = INSTALL_TIMEOUT) -> dict[str, Any]:
    """apt-install a Kali tool that is missing.

    Installing changes the machine, so `allow_install=true` is required -- pass
    it only after the human has agreed.

    Args:
        package: apt package name (e.g. "gobuster", "ffuf").
        allow_install: must be true to actually install.
        timeout: apt timeout in seconds.
    """
    package = package.strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", package):
        return {"ok": False, "error": f"invalid package name: {package!r}"}
    if not allow_install:
        return {
            "ok": False,
            "error": "allow_install is false",
            "hint": "confirm with the human, then call again with allow_install=true",
        }
    if package in _dpkg_installed():
        return {"ok": True, "already_installed": True, "package": package}

    update = _run(["apt-get", "update"], timeout=timeout)
    if not update["ok"]:
        return {"ok": False, "stage": "apt-get update", **update}
    install = _run(["apt-get", "install", "-y", "--no-install-recommends", package], timeout=timeout)
    _installed_bins.cache_clear()
    _dpkg_installed.cache_clear()
    if install["ok"]:
        return {"ok": True, "installed": package, "apt_update": update.get("stdout", "")[-500:]}
    return {"ok": False, "stage": "apt-get install", "package": package, **install}


@mcp.tool(name="run_shell_command")
def run_shell_command(command: str, timeout: int = COMMAND_TIMEOUT) -> dict[str, Any]:
    """Run an arbitrary shell command on this Kali machine (fallback tool).

    Args:
        command: the shell command line to run.
        timeout: seconds before the command is killed (default 600).
    """
    if not command.strip():
        return {"ok": False, "error": "empty command"}
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, env=_clean_env(), cwd="/root",
        )
    except subprocess.TimeoutExpired as exc:
        return {"ok": False, "error": f"timed out after {timeout}s", "exit_code": 124,
                "stdout": (exc.stdout or ""), "stderr": (exc.stderr or "")}
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "command": command,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-8000:],
    }


if __name__ == "__main__":
    mcp.run()
