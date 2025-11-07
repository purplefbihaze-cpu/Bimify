from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class PreprocessRuntimeConfig:
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    emit_stats: bool = True
    emit_wexbim: bool = False
    capture_logs: bool = True
    timeout: float | None = None


def _ensure_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, Sequence):
        return [str(v) for v in value]
    return None


def _normalize_command(cmd: Sequence[str]) -> list[str]:
    normalized = [str(part) for part in cmd]
    if not normalized:
        return normalized
    first = normalized[0]
    candidate = Path(first)
    if candidate.exists():
        normalized[0] = str(candidate.resolve())
    else:
        resolved = shutil.which(first)
        if resolved:
            normalized[0] = resolved
    return normalized


def resolve_preprocess_runtime(settings: Any) -> PreprocessRuntimeConfig | None:
    geometry = getattr(settings, "geometry", None)
    preprocess_cfg = getattr(geometry, "preprocess", None) if geometry else None

    command: list[str] | None = None
    env: dict[str, str] = {}
    emit_stats = True
    emit_wexbim = False
    capture_logs = True
    timeout_value: float | None = None

    if preprocess_cfg is not None:
        if not getattr(preprocess_cfg, "enabled", True):
            return None

        command = _ensure_str_list(getattr(preprocess_cfg, "command", None))
        if command is None:
            command = _ensure_str_list(getattr(preprocess_cfg, "executable", None))
        if command is None:
            path_value = getattr(preprocess_cfg, "path", None)
            if path_value:
                command = [str(path_value)]

        extra_args = _ensure_str_list(getattr(preprocess_cfg, "args", None))
        if command and extra_args:
            command += extra_args

        env_mapping = getattr(preprocess_cfg, "env", None)
        if isinstance(env_mapping, Mapping):
            env = {str(k): str(v) for k, v in env_mapping.items()}

        emit_stats = bool(getattr(preprocess_cfg, "emit_stats", emit_stats))
        emit_wexbim = bool(getattr(preprocess_cfg, "emit_wexbim", emit_wexbim))
        capture_logs = bool(getattr(preprocess_cfg, "capture_logs", capture_logs))

        timeout_raw = (
            getattr(preprocess_cfg, "timeout_seconds", None)
            or getattr(preprocess_cfg, "timeout_sec", None)
            or getattr(preprocess_cfg, "timeout", None)
        )
        if timeout_raw is not None:
            try:
                timeout_value = float(timeout_raw)
            except (TypeError, ValueError):
                timeout_value = None

    if command is None:
        env_command = os.getenv("XBIM_PREPROCESS_COMMAND")
        if env_command:
            command = shlex.split(env_command)
            env_args = os.getenv("XBIM_PREPROCESS_ARGS")
            if env_args:
                command.extend(shlex.split(env_args))
            env_timeout = os.getenv("XBIM_PREPROCESS_TIMEOUT")
            if env_timeout:
                try:
                    timeout_value = float(env_timeout)
                except ValueError:
                    timeout_value = None
            emit_stats = os.getenv("XBIM_PREPROCESS_STATS", "1") not in {"0", "false", "False"}
            emit_wexbim = os.getenv("XBIM_PREPROCESS_WEXBIM", "0") in {"1", "true", "True"}
            capture_logs = os.getenv("XBIM_PREPROCESS_CAPTURE", "1") not in {"0", "false", "False"}

    if not command:
        return None

    return PreprocessRuntimeConfig(
        command=_normalize_command(command),
        env=env,
        emit_stats=emit_stats,
        emit_wexbim=emit_wexbim,
        capture_logs=capture_logs,
        timeout=timeout_value,
    )


async def invoke_preprocessor(
    cfg: PreprocessRuntimeConfig,
    input_path: Path,
    output_path: Path,
    stats_path: Path | None = None,
    wexbim_path: Path | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if stats_path:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
    if wexbim_path:
        wexbim_path.parent.mkdir(parents=True, exist_ok=True)

    final_cmd: list[str] = [
        *cfg.command,
        "--in",
        str(input_path),
        "--out",
        str(output_path),
        "--overwrite",
    ]
    if stats_path:
        final_cmd += ["--stats", str(stats_path)]
    if wexbim_path:
        final_cmd += ["--wexbim", str(wexbim_path)]

    capture_logs = cfg.capture_logs
    env = os.environ.copy()
    if cfg.env:
        env.update(cfg.env)

    stdout_pipe = asyncio.subprocess.PIPE if capture_logs else None
    stderr_pipe = asyncio.subprocess.PIPE if capture_logs else None

    def decode(data: bytes | str | None) -> str:
        if data is None:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    try:
        process = await asyncio.create_subprocess_exec(
            *final_cmd,
            stdout=stdout_pipe,
            stderr=stderr_pipe,
            env=env,
        )
    except FileNotFoundError as exc:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Executable not found: {exc}",
            "command": final_cmd,
            "returncode": None,
            "timeout": False,
        }
    except NotImplementedError:
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                final_cmd,
                stdout=subprocess.PIPE if capture_logs else None,
                stderr=subprocess.PIPE if capture_logs else None,
                env=env,
                timeout=cfg.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout_bytes = exc.output if isinstance(exc.output, (bytes, str)) else None
            stderr_bytes = exc.stderr if isinstance(exc.stderr, (bytes, str)) else None
            stderr_text = (decode(stderr_bytes) + f"\nProcess timed out after {cfg.timeout} seconds.").strip()
            return {
                "success": False,
                "stdout": decode(stdout_bytes),
                "stderr": stderr_text,
                "command": final_cmd,
                "returncode": None,
                "timeout": True,
            }

        return {
            "success": result.returncode == 0,
            "stdout": decode(result.stdout),
            "stderr": decode(result.stderr),
            "command": final_cmd,
            "returncode": result.returncode,
            "timeout": False,
        }

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=cfg.timeout)
    except asyncio.TimeoutError:
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()
        stderr_text = (decode(stderr_bytes) + f"\nProcess timed out after {cfg.timeout} seconds.").strip()
        return {
            "success": False,
            "stdout": decode(stdout_bytes),
            "stderr": stderr_text,
            "command": final_cmd,
            "returncode": process.returncode,
            "timeout": True,
        }

    return {
        "success": process.returncode == 0,
        "stdout": decode(stdout_bytes),
        "stderr": decode(stderr_bytes),
        "command": final_cmd,
        "returncode": process.returncode,
        "timeout": False,
    }


__all__ = [
    "PreprocessRuntimeConfig",
    "resolve_preprocess_runtime",
    "invoke_preprocessor",
]


