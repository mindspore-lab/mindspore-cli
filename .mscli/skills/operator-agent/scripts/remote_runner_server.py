#!/usr/bin/env python3
"""Single-task remote deploy and test service for op-info-test.

This script implements a minimal single-client/single-task API server.
No queue is introduced. A global file lock guarantees mutual exclusion.
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import re
import selectors
import shlex
import signal
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

API_VERSION = "v1"
TERMINAL_STATUSES = {"success", "failed", "timeout", "canceled"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, data: Dict) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def read_json(path: Path, default: Dict) -> Dict:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


@dataclass
class RunningContext:
    job_id: str
    lock_fd: int
    cancel_event: threading.Event
    artifact_dir: Path
    workspace_dir: Path


class StateStore:
    """File-backed state store.

    State shape:
    {
      "api_version": "v1",
      "current_job_id": "..." | null,
      "jobs": {
         "job_xxx": { ... job status payload ... }
      }
    }
    """

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._lock = threading.Lock()
        self._init_state()

    def _init_state(self) -> None:
        with self._lock:
            if not self.state_file.exists():
                write_json(
                    self.state_file,
                    {"api_version": API_VERSION, "current_job_id": None, "jobs": {}},
                )

    def _load(self) -> Dict:
        return read_json(
            self.state_file,
            {"api_version": API_VERSION, "current_job_id": None, "jobs": {}},
        )

    def get_job(self, job_id: str) -> Optional[Dict]:
        with self._lock:
            state = self._load()
            return state.get("jobs", {}).get(job_id)

    def get_current_job(self) -> Optional[Dict]:
        with self._lock:
            state = self._load()
            job_id = state.get("current_job_id")
            if not job_id:
                return None
            return state.get("jobs", {}).get(job_id)

    def update_job(self, job_data: Dict, set_current: bool = False, clear_current: bool = False) -> None:
        with self._lock:
            state = self._load()
            jobs = state.setdefault("jobs", {})
            jobs[job_data["job_id"]] = job_data
            if set_current:
                state["current_job_id"] = job_data["job_id"]
            if clear_current and state.get("current_job_id") == job_data["job_id"]:
                state["current_job_id"] = None
            write_json(self.state_file, state)


class SingleTaskRunner:
    """Single-task runner with global file lock and cancellable subprocess."""

    def __init__(
        self,
        lock_file: Path,
        artifact_root: Path,
        workspace_root: Path,
        state_store: StateStore,
    ):
        self.lock_file = lock_file
        self.artifact_root = artifact_root
        self.workspace_root = workspace_root
        self.state_store = state_store

        self._mu = threading.Lock()
        self._running: Optional[RunningContext] = None
        self._active_process: Optional[subprocess.Popen] = None

        ensure_dir(self.artifact_root)
        ensure_dir(self.workspace_root)
        ensure_dir(self.lock_file.parent)

    def submit(self, payload: Dict) -> Tuple[int, Dict]:
        required = ["repo", "branch", "test_cmd", "timeout_sec"]
        missing = [k for k in required if not payload.get(k)]
        if missing:
            return 400, {
                "code": 400,
                "message": f"missing required fields: {', '.join(missing)}",
                "api_version": API_VERSION,
            }

        timeout_sec = payload.get("timeout_sec")
        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            return 400, {
                "code": 400,
                "message": "timeout_sec must be a positive integer",
                "api_version": API_VERSION,
            }

        with self._mu:
            lock_fd = os.open(self.lock_file, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(lock_fd)
                current = self.state_store.get_current_job()
                return 409, {
                    "code": 409,
                    "message": "runner busy",
                    "running_job_id": current.get("job_id") if current else None,
                    "api_version": API_VERSION,
                }

            job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            artifact_dir = self.artifact_root / job_id
            workspace_dir = self.workspace_root / job_id
            ensure_dir(artifact_dir)
            ensure_dir(workspace_dir)

            # Keep payload schema extensible, reserved fields are persisted.
            normalized_payload = {
                "repo": payload["repo"],
                "branch": payload["branch"],
                "commit": payload.get("commit", ""),
                "test_cmd": payload["test_cmd"],
                "timeout_sec": timeout_sec,
                "client_id": payload.get("client_id", "default"),
                "priority": payload.get("priority", 0),
                "resource_need": payload.get("resource_need", ""),
            }

            job_data = {
                "job_id": job_id,
                "status": "running",
                "payload": normalized_payload,
                "artifact_uri": f"/artifacts/{job_id}/",
                "artifact_bundle_uri": f"/jobs/{job_id}/artifacts.zip",
                "error_type": "",
                "created_at": utc_now(),
                "started_at": utc_now(),
                "finished_at": None,
                "api_version": API_VERSION,
            }
            self.state_store.update_job(job_data, set_current=True)

            running = RunningContext(
                job_id=job_id,
                lock_fd=lock_fd,
                cancel_event=threading.Event(),
                artifact_dir=artifact_dir,
                workspace_dir=workspace_dir,
            )
            self._running = running

            worker = threading.Thread(
                target=self._run_job,
                name=f"runner-{job_id}",
                args=(running, normalized_payload),
                daemon=True,
            )
            worker.start()

            return 200, {
                "job_id": job_id,
                "status": "running",
                "api_version": API_VERSION,
            }

    def get_job(self, job_id: str) -> Tuple[int, Dict]:
        job = self.state_store.get_job(job_id)
        if not job:
            return 404, {
                "code": 404,
                "message": f"job not found: {job_id}",
                "api_version": API_VERSION,
            }
        return 200, self._normalize_job_response(job)

    def get_current(self) -> Tuple[int, Dict]:
        current = self.state_store.get_current_job()
        if not current:
            return 200, {"job": None, "api_version": API_VERSION}
        return 200, {"job": self._normalize_job_response(current), "api_version": API_VERSION}

    def cancel(self, job_id: str) -> Tuple[int, Dict]:
        with self._mu:
            if not self._running or self._running.job_id != job_id:
                return 409, {
                    "code": 409,
                    "message": "job is not running",
                    "api_version": API_VERSION,
                }
            self._running.cancel_event.set()
            proc = self._active_process
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
            return 200, {
                "job_id": job_id,
                "status": "canceling",
                "api_version": API_VERSION,
            }

    def _set_active_process(self, proc: Optional[subprocess.Popen]) -> None:
        with self._mu:
            self._active_process = proc

    def _release_lock(self, lock_fd: int) -> None:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _build_summary(
        self,
        job_id: str,
        status: str,
        error_type: str,
        log_path: Path,
        junit_path: Path,
    ) -> Dict:
        log_text = ""
        if log_path.exists():
            try:
                log_text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                log_text = ""

        failed_cases = extract_failed_cases(log_text, junit_path)
        top_traceback = "" if status == "success" else extract_top_traceback(log_text)

        return {
            "job_id": job_id,
            "status": status,
            "failed_cases": failed_cases,
            "error_type": error_type,
            "top_traceback": top_traceback,
            "generated_at": utc_now(),
        }

    def _update_job_terminal(
        self,
        running: RunningContext,
        status: str,
        error_type: str,
        summary: Dict,
    ) -> None:
        job = self.state_store.get_job(running.job_id)
        if not job:
            return
        job["status"] = status
        job["error_type"] = error_type
        job["finished_at"] = utc_now()
        job["summary"] = summary
        self.state_store.update_job(job, clear_current=True)

    def _run_job(self, running: RunningContext, payload: Dict) -> None:
        artifact_dir = running.artifact_dir
        workspace_dir = running.workspace_dir
        log_path = artifact_dir / "pytest.log"
        junit_path = artifact_dir / "junit.xml"
        env_path = artifact_dir / "env.txt"
        deploy_meta_path = artifact_dir / "deploy_meta.json"
        summary_path = artifact_dir / "summary.json"

        deploy_ok = False
        status = "failed"
        error_type = "infra"

        try:
            self._write_env_info(env_path)
            src_dir = workspace_dir / "src"

            with log_path.open("a", encoding="utf-8") as log_fp:
                self._log(log_fp, f"[runner] job_id={running.job_id}")
                self._log(log_fp, f"[runner] payload={json.dumps(payload, ensure_ascii=True)}")

                deploy_ok = self._deploy_repo(payload, workspace_dir, log_fp, running.cancel_event)

                deploy_meta = {
                    "branch": payload["branch"],
                    "commit": payload.get("commit", ""),
                    "deploy_time": utc_now(),
                    "runner_id": os.uname().nodename,
                }
                write_json(deploy_meta_path, deploy_meta)

                if not deploy_ok:
                    if running.cancel_event.is_set():
                        status = "canceled"
                        error_type = "infra"
                    else:
                        status = "failed"
                        error_type = "infra"
                else:
                    test_cmd = ensure_junit_xml(payload["test_cmd"], junit_path)
                    code, canceled, timed_out = self._run_command(
                        test_cmd,
                        cwd=src_dir,
                        log_fp=log_fp,
                        timeout_sec=payload["timeout_sec"],
                        cancel_event=running.cancel_event,
                        extra_env={"PYTHONPATH": str(src_dir)},
                        shell=True,
                    )
                    if canceled:
                        status = "canceled"
                        error_type = "infra"
                    elif timed_out:
                        status = "timeout"
                        error_type = "infra"
                    elif code == 0:
                        status = "success"
                        error_type = ""
                    else:
                        status = "failed"
                        error_type = classify_error_type(log_path, junit_path)

            summary = self._build_summary(running.job_id, status, error_type, log_path, junit_path)
            write_json(summary_path, summary)
            self._update_job_terminal(running, status, error_type, summary)
        finally:
            self._set_active_process(None)
            with self._mu:
                self._running = None
            self._release_lock(running.lock_fd)

    def _write_env_info(self, env_path: Path) -> None:
        info = []
        info.append(f"time={utc_now()}")
        info.append(f"host={os.uname().nodename}")
        info.append(f"python={_safe_run_capture(['python', '-V']).strip()}")
        info.append(f"git={_safe_run_capture(['git', '--version']).strip()}")
        env_path.write_text("\n".join(info) + "\n", encoding="utf-8")

    def _deploy_repo(
        self,
        payload: Dict,
        workspace_dir: Path,
        log_fp,
        cancel_event: threading.Event,
    ) -> bool:
        repo = payload["repo"]
        branch = payload["branch"]
        commit = payload.get("commit", "")
        src_dir = workspace_dir / "src"

        if (src_dir / ".git").exists():
            self._log(log_fp, "[deploy] reuse existing workspace")
        else:
            ensure_dir(workspace_dir)
            code, canceled, timed_out = self._run_command(
                ["git", "clone", repo, str(src_dir)],
                cwd=workspace_dir,
                log_fp=log_fp,
                timeout_sec=600,
                cancel_event=cancel_event,
            )
            if code != 0 or canceled or timed_out:
                return False

        steps = [
            ["git", "fetch", "origin", "--prune"],
            ["git", "checkout", "-B", branch, f"origin/{branch}"],
        ]
        if commit:
            steps.append(["git", "checkout", commit])
        else:
            steps.append(["git", "reset", "--hard", f"origin/{branch}"])

        for cmd in steps:
            code, canceled, timed_out = self._run_command(
                cmd,
                cwd=src_dir,
                log_fp=log_fp,
                timeout_sec=300,
                cancel_event=cancel_event,
            )
            if code != 0 or canceled or timed_out:
                return False
        return True

    def _run_command(
        self,
        cmd,
        cwd: Path,
        log_fp,
        timeout_sec: int,
        cancel_event: threading.Event,
        extra_env: Optional[Dict[str, str]] = None,
        shell: bool = False,
    ) -> Tuple[int, bool, bool]:
        display_cmd = cmd if isinstance(cmd, str) else " ".join(cmd)
        self._log(log_fp, f"[exec] cwd={cwd}")
        self._log(log_fp, f"[exec] cmd={display_cmd}")

        env = os.environ.copy()
        if extra_env:
            # Keep existing PYTHONPATH while ensuring repo root takes precedence.
            for key, val in extra_env.items():
                if key == "PYTHONPATH" and env.get("PYTHONPATH"):
                    env[key] = f"{val}:{env['PYTHONPATH']}"
                else:
                    env[key] = val

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=shell,
            env=env,
            preexec_fn=os.setsid,
        )
        self._set_active_process(proc)

        selector = selectors.DefaultSelector()
        assert proc.stdout is not None
        selector.register(proc.stdout, selectors.EVENT_READ)

        start = time.monotonic()
        canceled = False
        timed_out = False
        try:
            while True:
                if cancel_event.is_set():
                    canceled = True
                    _terminate_process_group(proc)
                    break

                if timeout_sec > 0 and (time.monotonic() - start) > timeout_sec:
                    timed_out = True
                    _terminate_process_group(proc)
                    break

                events = selector.select(timeout=0.2)
                for key, _ in events:
                    line = key.fileobj.readline()
                    if line:
                        self._log(log_fp, line.rstrip("\n"))

                if proc.poll() is not None:
                    # Drain remaining output.
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            break
                        self._log(log_fp, line.rstrip("\n"))
                    break

            code = proc.wait(timeout=5)
            self._log(log_fp, f"[exec] return_code={code}")
            return code, canceled, timed_out
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            code = proc.wait(timeout=5)
            self._log(log_fp, f"[exec] return_code={code}")
            return code, canceled, True
        finally:
            selector.close()
            self._set_active_process(None)

    @staticmethod
    def _log(log_fp, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_fp.write(f"[{stamp}] {message}\n")
        log_fp.flush()

    @staticmethod
    def _normalize_job_response(job: Dict) -> Dict:
        normalized = dict(job)
        job_id = normalized.get("job_id", "")
        if job_id and not normalized.get("artifact_bundle_uri"):
            normalized["artifact_bundle_uri"] = f"/jobs/{job_id}/artifacts.zip"
        return normalized


class ApiHandler(BaseHTTPRequestHandler):
    runner: SingleTaskRunner = None  # type: ignore[assignment]

    def _respond(self, status: int, payload: Dict) -> None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None, "invalid content-length"
        if length <= 0:
            return None, "empty request body"
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8")), None
        except json.JSONDecodeError:
            return None, "invalid json"

    def do_POST(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/jobs":
            payload, err = self._read_json()
            if err:
                self._respond(400, {"code": 400, "message": err, "api_version": API_VERSION})
                return
            status, resp = self.runner.submit(payload or {})
            self._respond(status, resp)
            return

        parts = [p for p in path.split("/") if p]
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "cancel":
            status, resp = self.runner.cancel(parts[1])
            self._respond(status, resp)
            return

        self._respond(404, {"code": 404, "message": "not found", "api_version": API_VERSION})

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        if path == "/jobs/current":
            status, resp = self.runner.get_current()
            self._respond(status, resp)
            return

        parts = [p for p in path.split("/") if p]
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "artifacts.zip":
            job_id = parts[1]
            job = self.runner.state_store.get_job(job_id)
            if not job:
                self._respond(404, {"code": 404, "message": "job not found", "api_version": API_VERSION})
                return

            artifact_dir = self.runner.artifact_root / job_id
            if not artifact_dir.exists() or not artifact_dir.is_dir():
                self._respond(404, {"code": 404, "message": "artifact not found", "api_version": API_VERSION})
                return

            zip_data = _build_artifact_zip_bytes(artifact_dir)
            filename = f"{job_id}_artifacts.zip"
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Content-Length", str(len(zip_data)))
            self.end_headers()
            self.wfile.write(zip_data)
            return

        if len(parts) == 2 and parts[0] == "jobs":
            status, resp = self.runner.get_job(parts[1])
            self._respond(status, resp)
            return

        if len(parts) >= 3 and parts[0] == "artifacts":
            # Lightweight artifact file serving.
            job_id = parts[1]
            rel_file = "/".join(parts[2:])
            if ".." in rel_file or rel_file.startswith("/"):
                self._respond(400, {"code": 400, "message": "invalid artifact path", "api_version": API_VERSION})
                return
            target = self.runner.artifact_root / job_id / rel_file
            if not target.exists() or not target.is_file():
                self._respond(404, {"code": 404, "message": "artifact not found", "api_version": API_VERSION})
                return
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self._respond(404, {"code": 404, "message": "not found", "api_version": API_VERSION})

    def log_message(self, fmt: str, *args) -> None:
        # Keep server output concise.
        return


def _terminate_process_group(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        pass
    for _ in range(20):
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass


def _safe_run_capture(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except (OSError, subprocess.CalledProcessError) as ex:
        return f"<error: {ex}>"


def extract_failed_cases(log_text: str, junit_path: Path) -> List[str]:
    cases: List[str] = []

    if junit_path.exists():
        try:
            root = ET.parse(junit_path).getroot()
            for testcase in root.iter("testcase"):
                has_fail = testcase.find("failure") is not None or testcase.find("error") is not None
                if not has_fail:
                    continue
                classname = testcase.attrib.get("classname", "")
                name = testcase.attrib.get("name", "")
                full = f"{classname}::{name}" if classname else name
                if full:
                    cases.append(full)
        except ET.ParseError:
            pass

    if not cases:
        for match in re.finditer(r"^FAILED\s+(.+?)\s+-", log_text, flags=re.MULTILINE):
            cases.append(match.group(1).strip())

    # Dedupe while preserving order.
    seen = set()
    unique = []
    for item in cases:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def extract_top_traceback(log_text: str, max_lines: int = 30) -> str:
    marker = "Traceback (most recent call last):"
    pos = log_text.find(marker)
    if pos >= 0:
        tail = log_text[pos:].splitlines()
        return "\n".join(tail[:max_lines])

    lines = [ln for ln in log_text.splitlines() if ln.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def classify_error_type(log_path: Path, junit_path: Path) -> str:
    if not log_path.exists():
        return "infra"

    text = log_path.read_text(encoding="utf-8", errors="replace").lower()

    infra_patterns = [
        "fatal:",
        "connection refused",
        "permission denied",
        "command not found",
        "no module named",
        "segmentation fault",
        "internal error",
        "device not found",
        "killed",
    ]
    if any(pat in text for pat in infra_patterns):
        return "infra"

    if junit_path.exists():
        try:
            root = ET.parse(junit_path).getroot()
            tests = int(root.attrib.get("tests", "0"))
            failures = int(root.attrib.get("failures", "0"))
            errors = int(root.attrib.get("errors", "0"))
            if tests > 0 and (failures > 0 or errors > 0):
                return "testcase"
        except (ValueError, ET.ParseError):
            pass

    if "assertionerror" in text or re.search(r"^failed\s+", text, flags=re.MULTILINE):
        return "testcase"
    return "infra"


def ensure_junit_xml(test_cmd: str, junit_path: Path) -> str:
    """Add --junitxml for pytest command if user didn't set it."""
    if re.search(r"--junitxml(?:=|\s+)", test_cmd):
        return test_cmd
    if re.search(r"(^|\s)pytest(\s|$)", test_cmd):
        return f"{test_cmd} --junitxml={shlex.quote(str(junit_path))}"
    return test_cmd


def _build_artifact_zip_bytes(artifact_dir: Path) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zip_fp:
        for file_path in sorted(artifact_dir.rglob("*")):
            if not file_path.is_file():
                continue
            arcname = file_path.relative_to(artifact_dir).as_posix()
            zip_fp.write(file_path, arcname=arcname)
    return buffer.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-task remote deploy/test API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--state-file", default="/tmp/op_info_state.json")
    parser.add_argument("--lock-file", default="/tmp/op_info_runner.lock")
    parser.add_argument("--artifact-root", default="/tmp/op_info_artifacts")
    parser.add_argument("--workspace-root", default="/tmp/op_info_workspace")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_store = StateStore(Path(args.state_file))
    runner = SingleTaskRunner(
        lock_file=Path(args.lock_file),
        artifact_root=Path(args.artifact_root),
        workspace_root=Path(args.workspace_root),
        state_store=state_store,
    )

    ApiHandler.runner = runner
    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"[op_info_runner] listening on {args.host}:{args.port}")
    print(f"[op_info_runner] state_file={args.state_file}")
    print(f"[op_info_runner] artifact_root={args.artifact_root}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
