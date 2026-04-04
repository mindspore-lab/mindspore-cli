#!/usr/bin/env python3
"""Client utility for remote_runner_server.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

TERMINAL_STATUSES = {"success", "failed", "timeout", "canceled"}


def http_json(method: str, url: str, payload: Dict | None = None) -> Tuple[int, Dict]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = Request(url=url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8")
            return resp.status, json.loads(text) if text else {}
    except HTTPError as e:
        text = e.read().decode("utf-8")
        body = {}
        if text:
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = {"message": text}
        return e.code, body
    except URLError as e:
        return 599, {"message": str(e)}


def http_download(url: str, output_path: Path) -> Tuple[int, Dict]:
    req = Request(url=url, method="GET")
    try:
        with urlopen(req, timeout=60) as resp:
            status = resp.status
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("wb") as fp:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fp.write(chunk)
        return status, {"output_path": str(output_path), "size": output_path.stat().st_size}
    except HTTPError as e:
        text = e.read().decode("utf-8")
        body = {}
        if text:
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = {"message": text}
        return e.code, body
    except URLError as e:
        return 599, {"message": str(e)}


def cmd_submit(args: argparse.Namespace) -> int:
    payload = {
        "repo": args.repo,
        "branch": args.branch,
        "commit": args.commit,
        "test_cmd": args.test_cmd,
        "timeout_sec": args.timeout_sec,
        "client_id": args.client_id,
        "priority": args.priority,
        "resource_need": args.resource_need,
    }
    code, body = http_json("POST", f"{args.server}/jobs", payload)
    print(json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if code < 300 else 1


def cmd_status(args: argparse.Namespace) -> int:
    code, body = http_json("GET", f"{args.server}/jobs/{args.job_id}")
    print(json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if code < 300 else 1


def cmd_current(args: argparse.Namespace) -> int:
    code, body = http_json("GET", f"{args.server}/jobs/current")
    print(json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if code < 300 else 1


def cmd_cancel(args: argparse.Namespace) -> int:
    code, body = http_json("POST", f"{args.server}/jobs/{args.job_id}/cancel", {})
    print(json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if code < 300 else 1


def cmd_wait(args: argparse.Namespace) -> int:
    deadline = time.time() + args.wait_timeout_sec
    while True:
        code, body = http_json("GET", f"{args.server}/jobs/{args.job_id}")
        if code >= 300:
            print(json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True))
            return 1

        status = body.get("status", "")
        print(f"job_id={args.job_id} status={status}")
        if status in TERMINAL_STATUSES:
            print(json.dumps(body, ensure_ascii=True, indent=2, sort_keys=True))
            return 0 if status == "success" else 2

        if time.time() > deadline:
            print(f"wait timeout: {args.wait_timeout_sec}s", file=sys.stderr)
            return 3

        time.sleep(args.poll_interval_sec)


def cmd_download(args: argparse.Namespace) -> int:
    status_code, status_body = http_json("GET", f"{args.server}/jobs/{args.job_id}")
    if status_code >= 300:
        print(json.dumps(status_body, ensure_ascii=True, indent=2, sort_keys=True))
        return 1

    bundle_uri = status_body.get("artifact_bundle_uri", "")
    if not bundle_uri:
        print(
            json.dumps(
                {
                    "code": 400,
                    "message": "artifact_bundle_uri missing in job status response",
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )
        return 1

    output_path = Path(args.output) if args.output else Path(f"{args.job_id}_artifacts.zip")
    server = args.server.rstrip("/")
    download_url = f"{server}{bundle_uri}"
    download_code, download_body = http_download(download_url, output_path)
    if download_code >= 300:
        print(json.dumps(download_body, ensure_ascii=True, indent=2, sort_keys=True))
        return 1

    resp = {
        "job_id": args.job_id,
        "status": status_body.get("status", ""),
        "error_type": status_body.get("error_type", ""),
        "artifact_bundle_uri": bundle_uri,
        "download_url": download_url,
        "output_path": download_body["output_path"],
        "size": download_body["size"],
    }
    print(json.dumps(resp, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client for op_info remote runner")
    parser.add_argument("--server", default="http://127.0.0.1:18080")

    sub = parser.add_subparsers(dest="subcmd", required=True)

    submit = sub.add_parser("submit", help="Submit a new job")
    submit.add_argument("--repo", required=True)
    submit.add_argument("--branch", required=True)
    submit.add_argument("--commit", default="")
    submit.add_argument("--test-cmd", required=True)
    submit.add_argument("--timeout-sec", type=int, default=1800)
    submit.add_argument("--client-id", default="default")
    submit.add_argument("--priority", type=int, default=0)
    submit.add_argument("--resource-need", default="")
    submit.set_defaults(func=cmd_submit)

    status = sub.add_parser("status", help="Get job status")
    status.add_argument("--job-id", required=True)
    status.set_defaults(func=cmd_status)

    current = sub.add_parser("current", help="Get current running job")
    current.set_defaults(func=cmd_current)

    cancel = sub.add_parser("cancel", help="Cancel a running job")
    cancel.add_argument("--job-id", required=True)
    cancel.set_defaults(func=cmd_cancel)

    wait = sub.add_parser("wait", help="Wait until terminal status")
    wait.add_argument("--job-id", required=True)
    wait.add_argument("--poll-interval-sec", type=int, default=10)
    wait.add_argument("--wait-timeout-sec", type=int, default=7200)
    wait.set_defaults(func=cmd_wait)

    download = sub.add_parser("download", help="Download job artifacts bundle")
    download.add_argument("--job-id", required=True)
    download.add_argument("--output", default="")
    download.set_defaults(func=cmd_download)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
