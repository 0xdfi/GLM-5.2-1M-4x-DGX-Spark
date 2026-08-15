#!/usr/bin/env python3
"""Legacy exp1 64K fixture runner using vLLM engine-log throughput.

All deployment-specific endpoint/path inputs are required. Historical evidence only;
this script does not select an O14 profile.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from pathlib import Path

FILLER = "The quick brown fox jumps over the lazy dog beside the quiet river. "
PROMPT_RE = re.compile(r"Avg prompt throughput:\s*([0-9.]+)")
GEN_RE = re.compile(r"Avg generation throughput:\s*([0-9.]+)")
ACC_RE = re.compile(r"Mean acceptance length:\s*([0-9.]+)")
PEAK_TASK = "\n\nNow repeat this exact line 100 times, one per line: The quick brown fox jumps over the lazy dog."
FLOOR_TASK = "\n\nNow count from 1 to 400, one integer per line, nothing else."


def build(target: int, task: str) -> str:
    reps = max(1, int(target / 1.3) // len(FILLER.split()))
    return "Reference document (ignore its content):\n" + FILLER * reps + task


def fire(base: str, model: str, prompt: str, temperature: float, generation_tokens: int):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": generation_tokens,
        "temperature": temperature,
    }).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def new_text(log_path: Path, start: int) -> str:
    with log_path.open("r", errors="ignore") as handle:
        handle.seek(start)
        return handle.read()


def maximum(regex: re.Pattern[str], text: str):
    values = [float(value) for value in regex.findall(text)]
    return round(max(values), 1) if values else None


def run(args, label: str, task: str, temperature: float, status_log: Path):
    print(f"[{label}] firing {args.target_tokens}-token prompt ...", flush=True)
    start = os.path.getsize(args.log)
    response = fire(args.base, args.model, build(args.target_tokens, task), temperature, args.generation_tokens)
    time.sleep(args.log_settle_seconds)
    text = new_text(args.log, start)
    usage = response.get("usage", {})
    row = {
        "phase": label,
        "actual_prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "prefill_toks": maximum(PROMPT_RE, text),
        "decode_toks": maximum(GEN_RE, text),
        "max_accept_len": maximum(ACC_RE, text),
    }
    line = json.dumps(row)
    print("  " + line, flush=True)
    with status_log.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="OpenAI-compatible API base URL")
    parser.add_argument("--log", required=True, type=Path, help="vLLM runtime log path")
    parser.add_argument("--out", required=True, type=Path, help="result JSON path")
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--target-tokens", type=int, default=64000)
    parser.add_argument("--generation-tokens", type=int, default=500)
    parser.add_argument("--log-settle-seconds", type=float, default=4.0)
    args = parser.parse_args()

    if not args.log.is_file():
        parser.error(f"runtime log does not exist: {args.log}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    status_log = args.out.with_suffix(args.out.suffix + ".log")
    status_log.write_text("", encoding="utf-8")
    result = {"context_target": args.target_tokens, "fixture": "legacy-exp1-dcp1", "rows": []}
    result["rows"].append(run(args, "PEAK", PEAK_TASK, 0.0, status_log))
    result["rows"].append(run(args, "FLOOR", FLOOR_TASK, 0.1, status_log))
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"DONE {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
