#!/usr/bin/env python3
"""Legacy exp1 benchmark using vLLM engine-log throughput.

Client-side timing is intentionally not used. All endpoint/path inputs are explicit.
Historical evidence only; this script does not select an O14 profile.

Usage: bench_engine.py --base http://HOST:PORT --log /path/to/runtime.log \
         --levels 4000,8000,64000 --label legacy-exp1-dcp1 --out /path/to/out.json
"""

import argparse
import json
import os
import re
import time
import urllib.request

FILLER = "The quick brown fox jumps over the lazy dog beside the quiet river. "
PEAK = "\n\nCount from 1 to 600, one integer per line, nothing else."
BASELINE = "\n\nWrite a long, original, imaginative story; be creative and unpredictable."
PROMPT_RE = re.compile(r"Avg prompt throughput:\s*([0-9.]+)")
GEN_RE = re.compile(r"Avg generation throughput:\s*([0-9.]+)")


def build(target, task):
    reps = max(1, int(target / 1.3) // len(FILLER.split()))
    return "Doc (ignore):\n" + FILLER * reps + task


def fire(base, model, prompt, temp, generation_tokens):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": generation_tokens,
        "temperature": temp,
    }).encode()
    request = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5400) as response:
        return json.loads(response.read().decode("utf-8", "ignore"))


def logsize(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_new(path, start):
    with open(path, "r", errors="ignore") as handle:
        handle.seek(start)
        return handle.read()


def maxes(text):
    prompt_values = [float(value) for value in PROMPT_RE.findall(text)]
    generation_values = [float(value) for value in GEN_RE.findall(text)]
    return (
        max(prompt_values) if prompt_values else None,
        max(generation_values) if generation_values else None,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="glm-5.2")
    parser.add_argument("--generation-tokens", type=int, default=600)
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        parser.error(f"runtime log does not exist: {args.log}")
    results = {"label": args.label, "fixture": "legacy-exp1", "levels": []}
    for level in [int(value) for value in args.levels.split(",")]:
        print(f"[{args.label}] {level} ...", flush=True)
        row: dict[str, object] = {"target_tokens": level}
        try:
            start = logsize(args.log)
            response = fire(args.base, args.model, build(level, PEAK), 0.1, args.generation_tokens)
            row["actual_prompt_tokens"] = response.get("usage", {}).get("prompt_tokens")
            time.sleep(3)
            prefill, peak = maxes(read_new(args.log, start))
            row["prefill_toks"] = round(prefill) if prefill else None
            row["decode_peak_toks"] = round(peak, 1) if peak else None
            baseline_start = logsize(args.log)
            fire(args.base, args.model, build(level, BASELINE), 0.8, args.generation_tokens)
            time.sleep(3)
            _, baseline = maxes(read_new(args.log, baseline_start))
            row["decode_baseline_toks"] = round(baseline, 1) if baseline else None
        except Exception as exc:  # Preserve incremental evidence on a failed fixture.
            row["error"] = f"{type(exc).__name__}:{exc}"
        print("  " + json.dumps(row), flush=True)
        results["levels"].append(row)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")
    print("DONE", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
