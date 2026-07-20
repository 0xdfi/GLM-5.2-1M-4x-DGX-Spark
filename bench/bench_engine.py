#!/usr/bin/env python3
"""GLM-5.2 benchmark via vLLM's OWN logged throughput (ground truth). Client-side
timing proved unreliable (streaming + subtraction artifacts); the engine's
"Avg prompt throughput" (prefill) and "Avg generation throughput" (decode) are
authoritative. Fires one sustained generation per (level,task) at the target
context, then reads the max throughput the engine logged during that window.
Run on the head node (reads the runtime log). Stdlib only.

Usage: bench_engine.py --base http://IP:8210 --log /exp1-evidence/runtime-X.log \
         --levels 4000,8000,64000 --label dcp1-100k --out out.json
"""
import argparse, json, os, re, time, urllib.request

FILLER = "The quick brown fox jumps over the lazy dog beside the quiet river. "
PEAK = "\n\nCount from 1 to 600, one integer per line, nothing else."
BASE = "\n\nWrite a long, original, imaginative story; be creative and unpredictable."
PROMPT_RE = re.compile(r"Avg prompt throughput:\s*([0-9.]+)")
GEN_RE = re.compile(r"Avg generation throughput:\s*([0-9.]+)")


def build(target, task):
    reps = max(1, int(target / 1.3) // len(FILLER.split()))
    return "Doc (ignore):\n" + FILLER * reps + task


def fire(base, prompt, temp, gen):
    payload = json.dumps({"model": "glm-5.2", "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": gen, "temperature": temp}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5400) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def logsize(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def read_new(path, start):
    with open(path, "r", errors="ignore") as f:
        f.seek(start)
        return f.read()


def maxes(text):
    p = [float(x) for x in PROMPT_RE.findall(text)]
    g = [float(x) for x in GEN_RE.findall(text)]
    return (max(p) if p else None, max(g) if g else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--log", required=True)
    ap.add_argument("--levels", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = {"label": a.label, "levels": []}
    for lvl in [int(x) for x in a.levels.split(",")]:
        print(f"[{a.label}] {lvl} ...", flush=True)
        row = {"target_tokens": lvl}
        try:
            s = logsize(a.log)
            r = fire(a.base, build(lvl, PEAK), 0.1, 600)
            row["actual_prompt_tokens"] = r.get("usage", {}).get("prompt_tokens")
            time.sleep(3)
            pf, gp = maxes(read_new(a.log, s))
            row["prefill_toks"] = round(pf) if pf else None
            row["decode_peak_toks"] = round(gp, 1) if gp else None
            s2 = logsize(a.log)
            fire(a.base, build(lvl, BASE), 0.8, 600)
            time.sleep(3)
            _, gb = maxes(read_new(a.log, s2))
            row["decode_baseline_toks"] = round(gb, 1) if gb else None
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}:{exc}"
        print("  ", json.dumps(row), flush=True)
        results["levels"].append(row)
        json.dump(results, open(a.out, "w"), indent=1)
    print("DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
