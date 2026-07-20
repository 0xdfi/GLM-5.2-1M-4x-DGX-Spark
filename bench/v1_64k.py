#!/usr/bin/env python3
"""v1 close-out: ONE 64K run on the live engine. Measures prefill + decode PEAK
(high-acceptance repetitive output) + decode FLOOR (low-acceptance counting), all
from vLLM's OWN logged throughput/acceptance (ground truth). Writes incrementally
to OUT so it can be polled over a flaky SSH link. Stdlib only."""
import json, os, re, sys, time, urllib.request

BASE = "http://192.168.100.10:8210"
LOG = "/home/dfi/exp1-evidence/runtime-dcp1-100k-graphs.log"
OUT = "/tmp/v1_64k_result.json"
FILLER = "The quick brown fox jumps over the lazy dog beside the quiet river. "
PROMPT_RE = re.compile(r"Avg prompt throughput:\s*([0-9.]+)")
GEN_RE = re.compile(r"Avg generation throughput:\s*([0-9.]+)")
ACC_RE = re.compile(r"Mean acceptance length:\s*([0-9.]+)")

def log(m):
    print(m, flush=True)
    with open(OUT + ".log", "a") as f:
        f.write(m + "\n")

def build(target, task):
    reps = max(1, int(target / 1.3) // len(FILLER.split()))
    return "Reference document (ignore its content):\n" + FILLER * reps + task

def fire(prompt, temp, gen):
    payload = json.dumps({"model": "glm-5.2", "messages": [{"role": "user", "content": prompt}],
                          "max_tokens": gen, "temperature": temp}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))

def newtext(start):
    with open(LOG, "r", errors="ignore") as f:
        f.seek(start); return f.read()

def mx(rx, t):
    v = [float(x) for x in rx.findall(t)]
    return round(max(v), 1) if v else None

PEAK_TASK = "\n\nNow repeat this exact line 100 times, one per line: The quick brown fox jumps over the lazy dog."
FLOOR_TASK = "\n\nNow count from 1 to 400, one integer per line, nothing else."

def run(label, task, temp):
    log(f"[{label}] firing 64K prompt ...")
    s = os.path.getsize(LOG)
    r = fire(build(64000, task), temp, 500)
    u = r.get("usage", {})
    time.sleep(4)
    t = newtext(s)
    row = {"phase": label, "actual_prompt_tokens": u.get("prompt_tokens"),
           "completion_tokens": u.get("completion_tokens"),
           "prefill_toks": mx(PROMPT_RE, t), "decode_toks": mx(GEN_RE, t),
           "max_accept_len": mx(ACC_RE, t)}
    log("  " + json.dumps(row))
    return row

def main():
    res = {"context": "64K", "config": "DCP1/100K/graphs", "rows": []}
    open(OUT + ".log", "w").close()
    res["rows"].append(run("PEAK", PEAK_TASK, 0.0))
    res["rows"].append(run("FLOOR", FLOOR_TASK, 0.1))
    json.dump(res, open(OUT, "w"), indent=1)
    log("DONE " + OUT)

if __name__ == "__main__":
    main()
