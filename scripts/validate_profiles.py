#!/usr/bin/env python3
"""Fail-closed validation for the public O14 profile index."""

from __future__ import annotations

import ipaddress
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTOR = ROOT / "profiles" / "o14-profiles.json"
TEXT_SUFFIXES = {".md", ".py", ".sh", ".json", ".yml", ".yaml", ".txt"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_selector() -> None:
    data = json.loads(SELECTOR.read_text(encoding="utf-8"))
    require(data["schema"] == "glm52-o14-public-profile-index/1", "unexpected schema")
    require(set(data["profiles"]) == {"o14-fast", "o14-balanced"}, "unexpected selectable profile set")

    fast = data["profiles"]["o14-fast"]
    require(fast["status"] == "READY" and fast["deployable"] is True, "Fast must be READY/deployable")
    require(fast["topology"] == {
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }, "Fast topology drift")
    require(fast["capacity"] == {
        "public_total_kv_label": "250K",
        "display_total_kv_tokens": 250000,
        "allocator_total_kv_tokens": 250023,
        "kv_cache_memory_bytes_per_rank": 7995534848,
        "max_model_len": 249000,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 2048,
    }, "Fast exact recipe drift")
    require(fast["image"]["reference"] is None and fast["image"]["digest"] is None, "false Fast image claim")
    require(fast["matched_speed"] == {
        "fixture_set": "legacy-exp1-dcp1",
        "prefill_tokens_per_second": 819,
        "decode_peak_tokens_per_second": 42.3,
        "deep_prompt_tokens": 53000,
        "deep_decode_tokens_per_second_range": [29, 33],
    }, "Fast retained DCP1 speed drift")
    fast_blob = json.dumps(fast, sort_keys=True).lower()
    for banned in ("projected", "candidate", "not-validated", "not validated"):
        require(banned not in fast_blob, f"Fast contains banned qualifier: {banned}")

    balanced = data["profiles"]["o14-balanced"]
    require(balanced["status"] == "TESTING" and balanced["deployable"] is False, "Balanced must fail closed")
    require(balanced["status_label"] == "TESTING / DO NOT DEPLOY", "Balanced warning drift")
    require(balanced["topology"] == {
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 2,
        "pipeline_parallel_size": 1,
    }, "Balanced topology drift")
    cap = balanced["capacity"]
    require(cap["target_total_kv_tokens"] == 500000, "Balanced target drift")
    require(cap["expected_allocator_total_kv_tokens"] == 500237, "Balanced allocator drift")
    require(cap["kv_cache_memory_bytes_per_rank"] == 8000000000, "Balanced KV bytes drift")
    require(cap["max_model_len"] == 490000, "Balanced max length drift")
    require(all(balanced["image"][key] is None for key in ("reference", "digest")), "false Balanced image claim")
    require(all(balanced["source"][key] is None for key in ("repository", "url", "commit", "dockerfile", "checksum_manifest")), "false Balanced source claim")
    speed = balanced["matched_speed"]
    require(all(speed[key] is None for key in (
        "prefill_tokens_per_second", "decode_peak_tokens_per_second", "deep_prompt_tokens", "deep_decode_tokens_per_second_range"
    )), "Balanced speed must stay TBD")
    require(data["legacy"]["exp1"]["selectable"] is False, "legacy exp1 became selectable")


def iter_text_files():
    for path in sorted(ROOT.rglob("*")):
        if ".git" in path.parts or not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path, path.read_text(encoding="utf-8", errors="strict")


def validate_privacy_and_active_docs() -> None:
    forbidden_literals = (
        "/home" + "/dfi",
        "/exp1" + "-evidence",
        "rocep1" + "s0f1",
        "roceP2" + "p1s0f1",
        "enp1s0" + "f1np1",
    )
    private_hits: list[str] = []
    for path, text in iter_text_files():
        for literal in forbidden_literals:
            if literal in text:
                private_hits.append(f"{path.relative_to(ROOT)}:{literal}")
        for match in re.finditer(r"(?<![0-9.])(?:10|172|192)\.(?:\d{1,3}\.){2}\d{1,3}(?![0-9.])", text):
            try:
                address = ipaddress.ip_address(match.group(0))
            except ValueError:
                continue
            if address.is_private:
                private_hits.append(f"{path.relative_to(ROOT)}:{address}")
        if re.search(r"\b200\s*(?:Gb/s|GbE|Gbit/s)\b", text, re.IGNORECASE):
            private_hits.append(f"{path.relative_to(ROOT)}:stale-200G-wording")
    require(not private_hits, "private/stale defaults found: " + ", ".join(private_hits))

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    current, marker, legacy = readme.partition("## Legacy exp1")
    require(bool(marker), "README lacks Legacy exp1 boundary")
    require("O14 Fast" in current and current.index("O14 Fast") < current.index("O14 Balanced"), "Fast must appear first")
    require(not re.search(r"\b(?:375K|625K)\b", current), "stale capacity in current README")
    require("Legacy exp1 DCP4" in legacy and "O14 High" not in current, "legacy DCP4 labeling drift")

    fast_section = current.split("## O14 Fast", 1)[1].split("## O14 Balanced", 1)[0].lower()
    for banned in ("projected", "candidate", "not validated", "not-validated"):
        require(banned not in fast_section, f"Fast README contains banned qualifier: {banned}")

    launcher = (ROOT / "serve" / "serve-dcp2-nvfp4.sh").read_text(encoding="utf-8")
    require("LEGACY_EXP1_ACK" in launcher, "legacy launcher lacks acknowledgment gate")
    require("MAX_MODEL_LEN:-" not in launcher and "KV_CACHE_MEMORY_BYTES:-" not in launcher, "legacy capacity remains selectable by default")


def validate_chart() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "render_profile_chart.py"), "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    require(result.returncode == 0, result.stdout + result.stderr)


def main() -> int:
    validate_selector()
    validate_privacy_and_active_docs()
    validate_chart()
    print("O14 profile index validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
