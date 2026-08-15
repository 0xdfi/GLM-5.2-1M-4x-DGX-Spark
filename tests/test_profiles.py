#!/usr/bin/env python3
"""Tests for exact O14 profile selection and publication safety."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProfileIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = json.loads((ROOT / "profiles" / "o14-profiles.json").read_text(encoding="utf-8"))

    def test_fast_exact_ready_profile(self) -> None:
        fast = self.data["profiles"]["o14-fast"]
        self.assertEqual((fast["status"], fast["deployable"]), ("READY", True))
        self.assertEqual(fast["capacity"]["allocator_total_kv_tokens"], 250023)
        self.assertEqual(fast["capacity"]["kv_cache_memory_bytes_per_rank"], 7995534848)
        self.assertEqual(fast["capacity"]["max_model_len"], 249000)
        self.assertEqual(fast["capacity"]["max_num_seqs"], 4)
        self.assertEqual(fast["capacity"]["max_num_batched_tokens"], 2048)
        self.assertEqual(fast["matched_speed"]["prefill_tokens_per_second"], 819)
        self.assertEqual(fast["matched_speed"]["decode_peak_tokens_per_second"], 42.3)
        self.assertEqual(fast["matched_speed"]["deep_decode_tokens_per_second_range"], [29, 33])

    def test_balanced_fails_closed_without_artifacts_or_speed(self) -> None:
        balanced = self.data["profiles"]["o14-balanced"]
        self.assertEqual((balanced["status"], balanced["deployable"]), ("TESTING", False))
        self.assertEqual(balanced["capacity"]["expected_allocator_total_kv_tokens"], 500237)
        self.assertIsNone(balanced["image"]["reference"])
        self.assertIsNone(balanced["image"]["digest"])
        self.assertIsNone(balanced["source"]["commit"])
        self.assertIsNone(balanced["matched_speed"]["prefill_tokens_per_second"])
        self.assertIsNone(balanced["matched_speed"]["decode_peak_tokens_per_second"])

    def test_only_fast_is_selectable(self) -> None:
        selected = [
            key for key, value in self.data["profiles"].items()
            if value["status"] == "READY" and value["deployable"] is True
        ]
        self.assertEqual(selected, ["o14-fast"])
        self.assertFalse(self.data["legacy"]["exp1"]["selectable"])

    def test_chart_rebuild_is_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chart.svg"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "render_profile_chart.py"), "--output", str(output)],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(output.read_bytes(), (ROOT / "profile-index-chart.svg").read_bytes())

    def test_full_validator(self) -> None:
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_profiles.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
