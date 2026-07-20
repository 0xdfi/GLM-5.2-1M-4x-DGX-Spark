#!/usr/bin/env python3
"""Fix the DSA indexer's expanded_block_table_buffer being one block too small under MTP.

BUG
---
The DSA (DeepSeek sparse attention) indexer sizes its ``expanded_block_table_buffer``
from ``max_model_len`` alone::

    max_num_blocks_per_req = cdiv(max_model_len, block_size * cp_world_size)

With ``max_model_len=200000`` and ``block_size=64`` that is 200000/64 = **3125 blocks**.
But MTP speculative decoding (k=4) can extend a request past ``max_model_len`` by up to
``num_speculative_tokens``, requiring **3126 blocks**. At >= 3 concurrent requests this
reliably crashes the engine with:

    RuntimeError: The expanded size of the tensor (3125) must match the existing size (3126)

FIX
---
Size the buffer one block larger. This script patches the installed vLLM tree
(``vllm/v1/attention/backends/mla/indexer.py``), appending
``+ 1`` to the ``cdiv(...)`` result.

SCOPE
-----
* Applies to vLLM ref ab666069935c1f23e8ef56038b4659ac9e8f19f8 (jasl/eugr lineage).
* Discovered 2026-07-05 on this cluster (4x DGX Spark GB10, GLM-5.2 QuantTrio
  Int4-Int8Mix, 200K ctx, MTP k=4).
* Likely affects ANY ``max_model_len`` that is an exact multiple of ``block_size``
  when MTP is enabled — the overhang only lands in a fresh block when the context
  limit falls exactly on a block boundary.

USAGE
-----
Run inside the container (e.g. during the "bake mods" step, see README)::

    python3 /patches/fix-indexer-mtp-overhang.py

Idempotent: re-running after the patch is applied is a no-op.
"""

import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/indexer.py"

OLD = """\
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        )
        self.expanded_block_table_buffer = torch.zeros(
"""

NEW = """\
        max_num_blocks_per_req = cdiv(
            self.vllm_config.model_config.max_model_len,
            self.kv_cache_spec.block_size * get_total_cp_world_size(),
        ) + 1  # MTP spec tokens can extend a request one block past max_model_len
        self.expanded_block_table_buffer = torch.zeros(
"""


def main() -> int:
    with open(TARGET, "r", encoding="utf-8") as f:
        src = f.read()

    if NEW in src:
        print(f"already patched: {TARGET}")
        return 0

    if OLD not in src:
        print(f"ERROR: expected block not found in {TARGET} — vLLM ref mismatch?",
              file=sys.stderr)
        return 1

    with open(TARGET, "w", encoding="utf-8") as f:
        f.write(src.replace(OLD, NEW, 1))

    print(f"patched: {TARGET} (+1 block on expanded_block_table_buffer)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
