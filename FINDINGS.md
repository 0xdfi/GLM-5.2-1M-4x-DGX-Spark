# Legacy exp1 findings — GLM-5.2 NVFP4 on 4× DGX Spark

> **Archive status:** historical evidence only. Nothing in this file is a current selectable O14 profile, deployment promise, image claim, or network requirement. Use [`profiles/o14-profiles.json`](profiles/o14-profiles.json) for current O14 Fast and O14 Balanced status. Legacy exp1 DCP4 is not “O14 High.”

The exp1 measurements were collected in July 2026 on one four-node GB10 / `sm_121a` rig with a custom vLLM runtime, NVFP4 DS-MLA KV, B12X sparse-MLA, and MTP-5. Endpoints, paths, device names, and fabric interfaces are intentionally omitted; reproductions must supply their own.

## Legacy exp1 retained DCP1 speed fixtures

The same live DCP1 engine supplied the fixture values retained for O14 Fast's matched-speed row:

| Fixture | Historical result |
|---|---:|
| Repetitive/structured peak decode | 42.3 tok/s |
| Typical prose decode | ~28–32 tok/s |
| Adversarial counting decode | ~22 tok/s |
| ~53K actual prompt prefill | 819 tok/s |
| ~53K actual prompt decode | 29–33 tok/s |

Decode was MTP-acceptance-dependent at short context. At depth, the O(context) sparse indexer dominated and narrowed the difference between high- and low-acceptance tasks.

## Legacy exp1 topology exploration

| Legacy label | Historical max context with full graphs | Historical peak decode | Historical fastest prefill |
|---|---:|---:|---:|
| Legacy exp1 DCP1 | 375K | ~42 tok/s | ~820 tok/s |
| Legacy exp1 DCP2 | 625K | ~37 tok/s | 746 tok/s |
| Legacy exp1 DCP4 | 1M | ~38 tok/s short / ~30 tok/s deep | 614 tok/s |

These topology edges were experimental measurements, not current capacities. They must not be copied into the current selector, turned into image tags, or renamed as O14 profiles.

## Legacy exp1 engineering observations

1. **Sparse-indexer scratch scaled with `context × batch`.** The `fold_indices` allocation was not sharded by DCP, so large prefill batches reduced the practical long-context edge.
2. **NVFP4 KV was the fit enabler.** The 368-byte record was materially smaller than the FP8 path used in earlier experiments.
3. **CUDA graphs traded memory for decode speed.** Graph execution improved decode but reduced memory available for context.
4. **Decode-context parallelism imposed a collective tax.** More sharding extended context experiments but added per-step synchronization.
5. **Fabric bandwidth was not the primary lever.** One versus two active RoCE links changed throughput by only about 1.5–3% in the observed workload. No link count, fixed interface, or link-rate requirement is carried forward.
6. **Unified-memory load pressure was operationally significant.** Page-cache and swap state could affect model load stability.

## Reproduction boundary

- [`bench/bench_engine.py`](bench/bench_engine.py) requires explicit endpoint, log, levels, label, and output arguments.
- [`bench/v1_64k.py`](bench/v1_64k.py) requires explicit endpoint, log, and output arguments.
- [`serve/serve-dcp2-nvfp4.sh`](serve/serve-dcp2-nvfp4.sh) is a gated, parameterized legacy launcher. It has no private address/path/interface defaults and is not an O14 Balanced recipe.

Balanced matched-speed cells remain `TBD` until the exact retained DCP1 fixtures run against an accepted DCP2 runtime.
