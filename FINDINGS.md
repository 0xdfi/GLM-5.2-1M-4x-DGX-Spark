# GLM-5.2 NVFP4 on 4× DGX Spark (GB10/sm_121a) — Results Matrix

Date: 2026-07-20 · Runtime: custom vLLM fork `exp1sm121a368r4dtypefix` (NVFP4 368-byte
DS-MLA KV + B12X sparse-MLA + MTP-5). Model: QuantTrio GLM-5.2 INT4/INT8-mix (stock, ~200GB,
124 shards). **GLM-5.2 = ~744B total / ~40B active** (256 experts, 8+1 routed), DSA+MLA, 1M ctx.
At INT4 ≈ 372GB weights → **cannot fit one 128GB Spark; multi-node is mandatory.** Fabric: dual
RoCE HCA (2×100 Gb/s). All configs served **correct output** (47×89=4183).

## Speed reality — the 22-vs-42 question, settled (2026-07-20)

Decode speed is **MTP-acceptance-bound and content-dependent**, not misconfigured. Live engine
confirmed optimal: `enforce_eager=False`, CUDA graphs `FULL_AND_PIECEWISE` captured (0.6 GiB),
DCP1, MTP-5, flashinfer-cutlass MoE. Measured on the SAME live DCP1/100K engine:

| Content acceptance | Mean accept len | Draft accept % | Decode tok/s |
|---|---|---|---|
| Adversarial (multi-digit counting) | 2.1 | 22–25% | ~22 (floor) |
| Typical prose | 3.5–4.1 | 49–63% | ~28–32 |
| Repetitive/structured | 5.4–6.0 | 88.5–**100%** | **42.3 (confirmed peak)** |

**64K-context run (v1 close-out datapoint, DCP1/100K/graphs, 53K actual prompt tokens):**
prefill **819 tok/s** · decode **29–33 tok/s** (accept len 4.8–6.0). At 64K both high- and low-acceptance
tasks converge to ~30 tok/s — the O(context) sparse indexer dominates per-step cost, so acceptance
matters less at depth. Decode-vs-depth: ~42 (short) → ~30 (64K), the expected graceful slope.
**v1 FORMALLY CLOSED 2026-07-20**: at/above community SOTA for GLM-5.2 on 4× Spark, unpruned.

- **42.3 tok/s peak matches/beats the ~40 tok/s community reports** — those are the high-acceptance
  ceiling, not hype. The 22–42 swing is inherent to *any* speculative decoder.
- **Bandwidth ceiling check**: TP4 per-node reads ~5GB/token → ~56 tok/s naive ceiling. Peak 42.3
  = **~75% of ceiling** — excellent; little headroom left on THIS drafter. Floor 22 = low-acceptance
  waste (drafting 5, accepting 2), the only real speed lever left → **better drafter (v2)**.
- **Competitive positioning (publication gold)**: best public 4× GB10 GLM-5.2 report is ~20–22 tok/s
  single-stream (CosmicRaisins, *pruned* to 218 experts). We run **unpruned** at 22 floor / 42 peak.
  8-node reports: 26–27 (unpruned) / 33–55 (W4A8, coding). **We are at or above community SOTA for
  GLM-5.2 on 4× Spark, unpruned.**

## v2/v3 speed roadmap (the "new tech" eval — 2026-07-20)

1. **DSpark speculator — EVALUATED & DECLINED for v1/v2 (2026-07-20). MTP-5 is the answer.**
   Red Hat's `RedHatAI/GLM-5.2-speculator.dspark` (downloaded to `/home/dfi/models/glm52-dspark-speculator`,
   kept for reference). Two blockers found by direct fork inspection:
   - **Not a swap — a multi-day port.** Fork registers `eagle3/peagle/mtp/deepseek_mtp/dflash(Qwen3)` only;
     NO `dspark`, no `speculators` lib. GLM impl (`glm4_moe*.py`) has NO aux-hidden-state/Eagle3 support,
     so it can't feed the DSpark draft (needs target hidden states at layers [8,23,39,55,70]). Would require
     implementing `DSparkDraftModel` (block-diffusion + markov + confidence heads), registering the algo,
     adding aux-hidden emission to GLM, installing speculators, and eating a verifier mismatch
     (draft trained on GLM-5.2-**FP8**; we serve INT4/INT8-mix).
   - **Physics says it won't pay off at depth.** Advertised 2.15× was short-ctx on 4×B300. We MEASURED that
     at 64K, accept-len 4.8 vs 6.0 both give ~30 tok/s — the O(context) sparse indexer dominates, so a better
     drafter (which only helps when acceptance-bound) converges to the same ~30 at the 64K–1M depths this
     program targets. A better drafter can't beat the indexer-bound ceiling at depth.
   - **Verdict: MTP-5 stays.** Revisit only if the real depth ceiling (indexer/DCP-collective cost) is
     attacked first, or when native long-ctx support lands upstream (#45317).
2. **"Native vLLM for all models on Spark" = DEBUNKED as stated.** vLLM nightly (`cu130-nightly`)
   validated sweet spot is 100–130B NVFP4 ~10–15B active — NOT 744B. sm_121 still fragmented:
   MXFP8 MoE falls back to slow Marlin (#43906), PR #37700 unmerged, NVIDIA NGC container lags
   upstream. Official multi-node tops out at a 2-node playbook. **Our 4-node fork is not replaceable
   by stock vLLM today.** Track for when #37700/#43906 land → clean rebase candidate.
3. **DFlash proper**: no GLM checkpoint (validated on Qwen/Llama/Gemma/gpt-oss only). Skip unless we train one.

## Design philosophy (Don, 2026-07-20): SPEED-FIRST — full CUDA graphs mandatory at every DCP level

Full CUDA graphs are treated as a REQUIREMENT (for decode speed); each topology's production
context = **the max context that still fits full graphs**. 1M is a target, NOT at the cost of speed.
Record the context-for-speed tradeoff at each level.

**Clean 1:2:4 scaling** (per-token KV halves with each DCP doubling; weights+graph-pool+runtime ~constant,
so max-context-with-graphs scales inversely with per-token KV):

| Option | per-token KV | Max ctx WITH full graphs | Peak decode | Floor | Confirmed |
|---|---|---|---|---|---|
| **DCP1 (fastest)** | 31,976 B | **375K** | ~42 peak | — | ✓ MEASURED 2026-07-20 (400K hangs) |
| **DCP2 (balanced)** | 15,988 B | **625K** | ~32 | — | ✓ MEASURED (700K crashes, 750K hangs) |
| **DCP4 (max ctx)** | 7,994 B | **1M** | ~27-32 | ~1.2 GB free | ✓ MEASURED w/ full graphs (no swap/thrash) |

**Max-context correction (2026-07-20):** the earlier ~200K/400K/813K figures were computed as
`KV_budget ÷ per-token-KV` — a KV-only model that ignored the **sparse index-cache + graph-capture
memory** (~46 GB at 500K, measured), which is the actual ceiling driver and is **NOT sharded by DCP**.
Re-measured to the real clean-boot edge with full graphs, seqs=1. All three were understated. Context
scales sub-linearly with DCP (375K→625K→1M ≈ 1.6×/doubling). Swap is OFF on all nodes and NVMe reads are
~0 during decode at every max — no swap or SSD thrash. Failure modes at the edge: hang / worker-crash / OOM.

Graphs beat eager on every axis except raw context: DCP4-800K-graphs (32 peak, 4.9GB floor) >
DCP4-1M-eager (27 peak, 2.1GB floor). Note: graphs cut kernel-LAUNCH overhead (big win at low
context, smaller at deep context where the O(context) sparse-indexer compute dominates) — so the
decode-vs-depth curve shifts up but still slopes down. Each topology decodes FASTER at a given depth
with LESS DCP sharding (DCP1 fastest, DCP4 reaches furthest).

## Three production options (the endgame)

| Option | Peak decode (MTP) | Prefill | Max context (OOM edge) | Hardened production ctx | Floor@hardened | Use |
|---|---|---|---|---|---|---|
| **DCP1 — fastest** | ~40 tok/s (39.5) | 828 tok/s | ~156K | ~156K | 6.1 GB | speed |
| **DCP2 — balanced** | ~32 tok/s | ~690 tok/s | ~500K | ~480K | (tight) | balance |
| **DCP4 — max context** | ~27 tok/s (26.6) | TBD sweep | **1.06M** | ~950K | ~2-4 GB | 1M context |

(Decode is PEAK with MTP-5 on high-acceptance content; creative/low-acceptance floors ~15-25% lower.)

## Key engineering discoveries (the non-obvious stuff — publication gold)

1. **B12X sparse-indexer scratch (`fold_indices`) scales with `context × batch`, NOT sharded by DCP.**
   Measured: 11.45 GiB at 600K×2048 → coefficient ≈ 9.32 bytes per (token×batch-token). This — not
   the KV cache — is the real ceiling driver at long context. It forces **small batch (512-1024) at
   high context**, which caps prefill. DCP4 quarters the KV but the indexer stays full-size, so 1M
   requires batch-512.
2. **The workload is compute-bound (MoE GEMMs) + collective-latency-bound, NOT bandwidth-bound.**
   We were using 1 of 2 RoCE ports (100/200 Gb/s = the observed "half bandwidth"); enabling both
   (`NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`) gave only ~1.5-3%. Kept it (free), but bandwidth is not
   the lever. (DCP2/DCP4 heavier collectives may benefit more — to confirm in sweep.)
3. **NVFP4 KV enables the whole thing.** NVFP4 (368B) is 58% of FP8 (656B) → ~2× the context per rank
   vs FP8. Stock production was FP8/DCP2/320K; NVFP4/DCP2 reaches ~480K and NVFP4/DCP4 reaches 1M+.
4. **DCP collective tax on decode is real and inherent**: DCP1 ~40 → DCP2 ~32 → DCP4 ~27 peak. More
   sharding = more per-step all-gather/reduce-scatter = slower decode. The context-vs-speed tradeoff.
5. **CUDA graphs (decode speed lever) need ~3 GB and only fit at reduced context** — so graphs+high-context
   don't coexist; graphs live in the DCP1/low-context fast option.
6. **r4 OOM fix proven**: transient NVRM `NV_ERR_NO_MEMORY` during load is absorbed by swapoff + the
   drop-caches loop; load recovers instead of cascading to worker-death. (Phase-A/B probe pre-proved it.)

## Decisions settled
- **Drafter: MTP-5 stays.** DSpark (untested at long ctx, needs uncensored base), EAGLE-3 (no GLM-5.2
  checkpoint), DFlash (no GLM-5.2 checkpoint, degrades at long ctx) — none beat MTP-5 for long context.
- **RoCE: dual-HCA active** (~1.5-3%, kept as free gain).
- **V2 rebase onto latest vLLM: DEFER.** NVFP4/MTP/MRv2 are tractable, but B12X sparse-MLA is
  "major surgery + indefinite fork" (upstream sparse-MLA broken on SM121, issue #45317). Newer kernels
  only ~4-14%. Revisit when #45317 lands upstream → then a clean cherry-pick. V1 ships on this fork.

## Remaining before publish
- [ ] Benchmark sweep: prefill + decode at 64K/128K/50%/90% for each topology (harness: shared/bench/).
- [ ] Harden DCP2@480K, DCP4@950K (5% below OOM edge); stress-test at full depth (real long prompts).
- [ ] NVFP4-vs-FP8 KV quality arm (retrieval at long context) to substantiate the quality claim honestly.
- [ ] Reproducible recipe + image for HF Hub + X (serve configs, image digest, bench scripts).
