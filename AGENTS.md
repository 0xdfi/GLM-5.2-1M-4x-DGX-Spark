# Agent instructions

[`profiles/o14-profiles.json`](profiles/o14-profiles.json) is the public selector of record. Read it before proposing any O14 action.

## Fail-closed profile selection

1. Match the requested key exactly (`o14-fast` or `o14-balanced`). Unknown names are plan-only.
2. Pull, build, or deploy only when `status` is exactly `READY` **and** `deployable` is `true`.
3. Treat `TESTING`, missing fields, `null` artifact identities, and legacy exp1 material as non-deployable. Do not fill gaps by inference.
4. Never turn legacy DCP1/DCP2/DCP4 settings into named profiles. In particular, never relabel DCP4 as “O14 High.”
5. Model weights, addresses, ports, device/interface names, and filesystem paths are operator inputs. Never copy private or example values into defaults.

## O14 Fast: select and build

`o14-fast` is **READY** and deployable at **250K total KV**, TP4/DCP1/PP1. Preserve these exact machine values:

```text
allocator_total_kv_tokens=250023
KV_CACHE_MEMORY_BYTES=7995534848
MAX_MODEL_LEN=249000
MAX_NUM_SEQS=4
MAX_NUM_BATCHED_TOKENS=2048
```

The selector pins canonical Harness commit `9ba54db8f7c852eabb90372f7dcf47f87eafbe20`. Check out that exact commit before validating or building.

```bash
git clone https://github.com/0xdfi/GLM-5.2-Harness-O14-4x-DGX-Spark.git
cd GLM-5.2-Harness-O14-4x-DGX-Spark
git checkout 9ba54db8f7c852eabb90372f7dcf47f87eafbe20
python3 scripts/validate.py
python3 reproducibility/verify.py
```

Verify `SHA256SUMS`, then build from `docker/Dockerfile.repro` with the exact Fast values above. The model checkpoint remains operator-supplied and outside the runtime image.

Fast has no published OCI image in this phase. Because `image.reference` and `image.digest` are `null`, do **not** run `docker pull` and do not invent `fast`, `o14-fast`, `latest`, or any derived tag. Build from the pinned source.

## O14 Balanced: do not pull, build, or deploy

`o14-balanced` is **TESTING / DO NOT DEPLOY** at a 500K target, TP4/DCP2/PP1. Its expected allocator total is 500237, KV allocation is 8000000000 bytes per rank, and `MAX_MODEL_LEN` is 490000.

Balanced has no accepted source recipe, image, digest, or speed result. Do not build, pull, launch, or adapt Fast. Never synthesize a Balanced tag. Acceptance must first publish a separate immutable DCP2 runtime source/recipe and image identity; it must not overwrite or retag Fast. Only then may Balanced change to `READY`.

## Legacy exp1

README and FINDINGS material under **Legacy exp1** is historical evidence, not a selector. The legacy launcher requires explicit operator parameters and an acknowledgment gate. It must not be used to infer current profile capacities, statuses, artifact names, or deployment commands.
