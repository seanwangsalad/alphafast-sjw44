# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

AlphaFast: drop-in framework wrapping AlphaFold 3 (AF3) that replaces Jackhmmer with MMseqs2-GPU for MSA search. ~68x faster MSA, ~22x faster end-to-end on single H200. Supports Docker, HPC (Singularity/SLURM), and Modal serverless.

The package is a fork/derivative of `google-deepmind/alphafold3`. Source lives in `src/alphafold3/`. Runtime requires AF3 model weights (`af3.bin.zst`) obtained separately from Google.

## Running Inference

All production runs go through `scripts/run_alphafast.sh`, which launches Docker or Singularity and orchestrates the two-stage pipeline internally:

```bash
# Single GPU
./scripts/run_alphafast.sh \
    --input_dir /path/to/inputs \
    --output_dir /path/to/outputs \
    --db_dir /path/to/databases \
    --weights_dir /path/to/weights

# Multi-GPU (delegates to scripts/run_multigpu.sh internally)
./scripts/run_alphafast.sh \
    --input_dir /path/to/inputs \
    --output_dir /path/to/outputs \
    --db_dir /path/to/databases \
    --weights_dir /path/to/weights \
    --gpu_devices 0,1,2,3

# HPC/memory-constrained
./scripts/run_alphafast.sh ... \
    --temp_dir /scratch/$USER/alphafast_tmp \
    --lowram \
    --mmseqs_split_memory_limit 8G
```

## Two-Stage Pipeline (inside container)

AF3 inference runs in two sequential stages, intentionally separated so MMseqs2-GPU gets full GPU VRAM without competing with JAX:

**Stage 1 — Data pipeline** (`run_data_pipeline.py`): MSA search + template search. No JAX imports. Outputs `*_data.json` files.

**Stage 2 — Inference** (`run_alphafold.py --norun_data_pipeline`): JAX/Haiku model. Reads `*_data.json` from Stage 1.

Multi-GPU mode runs both stages in parallel across N GPUs via phase separation (all GPUs MSA first, barrier, all GPUs fold).

## Key Source Files

| File | Role |
|------|------|
| `src/alphafold3/data/pipeline.py` | `DataPipelineConfig`; orchestrates MSA + template search per chain |
| `src/alphafold3/data/msa.py` | Single-chain MSA dispatch; pipelined multi-DB search |
| `src/alphafold3/data/msa_config.py` | `MmseqsConfig`, `DataPipelineConfig` dataclasses |
| `src/alphafold3/data/tools/mmseqs.py` | `Mmseqs` tool wrapper; calls `mmseqs search` |
| `src/alphafold3/data/tools/mmseqs_batch.py` | `MmseqsBatch` / `MmseqsMultiDBBatch`; batch all inputs into one GPU search |
| `src/alphafold3/common/folding_input.py` | Input JSON parsing; `ProteinChain`, `RnaChain` etc. |
| `src/alphafold3/model/inference.py` | `ModelRunner`, `process_fold_input` |
| `run_data_pipeline.py` | CLI entrypoint for Stage 1 |
| `run_alphafold.py` | CLI entrypoint for Stage 2 (or combined) |
| `scripts/run_alphafast.sh` | Main orchestration shell script |
| `scripts/run_multigpu.sh` | Multi-GPU phase-separated orchestration |

## Sean's Local Modifications (claude_seanedits.md)

Changes on top of upstream AlphaFast:

1. **`--mmseqs_split_memory_limit`** — caps MMseqs2 RAM so it chunks large DBs; default = ~50% of `MemAvailable`. Plumbed through `MmseqsConfig` → `Mmseqs`/`MmseqsBatch`/`MmseqsMultiDBBatch` → `run_data_pipeline.py` → shell scripts.

2. **`--lowram`** — disables pipelining; waits for `result2msa` to finish before starting next DB search. Slower, but caps peak RAM to one DB at a time.

3. **JSON null semantics** — `"unpairedMsa": null` now means "skip search, use empty MSA" (not "run search"). Key absent = run search. Applies to `pairedMsa` and `templates` too. Logic in `folding_input.py`.

4. **Template/MSA branch logic** — `pipeline.py::process_protein_chain` respects user-provided templates end-to-end (skips PDB search and Foldseek when `chain.templates` is already set).

5. **Container source-mount** — Singularity branches in `run_alphafast.sh` bind-mount `../` into `/app/alphafold` so host edits apply without `.sif` rebuild. Two pre-built pickles extracted from container:
   - `src/alphafold3/constants/converters/ccd.pickle`
   - `src/alphafold3/constants/converters/chemical_component_sets.pickle`
   Both are `.gitignore`d (large binaries).

## Edit Scope

Most of `src/alphafold3/` is upstream AF3. Tread carefully — patches diverge from `google-deepmind/alphafold3` and complicate future rebases.

**Sean's mod surface (safe to extend):**
- `src/alphafold3/data/pipeline.py` — chain dispatch, template/MSA branch logic
- `src/alphafold3/data/msa.py`, `msa_config.py` — MSA orchestration + config
- `src/alphafold3/data/tools/mmseqs.py`, `mmseqs_batch.py` — MMseqs wrappers
- `src/alphafold3/common/folding_input.py` — JSON null semantics
- `run_data_pipeline.py`, `run_alphafold.py` — CLI flags
- `scripts/*.sh` — orchestration

**Upstream (avoid unless required):**
- `src/alphafold3/model/` — JAX/Haiku model code
- `src/alphafold3/constants/` — chem constants, generated pickles
- `src/alphafold3/structure/`, `cpp/` — pybind11 C++ ext

If upstream edit needed, note in `claude_seanedits.md`.

## Development Setup

Install (Linux x86_64/aarch64, Python 3.12 required):
```bash
uv sync
```

Build C++ extension (pybind11 via scikit-build-core):
```bash
uv run build_data  # also generates .pickle constants
```

Test data lives in `src/alphafold3/test_data/`. No test runner is currently configured; `pytest` is available as a dev dependency.

## Input Format

Input JSON files follow AF3 dialect. Key AlphaFast extensions:
- `"unpairedMsa": null` or `"pairedMsa": null` → skip MSA search, use empty
- `"templates": null` → skip template search, use empty
- Key absent (not present in JSON) → run search (AF3 default behavior)
- Custom templates: `"templates": [{"mmcifPath": "...", "queryIndices": [...], "templateIndices": [...]}]`

See `docs/input_format.md` for full reference.

## Environment Variables (inside container)

```
XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"   # required; avoids XLA slowdown
XLA_CLIENT_MEM_FRACTION=0.95                      # JAX pre-alloc fraction
```

For V100 (compute 7.x): `XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"`
