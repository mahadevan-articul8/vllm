# Gaudi FP8 and sparse attention changes (a8/gaudi-fp8-deepseek)

## Overview

This branch adds Gaudi (HPU) FP8 support and sparse attention for DeepSeek-style models, and fixes version handling when the repository is built from a non-git or manually synced tree.

**Commits:**

| Hash | Message |
|------|---------|
| `f073e15` | Handle version string for manually synced code |
| `c508a03` | Add gaudi fp8 support and sparse attention |

---

## Commit 1: Version string for manually synced code

**Why:** setuptools-scm can fail when the repo is not a full git clone or is synced without history (e.g. copy/sync without `.git`). The build would then fail when computing the package version.

**What:** Use a fallback version when SCM cannot determine the version:

- **pyproject.toml:** `fallback_version = "0.0.0"` under `[tool.setuptools_scm]` so setuptools-scm has a default when git metadata is missing.
- **setup.py:** In `get_vllm_version()`, pass `fallback_version="0.0.0"` to `get_version()` so a valid version string is always produced.

---

## Commit 2: Gaudi FP8 support and sparse attention

### fp8_utils.py

- **HPU path in `per_token_group_quant_fp8`:** When `current_platform.device_type == "hpu"`, use a pure PyTorch implementation (no Triton; Triton is not supported on HPU). This path computes per-group absmax, scale, and quantized values in PyTorch and handles both column-major and row-major scale layouts.
- **Triton fallback scope:** The Triton fallback is now explicitly for non-CUDA, non-HPU platforms. Comment updated accordingly.
- **BLOCK computation:** Use `2 ** math.ceil(math.log2(N))` instead of `triton.next_power_of_2(N)` for dynamo compatibility in the Triton fallback.

### sparse_attn_indexer.py

- **Workspace reservation on HPU:** During the profiling/dummy run, skip calling `current_workspace_manager().get_simultaneous(...)` when on HPU, because the workspace manager may not be initialized there.
- **`forward_hpu`:** New method that dispatches to the module-level `sparse_attn_indexer` (PyTorch implementation). Used when `current_platform.device_type == "hpu"`. Handles uninitialized KV cache during profiling.
- **`forward_native`:** Added branch for HPU that calls `forward_hpu`; the `NotImplementedError` message updated to "CUDA, ROCm, and HPU platforms."

### deepseek_v2.py

- **Indexer.forward:** Reshape `k` to `(-1, self.head_dim)` after `k_norm` and before the RoPE/split so that `k` has the same 2D shape as `q` for consistent handling. Required for HPU and compilation.

---

## Files changed

| File | Commit(s) | Summary |
|------|-----------|---------|
| `pyproject.toml` | f073e15 | setuptools-scm `fallback_version` for non-git/manual sync |
| `setup.py` | f073e15 | `get_version(..., fallback_version="0.0.0")` in `get_vllm_version()` |
| `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | c508a03 | HPU PyTorch path in `per_token_group_quant_fp8`; Triton fallback comment and BLOCK via `math` |
| `vllm/model_executor/layers/sparse_attn_indexer.py` | c508a03 | Skip workspace reservation on HPU; `forward_hpu`; error message includes HPU |
| `vllm/model_executor/models/deepseek_v2.py` | c508a03 | `k.view(-1, self.head_dim)` in Indexer forward for RoPE/split consistency |

---

## Platform notes

- On HPU (Gaudi), FP8 per-token-group quantization and the sparse attention indexer use **PyTorch fallbacks** only; there are no Triton or CUDA-style custom kernels for these paths on HPU.
- HPU-specific TPC or other kernels can be added later to optimize these paths.
