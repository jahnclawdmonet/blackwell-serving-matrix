# Blackwell SM120 Serving Matrix

Measured LLM serving results on NVIDIA Blackwell (`sm_120`, RTX PRO 6000, 96 GB) across vLLM, SGLang and llama.cpp, with FP8 and NVFP4 passes. Each row in [`serving-matrix.jsonl`](serving-matrix.jsonl) is one exact cell: a model, a runtime and version, a quantization, a topology and a load point, with the startup result, throughput or latency where measured, memory profile, and a link to the raw artifact.

The point of the matrix is the cells that are **known-broken** as much as the known-good ones: serving a given model on Blackwell often fails on one specific runtime and backend combination while another boots fine, and the failure is rarely in the model. This records both, on real hardware.

## Schema (one JSON object per line)

`model_id`, `runtime`, `runtime_version`, `gpu`, `gpu_arch`, `vram_gb`, `quantization`, `kv_cache_dtype`, `tp`, `backend`, `context_len`, `concurrency`, `startup_status`, `correctness_status`, `agg_tokens_per_second`, `ttft_ms_p50`, `e2e_s_p99`, `available_kv_cache_gib`, `max_concurrency_x`, `peak_vram_gb`, `artifact_url`, `tested_at`, `note`.

## Notable cells

- `nvidia/Qwen3.6-35B-A3B-NVFP4` on vLLM 0.25.1 with `--moe-backend=flashinfer_b12x` reserves 11.47 GiB more than the `marlin` baseline before the KV cache is sized, so it OOMs 16-32 GB Blackwell cards during `profile_run` but starts on 96 GB. [Full repro.](https://conatus.jahn.ai/ai-engineering/sm120-b12x-workspace/)
- `Qwen/Qwen3-8B` FP8 on vLLM gives roughly 1.5x aggregate throughput over BF16 with no regression on a fixed 20-prompt check.

## Related

- Environment probe: [`blackwell-doctor`](https://github.com/jahnclawdmonet/blackwell-doctor) (`uvx blackwell-doctor`) prints your GPU, stack and a matrix key that indexes into this table.
- Error index: [Blackwell serving error index](https://conatus.jahn.ai/ai-engineering/blackwell-serving-errors/).

## Missing a cell?

If the exact combination you need is not here, it can be measured on the reference hardware and added: one public model and revision, one runtime, one quantization, one topology, one load point, returned with the exact command and raw logs. Fixed-scope single-cell verification: [conatus.jahn.ai/ai-engineering#sku-e](https://conatus.jahn.ai/ai-engineering/#sku-e).

MIT licensed. Operator: Conatus AI. Corrections and additions welcome.
