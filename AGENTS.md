# MiniServe

Goal: a from-scratch LLM inference engine (KV cache, continuous batching,
optional paged cache), profiled and compared against vLLM.
Model: Qwen2.5-0.5B (draft) and Qwen2.5-1.5B (target), fp16, single T4.

## Rules
- Correctness first: any engine or kernel change needs a test that matches the
  HF / PyTorch reference before it is benchmarked.
- Never invent benchmark numbers. Results come from bench/results/*.json.
- Keep files small, typed, and documented with the why.
- CPU-testable logic (scheduler, allocator, sampler) must have unit tests.
- Do not add dependencies without updating pyproject.toml.

## Commands
- Unit tests (CPU): pytest tests -q
- Kernel tests (GPU): pytest kernels/tests -q
- Baseline: python bench/run_baseline.py --model Qwen/Qwen2.5-0.5B
- Benchmark: python bench/run_matrix.py --config bench/configs/<name>.yaml
- Regression check: python bench/regression_check.py --new <json> --base <json> --max-regress 0.10

## Layout
engine/ kernels/ quant/ bench/ profiles/ docs/ tests/ scripts/ notebooks/

## Where to run
- All logic lives in this repo. Notebooks are thin runners only.
- CPU (this workspace): pytest tests -q, tiny random-weight models.
- GPU: Kaggle is primary; Colab is the alternative. The same scripts run in both.
- The runner notebook clones/pulls the repo, installs deps, runs ONE script,
  and writes JSON to the persistent output directory chosen by bench/env.py.
- Download the JSON from Kaggle/Colab, commit it under bench/results, and plot
  from JSON. Never copy numbers by hand.

## Environment notes
- GPU runs happen on Kaggle (T4, fp16 only, no FP8, no FlashAttention 2) or
  Colab (T4 free tier; L4/A100 only if paid). The engine targets sm_75.
- Kaggle: internet is OFF by default; enable it. /kaggle/working is the only
  persistent path. Attach model weights as a Kaggle Dataset to avoid re-downloads.
- Colab: mount Drive for persistence; /content is wiped when the session ends.
- Nsight hardware counters are usually unavailable in both; fall back to
  torch.profiler + CUDA events and record that limitation in docs/profile_report.md.
