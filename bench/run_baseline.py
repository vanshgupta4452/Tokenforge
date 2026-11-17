"""Phase 1 baseline: HF generate + TTFT/ITL/throughput, written as resumable JSON."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from threading import Thread

from bench.env import detect_platform, env_info, output_dir
from bench.metrics import RequestMetrics, StepTimer, summarize

PROMPTS = {
    32: "The capital of France is Paris. Explain why " + "this is true. " * 4,
    128: "Write a short story about a robot learning to paint. " * 8,
    512: "The history of computing can be traced through several eras. " * 32,
    1024: "Summarize the trade-offs of memory-bound inference step by step. " * 64,
}


def build_prompt(length: int) -> str:
    base = PROMPTS.get(length, PROMPTS[128])
    return (base * (length // len(base) + 1))[:length]


def load_model(name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return tokenizer, model


def run_once(tokenizer, model, prompt: str, max_new_tokens: int) -> RequestMetrics:
    from transformers import TextIteratorStreamer

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    record = RequestMetrics(arrival_s=time.perf_counter())
    worker = Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "streamer": streamer,
            "pad_token_id": tokenizer.eos_token_id,
        },
    )
    worker.start()
    for _ in streamer:
        record.record_token()
    worker.join()
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--prompt-lens", default="32,128,512,1024")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    platform_name = detect_platform()
    out = Path(args.out) if args.out else output_dir(platform_name) / "baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": time.strftime("%Y-%m-%d_%H-%M-%S") + "_baseline",
        "env": env_info().__dict__,
        "config": {
            "engine": "hf_generate",
            "model": args.model,
            "prompt_lens": args.prompt_lens,
            "output_len": args.output_len,
            "warmups": args.warmups,
            "runs": args.runs,
        },
        "results": [],
    }
    completed = set()
    if out.exists():
        previous = json.loads(out.read_text())
        completed = {row["prompt_len"] for row in previous.get("results", [])}
        result["results"] = previous.get("results", [])

    tokenizer, model = load_model(args.model)
    timer = StepTimer()

    for prompt_len in (int(value) for value in args.prompt_lens.split(",")):
        if prompt_len in completed:
            continue
        import torch

        for _ in range(args.warmups):
            run_once(tokenizer, model, build_prompt(prompt_len), 8)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            timer.start()
        records = [
            run_once(tokenizer, model, build_prompt(prompt_len), args.output_len)
            for _ in range(args.runs)
        ]
        if torch.cuda.is_available():
            elapsed = timer.stop_ms()
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            elapsed, peak_mb = 0.0, 0.0
        metrics = summarize(records, peak_mb)
        row = {"prompt_len": prompt_len, "elapsed_ms": elapsed, **metrics}
        result["results"].append(row)
        out.write_text(json.dumps(result, indent=2))

    print(out)
    print(json.dumps(result["results"], indent=2))


if __name__ == "__main__":
    main()
