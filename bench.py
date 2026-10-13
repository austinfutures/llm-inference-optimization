"""Latency benchmark harness: TTFT, per-token decode latency (p50/p99), tokens/sec, peak memory.

    python bench.py                          # all variants
    python bench.py --variants sdpa_fp16 nf4 --runs 5

Batch size 1, greedy decoding, fixed prompt length and fixed number of generated tokens,
so every variant does identical work. Timing uses a streamer that synchronizes the GPU
after each token, giving true per-token latencies rather than an average.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import traceback

import torch
from transformers.generation.streamers import BaseStreamer

from common import DEFAULT_MODEL, VARIANTS, free, load

TEXT = ("The history of computing is a story of abstraction layered on abstraction, "
        "from vacuum tubes to transistors to compilers to neural networks. ")


class TimingStreamer(BaseStreamer):
    """Records a GPU-synchronized timestamp every time generate() emits tokens."""
    def __init__(self):
        self.stamps = []

    def put(self, value):
        torch.cuda.synchronize()
        self.stamps.append(time.perf_counter())

    def end(self):
        pass


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def run_once(model, input_ids, new_tokens):
    s = TimingStreamer()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.generate(input_ids, max_new_tokens=new_tokens, min_new_tokens=new_tokens,
                   do_sample=False, streamer=s, pad_token_id=model.config.eos_token_id)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    # stamps[0] = prompt echo, stamps[1] = first generated token, ...
    ttft = s.stamps[1] - t0
    itl = [b - a for a, b in zip(s.stamps[1:], s.stamps[2:])]
    return ttft, itl, t_end - t0


def bench_variant(variant, args):
    print(f"\n=== {variant}: {VARIANTS[variant]}")
    torch.cuda.reset_peak_memory_stats()
    tok, model = load(variant, args.model)

    ids = tok(TEXT * 64, return_tensors="pt").input_ids[:, : args.prompt_len].cuda()

    warmups = 3 if variant == "compile" else 1     # compile needs extra passes to settle
    for i in range(warmups):
        t = time.perf_counter()
        run_once(model, ids, args.new_tokens)
        print(f"  warmup {i + 1}/{warmups}: {time.perf_counter() - t:.1f}s")

    ttfts, itls, totals = [], [], []
    for _ in range(args.runs):
        ttft, itl, total = run_once(model, ids, args.new_tokens)
        ttfts.append(ttft); itls.extend(itl); totals.append(total)

    decode_time = statistics.mean(totals) - statistics.mean(ttfts)
    res = dict(
        variant=variant, desc=VARIANTS[variant], model=args.model,
        gpu=torch.cuda.get_device_name(), prompt_len=ids.shape[1],
        new_tokens=args.new_tokens, runs=args.runs,
        ttft_ms_p50=1e3 * pct(ttfts, 50), ttft_ms_p99=1e3 * pct(ttfts, 99),
        itl_ms_p50=1e3 * pct(itls, 50), itl_ms_p99=1e3 * pct(itls, 99),
        decode_tok_s=(args.new_tokens - 1) / decode_time,
        e2e_ms_p50=1e3 * pct(totals, 50),
        peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20,
    )
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:>14}: {v:,.2f}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, f"bench_{variant}.json"), "w") as f:
        json.dump(res, f, indent=2)
    free(model)
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt-len", type=int, default=256)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    if len(args.variants) > 1:
        # Run each variant in its own process: a hard crash in one backend
        # (compiler, quantization library) can't take the others down with it.
        print(f"GPU: {torch.cuda.get_device_name()}   model: {args.model}", flush=True)
        failed = []
        for v in args.variants:
            cmd = [sys.executable, __file__, "--variants", v, "--model", args.model,
                   "--prompt-len", str(args.prompt_len), "--new-tokens", str(args.new_tokens),
                   "--runs", str(args.runs), "--out", args.out]
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                failed.append(v)
                print(f"  !! {v} exited with code {rc}", flush=True)
        print(f"\ndone. failed: {failed or 'none'}")
        return

    try:
        bench_variant(args.variants[0], args)
    except Exception as e:
        traceback.print_exc()
        print(f"  !! {args.variants[0]} failed: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
