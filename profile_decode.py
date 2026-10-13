"""Where does decode time go? Profiles 32 decode steps and prints the top GPU kernels.

    python profile_decode.py --variant sdpa_fp16

Also writes a Chrome trace (open at chrome://tracing or ui.perfetto.dev) so you can see
the launch gaps between tiny kernels — the thing CUDA graphs (compile variant) remove.
"""
import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile

from common import DEFAULT_MODEL, VARIANTS, load


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="sdpa_fp16", choices=list(VARIANTS))
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    tok, model = load(args.variant, args.model)
    ids = tok("Explain how a GPU executes a matrix multiply.", return_tensors="pt").input_ids.cuda()
    gen = dict(max_new_tokens=32, min_new_tokens=32, do_sample=False,
               pad_token_id=model.config.eos_token_id)
    for _ in range(3 if args.variant == "compile" else 1):
        model.generate(ids, **gen)            # warmup
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model.generate(ids, **gen)
        torch.cuda.synchronize()

    events = prof.key_averages()
    for key in ("cuda_time_total", "device_time_total"):
        try:
            print(events.table(sort_by=key, row_limit=15))
            break
        except Exception:
            continue

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"trace_{args.variant}.json")
    prof.export_chrome_trace(path)
    print(f"trace -> {path}")


if __name__ == "__main__":
    main()
