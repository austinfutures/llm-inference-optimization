"""Accuracy cost of each optimization: WikiText-2 perplexity."""
import argparse, json, math, os, subprocess, sys, time

import torch
from datasets import load_dataset

from common import DEFAULT_MODEL, free, load

CACHE = os.path.expanduser("~/.cache/wikitext2_test.txt")


def wikitext_text():
    """Download the WikiText-2 test split once (with retries, the Hub rate-limits
    anonymous requests) and cache it as plain text for every variant's process."""
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return f.read()
    for attempt in range(6):
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            break
        except Exception as e:
            if attempt == 5:
                raise
            wait = 5 * 2 ** attempt
            print(f"  dataset download failed ({type(e).__name__}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    text = "\n\n".join(ds["text"])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as f:
        f.write(text)
    return text


@torch.no_grad()
def perplexity(model, tok, max_tokens, window):
    ids = tok(wikitext_text(), return_tensors="pt").input_ids[:, :max_tokens].cuda()
    nll, count = 0.0, 0
    for i in range(0, ids.shape[1] - 1, window):
        chunk = ids[:, i: i + window]
        if chunk.shape[1] < 2:
            break
        loss = model(chunk, labels=chunk).loss
        nll += loss.item() * (chunk.shape[1] - 1)
        count += chunk.shape[1] - 1
    return math.exp(nll / count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["sdpa_fp16", "triton_rmsnorm", "int8", "nf4"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--max-tokens", type=int, default=40_000)
    p.add_argument("--window", type=int, default=1024)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if len(args.variants) > 1:     # one process per variant, same reason as bench.py
        wikitext_text()            # download once up front
        for v in args.variants:
            cmd = [sys.executable, __file__, "--variants", v, "--model", args.model,
                   "--max-tokens", str(args.max_tokens), "--window", str(args.window),
                   "--out", args.out]
            if subprocess.run(cmd).returncode != 0:
                print(f"  !! {v} failed", flush=True)
        return

    for v in args.variants:
        tok, model = load(v, args.model)
        ppl = perplexity(model, tok, args.max_tokens, args.window)
        print(f"{v:>15}: perplexity {ppl:.3f}")
        with open(os.path.join(args.out, f"ppl_{v}.json"), "w") as f:
            json.dump({"variant": v, "perplexity": ppl, "tokens": args.max_tokens}, f, indent=2)
        free(model)


if __name__ == "__main__":
    main()
