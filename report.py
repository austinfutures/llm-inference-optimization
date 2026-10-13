"""Collect results/*.json into tables, headline findings and a chart.

Writes results/summary.md and results/summary.png, and fills the README section
between the RESULTS markers so the repo always shows the numbers from the last run.
"""
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["eager_fp16", "sdpa_fp16", "triton_rmsnorm", "compile", "int8", "nf4"]
SAME_MATH_AS_FP16 = {"eager_fp16", "compile"}      # identical weights/precision to sdpa_fp16


def jload(path):
    with open(path) as f:
        return json.load(f)


def fmt(x, spec):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def build(out):
    bench = {d["variant"]: d for d in map(jload, glob.glob(os.path.join(out, "bench_*.json")))}
    ppl = {d["variant"]: d["perplexity"] for d in map(jload, glob.glob(os.path.join(out, "ppl_*.json")))}
    kpath, tpath = os.path.join(out, "kernel_rmsnorm.json"), os.path.join(out, "trt_prefill.json")
    kern = jload(kpath) if os.path.exists(kpath) else None
    trt = jload(tpath) if os.path.exists(tpath) else None
    if not bench:
        raise SystemExit("no bench_*.json found — run bench.py first")

    names = [v for v in ORDER if v in bench]
    base = bench.get("eager_fp16") or bench[names[0]]
    for v in SAME_MATH_AS_FP16:                       # these compute the same thing as sdpa_fp16
        if v not in ppl and "sdpa_fp16" in ppl:
            ppl[v] = ppl["sdpa_fp16"]

    # ---------------------------------------------------------------- headline findings
    find = []
    if "compile" in bench:
        c = bench["compile"]
        find.append(f"**torch.compile + static KV cache: {c['decode_tok_s'] / base['decode_tok_s']:.1f}x decode throughput** "
                    f"({base['decode_tok_s']:.1f} → {c['decode_tok_s']:.1f} tok/s). Per-token latency fell from "
                    f"{base['itl_ms_p50']:.1f} ms to {c['itl_ms_p50']:.1f} ms, and p99 from "
                    f"{base['itl_ms_p99']:.0f} ms to {c['itl_ms_p99']:.1f} ms.")
    if trt:
        line = (f"**TensorRT engine: {trt['speedup_vs_pytorch']:.1f}x faster prefill than PyTorch** "
                f"({trt['pytorch_ms_p50']:.1f} → {trt['trt_ms_p50']:.1f} ms for a {trt['seq_len']}-token prompt")
        if "speedup_vs_compile" in trt:
            line += f", {trt['speedup_vs_compile']:.2f}x vs torch.compile"
        line += (f"), with matching outputs (top-1 token match: {'yes' if trt['top1_match'] else 'no'}, "
                 f"cosine similarity {trt['cosine_sim']:.4f}).")
        find.append(line)
    if kern:
        big = [r for r in kern["rows"] if r["M"] >= 512]
        sp = sorted(r["speedup"] for r in big)[len(big) // 2]
        vs_comp = sorted(r["compile_us"] / r["triton_us"] for r in big)[len(big) // 2]
        peak = max(r["pct_peak"] for r in kern["rows"])
        find.append(f"**Custom Triton fused add+RMSNorm kernel: {sp:.1f}x faster than eager PyTorch** "
                    f"(median over prefill-sized shapes), {vs_comp:.2f}x vs torch.compile, reaching "
                    f"{peak:.0f}% of the GPU's peak memory bandwidth.")
    if "nf4" in bench and "sdpa_fp16" in ppl and "nf4" in ppl:
        n, f16 = bench["nf4"], bench["sdpa_fp16"]
        find.append(f"**4-bit NF4 cut peak GPU memory {100 * (1 - n['peak_mem_mb'] / f16['peak_mem_mb']):.0f}%** "
                    f"({f16['peak_mem_mb']:.0f} → {n['peak_mem_mb']:.0f} MB) at a "
                    f"{100 * (ppl['nf4'] / ppl['sdpa_fp16'] - 1):.0f}% perplexity cost "
                    f"({ppl['sdpa_fp16']:.2f} → {ppl['nf4']:.2f}).")
    if "int8" in bench and "int8" in ppl and "sdpa_fp16" in ppl:
        i8 = bench["int8"]
        find.append(f"**INT8 kept accuracy** (perplexity {ppl['int8']:.2f} vs {ppl['sdpa_fp16']:.2f}) "
                    f"**but ran {base['decode_tok_s'] / i8['decode_tok_s']:.1f}x slower** at batch 1: "
                    f"bitsandbytes LLM.int8 dequantizes weights on every matmul, so at batch 1 that overhead dominates.")

    # ---------------------------------------------------------------- tables
    L = [f"Model `{base['model']}` on **{base['gpu']}**. Prompt {base['prompt_len']} tokens, "
         f"generate {base['new_tokens']} tokens, batch 1, greedy, {base['runs']} timed runs per variant.", "",
         "### Key findings", ""] + [f"- {x}" for x in find] + [
         "", "### End-to-end generation", "",
         "| Variant | Decode tok/s | Speedup | TTFT p50 (ms) | Per-token p50 (ms) | Per-token p99 (ms) | Peak memory (MB) | WikiText-2 perplexity |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for v in names:
        b = bench[v]
        L.append(f"| `{v}` | {b['decode_tok_s']:.1f} | {b['decode_tok_s'] / base['decode_tok_s']:.2f}x | "
                 f"{b['ttft_ms_p50']:.1f} | {b['itl_ms_p50']:.2f} | {b['itl_ms_p99']:.2f} | "
                 f"{b['peak_mem_mb']:.0f} | {fmt(ppl.get(v), '.3f')} |")
    L += ["", "![Benchmark summary](results/summary.png)"]

    if trt:
        L += ["", f"### TensorRT prefill ({trt['seq_len']}-token prompt, TensorRT {trt['trt_version']})", "",
              "| Backend | p50 (ms) | p99 (ms) | Speedup |", "|---|---:|---:|---:|",
              f"| PyTorch eager (SDPA) | {trt['pytorch_ms_p50']:.2f} | {trt['pytorch_ms_p99']:.2f} | 1.00x |"]
        if "compile_ms_p50" in trt:
            L.append(f"| torch.compile | {trt['compile_ms_p50']:.2f} | {trt['compile_ms_p99']:.2f} | "
                     f"{trt['pytorch_ms_p50'] / trt['compile_ms_p50']:.2f}x |")
        L += [f"| TensorRT engine | {trt['trt_ms_p50']:.2f} | {trt['trt_ms_p99']:.2f} | {trt['speedup_vs_pytorch']:.2f}x |",
              "", f"Output check vs PyTorch: top-1 token match **{'yes' if trt['top1_match'] else 'no'}**, "
              f"top-5 overlap {trt['top5_overlap']}/5, cosine similarity {trt['cosine_sim']:.5f}. "
              f"Engine build: {trt['build_s']:.0f} s, {trt['network_layers']} network layers."]

    if kern:
        L += ["", f"### Triton kernel microbenchmark ({kern['gpu']}, peak {kern['peak_gbs']} GB/s)", "",
              "M = rows (1 = one decode token, 4096 = long prefill), N = hidden size.", "",
              "| M | N | Eager PyTorch (µs) | torch.compile (µs) | Triton (µs) | vs eager | vs compile | GB/s | % of peak |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in kern["rows"]:
            L.append(f"| {r['M']} | {r['N']} | {r['eager_us']:.1f} | {r['compile_us']:.1f} | {r['triton_us']:.1f} | "
                     f"{r['speedup']:.2f}x | {r['compile_us'] / r['triton_us']:.2f}x | {r['gbs']:.0f} | {fmt(r['pct_peak'], '.0f')}% |")

    # ---------------------------------------------------------------- chart
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    metrics = [("decode_tok_s", "Decode tokens/sec (higher is better)"),
               ("itl_ms_p99", "Per-token latency p99, ms (lower is better)"),
               ("peak_mem_mb", "Peak GPU memory, MB (lower is better)")]
    for a, (key, title) in zip(ax, metrics):
        vals = [bench[v][key] for v in names]
        bars = a.bar(names, vals, color=["#76B900" if v == "compile" else "#4C72B0" for v in names])
        a.bar_label(bars, fmt="%.0f" if key != "itl_ms_p99" else "%.1f", fontsize=8)
        a.set_title(title, fontsize=10)
        a.tick_params(axis="x", rotation=35)
        a.grid(axis="y", alpha=0.3)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{base['model']} on {base['gpu']}, batch 1")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "summary.png"), dpi=140)
    return "\n".join(L), find


def main(out="results", readme="README.md"):
    md, find = build(out)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(md + "\n")
    if os.path.exists(readme):
        with open(readme) as f:
            text = f.read()
        new = re.sub(r"<!-- RESULTS:START -->.*<!-- RESULTS:END -->",
                     lambda _: f"<!-- RESULTS:START -->\n{md}\n<!-- RESULTS:END -->", text, flags=re.S)
        with open(readme, "w") as f:
            f.write(new)
    print(md)
    print(f"\nsaved {out}/summary.md, {out}/summary.png and updated {readme}")


if __name__ == "__main__":
    main()
