"""Fused residual-add + RMSNorm in Triton.

One program per row: load x (and the residual), add, reduce sum of squares in FP32,
normalize, scale by weight, write out. The unfused PyTorch path reads/writes the
hidden state several times (add -> pow -> mean -> rsqrt -> mul -> mul); this kernel
touches each element once, so for a memory-bound op it should approach peak DRAM
bandwidth.

Run directly for a correctness check + microbenchmark:
    python -m kernels.rmsnorm_triton
"""
import argparse
import types

import torch
import triton
import triton.language as tl


@triton.jit
def _add_rmsnorm_kernel(X, R, W, Y, RES_OUT, N, eps,
                        HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    offs = row * N + cols

    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        r = tl.load(R + offs, mask=mask, other=0.0).to(tl.float32)
        x = x + r
        tl.store(RES_OUT + offs, x.to(RES_OUT.dtype.element_ty), mask=mask)

    var = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # Match HF: normalize in FP32, cast to input dtype, then scale by weight.
    y = (x * rstd).to(w.dtype) * w
    tl.store(Y + offs, y.to(Y.dtype.element_ty), mask=mask)


def rmsnorm(x, weight, eps=1e-6, residual=None):
    """Returns norm(x [+ residual]); if residual is given also returns the summed tensor."""
    shape = x.shape
    N = shape[-1]
    x2 = x.reshape(-1, N).contiguous()
    M = x2.shape[0]
    y = torch.empty_like(x2)
    has_res = residual is not None
    r2 = residual.reshape(-1, N).contiguous() if has_res else x2
    res_out = torch.empty_like(x2) if has_res else y

    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 1024 else 8 if BLOCK <= 4096 else 16
    _add_rmsnorm_kernel[(M,)](x2, r2, weight, y, res_out, N, eps,
                              HAS_RES=has_res, BLOCK=BLOCK, num_warps=num_warps)
    y = y.view(shape)
    return (y, res_out.view(shape)) if has_res else y


# ---------------------------------------------------------------- model patching
def patch_rmsnorm(model) -> int:
    """Swap the forward of every *RMSNorm module (Qwen2, Llama, Mistral...) for the Triton kernel."""
    n = 0
    for m in model.modules():
        if type(m).__name__.endswith("RMSNorm"):
            eps = getattr(m, "variance_epsilon", getattr(m, "eps", 1e-6))

            def fwd(self, hidden_states, _eps=eps):
                return rmsnorm(hidden_states, self.weight, _eps)

            m.forward = types.MethodType(fwd, m)
            n += 1
    return n


# ---------------------------------------------------------------- reference + bench
def torch_add_rmsnorm(x, r, w, eps=1e-6):
    h = x + r
    hf = h.float()
    hf = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + eps)
    return w * hf.to(x.dtype), h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peak-gbs", type=float, default=None,
                   help="GPU peak DRAM bandwidth in GB/s (auto-detected for common GPUs)")
    args = p.parse_args()

    name = torch.cuda.get_device_name()
    known = {"T4": 320, "L4": 300, "A10": 600, "A100": 1555, "H100": 3350, "V100": 900}
    peak = args.peak_gbs or next((v for k, v in known.items() if k in name), None)
    print(f"GPU: {name}   peak DRAM BW: {peak or 'unknown'} GB/s\n")

    compiled = torch.compile(torch_add_rmsnorm)
    rows = []
    for N in (896, 2048, 4096):              # 896 = Qwen2.5-0.5B hidden size
        for M in (1, 32, 512, 4096):          # 1 = decode, larger = prefill / batched
            x = torch.randn(M, N, device="cuda", dtype=torch.float16)
            r = torch.randn_like(x)
            w = torch.randn(N, device="cuda", dtype=torch.float16)

            ref_y, ref_h = torch_add_rmsnorm(x, r, w)
            y, h = rmsnorm(x, w, residual=r)
            torch.testing.assert_close(y, ref_y, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(h, ref_h, atol=1e-3, rtol=1e-3)

            # Warm up every path first so first-call costs (cuBLAS init, Inductor
            # recompiling for a new shape, Triton JIT) don't land in the timings.
            fns = {"eager": lambda: torch_add_rmsnorm(x, r, w),
                   "comp": lambda: compiled(x, r, w),
                   "tri": lambda: rmsnorm(x, w, residual=r)}
            for f in fns.values():
                for _ in range(5):
                    f()
            torch.cuda.synchronize()
            t_eager, t_comp, t_tri = (triton.testing.do_bench(f, warmup=50, rep=200)
                                      for f in fns.values())

            bytes_moved = (4 * M * N + N) * 2          # read x, r, w; write y, h (fp16)
            gbs = bytes_moved / (t_tri * 1e-3) / 1e9
            rows.append((M, N, t_eager * 1e3, t_comp * 1e3, t_tri * 1e3,
                         t_eager / t_tri, gbs, (100 * gbs / peak) if peak else float("nan")))

    hdr = f"{'M':>5} {'N':>5} {'eager us':>9} {'compile us':>10} {'triton us':>9} {'speedup':>8} {'GB/s':>7} {'%peak':>6}"
    print("All correctness checks passed.\n")
    print(hdr)
    print("-" * len(hdr))
    for M, N, te, tc, tt, sp, gbs, pct in rows:
        print(f"{M:>5} {N:>5} {te:>9.1f} {tc:>10.1f} {tt:>9.1f} {sp:>7.2f}x {gbs:>7.0f} {pct:>5.0f}%")

    import json, os
    os.makedirs("results", exist_ok=True)
    with open("results/kernel_rmsnorm.json", "w") as f:
        json.dump({"gpu": name, "peak_gbs": peak, "rows": [
            dict(M=a, N=b, eager_us=c, compile_us=d, triton_us=e, speedup=s, gbs=g, pct_peak=pc)
            for a, b, c, d, e, s, g, pc in rows]}, f, indent=2)
    print("\nsaved results/kernel_rmsnorm.json")


if __name__ == "__main__":
    main()
