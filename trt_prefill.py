"""TensorRT path: export the model to ONNX, build a TensorRT engine, and compare prefill
latency (the work behind time-to-first-token) and outputs against PyTorch.

    python trt_prefill.py                 # batch 1, 256-token prompt

The engine is built as a *strongly typed* network: TensorRT keeps the exact precisions
in the ONNX graph (FP16 matmuls, FP32 softmax/RMSNorm accumulation the model asks for)
instead of guessing, which keeps outputs numerically close to PyTorch.

Large build artifacts (ONNX weights, engine) go in --build-dir, outside the repo.
"""
import argparse
import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

from common import DEFAULT_MODEL


class LastTokenLogits(torch.nn.Module):
    """Prefill-only graph: token ids in, next-token logits out. No KV cache, fixed shape."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids.long(), use_cache=False)
        return out.logits[:, -1, :].float()


def load_model(model_id, attn):
    return AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="cuda", attn_implementation=attn).eval()


def time_cuda(fn, warmup=10, runs=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return statistics.median(times), times[min(len(times) - 1, int(0.99 * (len(times) - 1) + 0.5))]


def export_onnx(wrapper, ids, path):
    t = time.time()
    try:   # torch.export-based exporter; handles current transformers models
        prog = torch.onnx.export(wrapper, (ids,), dynamo=True)
        prog.save(path, external_data=True)
        how = "dynamo (torch.export) exporter"
    except Exception as e:
        print(f"  dynamo exporter failed ({type(e).__name__}: {str(e)[:200]}); trying TorchScript exporter")
        torch.onnx.export(wrapper, (ids,), path, input_names=["input_ids"],
                          output_names=["logits"], opset_version=17, dynamo=False)
        how = "TorchScript exporter"
    print(f"  exported ONNX with the {how} in {time.time() - t:.0f}s")
    return how


def build_engine(onnx_path, trt):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    strongly_typed = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED) if strongly_typed else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the ONNX model:\n{errs}")
    config = builder.create_builder_config()
    if not strongly_typed:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    t = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed (see warnings above)")
    return serialized, time.time() - t, network.num_layers, strongly_typed


class TRTRunner:
    def __init__(self, serialized, trt):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.outn = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)
        dtypes = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.out = torch.empty(tuple(self.ctx.get_tensor_shape(self.outn)),
                               dtype=dtypes[self.engine.get_tensor_dtype(self.outn)], device="cuda")

    def __call__(self, ids):
        self.ctx.set_tensor_address(self.inp, ids.data_ptr())
        self.ctx.set_tensor_address(self.outn, self.out.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return self.out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--runs", type=int, default=50)
    p.add_argument("--build-dir", default="../trt_build")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    import tensorrt as trt
    print(f"GPU: {torch.cuda.get_device_name()}   TensorRT {trt.__version__}   torch {torch.__version__}")
    os.makedirs(args.build_dir, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(0)
    ids = torch.randint(100, 20000, (1, args.seq_len), device="cuda", dtype=torch.int32)
    res = dict(model=args.model, gpu=torch.cuda.get_device_name(), seq_len=args.seq_len,
               trt_version=trt.__version__, runs=args.runs)

    # --- PyTorch baselines --------------------------------------------------------
    with torch.no_grad():
        sdpa = LastTokenLogits(load_model(args.model, "sdpa"))
        ref = sdpa(ids).clone()
        res["pytorch_ms_p50"], res["pytorch_ms_p99"] = time_cuda(lambda: sdpa(ids), runs=args.runs)
        print(f"  PyTorch eager (SDPA)  : {res['pytorch_ms_p50']:.2f} ms")
        try:
            comp = torch.compile(sdpa)
            res["compile_ms_p50"], res["compile_ms_p99"] = time_cuda(lambda: comp(ids), runs=args.runs)
            print(f"  torch.compile         : {res['compile_ms_p50']:.2f} ms")
        except Exception as e:
            print(f"  torch.compile failed: {type(e).__name__}: {e}")
        del sdpa
        torch._dynamo.reset()
        torch.cuda.empty_cache()

        # --- ONNX export -> TensorRT engine ----------------------------------------
        onnx_path = os.path.join(args.build_dir, "model.onnx")
        src = LastTokenLogits(load_model(args.model, "eager"))   # plain ops export cleanly
        res["onnx_exporter"] = export_onnx(src, ids, onnx_path)
        del src
        torch.cuda.empty_cache()

    print("  building TensorRT engine (a few minutes on a T4)...")
    serialized, build_s, n_layers, strongly_typed = build_engine(onnx_path, trt)
    with open(os.path.join(args.build_dir, "model.engine"), "wb") as f:
        f.write(serialized)
    res.update(build_s=build_s, network_layers=n_layers, strongly_typed=strongly_typed)
    print(f"  built in {build_s:.0f}s ({n_layers} network layers, strongly typed={strongly_typed})")

    runner = TRTRunner(serialized, trt)
    out = runner(ids).float().clone()
    torch.cuda.synchronize()
    res["trt_ms_p50"], res["trt_ms_p99"] = time_cuda(lambda: runner(ids), runs=args.runs)
    print(f"  TensorRT              : {res['trt_ms_p50']:.2f} ms")

    # --- correctness -----------------------------------------------------------------
    res["top1_match"] = bool(out.argmax(-1).eq(ref.argmax(-1)).all())
    top5_ref, top5_trt = set(ref.topk(5).indices[0].tolist()), set(out.topk(5).indices[0].tolist())
    res["top5_overlap"] = len(top5_ref & top5_trt)
    res["max_abs_logit_diff"] = (out - ref).abs().max().item()
    res["cosine_sim"] = torch.nn.functional.cosine_similarity(out, ref).item()
    res["speedup_vs_pytorch"] = res["pytorch_ms_p50"] / res["trt_ms_p50"]
    if "compile_ms_p50" in res:
        res["speedup_vs_compile"] = res["compile_ms_p50"] / res["trt_ms_p50"]

    print(f"  top-1 token match: {res['top1_match']}   top-5 overlap: {res['top5_overlap']}/5   "
          f"cosine sim: {res['cosine_sim']:.5f}   max |diff|: {res['max_abs_logit_diff']:.3f}")
    print(f"  TensorRT speedup: {res['speedup_vs_pytorch']:.2f}x vs PyTorch"
          + (f", {res['speedup_vs_compile']:.2f}x vs torch.compile" if "speedup_vs_compile" in res else ""))
    with open(os.path.join(args.out, "trt_prefill.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {args.out}/trt_prefill.json")


if __name__ == "__main__":
    main()
