"""Shared model loading for every variant in the study."""
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"   # ~1 GB in FP16, fits any free Colab GPU

VARIANTS = {
    "eager_fp16":     "PyTorch eager, naive attention (baseline)",
    "sdpa_fp16":      "PyTorch SDPA fused attention",
    "triton_rmsnorm": "SDPA + custom Triton RMSNorm kernel patched in",
    "compile":        "SDPA + static KV cache + torch.compile (TorchInductor)",
    "int8":           "INT8 weight quantization (bitsandbytes LLM.int8)",
    "nf4":            "4-bit NF4 weight quantization (bitsandbytes)",
}


def load(variant: str, model_id: str = DEFAULT_MODEL):
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; pick from {list(VARIANTS)}")

    tok = AutoTokenizer.from_pretrained(model_id)
    kw = dict(torch_dtype=torch.float16, device_map="cuda")
    kw["attn_implementation"] = "eager" if variant == "eager_fp16" else "sdpa"

    if variant == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif variant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()

    if variant == "triton_rmsnorm":
        from kernels.rmsnorm_triton import patch_rmsnorm
        print(f"  patched {patch_rmsnorm(model)} RMSNorm modules with the Triton kernel")

    if variant == "compile":
        # A static KV cache gives fixed tensor shapes; recent transformers versions then
        # torch.compile the decode step automatically inside generate(). Wrapping
        # model.forward ourselves on top of that double-compiles and can crash.
        model.generation_config.cache_implementation = "static"

    return tok, model


def free(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch._dynamo.reset()
