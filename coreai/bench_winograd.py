#!/usr/bin/env python3
"""ANE conv2d chain: does a 3x3 conv get a Winograd-style speedup? (Core AI)

Core AI has no public Winograd switch (the ANE compiler's DisableWinograd is
internal), so "with / without" is done by shape: a stride-1, dilation-1 3x3
conv is Winograd-eligible; the same 3x3 conv with dilation 2 does the same
multiply count but is not. A 1x1 chain is the reference MAC rate.

S sequential nn.Conv2d(C, C, k) on a (1, C, 64, 64) input (padding keeps the
size), ANE preferred, dense random activations (signed), weights scaled to keep
unit variance. Warmup 5, median of 50.

Variants (NAME):
  k1     1x1 conv (reference rate)
  k3     3x3, stride 1, dilation 1, padding 1  (Winograd-eligible)
  k3d2   3x3, stride 1, dilation 2, padding 2  (same work, not eligible)

FP8 TFLOPS = 2*H*W*C*C*k*k*S / wall_clock, the direct-conv work. A Winograd
path does fewer multiplies, so it shows up as a higher rate than k1 / k3d2.

Usage:
  uv run python bench_winograd.py                 # k1 k3 k3d2, f8f8
  uv run python bench_winograd.py k3 k3d2 --stack 32
Do NOT set USE_LOCAL_COREAI.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import coreai_opt as opt
import coreai_torch
import numpy as np
import torch
import torch.nn as nn
from coreai_opt.casting import cast_to_16_bit_precision
from coreai_opt.quantization import Quantizer

from bench_sparsity import time_inference
from bench_stacked import SEED, make_f8f8_config, remove_path

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts_winograd"

H = W = 64
VARIANTS = {
    #        kernel, dilation, padding
    "k1": (1, 1, 0),
    "k3": (3, 1, 1),
    "k3d2": (3, 2, 2),
}
DEFAULT_STACK = {"k1": 256, "k3": 64, "k3d2": 64}


class ConvChain(nn.Module):
    def __init__(self, c: int, k: int, dilation: int, padding: int, stack: int) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(c, c, k, padding=padding, dilation=dilation, bias=False)
            for _ in range(stack)
        )
        with torch.no_grad():
            for conv in self.convs:
                # Unit-variance output for unit-variance input (fan_in = c*k*k).
                conv.weight.normal_(0.0, (c * k * k) ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
        return x


def export(name: str, c: int, stack: int, dtype: str, force: bool) -> Path:
    k, dil, pad = VARIANTS[name]
    out = ARTIFACTS / f"{dtype}_{name}_C{c}_S{stack}.aimodel"
    if out.exists() and not force:
        return out
    torch.manual_seed(SEED)
    model = ConvChain(c, k, dil, pad, stack).eval()
    shape = (1, c, H, W)
    example = torch.randn(*shape)
    if dtype == "fp16":
        model, example = model.to(torch.float16), example.to(torch.float16)
    else:
        quantizer = Quantizer(model, make_f8f8_config())
        prepared = quantizer.prepare((example,))
        with quantizer.calibration_mode(), torch.no_grad():
            for _ in range(4):
                prepared(torch.randn(*shape))
        model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
    exported = torch.export.export(model, (example,), strict=False).run_decompositions(
        coreai_torch.get_decomp_table()
    )
    cast_to_16_bit_precision(exported)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=["x"], output_names=["y"])
    program = converter.to_coreai()
    program.optimize()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    remove_path(out)
    program.save_asset(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("variants", nargs="*", default=["k1", "k3", "k3d2"],
                    help=f"any of {', '.join(VARIANTS)}")
    ap.add_argument("--dtype", choices=("f8f8", "fp16"), default="f8f8")
    ap.add_argument("--compute", choices=("ane", "gpu"), default="ane")
    ap.add_argument("--channels", type=int, default=512)
    ap.add_argument("--stack", type=int, default=None,
                    help="layers per chain (default: 256 for k1, 64 for 3x3)")
    ap.add_argument("--force", action="store_true", help="re-export cached models")
    args = ap.parse_args()

    c = args.channels
    rows = []
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}")
        k, dil, _ = VARIANTS[name]
        stack = args.stack or DEFAULT_STACK[name]
        print(f"== {args.dtype} {name} ({k}x{k}, dilation {dil}) C={c} S={stack}", flush=True)
        path = export(name, c, stack, args.dtype, args.force)
        torch.manual_seed(SEED)
        x = torch.randn(1, c, H, W).to(torch.float16).numpy()
        med, out_zero = asyncio.run(time_inference(path, x, args.compute))
        flops = 2 * H * W * c * c * k * k * stack
        rows.append((name, k, dil, stack, flops, med, out_zero))
        print(f"  {med * 1e3:.3f} ms  {flops / med / 1e12:.2f} TFLOPS  "
              f"output zero fraction {out_zero:.3f}", flush=True)

    print(f"\n{'dtype':5s} {'variant':7s} {'kernel':>6s} {'dil':>3s} {'S':>4s} "
          f"{'GFLOP':>8s} {'ms':>8s} {'TFLOPS':>7s}")
    for name, k, dil, stack, flops, med, _ in rows:
        print(f"{args.dtype:5s} {name:7s} {f'{k}x{k}':>6s} {dil:3d} {stack:4d} "
              f"{flops / 1e9:8.1f} {med * 1e3:8.3f} {flops / med / 1e12:7.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
