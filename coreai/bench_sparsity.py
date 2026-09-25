#!/usr/bin/env python3
"""ANE conv2d chain throughput vs activation / weight sparsity (Core AI).

S sequential 1x1 nn.Conv2d(512, 512) on a (1, 512, 64, 64) input (= conv512,
N=4096) in one .aimodel, ANE preferred. Weights are orthogonal so activations
keep unit scale through all S layers instead of decaying to zero. Input is
torch.randn (seed 42) unless the mode says otherwise. Warmup 5, median of 50.

Modes (MODE:P, P = zero fraction):
  plain:0  — plain conv, dense random activations (the honest dense number)
  zero:0   — plain conv, all-zero input (every activation is 0)
  act:P    — conv + bias + ReLU per layer; bias calibrated per layer so exactly
             a fraction P of activations are zero, rescaled to unit RMS
  act:P:Q  — act:P combined with a fraction Q of weights zeroed (random positions)
  w:P      — plain conv, fraction P of weights zeroed at random positions
  w24:P    — plain conv, P of every 4 consecutive input-channel weights zeroed

TFLOPS is dense-equivalent: 2*N*C*C*S / wall_clock, zeros counted as work.

Usage:
  uv run python bench_sparsity.py plain:0 zero:0 act:0.5 act:0.75 w:0.5
  uv run python bench_sparsity.py act:0.75:0.75
  uv run python bench_sparsity.py --dtype fp16 plain:0
Do NOT set USE_LOCAL_COREAI.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import coreai_opt as opt
import coreai_torch
import numpy as np
import torch
import torch.nn as nn
from coreai.runtime import AIModel, NDArray
from coreai_opt.casting import cast_to_16_bit_precision
from coreai_opt.quantization import Quantizer

from bench_stacked import SEED, make_f8f8_config, remove_path, specialization_for

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts_sparsity"

C, H, W = 512, 64, 64
SHAPE = (1, C, H, W)
WARMUP = 5
ITERS = 50


class ConvChain(nn.Module):
    def __init__(self, stack: int, relu: bool) -> None:
        super().__init__()
        self.relu = relu
        self.convs = nn.ModuleList(
            nn.Conv2d(C, C, 1, bias=relu) for _ in range(stack)
        )
        with torch.no_grad():
            for conv in self.convs:
                nn.init.orthogonal_(conv.weight.view(C, C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x)
            if self.relu:
                x = torch.relu(x)
        return x


def calibrate_act_sparsity(model: ConvChain, p: float) -> list[float]:
    """Set each layer's bias so a fraction p of its ReLU outputs are zero."""
    x = torch.randn(*SHAPE)
    zero_frac = []
    for conv in model.convs:
        conv.bias.zero_()
        z = conv(x).flatten()
        if p > 0:
            t = z.kthvalue(max(1, int(p * z.numel()))).values
        else:
            t = z.min() - 1
        s = torch.relu(z - t).pow(2).mean().sqrt()
        conv.weight.div_(s)
        conv.bias.fill_(float(-t / s))
        x = torch.relu(conv(x))
        zero_frac.append((x == 0).float().mean().item())
    return zero_frac


def zero_weights(model: ConvChain, p: float, structured: bool) -> None:
    g = torch.Generator().manual_seed(123)
    for conv in model.convs:
        w = conv.weight.view(C, C)
        if structured:
            keep = round(4 * (1 - p))
            idx = torch.rand(C, C // 4, 4, generator=g).argsort(-1)[..., :keep]
            mask = torch.zeros(C, C // 4, 4, dtype=torch.bool).scatter_(-1, idx, True)
            mask = mask.view(C, C)
        else:
            mask = torch.rand(C, C, generator=g) >= p
        w.mul_(mask)
        w.mul_((1 - p) ** -0.5)  # keep gain ~1


def build(mode: str, p: float, stack: int, q: float = 0.0) -> nn.Module:
    torch.manual_seed(SEED)
    model = ConvChain(stack, relu=(mode == "act")).eval()
    with torch.no_grad():
        if mode == "act":
            if q > 0:
                zero_weights(model, q, structured=False)
            zf = calibrate_act_sparsity(model, p)
            print(f"  calibrated zero fraction: first={zf[0]:.3f} "
                  f"last={zf[-1]:.3f} mean={np.mean(zf):.3f}")
        elif mode in ("w", "w24"):
            zero_weights(model, p, structured=(mode == "w24"))
    return model


def export(mode: str, p: float, stack: int, dtype: str, force: bool, q: float = 0.0) -> Path:
    wtag = f"_w{q:g}" if q > 0 else ""
    out = ARTIFACTS / f"{dtype}_{mode}{p:g}{wtag}_S{stack}.aimodel"
    if out.exists() and not force:
        return out
    model = build(mode, p, stack, q)
    example = torch.randn(*SHAPE)
    if dtype == "fp16":
        model, example = model.to(torch.float16), example.to(torch.float16)
    else:
        quantizer = Quantizer(model, make_f8f8_config())
        prepared = quantizer.prepare((example,))
        with quantizer.calibration_mode(), torch.no_grad():
            for _ in range(4):
                prepared(torch.randn(*SHAPE))
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


def to_numpy(value) -> np.ndarray:
    for attr in ("numpy", "to_numpy"):
        if hasattr(value, attr):
            return np.asarray(getattr(value, attr)())
    return np.asarray(value)


async def time_inference(path: Path, x: np.ndarray, compute: str) -> tuple[float, float]:
    """Return (median seconds, zero fraction of the output)."""
    model = await AIModel.load(path, specialization_options=specialization_for(compute))
    fn = model.load_function("main")
    name = list(fn.desc.input_names)[0]
    nd = NDArray(x)
    for _ in range(WARMUP):
        out = await fn(inputs={name: nd})
    times = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        out = await fn(inputs={name: nd})
        times.append(time.perf_counter() - t0)
    y = to_numpy(list(out.values())[0] if isinstance(out, dict) else out)
    return statistics.median(times), float((y == 0).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("specs", nargs="+", help="MODE:P, e.g. act:0.75 w:0.5 plain:0")
    ap.add_argument("--dtype", choices=("f8f8", "fp16"), default="f8f8")
    ap.add_argument("--compute", choices=("ane", "gpu"), default="ane")
    ap.add_argument("--stack", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="re-export cached models")
    args = ap.parse_args()

    flops = 2 * H * W * C * C * args.stack
    rows = []
    for spec in args.specs:
        mode, p, *rest = spec.split(":")
        p = float(p)
        q = float(rest[0]) if rest else 0.0
        if mode not in ("plain", "zero", "act", "w", "w24"):
            raise SystemExit(f"unknown mode {mode!r}")
        if q > 0 and mode != "act":
            raise SystemExit(f"weight fraction :Q only combines with act, got {spec!r}")
        print(f"== {args.dtype} {spec} S={args.stack}", flush=True)
        build_mode = "plain" if mode == "zero" else mode
        path = export(build_mode, p, args.stack, args.dtype, args.force, q)
        torch.manual_seed(SEED)
        x = torch.randn(*SHAPE).to(torch.float16).numpy()
        if mode == "zero":
            x = np.zeros_like(x)
        med, out_zero = asyncio.run(time_inference(path, x, args.compute))
        rows.append((spec, med, out_zero))
        print(f"  {med * 1e3:.3f} ms  {flops / med / 1e12:.2f} TFLOPS  "
              f"output zero fraction {out_zero:.3f}", flush=True)

    print(f"\n{'dtype':5s} {'spec':14s} {'ms':>8s} {'TFLOPS':>7s} {'out0%':>6s}")
    for spec, med, out_zero in rows:
        print(f"{args.dtype:5s} {spec:14s} {med * 1e3:8.3f} "
              f"{flops / med / 1e12:7.2f} {out_zero * 100:6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
