#!/usr/bin/env python3
"""Stacked GEMM call-overhead probe for Core AI.

One exported .aimodel contains S sequential bias-free GEMMs of the same shape
(Y = X @ W, each feeding the next). Compare wall-clock T(S) across S∈{1,4,16,32}
to estimate:

    T(S) ≈ T_call + S * T_kernel

Timings are Python wall-clock around the InferenceFunction call only (model
already loaded and specialized). Warmup 2, median of 5. Same input each time.

Dtypes:
  fp16  — activations + weights float16 (existing path)
  i8i8  — coreai-opt default int8 weight + int8 activation QuantizerConfig.
          IR is inspected; if the matmul stays fp16 with quant/dequant wrappers,
          results are labeled \"int8 quant around fp16\" (not a true integer matmul).
  f8f8  — FP8 E4M3FN weight + activation (symmetric per-tensor, graph mode),
          same preset as bench.py.
  fp8   — FP8 E4M3FN weight-only (activations stay fp16), same preset as bench.py.

Compute: GPU or ANE (preferred, not exclusive) via
SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu() /
neural_engine()). Select with --compute {gpu,ane}. One process.
Do NOT set USE_LOCAL_COREAI.

Shapes: smoke 256³ (required first), square 4096³, and gist-style conv
chains conv512/conv384/conv256 (N=4096 = 64×64 spatial).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Prefer OS Core AI so SpecializationOptions / ComputeUnitKind work.
if "USE_LOCAL_COREAI" in os.environ:
    del os.environ["USE_LOCAL_COREAI"]

import coreai_opt as opt
import coreai_torch
import numpy as np
import torch
import torch.nn as nn
from coreai.runtime import (
    AIModel,
    ComputeUnitKind,
    NDArray,
    SpecializationOptions,
    _use_os_coreai,
)
from coreai_opt.casting import cast_to_16_bit_precision
from coreai_opt.quantization import ModuleQuantizerConfig, Quantizer, QuantizerConfig
from coreai_opt.quantization.spec import (
    PerTensorGranularity,
    QuantizationScheme,
    QuantizationSpec,
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts_stacked"
SEED = 42
WARMUP = 2
TIMED_ITERS = 5
STACK_DEPTHS = (1, 4, 16, 32)
EXPORT_BUDGET_S = 20 * 60  # soft skip for a single export

SHAPES: dict[str, tuple[int, int, int]] = {
    "smoke": (256, 256, 256),
    "square": (4096, 4096, 4096),
    # gist-style compute-bound conv chains: a 1x1 conv over (1, ch, sp, sp) is
    # exactly Linear(ch, ch) over N = sp*sp activations. These mirror
    # Anemll's ANE INT8 W8A8 gist configs (sp=64 -> N=4096):
    "conv512": (4096, 512, 512),  # 128x conv 512ch  (274.88 GFLOP/chain)
    "conv384": (4096, 384, 384),  # 256x conv 384ch
    "conv256": (4096, 256, 256),  # 256x conv 256ch
    # wide chains: FLOPs/layer grow with ch^2, activation bytes with ch, so
    # these separate an 8-bit MAC rate from activation-bandwidth limits.
    "conv1024": (4096, 1024, 1024),
    "conv2048": (4096, 2048, 2048),
}


@dataclass
class DepthTiming:
    stack: int
    median_ms: float | None
    status: str
    error: str = ""
    export_s: float = 0.0
    ir_label: str = ""


@dataclass
class SeriesResult:
    dtype: str
    shape: str
    n: int
    k: int
    m: int
    ir_label: str
    timings: dict[int, DepthTiming] = field(default_factory=dict)
    t_call_ms: float | None = None
    t_kernel_ms: float | None = None
    kernel_tflops: float | None = None
    t1_wall_tflops: float | None = None
    fit_note: str = ""


class StackedGemm(nn.Module):
    """S sequential bias-free Linears: x → L0 → L1 → … → L{S-1}."""

    def __init__(self, k: int, m: int, stack: int) -> None:
        super().__init__()
        if stack < 1:
            raise ValueError(f"stack must be >= 1, got {stack}")
        self.layers = nn.ModuleList(
            [nn.Linear(k if i == 0 else m, m, bias=False) for i in range(stack)]
        )
        # For square GEMMs k==m so every layer is Linear(k, m). For non-square
        # smoke/square we always use square shapes in this script.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def flops_one(n: int, k: int, m: int) -> float:
    return 2.0 * float(n) * float(k) * float(m)


def tflops_from_seconds(n: int, k: int, m: int, seconds: float) -> float:
    if seconds <= 0:
        return float("nan")
    return flops_one(n, k, m) / seconds / 1e12


def specialization_for(compute: str) -> SpecializationOptions:
    if not SpecializationOptions.is_supported():
        raise RuntimeError(
            "SpecializationOptions not supported "
            f"(_use_os_coreai={_use_os_coreai}). "
            "Unset USE_LOCAL_COREAI; wheel installs need the OS Core AI framework."
        )
    kind = {
        "gpu": ComputeUnitKind.gpu(),
        "ane": ComputeUnitKind.neural_engine(),
    }[compute]
    return SpecializationOptions.from_preferred_compute_unit_kind(kind)


def asset_path(dtype: str, shape_name: str, n: int, k: int, m: int, stack: int) -> Path:
    return (
        ARTIFACTS
        / f"stacked_{dtype}_{shape_name}_N{n}_K{k}_M{m}_S{stack}.aimodel"
    )


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        import shutil

        shutil.rmtree(path)
    else:
        path.unlink()


def classify_ir(mlir_text: str) -> str:
    """Classify whether matmul operands are integer or fp16-around-quant."""
    # Look at broadcasting_batch_matmul / batch_matmul type signatures.
    matmul_sigs = re.findall(
        r"(?:broadcasting_batch_matmul|batch_matmul)\s+[^:]+:\s*\(([^)]+)\)",
        mlir_text,
    )
    if not matmul_sigs:
        # Fallback: any matmul-like line.
        if "broadcasting_batch_matmul" in mlir_text or "batch_matmul" in mlir_text:
            if "xsi8" in mlir_text and re.search(
                r"broadcasting_batch_matmul[^\\n]*xsi8[^\\n]*xsi8", mlir_text
            ):
                return "true int8×int8 matmul (si8 operands)"
            return "matmul present; could not parse operand dtypes"
        return "no matmul op found in IR"

    saw_fp16 = False
    saw_si8_pair = False
    for sig in matmul_sigs:
        types = re.findall(r"tensor<[^>]+>", sig)
        if len(types) >= 2:
            a, b = types[0], types[1]
            if "xsi8" in a and "xsi8" in b:
                saw_si8_pair = True
            if "xf16" in a or "xf16" in b or "xf32" in a or "xf32" in b:
                saw_fp16 = True

    has_quant = "coreai.quantize" in mlir_text or "coreai.dequantize" in mlir_text
    if saw_si8_pair and not saw_fp16:
        return "true int8×int8 matmul (si8 operands)"
    if saw_fp16 and has_quant:
        return "int8 quant around fp16 (dequant → broadcasting_batch_matmul fp16)"
    if saw_fp16:
        return "fp16 broadcasting_batch_matmul"
    if saw_si8_pair:
        return "si8 matmul operands (check accumulate dtype)"
    return "unknown matmul dtype pattern"


def make_f8f8_config() -> QuantizerConfig:
    """FP8 E4M3FN weights + activations, symmetric per-tensor (graph)."""
    weight_spec = QuantizationSpec(
        dtype=torch.float8_e4m3fn,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        qparam_calculator_cls="static",
    )
    activation_spec = QuantizationSpec(
        dtype=torch.float8_e4m3fn,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        qparam_calculator_cls="global_minmax",
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_spec},
            op_input_spec={"*": activation_spec},
            op_output_spec={"*": activation_spec},
        ),
        execution_mode="graph",
    )


def make_fp8_weight_only_config() -> QuantizerConfig:
    """FP8 E4M3FN weights only (fp16 activations) — metal-bench `fp8`."""
    weight_spec = QuantizationSpec(
        dtype=torch.float8_e4m3fn,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=PerTensorGranularity(),
        qparam_calculator_cls="static",
    )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight_spec},
            op_input_spec=None,
            op_output_spec=None,
        ),
        execution_mode="graph",
    )


def build_exported_model(
    dtype: str,
    n: int,
    k: int,
    m: int,
    stack: int,
) -> tuple[nn.Module, torch.Tensor]:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    base = StackedGemm(k, m, stack).eval()
    example = torch.randn(n, k)

    if dtype == "fp16":
        return base.to(dtype=torch.float16), example.to(torch.float16)

    if dtype == "i8i8":
        # Default QuantizerConfig: int8 weights (per-channel) + int8 activations
        # (per-tensor). Converter currently emits quant/dequant around fp16 BMM;
        # classify_ir() labels honestly after export.
        config = QuantizerConfig()
        quantizer = Quantizer(base, config)
        prepared = quantizer.prepare((example,))
        with quantizer.calibration_mode(), torch.no_grad():
            for _ in range(4):
                prepared(torch.randn(n, k))
        finalized = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
        return finalized, example

    if dtype == "f8f8":
        config = make_f8f8_config()
        quantizer = Quantizer(base, config)
        prepared = quantizer.prepare((example,))
        with quantizer.calibration_mode(), torch.no_grad():
            for _ in range(4):
                prepared(torch.randn(n, k))
        finalized = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
        return finalized, example

    if dtype == "fp8":
        config = make_fp8_weight_only_config()
        quantizer = Quantizer(base, config)
        prepared = quantizer.prepare((example,))
        _ = prepared
        finalized = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
        return finalized, example

    raise ValueError(f"Unsupported dtype mode: {dtype!r}")


def export_aimodel(
    dtype: str,
    shape_name: str,
    n: int,
    k: int,
    m: int,
    stack: int,
    *,
    force: bool = False,
    budget_s: float = EXPORT_BUDGET_S,
) -> tuple[Path, str, float]:
    """Return (path, ir_label, export_seconds)."""
    out = asset_path(dtype, shape_name, n, k, m, stack)
    if out.exists() and not force:
        print(f"  reuse asset: {out}")
        # Re-inspect IR from saved asset if possible via reload convert is heavy;
        # read mlirb is binary — re-export cheap path: load via converter not needed.
        # Quick classify from a tiny re-convert is expensive; stash label beside asset.
        label_path = out.with_suffix(out.suffix + ".ir_label")
        if label_path.exists():
            return out, label_path.read_text(encoding="utf-8").strip(), 0.0
        return out, "reused asset (ir label missing; re-run --force-export)", 0.0

    print(
        f"  export {dtype} {shape_name} N={n} K={k} M={m} stack={stack} → {out}"
    )
    t0 = time.perf_counter()
    model, example = build_exported_model(dtype, n, k, m, stack)
    if time.perf_counter() - t0 > budget_s:
        raise TimeoutError(
            f"build_exported_model exceeded {budget_s:.0f}s budget before convert"
        )

    exported = torch.export.export(model, (example,), strict=False).run_decompositions(
        coreai_torch.get_decomp_table()
    )
    cast_to_16_bit_precision(exported)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_exported_program(exported, input_names=["x"], output_names=["y"])
    ai_program = converter.to_coreai()
    ai_program.optimize()
    mlir_text = str(ai_program)
    ir_label = classify_ir(mlir_text)
    if dtype == "fp16":
        ir_label = "fp16 broadcasting_batch_matmul"
    elif dtype == "f8f8":
        ir_label = "f8f8 quant around fp16 (dequant → broadcasting_batch_matmul fp16)"
    elif dtype == "fp8":
        ir_label = "fp8 weight-only quant around fp16 (dequant → broadcasting_batch_matmul fp16)"
    print(f"  IR: {ir_label}")

    elapsed = time.perf_counter() - t0
    if elapsed > budget_s:
        raise TimeoutError(
            f"export exceeded {budget_s:.0f}s budget ({elapsed:.1f}s); "
            "skipping save for this config"
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    remove_path(out)
    ai_program.save_asset(out)
    label_path = out.with_suffix(out.suffix + ".ir_label")
    label_path.write_text(ir_label + "\n", encoding="utf-8")
    print(f"  export done in {elapsed:.1f}s")
    return out, ir_label, elapsed


async def time_inference(
    aimodel_path: Path,
    x: np.ndarray,
    options: SpecializationOptions,
    *,
    warmup: int,
    iters: int,
) -> float:
    """Return median seconds around await fn(...)."""
    model = await AIModel.load(aimodel_path, specialization_options=options)
    fn = model.load_function("main")
    input_names = list(fn.desc.input_names)
    if len(input_names) != 1:
        raise RuntimeError(f"Expected 1 input, got {input_names}")
    name = input_names[0]
    nd = NDArray(x)

    for _ in range(warmup):
        await fn(inputs={name: nd})

    times: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        await fn(inputs={name: nd})
        times.append(time.perf_counter() - t0)

    return statistics.median(times)


def fit_overhead(
    timings: dict[int, float],
) -> tuple[float, float, str]:
    """Fit T(S) = T_call + S * T_kernel.

    Primary estimate is least-squares over all available depths.
    Also reports a two-point slope using the largest S vs S=1 when both exist
    (e.g. (T(32)-T(1))/31).
    Returns (t_call_s, t_kernel_s, note) where call/kernel are the LS values.
    """
    available = sorted(s for s, t in timings.items() if t is not None and t > 0)
    if len(available) < 2:
        raise ValueError("need at least two stack depths to fit overhead")

    xs = np.array(available, dtype=np.float64)
    ys = np.array([timings[s] for s in available], dtype=np.float64)
    # ys = a + b * xs  → T_call=a, T_kernel=b
    b, a = np.polyfit(xs, ys, 1)
    note = f"least-squares over S={available}"

    s_lo = 1 if 1 in timings and timings[1] else available[0]
    s_hi = available[-1]
    if s_hi > s_lo and timings.get(s_lo) and timings.get(s_hi):
        t_kernel_2p = (timings[s_hi] - timings[s_lo]) / float(s_hi - s_lo)
        t_call_2p = timings[s_lo] - t_kernel_2p
        note += (
            f"; two-point (T({s_hi})-T({s_lo}))/{s_hi - s_lo}: "
            f"T_call={t_call_2p * 1e3:.4f}ms T_kernel={t_kernel_2p * 1e3:.4f}ms"
        )

    return float(a), float(b), note


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--compute",
        default="gpu",
        choices=["gpu", "ane"],
        help="Preferred compute unit: gpu (default) or ane (preferred, not exclusive).",
    )
    p.add_argument(
        "--shapes",
        default="smoke,square",
        help="Comma list from {smoke,square}. Default: smoke,square",
    )
    p.add_argument(
        "--dtypes",
        default="fp16,i8i8",
        help="Comma list from {fp16,f8f8,fp8,i8i8}. Default: fp16,i8i8",
    )
    p.add_argument(
        "--stacks",
        default="1,4,16,32",
        help="Comma list of stack depths. Default: 1,4,16,32",
    )
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--iters", type=int, default=TIMED_ITERS)
    p.add_argument("--force-export", action="store_true")
    p.add_argument(
        "--export-budget-s",
        type=float,
        default=EXPORT_BUDGET_S,
        help="Skip a single export if it exceeds this many seconds (default 900).",
    )
    p.add_argument(
        "--smoke-only",
        action="store_true",
        help="Only 256³ for all dtypes/stacks (catch failures early).",
    )
    return p.parse_args(argv)


def format_series(series: SeriesResult) -> list[str]:
    lines: list[str] = []
    unit = "TOPS" if series.dtype == "i8i8" else "TFLOPs"
    lines.append(
        f"## {series.dtype} {series.shape} "
        f"N={series.n} K={series.k} M={series.m}"
    )
    lines.append(f"IR: {series.ir_label}")
    lines.append(
        f"{'S':>4} {'median_ms':>12}  status"
    )
    lines.append("-" * 40)
    depth_order = sorted(series.timings.keys()) or list(STACK_DEPTHS)
    for s in depth_order:
        t = series.timings.get(s)
        if t is None:
            lines.append(f"{s:4d} {'—':>12}  missing")
            continue
        med = f"{t.median_ms:12.4f}" if t.median_ms is not None else f"{'—':>12}"
        err = f"  {t.error}" if t.error else ""
        lines.append(f"{s:4d} {med}  {t.status}{err}")

    if series.t_call_ms is not None and series.t_kernel_ms is not None:
        lines.append(
            f"estimated T_call={series.t_call_ms:.4f} ms  "
            f"T_kernel={series.t_kernel_ms:.4f} ms"
        )
        lines.append(f"fit: {series.fit_note}")
        k_tf = (
            f"{series.kernel_tflops:.3f}"
            if series.kernel_tflops is not None
            else "—"
        )
        t1_tf = (
            f"{series.t1_wall_tflops:.3f}"
            if series.t1_wall_tflops is not None
            else "—"
        )
        lines.append(
            f"per-GEMM {unit}: kernel-only={k_tf}  "
            f"T(1) wall-clock={t1_tf}"
        )
    return lines


def write_outputs(
    series_list: list[SeriesResult], meta: list[str], compute: str = "gpu"
) -> None:
    txt_path = ROOT / f"results_stacked_{compute}.txt"
    blocks: list[str] = list(meta) + [""]
    for series in series_list:
        blocks.extend(format_series(series))
        blocks.append("")

    # Summary table
    blocks.append("## Summary table")
    blocks.append(
        f"{'dtype':6} {'shape':7} {'T1_ms':>10} {'T4_ms':>10} {'T16_ms':>10} "
        f"{'T32_ms':>10} {'T_call_ms':>10} {'T_kern_ms':>10} "
        f"{'kern_TF/TOPS':>12} {'T1_TF/TOPS':>11}  IR"
    )
    blocks.append("-" * 130)
    for series in series_list:
        def ms(s: int) -> str:
            t = series.timings.get(s)
            if t is None or t.median_ms is None:
                return f"{'—':>10}"
            return f"{t.median_ms:10.4f}"

        tc = (
            f"{series.t_call_ms:10.4f}"
            if series.t_call_ms is not None
            else f"{'—':>10}"
        )
        tk = (
            f"{series.t_kernel_ms:10.4f}"
            if series.t_kernel_ms is not None
            else f"{'—':>10}"
        )
        kt = (
            f"{series.kernel_tflops:12.3f}"
            if series.kernel_tflops is not None
            else f"{'—':>12}"
        )
        t1t = (
            f"{series.t1_wall_tflops:11.3f}"
            if series.t1_wall_tflops is not None
            else f"{'—':>11}"
        )
        blocks.append(
            f"{series.dtype:6} {series.shape:7} {ms(1)} {ms(4)} {ms(16)} {ms(32)} "
            f"{tc} {tk} {kt} {t1t}  {series.ir_label}"
        )

    # Gap analysis vs Metal
    blocks.append("")
    blocks.append("## Overhead vs Metal gap (fp16 square reference)")
    blocks.append(
        "Metal square fp16 ~2.15 ms (GPU timestamps); "
        "JIT Core AI ~5.45 ms; AOT Core AI ~3.09 ms (wall-clock)."
    )
    for series in series_list:
        if series.dtype != "fp16" or series.shape != "square":
            continue
        if series.t_call_ms is None or series.t_kernel_ms is None:
            blocks.append("fp16 square: insufficient timings to estimate overhead.")
            continue
        t1 = series.timings.get(1)
        t1_ms = t1.median_ms if t1 and t1.median_ms is not None else float("nan")
        metal_ms = 2.15
        gap_jit = t1_ms - metal_ms
        blocks.append(
            f"Measured T(1)={t1_ms:.4f} ms, T_call≈{series.t_call_ms:.4f} ms, "
            f"T_kernel≈{series.t_kernel_ms:.4f} ms."
        )
        blocks.append(
            f"Gap vs Metal device time: T(1)-2.15 ≈ {gap_jit:.4f} ms. "
            f"If T_call explains most of that gap, overhead is a major factor; "
            f"if T_kernel alone is still >> 2.15 ms, the kernel itself is slower "
            f"(or wall-clock includes more than device time)."
        )
        # Compare kernel-only implied time to Metal
        blocks.append(
            f"Kernel-only vs Metal: T_kernel={series.t_kernel_ms:.4f} ms vs "
            f"~2.15 ms → ratio {series.t_kernel_ms / metal_ms:.2f}×."
        )

    text = "\n".join(blocks) + "\n"
    txt_path.write_text(text, encoding="utf-8")
    print(f"wrote {txt_path}")

    md_path = ROOT / "results.md"
    md_header = f"## Stacked GEMM call-overhead (JIT {compute.upper()})"
    md_block = ["", md_header, ""] + blocks + [""]
    md_text = "\n".join(md_block)
    if md_path.exists():
        existing = md_path.read_text(encoding="utf-8")
        idx = existing.find(md_header)
        if idx >= 0:
            md_path.write_text(existing[:idx].rstrip() + "\n" + md_text, encoding="utf-8")
            print(f"replaced stacked section in {md_path}")
        else:
            with md_path.open("a", encoding="utf-8") as f:
                f.write(md_text)
            print(f"appended {md_path}")
    else:
        md_path.write_text(md_text, encoding="utf-8")
        print(f"wrote {md_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    dtypes = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    stacks = [int(s.strip()) for s in args.stacks.split(",") if s.strip()]
    if args.smoke_only:
        shape_names = ["smoke"]
    else:
        shape_names = [s.strip() for s in args.shapes.split(",") if s.strip()]

    shapes: list[tuple[str, int, int, int]] = []
    for name in shape_names:
        if name not in SHAPES:
            raise SystemExit(f"Unknown shape {name!r}; expected smoke|square")
        n, k, m = SHAPES[name]
        shapes.append((name, n, k, m))

    # Smoke first always if present.
    shapes.sort(key=lambda t: 0 if t[0] == "smoke" else 1)

    print(f"torch={torch.__version__}")
    print(f"coreai_opt={getattr(opt, '__version__', '?')}")
    print(f"coreai_torch={coreai_torch.__version__}")
    print(f"_use_os_coreai={_use_os_coreai}")
    print(f"SpecializationOptions.is_supported()={SpecializationOptions.is_supported()}")
    try:
        opts = specialization_for(args.compute)
        print(f"specialization:\n{opts}")
    except Exception as e:
        print(f"FATAL specialization setup: {e}")
        return 2

    meta = [
        f"# Core AI stacked GEMM call-overhead — compute={args.compute} (JIT)",
        f"# timing: wall-clock around InferenceFunction (median of {args.iters} "
        f"after {args.warmup} warmup)",
        "# model: S sequential bias-free GEMMs in ONE .aimodel",
        "# fit: T(S) ≈ T_call + S * T_kernel via least-squares over all S; "
        "also report two-point (T(Smax)-T(1))/(Smax-1)",
        "# TFLOPs/TOPS = 2*N*K*M / seconds / 1e12 (per single GEMM)",
        f"# specialization: {opts}",
        f"# _use_os_coreai={_use_os_coreai}",
        "# reference (single GEMM, prior runs): JIT fp16 square 5.45 ms / 25.2 TFLOPs; "
        "AOT 3.09 ms / 44.5 TFLOPs; Metal ~2.15 ms / ~64 TFLOPs",
    ]

    series_list: list[SeriesResult] = []

    for shape_name, n, k, m in shapes:
        for dtype in dtypes:
            print(f"\n======== {dtype} {shape_name} N={n} K={k} M={m} ========")
            series = SeriesResult(
                dtype=dtype,
                shape=shape_name,
                n=n,
                k=k,
                m=m,
                ir_label="",
            )
            # Prefer smaller stacks first so a failure is cheap.
            for stack in sorted(stacks):
                print(f"\n--- stack={stack} ---")
                # Soft skip: if square × S=16 and previous exports were slow, still try
                # but honor per-export budget.
                try:
                    path, ir_label, export_s = export_aimodel(
                        dtype,
                        shape_name,
                        n,
                        k,
                        m,
                        stack,
                        force=args.force_export,
                        budget_s=args.export_budget_s,
                    )
                    if not series.ir_label:
                        series.ir_label = ir_label
                    elif ir_label and ir_label != series.ir_label:
                        series.ir_label = ir_label  # latest authoritative
                except TimeoutError as e:
                    err = str(e)
                    print(f"  EXPORT SKIP (budget): {err}")
                    series.timings[stack] = DepthTiming(
                        stack, None, "SKIP", err, ir_label=series.ir_label
                    )
                    # For square S=16, continue to allow other configs; document skip.
                    continue
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    print(f"  EXPORT FAIL: {err}")
                    traceback.print_exc()
                    series.timings[stack] = DepthTiming(
                        stack, None, "FAIL", err
                    )
                    continue

                try:
                    torch.manual_seed(SEED)
                    example = torch.randn(n, k)
                    x_np = example.detach().cpu().to(torch.float16).numpy()
                    median_s = asyncio.run(
                        time_inference(
                            path,
                            x_np,
                            opts,
                            warmup=args.warmup,
                            iters=args.iters,
                        )
                    )
                    median_ms = median_s * 1e3
                    print(
                        f"  median_ms={median_ms:.4f} "
                        f"(export_s={export_s:.1f}) IR={ir_label}"
                    )
                    series.timings[stack] = DepthTiming(
                        stack,
                        median_ms,
                        "OK",
                        export_s=export_s,
                        ir_label=ir_label,
                    )
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    print(f"  RUN FAIL: {err}")
                    traceback.print_exc()
                    series.timings[stack] = DepthTiming(
                        stack, None, "FAIL", err, ir_label=ir_label
                    )

            # Fit overhead if we have enough points.
            timed = {
                s: t.median_ms / 1e3
                for s, t in series.timings.items()
                if t.median_ms is not None
            }
            if len(timed) >= 2:
                try:
                    t_call_s, t_kernel_s, note = fit_overhead(timed)
                    series.t_call_ms = t_call_s * 1e3
                    series.t_kernel_ms = t_kernel_s * 1e3
                    series.fit_note = note
                    if t_kernel_s > 0:
                        series.kernel_tflops = tflops_from_seconds(
                            n, k, m, t_kernel_s
                        )
                    if 1 in timed:
                        series.t1_wall_tflops = tflops_from_seconds(
                            n, k, m, timed[1]
                        )
                    print(
                        f"  fit: T_call={series.t_call_ms:.4f} ms "
                        f"T_kernel={series.t_kernel_ms:.4f} ms "
                        f"kernel_TF/TOPS={series.kernel_tflops} "
                        f"T1_wall_TF/TOPS={series.t1_wall_tflops}"
                    )
                except Exception as e:
                    series.fit_note = f"fit failed: {e}"
                    print(f"  fit failed: {e}")
            else:
                series.fit_note = "insufficient timings for fit"

            series_list.append(series)

    print("\n" + "=" * 60)
    for series in series_list:
        print("\n".join(format_series(series)))
        print()

    write_outputs(series_list, meta, args.compute)

    any_fail = any(
        t.status == "FAIL"
        for series in series_list
        for t in series.timings.values()
    )
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
