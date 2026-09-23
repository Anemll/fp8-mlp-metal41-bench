#!/usr/bin/env python3
"""Core AI GEMM throughput check: FP16, FP8 and INT8 on GPU and ANE.

Same GEMM as the Metal bench one directory up:

    Y[N, M] = X[N, K] @ W[K, M]      (nn.Linear(K, M, bias=False))

Pipeline per (dtype, shape, S):
    torch module -> coreai-opt quantize -> torch.export -> coreai-torch
    -> .aimodel -> `xcrun coreai-build compile --preferred-compute {gpu,neural-engine}`
    -> .aimodelc -> coreai.runtime.AIModel (same preferred compute unit).

Core AI exposes only wall-clock timing around the inference call. That
includes a fixed per-call cost (≈1 ms), which hides device throughput. So each
row times a model with one GEMM (S=1) and a model with S chained GEMMs of the
same shape, and reports the slope:

    T(S) ≈ T_call + S * T_gemm      ->  T_gemm = (T(S) - T(1)) / (S - 1)

`tflops` is 2*N*K*M / T_gemm. `t1_tflops` is the plain one-call number.

Preferred compute is not exclusive (Core AI has no ANE-only option). The
compiled package says where it landed: an `ANE_region*.bc` bundle means the
graph was placed on the Neural Engine, otherwise it runs on the GPU. Every
row reports that placement next to the number.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

# The in-wheel runtime has no delegates (SpecializationOptions unsupported);
# the OS Core AI framework is required for GPU / ANE.
os.environ.pop("USE_LOCAL_COREAI", None)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
SEED = 42

# dtype -> (weight, activation) as in the Metal bench table.
DTYPES: dict[str, str] = {
    "fp16": "FP16 weights and activations",
    "fp8": "W8A16: FP8 E4M3 weights (per-tensor), FP16 activations",
    "f8f8": "W8A8: FP8 E4M3 weights and activations (per-tensor)",
    "int8": "W8A16: INT8 weights (per-channel), FP16 activations",
    "i8i8": "W8A8: INT8 weights (per-channel) and activations (per-tensor)",
}

SHAPES: dict[str, tuple[int, int, int]] = {
    "smoke": (256, 256, 256),
    "square": (4096, 4096, 4096),
    "fat": (2048, 4096, 11008),
}

COMPUTES = ("gpu", "ane")
PREFERRED_FLAG = {"gpu": "gpu", "ane": "neural-engine"}

# Output tolerance vs the FP32 torch reference, relative to max|ref|.
TOLERANCE = {"fp16": 1e-2, "fp8": 0.08, "f8f8": 0.12, "int8": 0.03, "i8i8": 0.06}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested without Core AI)
# --------------------------------------------------------------------------


def gemm_flops(n: int, k: int, m: int) -> float:
    return 2.0 * n * k * m


def tflops(n: int, k: int, m: int, seconds: float | None) -> float | None:
    if seconds is None or not seconds > 0:
        return None
    return gemm_flops(n, k, m) / seconds / 1e12


def fit_call_overhead(times: dict[int, float]) -> tuple[float, float]:
    """Least-squares fit of T(S) = T_call + S * T_gemm. Returns (T_call, T_gemm)."""
    depths = sorted(times)
    if len(depths) < 2:
        raise ValueError("need timings for at least two stack depths")
    slope, intercept = np.polyfit(
        np.array(depths, dtype=np.float64),
        np.array([times[s] for s in depths], dtype=np.float64),
        1,
    )
    return float(intercept), float(slope)


def stack_shape_ok(k: int, m: int, stack: int) -> bool:
    """Chained GEMMs need the output width to feed the next input (K == M)."""
    return stack == 1 or k == m


def _elem(tensor_type: str) -> str:
    """'tensor<256x256xf8E4M3FN>' -> 'f8E4M3FN'."""
    m = re.search(r"tensor<(?:[\d?]+x)*([a-zA-Z]\w*)>", tensor_type)
    return m.group(1) if m else "?"


def _short(elem: str) -> str:
    return {"f8E4M3FN": "fp8", "si8": "int8", "f16": "fp16", "f32": "fp32"}.get(elem, elem)


def classify_ir(ir_text: str) -> str:
    """Describe the GEMM as exported: weight storage and activation quantization.

    coreai-torch always emits an f16 x f16 broadcasting_batch_matmul. Low
    precision appears as a blockwise_shift_scale (weight dequant) and a
    quantize -> dequantize pair on activations. The backend compiler decides
    whether that fuses into a native low-precision matmul.
    """
    if "batch_matmul" not in ir_text:
        return "no-matmul"
    weight = act = None
    for line in ir_text.splitlines():
        sig = line.rsplit(" : ", 1)[-1]
        if "blockwise_shift_scale" in line and weight is None:
            weight = _elem(sig.split(",")[0])
        elif re.search(r"coreai\.quantize\b", line) and act is None:
            act = _elem(sig.rsplit("->", 1)[-1])
    return f"w={_short(weight or 'f16')} a={_short(act or 'f16')}"


@dataclass
class Placement:
    preferred: str  # preferredDevice string from the compiled graph
    ane_regions: int  # ANE_region*.bc bundles in the package
    fully_on_ane: bool  # manifest says mps.fullyPlacedOnANE + noGPUActivity

    @property
    def device(self) -> str:
        if self.ane_regions == 0:
            return "GPU"
        return "ANE" if self.fully_on_ane else "ANE+GPU"


def inspect_placement(aimodelc: Path) -> Placement:
    preferred = "?"
    for mlirb in aimodelc.glob("main-*.mlirb"):
        m = re.search(rb"preferredDevice=([A-Za-z]+)", mlirb.read_bytes())
        if m:
            preferred = m.group(1).decode()
    regions = {
        p.name
        for p in aimodelc.rglob("*ANE_region*")
        if p.name.endswith(".bc") and not p.name.endswith(".mlir.bc")
    }
    fully = False
    for manifest in aimodelc.rglob("manifest.plist"):
        data = manifest.read_bytes()
        if b"mps.fullyPlacedOnANE" in data and b"mps.noGPUActivity" in data:
            fully = True
    return Placement(preferred=preferred, ane_regions=len(regions), fully_on_ane=fully)


# --------------------------------------------------------------------------
# Model building / export (needs torch + coreai)
# --------------------------------------------------------------------------


def _torch():
    import torch

    return torch


def make_stack(k: int, m: int, stack: int):
    """S chained bias-free Linears. Weights ~ N(0, 1/K) keep activations near unit scale."""
    torch = _torch()
    nn = torch.nn

    class StackedGemm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                nn.Linear(k, m, bias=False) for _ in range(stack)
            )
            for layer in self.layers:
                nn.init.normal_(layer.weight, std=k**-0.5)

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return StackedGemm().eval()


def quantizer_config(dtype: str):
    """coreai-opt config for a quantized dtype, or None for fp16."""
    torch = _torch()
    from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig
    from coreai_opt.quantization.spec import (
        PerChannelGranularity,
        PerTensorGranularity,
        QuantizationScheme,
        QuantizationSpec,
    )

    if dtype == "fp16":
        return None
    elem = torch.float8_e4m3fn if dtype in ("fp8", "f8f8") else torch.int8
    weight_gran = PerTensorGranularity() if elem is torch.float8_e4m3fn else PerChannelGranularity(axis=0)
    weight = QuantizationSpec(
        dtype=elem,
        qscheme=QuantizationScheme.SYMMETRIC,
        granularity=weight_gran,
        qparam_calculator_cls="static",
    )
    act = None
    if dtype in ("f8f8", "i8i8"):
        act = QuantizationSpec(
            dtype=elem,
            qscheme=QuantizationScheme.SYMMETRIC,
            granularity=PerTensorGranularity(),
            qparam_calculator_cls="global_minmax",
        )
    return QuantizerConfig(
        global_config=ModuleQuantizerConfig(
            op_state_spec={"weight": weight},
            op_input_spec={"*": act} if act else None,
            op_output_spec={"*": act} if act else None,
        ),
        execution_mode="graph",
    )


def example_input(n: int, k: int):
    torch = _torch()
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(n, k, generator=g)


def build_module(dtype: str, n: int, k: int, m: int, stack: int):
    """Return (exportable module, example input, fp32 reference output)."""
    torch = _torch()
    import coreai_opt as opt
    from coreai_opt.quantization import Quantizer

    torch.manual_seed(SEED)
    base = make_stack(k, m, stack)
    x = example_input(n, k)
    with torch.no_grad():
        ref = base(x)
    if dtype == "fp16":
        return base.half(), x.half(), ref
    quantizer = Quantizer(base, quantizer_config(dtype))
    prepared = quantizer.prepare((x,))
    if dtype in ("f8f8", "i8i8"):
        with quantizer.calibration_mode(), torch.no_grad():
            prepared(x)
            for i in range(3):
                g = torch.Generator().manual_seed(SEED + 100 + i)
                prepared(torch.randn(n, k, generator=g))
    return quantizer.finalize(backend=opt.ExportBackend.CoreAI), x, ref


def asset_stem(dtype: str, n: int, k: int, m: int, stack: int) -> str:
    return f"gemm_{dtype}_N{n}_K{k}_M{m}_S{stack}"


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def export_aimodel(
    dtype: str, n: int, k: int, m: int, stack: int, *, force: bool = False
) -> tuple[Path, str]:
    """Export one .aimodel. Returns (path, matmul operand-type label)."""
    torch = _torch()
    import coreai_torch
    from coreai_opt.casting import cast_to_16_bit_precision

    out = ARTIFACTS / f"{asset_stem(dtype, n, k, m, stack)}.aimodel"
    label_file = out.with_suffix(".ir.txt")
    if out.exists() and label_file.exists() and not force:
        return out, classify_ir(label_file.read_text())

    module, x, _ = build_module(dtype, n, k, m, stack)
    exported = torch.export.export(module, (x,), strict=False).run_decompositions(
        coreai_torch.get_decomp_table()
    )
    cast_to_16_bit_precision(exported)
    conv = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    conv.add_exported_program(exported, input_names=["x"], output_names=["y"])
    program = conv.to_coreai()
    program.optimize()
    ir = str(program)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    remove_path(out)
    program.save_asset(out)
    label_file.write_text(ir)
    return out, classify_ir(ir)


_ARCH: str | None = os.environ.get("COREAI_ARCH")


def parse_device_arch(inspect_output: str) -> str | None:
    m = re.search(r"This device's architecture:\s*(\S+)", inspect_output)
    return m.group(1) if m else None


def device_architecture(asset: Path) -> str:
    """Architecture coreai-build targets on this Mac (h17c on M5 Max).

    Only a compiled package reports it, so compile `asset` for every
    architecture once (fast without --preferred-compute) and inspect one.
    Set COREAI_ARCH to skip the probe.
    """
    global _ARCH
    if _ARCH is None:
        probe = ARTIFACTS / "_arch_probe"
        remove_path(probe)
        probe.mkdir(parents=True)
        subprocess.run(
            ["xcrun", "coreai-build", "compile", str(asset), "--output", str(probe)],
            capture_output=True,
            check=True,
        )
        pkg = next(probe.glob("*.aimodelc"))
        proc = subprocess.run(
            ["xcrun", "coreai-build", "inspect", str(pkg), "--no-io"],
            capture_output=True,
            text=True,
        )
        remove_path(probe)
        _ARCH = parse_device_arch(proc.stdout + proc.stderr)
        if _ARCH is None:
            raise RuntimeError(f"coreai-build inspect gave no architecture:\n{proc.stdout}")
    return _ARCH


def compile_aimodelc(
    src: Path, compute: str, *, force: bool = False, timeout_s: float = 1800
) -> Path:
    arch = device_architecture(src)
    dst = src.with_name(f"{src.stem}_{compute}_{arch}.aimodelc")
    if dst.exists() and not force:
        return dst
    remove_path(dst)
    cmd = [
        "xcrun", "coreai-build", "compile", str(src),
        "--output", str(dst),
        "--platform", "macOS",
        "--preferred-compute", PREFERRED_FLAG[compute],
        "--architecture", arch,
    ]  # fmt: skip
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0 or not dst.exists():
        noise = lambda s: "\n".join(l for l in s.splitlines() if not l.startswith("objc["))  # noqa: E731
        raise RuntimeError(
            f"coreai-build compile failed rc={proc.returncode}\n"
            f"{noise(proc.stdout)}\n{noise(proc.stderr)}"
        )
    return dst


def specialization(compute: str):
    from coreai.runtime import ComputeUnitKind, SpecializationOptions, _use_os_coreai

    if not SpecializationOptions.is_supported():
        raise RuntimeError(
            f"SpecializationOptions unsupported (_use_os_coreai={_use_os_coreai}); "
            "unset USE_LOCAL_COREAI so the OS Core AI framework is used"
        )
    kind = ComputeUnitKind.gpu() if compute == "gpu" else ComputeUnitKind.neural_engine()
    return SpecializationOptions.from_preferred_compute_unit_kind(kind)


async def _run(path: Path, compute: str, x: np.ndarray, warmup: int, iters: int):
    from coreai.runtime import AIModel, NDArray

    model = await AIModel.load(path, specialization_options=specialization(compute))
    fn = model.load_function("main")
    name = list(fn.desc.input_names)[0]
    inp = {name: NDArray(x)}
    out = None
    for _ in range(max(warmup, 1)):
        out = await fn(inputs=inp)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        await fn(inputs=inp)
        times.append(time.perf_counter() - t0)
    y = next(iter(out.values())).numpy()
    return statistics.median(times), y


def run_model(
    path: Path, compute: str, x: np.ndarray, *, warmup: int = 3, iters: int = 10
) -> tuple[float, np.ndarray]:
    """Median seconds per call, and the output of the last warmup call."""
    return asyncio.run(_run(path, compute, x, warmup, iters))


def rel_error(y: np.ndarray, ref: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float32).reshape(ref.shape)
    return float(np.max(np.abs(y - ref)) / max(float(np.max(np.abs(ref))), 1e-12))


# --------------------------------------------------------------------------
# Benchmark rows
# --------------------------------------------------------------------------


@dataclass
class Row:
    shape: str
    n: int
    k: int
    m: int
    dtype: str
    compute: str
    stack: int
    placement: str = "?"
    matmul: str = "?"
    t1_ms: float | None = None
    ts_ms: float | None = None
    gemm_ms: float | None = None
    tflops: float | None = None
    t1_tflops: float | None = None
    rel_err: float | None = None
    status: str = "OK"
    note: str = ""
    extra: dict = field(default_factory=dict)


def bench_row(
    shape: str,
    n: int,
    k: int,
    m: int,
    dtype: str,
    compute: str,
    *,
    stack: int,
    warmup: int,
    iters: int,
    force: bool = False,
) -> Row:
    row = Row(shape, n, k, m, dtype, compute, stack)
    if not stack_shape_ok(k, m, stack):
        stack = row.stack = 1
        row.note = "K!=M: single GEMM only, tflops includes call overhead"
    x = example_input(n, k).half().numpy()
    placements = []
    times: dict[int, float] = {}
    for s in sorted({1, stack}):
        src, label = export_aimodel(dtype, n, k, m, s, force=force)
        pkg = compile_aimodelc(src, compute, force=force)
        placement = inspect_placement(pkg)
        placements.append(placement.device)
        if s == 1:
            row.matmul = label
            row.extra["preferred"] = placement.preferred
        t, y = run_model(pkg, compute, x, warmup=warmup, iters=iters)
        times[s] = t
        if s == 1:
            _, _, ref = build_module("fp16", n, k, m, 1)  # same seed -> same weights
            row.rel_err = rel_error(y, ref.numpy())
    row.placement = "/".join(sorted(set(placements)))
    row.t1_ms = times[1] * 1e3
    row.t1_tflops = tflops(n, k, m, times[1])
    if stack > 1:
        row.ts_ms = times[stack] * 1e3
        _, t_gemm = fit_call_overhead(times)
        row.gemm_ms = t_gemm * 1e3
        row.tflops = tflops(n, k, m, t_gemm)
    else:
        row.gemm_ms = row.t1_ms
        row.tflops = row.t1_tflops
    if row.rel_err is not None and row.rel_err > TOLERANCE[dtype]:
        row.status = "BAD"
        row.note = (row.note + "; " if row.note else "") + f"rel_err>{TOLERANCE[dtype]}"
    if compute == "ane" and row.placement != "ANE":
        row.note = (row.note + "; " if row.note else "") + "ANE requested, not placed on ANE"
    return row


def _f(v: float | None, fmt: str) -> str:
    return format(v, fmt) if v is not None else "-"


HEADER = (
    f"{'shape':7} {'N':>5} {'K':>5} {'M':>5} {'dtype':5} {'compute':7} {'placed':7} "
    f"{'S':>3} {'t1_ms':>8} {'tS_ms':>9} {'gemm_ms':>8} {'tflops':>7} {'t1_tf':>7} "
    f"{'rel_err':>8} {'status':6} matmul"
)


def format_row(r: Row) -> str:
    line = (
        f"{r.shape:7} {r.n:5d} {r.k:5d} {r.m:5d} {r.dtype:5} {r.compute:7} {r.placement:7} "
        f"{r.stack:3d} {_f(r.t1_ms, '8.3f')} {_f(r.ts_ms, '9.3f')} {_f(r.gemm_ms, '8.3f')} "
        f"{_f(r.tflops, '7.2f')} {_f(r.t1_tflops, '7.2f')} {_f(r.rel_err, '8.1e')} "
        f"{r.status:6} {r.matmul}"
    )
    return line + (f"  # {r.note}" if r.note else "")


def chip_name() -> str:
    try:
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        return platform.machine()


def parse_list(s: str, allowed) -> list[str]:
    items = [p.strip() for p in s.split(",") if p.strip()]
    if items == ["all"]:
        return list(allowed)
    bad = [i for i in items if i not in allowed]
    if bad:
        raise SystemExit(f"unknown {bad}; choose from {list(allowed)}")
    return items


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dtypes", default="all", help=f"comma list of {list(DTYPES)} or all")
    p.add_argument("--compute", default="all", help="gpu,ane or all")
    p.add_argument("--shapes", default="square", help=f"comma list of {list(SHAPES)}")
    p.add_argument("--stack", type=int, default=8, help="chained GEMMs for the slope (1 = no fit)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--force", action="store_true", help="re-export and re-compile")
    p.add_argument("--json", type=Path, help="also write rows as JSON")
    args = p.parse_args(argv)

    dtypes = parse_list(args.dtypes, DTYPES)
    computes = parse_list(args.compute, COMPUTES)
    shapes = parse_list(args.shapes, SHAPES)

    import coreai_opt
    import coreai_torch

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    print(f"device: {chip_name()}")
    print(f"coreai_opt={getattr(coreai_opt, '__version__', '?')} coreai_torch={coreai_torch.__version__}")
    print("timing: wall-clock median around InferenceFunction; tflops from T(S)-T(1) slope")
    for d in dtypes:
        print(f"  {d:5} {DTYPES[d]}")

    rows: list[Row] = []
    for shape in shapes:
        n, k, m = SHAPES[shape]
        for compute in computes:
            for dtype in dtypes:
                print(f"running {shape} {dtype} {compute}", flush=True)
                try:
                    row = bench_row(
                        shape, n, k, m, dtype, compute,
                        stack=args.stack,
                        warmup=args.warmup, iters=args.iters, force=args.force,
                    )  # fmt: skip
                except Exception as e:
                    traceback.print_exc()
                    row = Row(shape, n, k, m, dtype, compute, args.stack, status="FAIL",
                              note=f"{type(e).__name__}: {str(e).splitlines()[0][:120]}")  # fmt: skip
                rows.append(row)
                print("  " + format_row(row), flush=True)

    print()
    print(HEADER)
    print("-" * len(HEADER))
    for r in rows:
        print(format_row(r))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in rows], indent=2))
    return 1 if any(r.status == "FAIL" for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
