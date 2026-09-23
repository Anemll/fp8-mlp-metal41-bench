"""On-device smoke: export, AOT compile and run 256^3 for every dtype on GPU and ANE.

Slow (~1 min on M5 Max with a cold artifacts/ cache). Skip with `-m "not device"`.
"""

import shutil

import pytest

import coreai_gemm as g

pytestmark = pytest.mark.device


def _coreai_available() -> bool:
    if shutil.which("xcrun") is None:
        return False
    try:
        from coreai.runtime import SpecializationOptions

        return SpecializationOptions.is_supported()
    except Exception:
        return False


if not _coreai_available():
    pytest.skip("OS Core AI runtime / coreai-build not available", allow_module_level=True)

N = K = M = 256


@pytest.mark.parametrize("compute", g.COMPUTES)
@pytest.mark.parametrize("dtype", list(g.DTYPES))
def test_smoke_row(dtype, compute):
    row = g.bench_row("smoke", N, K, M, dtype, compute, stack=2, warmup=1, iters=3)
    assert row.status == "OK", g.format_row(row)
    assert row.rel_err is not None and row.rel_err <= g.TOLERANCE[dtype]
    assert row.t1_ms and row.t1_ms > 0
    assert row.ts_ms and row.ts_ms > 0
    assert row.extra["preferred"] == ("GPU" if compute == "gpu" else "NeuralEngine")
    if compute == "gpu":
        assert row.placement == "GPU"


def test_ir_labels_match_dtype():
    want = {
        "fp16": "w=fp16 a=fp16",
        "fp8": "w=fp8 a=fp16",
        "f8f8": "w=fp8 a=fp8",
        "int8": "w=int8 a=fp16",
        "i8i8": "w=int8 a=int8",
    }
    for dtype, label in want.items():
        _, got = g.export_aimodel(dtype, N, K, M, 1)
        assert got == label, dtype


def test_fp16_lands_on_ane():
    """The known-good ANE path: FP16 with ANE preferred compiles to an ANE region."""
    src, _ = g.export_aimodel("fp16", N, K, M, 1)
    placement = g.inspect_placement(g.compile_aimodelc(src, "ane"))
    assert placement.device == "ANE", placement


def test_device_architecture_detected():
    src, _ = g.export_aimodel("fp16", N, K, M, 1)
    assert g.device_architecture(src).startswith("h")
