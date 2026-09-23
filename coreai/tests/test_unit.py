"""Pure-Python checks: math, IR labels, placement parsing. No Core AI needed."""

from pathlib import Path

import pytest

import coreai_gemm as g

FP16_IR = """
coreai.graph @main(%arg0: tensor<256x256xf16> {coreai.name = "x"}) -> (tensor<256x256xf16>) {
  %0 = coreai.constant dense_resource<r> : tensor<256x256xf16>
  %2 = coreai.transpose %0, %1 : (tensor<256x256xf16>, tensor<2xui32>) -> tensor<256x256xf16>
  %3 = coreai.decomposable.broadcasting_batch_matmul %arg0, %2 : (tensor<256x256xf16>, tensor<256x256xf16>) -> tensor<256x256xf16>
"""

F8F8_IR = """
  %10 = coreai.blockwise_shift_scale %5, %4, %8, %9 : (tensor<256x256xf8E4M3FN>, tensor<1x1xf16>, tensor<1x1xf8E4M3FN>, tensor<1x1xf16>) -> tensor<256x256xf16>
  %11 = coreai.quantize %arg0, %6, %2, %1, %3 : (tensor<256x256xf16>, tensor<f16>, tensor<f8E4M3FN>, tensor<f16>, tensor<si32>) -> tensor<256x256xf8E4M3FN>
  %12 = coreai.dequantize %11, %6, %2, %1, %3 : (tensor<256x256xf8E4M3FN>, tensor<f16>, tensor<f8E4M3FN>, tensor<f16>, tensor<si32>) -> tensor<256x256xf16>
  %14 = coreai.decomposable.broadcasting_batch_matmul %12, %13 : (tensor<256x256xf16>, tensor<256x256xf16>) -> tensor<256x256xf16>
"""

INT8_W_IR = """
  %5 = coreai.blockwise_shift_scale %3, %1, %2, %4 : (tensor<256x256xsi8>, tensor<256x1xf16>, tensor<256x1xsi8>, tensor<256x1xf16>) -> tensor<256x256xf16>
  %7 = coreai.decomposable.broadcasting_batch_matmul %arg0, %6 : (tensor<256x256xf16>, tensor<256x256xf16>) -> tensor<256x256xf16>
"""

I8I8_IR = INT8_W_IR + """
  %11 = coreai.quantize %arg0, %6, %7, %1, %2 : (tensor<256x256xf16>, tensor<f16>, tensor<si8>, tensor<f16>, tensor<si32>) -> tensor<256x256xsi8>
"""


def test_flops_and_tflops():
    assert g.gemm_flops(4096, 4096, 4096) == 2 * 4096**3
    assert g.tflops(4096, 4096, 4096, 2 * 4096**3 / 50e12) == pytest.approx(50.0)
    assert g.tflops(1, 1, 1, 0.0) is None
    assert g.tflops(1, 1, 1, -1e-3) is None
    assert g.tflops(1, 1, 1, None) is None


def test_fit_recovers_call_and_gemm_time():
    t_call, t_gemm = 1.2e-3, 2.5e-3
    times = {s: t_call + s * t_gemm for s in (1, 4, 8)}
    call, gemm = g.fit_call_overhead(times)
    assert call == pytest.approx(t_call)
    assert gemm == pytest.approx(t_gemm)


def test_fit_two_points_is_the_slope():
    call, gemm = g.fit_call_overhead({1: 3.0, 8: 10.0})
    assert gemm == pytest.approx(1.0)
    assert call == pytest.approx(2.0)


def test_fit_needs_two_depths():
    with pytest.raises(ValueError):
        g.fit_call_overhead({1: 1.0})


def test_stack_shape_ok():
    assert g.stack_shape_ok(4096, 4096, 8)
    assert g.stack_shape_ok(4096, 11008, 1)
    assert not g.stack_shape_ok(4096, 11008, 2)


@pytest.mark.parametrize(
    "ir, label",
    [
        (FP16_IR, "w=fp16 a=fp16"),
        (F8F8_IR, "w=fp8 a=fp8"),
        (INT8_W_IR, "w=int8 a=fp16"),
        (I8I8_IR, "w=int8 a=int8"),
        ("coreai.graph @main() {}", "no-matmul"),
    ],
)
def test_classify_ir(ir, label):
    assert g.classify_ir(ir) == label


def _fake_package(root: Path, *, preferred: str, ane: bool, fully: bool) -> Path:
    pkg = root / "m.aimodelc"
    mps = pkg / "main-h17c-delegates/MPSGraph/mpsExecutable.mpsgraphpackage"
    mps.mkdir(parents=True)
    (pkg / "main-h17c.mlirb").write_bytes(b"\x00junk preferredDevice=" + preferred.encode() + b"\x00")
    manifest = b"<plist>"
    if fully:
        manifest += b"<string>mps.fullyPlacedOnANE</string><string>mps.noGPUActivity</string>"
    (mps / "manifest.plist").write_bytes(manifest)
    if ane:
        bc = mps / "binary_0.llir.bundle/main_x_0_ANE_region_0_0.bc/h17c"
        bc.mkdir(parents=True)
        (bc / "main_x_0_ANE_region_0_0.mlir.bc").write_bytes(b"")
    return pkg


def test_placement_ane(tmp_path):
    p = g.inspect_placement(_fake_package(tmp_path, preferred="NeuralEngine", ane=True, fully=True))
    assert (p.preferred, p.ane_regions, p.device) == ("NeuralEngine", 1, "ANE")


def test_placement_partial_ane(tmp_path):
    p = g.inspect_placement(_fake_package(tmp_path, preferred="NeuralEngine", ane=True, fully=False))
    assert p.device == "ANE+GPU"


def test_placement_ane_preferred_but_gpu(tmp_path):
    # What FP8 does today: ANE requested, package has no ANE region.
    p = g.inspect_placement(_fake_package(tmp_path, preferred="NeuralEngine", ane=False, fully=False))
    assert p.device == "GPU"


def test_parse_device_arch():
    out = "This device's architecture: h17c\nSupported architectures: h13c\n"
    assert g.parse_device_arch(out) == "h17c"
    assert g.parse_device_arch("Compatibility: 27.0") is None


def test_rel_error():
    import numpy as np

    ref = np.array([[1.0, -2.0], [0.5, 4.0]], dtype=np.float32)
    assert g.rel_error(ref.astype(np.float16), ref) == 0.0
    assert g.rel_error(ref + np.float32(0.4), ref) == pytest.approx(0.1)


def test_parse_list():
    assert g.parse_list("all", g.DTYPES) == list(g.DTYPES)
    assert g.parse_list("fp16, i8i8", g.DTYPES) == ["fp16", "i8i8"]
    with pytest.raises(SystemExit):
        g.parse_list("fp4", g.DTYPES)


def test_format_row_handles_missing_values():
    row = g.Row("square", 4096, 4096, 4096, "fp8", "ane", 8, status="FAIL", note="boom")
    line = g.format_row(row)
    assert "FAIL" in line and "# boom" in line
    assert len(g.HEADER.split()) == 16
