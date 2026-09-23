"""Try W8A8 INT8 quantizer variants with ANE preferred and report placement.

Usage: uv run python probe_i8i8_ane.py [size]   (default 256)
"""
import sys; sys.path.insert(0, ".")
import coreai_gemm as g, torch, coreai_opt as opt, coreai_torch
from pathlib import Path
from coreai_opt.quantization import ModuleQuantizerConfig, QuantizerConfig, Quantizer
from coreai_opt.quantization.spec import PerChannelGranularity, PerTensorGranularity, QuantizationScheme, QuantizationSpec
from coreai_opt.casting import cast_to_16_bit_precision

def spec(gran, calc, scheme=QuantizationScheme.SYMMETRIC):
    return QuantizationSpec(dtype=torch.int8, qscheme=scheme, granularity=gran, qparam_calculator_cls=calc)
W_PC = spec(PerChannelGranularity(axis=0), "static")
W_PT = spec(PerTensorGranularity(), "static")
A = spec(PerTensorGranularity(), "global_minmax")
variants = {
  "wpc_ain_aout": (W_PC, A, A),
  "wpc_ain":      (W_PC, A, None),
  "wpt_ain_aout": (W_PT, A, A),
  "wpt_ain":      (W_PT, A, None),
  "default_cfg":  None,
}
n=k=m=int(sys.argv[1]) if len(sys.argv)>1 else 256
out_dir = g.ARTIFACTS / "i8i8_variants"; out_dir.mkdir(parents=True, exist_ok=True)
for name, v in variants.items():
    torch.manual_seed(g.SEED); base = g.make_stack(k, m, 1); x = g.example_input(n, k)
    with torch.no_grad(): ref = base(x)
    if v is None: cfg = QuantizerConfig()
    else:
        w, ai, ao = v
        cfg = QuantizerConfig(global_config=ModuleQuantizerConfig(op_state_spec={"weight": w},
              op_input_spec={"*": ai}, op_output_spec={"*": ao} if ao else None), execution_mode="graph")
    q = Quantizer(base, cfg); p = q.prepare((x,))
    with q.calibration_mode(), torch.no_grad(): p(x)
    mod = q.finalize(backend=opt.ExportBackend.CoreAI)
    ep = torch.export.export(mod, (x,), strict=False).run_decompositions(coreai_torch.get_decomp_table())
    cast_to_16_bit_precision(ep)
    c = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    c.add_exported_program(ep, input_names=["x"], output_names=["y"])
    prog = c.to_coreai(); prog.optimize(); ir = str(prog)
    src = out_dir / f"{name}_{n}.aimodel"; g.remove_path(src); prog.save_asset(src)
    pkg = g.compile_aimodelc(src, "ane", force=True); pl = g.inspect_placement(pkg)
    t, y = g.run_model(pkg, "ane", x.half().numpy(), warmup=2, iters=5)
    nq = ir.count("coreai.quantize "); 
    print(f"RESULT {name:14} label={g.classify_ir(ir):16} quant_ops={nq} placed={pl.device:7} ms={t*1e3:.3f} rel_err={g.rel_error(y, ref.numpy()):.1e}", flush=True)
