#!/usr/bin/env bash
# Clean Core AI ANE reproduce: gist-style FP8 W8A8 conv-chain.
#
# Defaults to the 72 TFLOPs row: conv512 (N=4096, K=512, M=512), 256 layers,
# f8f8 (FP8 E4M3FN weights + activations), ANE preferred.
#
# Usage:
#   ./bench_ane.sh                        # conv512, 256 layers, f8f8, ANE  (~72 TFLOPS)
#   ./bench_ane.sh --stacks 128           # conv512, 128 layers, f8f8, ANE
#   ./bench_ane.sh --dtypes fp16 --stacks 128   # FP16 baseline
#   ./bench_ane.sh --compute gpu --stacks 256   # GPU comparison (~20 TFLOPS)
#
# Any bench_stacked.py flags are forwarded. Prints a clean table:
#   dtype  shape  S  median_ms  status  tflops
set -euo pipefail
cd "$(dirname "$0")"
unset USE_LOCAL_COREAI

args=("$@")
if [[ ${#args[@]} -eq 0 ]]; then
    args=(--compute ane --dtypes f8f8 --shapes conv512 --stacks 256)
fi

compute=ane
for i in "${!args[@]}"; do
    if [[ "${args[$i]}" == "--compute" ]]; then
        compute="${args[$((i + 1))]}"
    fi
done

echo "== running: uv run python bench_stacked.py ${args[*]} =="
uv run python bench_stacked.py "${args[@]}" 2>&1 \
    | grep -v -E '^objc\[|SyntaxWarning|escape sequence|NOTE: Redirects|coremltools|FutureWarning|^\s+\* regex|^\s+- "\\\\\*"|The special key|n_bits|Cannot infer nbits|converting 1 program|return cls.__new__'

results="results_stacked_${compute}.txt"

echo
echo "== clean table: $results =="
awk '
/^## / {
    dtype=$2; shape=$3
    n=k=m=0
    for (i=1; i<=NF; i++) {
        if ($i ~ /^N=/) n=substr($i,3)+0
        if ($i ~ /^K=/) k=substr($i,3)+0
        if ($i ~ /^M=/) m=substr($i,3)+0
    }
    next
}
/^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+[[:space:]]+OK/ {
    s=$1; ms=$2
    tf=2*n*k*m*s/(ms*1e9)
    printf "%-6s %-8s %5d %12.4f  %-6s %8.2f\n", dtype, shape, s, ms, "OK", tf
}
' "$results" | (
    printf "%-6s %-8s %5s %12s  %-6s %8s\n" dtype shape S median_ms status tflops
    cat
)
