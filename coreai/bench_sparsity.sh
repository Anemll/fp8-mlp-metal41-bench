#!/usr/bin/env bash
# ANE conv2d chain (conv512, 256 layers) throughput vs activation / weight sparsity.
#
# Usage:
#   ./bench_sparsity.sh                          # full f8f8 sweep (~3.5 min cold, cached after)
#   ./bench_sparsity.sh act:0.75 plain:0         # pick modes (MODE:P, P = zero fraction)
#   ./bench_sparsity.sh --dtype fp16 plain:0 zero:0
#   ./bench_sparsity.sh --stack 128 act:0.5
#
# Modes: plain (dense random), zero (all-zero input), act:P (P zero activations
# per layer), w:P (P zero weights, random), w24:P (P of every 4 weights zero).
# Any bench_sparsity.py flags are forwarded. Prints a clean table at the end.
set -euo pipefail
cd "$(dirname "$0")"
unset USE_LOCAL_COREAI

args=("$@")
has_spec=0
for a in "${args[@]+"${args[@]}"}"; do
    [[ "$a" == *:* ]] && has_spec=1
done
if [[ $has_spec -eq 0 ]]; then
    args+=(plain:0 zero:0 act:0 act:0.25 act:0.5 act:0.75 act:0.9 w:0.5 w:0.75 w24:0.5)
fi

echo "== running: uv run python bench_sparsity.py ${args[*]} =="
uv run python bench_sparsity.py "${args[@]}" 2>&1 \
    | grep -v -E '^objc\[|SyntaxWarning|escape sequence|NOTE: Redirects|coremltools|FutureWarning|^\s+\* regex|^\s+- "\\\\\*"|The special key|n_bits|Cannot infer nbits|converting 1 program|return cls.__new__|^W[0-9]{4} '
