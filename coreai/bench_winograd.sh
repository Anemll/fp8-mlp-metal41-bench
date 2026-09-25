#!/usr/bin/env bash
# ANE conv2d chain: does 3x3 get a Winograd-style speedup? (Core AI)
#
# Core AI has no public Winograd switch, so "with / without" is by shape:
#   k1    1x1 conv (reference MAC rate)
#   k3    3x3, dilation 1 (Winograd-eligible)
#   k3d2  3x3, dilation 2 (same multiply count, not eligible)
# TFLOPS counts the direct-conv work, so a Winograd path shows as k3 > k1.
#
# Usage:
#   ./bench_winograd.sh                      # k1 k3 k3d2, f8f8 (~2 min cold, cached after)
#   ./bench_winograd.sh --dtype fp16         # same in FP16
#   ./bench_winograd.sh k3 k3d2 --stack 32
# Any bench_winograd.py flags are forwarded. Prints a clean table at the end.
set -euo pipefail
cd "$(dirname "$0")"
unset USE_LOCAL_COREAI

echo "== running: uv run python bench_winograd.py $* =="
uv run python bench_winograd.py "$@" 2>&1 \
    | grep -v -E '^objc\[|SyntaxWarning|escape sequence|NOTE: Redirects|coremltools|FutureWarning|^\s+\* regex|^\s+- "\\\\\*"|The special key|n_bits|Cannot infer nbits|converting 1 program|return cls.__new__|^W[0-9]{4} '
