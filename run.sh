#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
out="${TMPDIR:-/tmp}/fp8_mlp_bench_$$"
mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
echo "== hardware =="
sysctl -n machdep.cpu.brand_string 2>/dev/null || true
system_profiler SPHardwareDataType 2>/dev/null | awk '/Chip|Model Name|Model Identifier|Memory/{print}' || true
echo "== compile metallib (-std=metal4.1) =="
xcrun -sdk macosx metal -std=metal4.1 -c fp8_mlp.metal -o "$out/fp8_mlp.air"
xcrun -sdk macosx metallib "$out/fp8_mlp.air" -o "$out/default.metallib"
echo "== compile host =="
swiftc -O -framework Metal -framework Foundation main.swift -o "$out/fp8_mlp_bench"
echo "== run =="
(cd "$out" && cp "$ROOT/fp8_mlp.metal" . && ./fp8_mlp_bench "$@")
