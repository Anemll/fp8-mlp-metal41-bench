#!/usr/bin/env bash
# Build and run the tile sweep (tune.metal + tune.swift), or the pure-compute probe with
# `./run_tune.sh peak ...` (peak.metal + peak.swift). Extra args go to the host.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
name=tune
if [[ "${1:-}" == "peak" ]]; then
    name=peak
    shift
fi
out="${TMPDIR:-/tmp}/fp8_mlp_${name}_$$"
mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
xcrun -sdk macosx metal -std=metal4.1 -c "$name.metal" -o "$out/$name.air"
xcrun -sdk macosx metallib "$out/$name.air" -o "$out/$name.metallib"
swiftc -O -framework Metal -framework Foundation "$name.swift" -o "$out/$name"
"$out/$name" --metallib "$out/$name.metallib" "$@"
