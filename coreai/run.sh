#!/bin/zsh
# Core AI GEMM check. Pass flags through, e.g. ./run.sh --shapes smoke --stack 4
set -e
cd "$(dirname "$0")"
unset USE_LOCAL_COREAI
uv run python coreai_gemm.py "$@" 2>&1 | grep -v -E '^objc\[|SyntaxWarning|escape sequence|NOTE: Redirects|coremltools|^\s+\* regex|^\s+- "\\\\\*"|The special key|n_bits|Cannot infer nbits|converting 1 program|FutureWarning|return cls.__new__'
