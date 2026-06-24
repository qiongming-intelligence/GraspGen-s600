#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: export_and_compile_graspgen.sh [generator-ckpt] [discriminator-ckpt] [output_dir]

Defaults to the Robotiq checkpoints in models/upstream and compiles the
resulting ONNX models to HBM under models/hbm.
EOF
}

if [[ $# -gt 3 ]]; then
  usage
  exit 2
fi

gen_ckpt="${1:-models/upstream/graspgen_robotiq_2f_140_gen.pth}"
dis_ckpt="${2:-models/upstream/graspgen_robotiq_2f_140_dis.pth}"
output_dir="${3:-models/hbm}"

if [[ ! -f "$gen_ckpt" ]]; then
  echo "missing generator checkpoint: $gen_ckpt" >&2
  exit 1
fi
if [[ ! -f "$dis_ckpt" ]]; then
  echo "missing discriminator checkpoint: $dis_ckpt" >&2
  exit 1
fi

python src/python/scripts/export_graspgen_onnx.py \
  --generator-ckpt "$gen_ckpt" \
  --discriminator-ckpt "$dis_ckpt"

bash src/python/scripts/compile_graspgen_hbm.sh configs/manifests "$output_dir"
