#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: compile_graspgen_hbm.sh [CONTRACT_DIR] [OUTPUT_DIR]

Compile GraspGen ONNX models to Horizon HBM using hb_compile.

Requires a working x86_64 hb_compile environment. On ws-wan, use the
sam3-compile conda environment because sam3-hbm is currently broken by NumPy 2.x.

Defaults:
  CONTRACT_DIR: configs/manifests
  OUTPUT_DIR:    models/hbm
EOF
}

contract_dir="${1:-configs/manifests}"
output_dir="${2:-models/hbm}"

if [[ ! -d "$contract_dir" ]]; then
  echo "contract directory does not exist: $contract_dir" >&2
  exit 1
fi

hb_compile_bin="${HB_COMPILE_BIN:-}"
if [[ -z "$hb_compile_bin" ]]; then
  for candidate in \
    /home/yufeng.lv/.conda/envs/sam3-compile/bin/hb_compile \
    /home/yufeng.lv/.conda/envs/sam3-hbm/bin/hb_compile \
    "$(command -v hb_compile 2>/dev/null || true)"; do
    if [[ -n "${candidate:-}" && -x "$candidate" ]]; then
      hb_compile_bin="$candidate"
      break
    fi
  done
fi

if [[ -z "$hb_compile_bin" ]]; then
  echo "hb_compile not found; set HB_COMPILE_BIN or use the sam3-compile environment" >&2
  exit 1
fi

mkdir -p "$output_dir"
output_dir="$(cd "$output_dir" && pwd)"
config_dir="$output_dir/hb_compile_configs"
mkdir -p "$config_dir"

contracts=("$contract_dir"/graspgen_*.json)
if [[ ! -e "${contracts[0]}" ]]; then
  echo "no GraspGen contract JSON files found in: $contract_dir" >&2
  exit 1
fi

for contract in "${contracts[@]}"; do
  mapfile -t fields < <(python3 - "$contract" "$output_dir" <<'PY'
import json
import sys
from pathlib import Path
contract = json.load(open(sys.argv[1], encoding='utf-8'))
out_dir = Path(sys.argv[2])
if not out_dir.is_absolute():
    out_dir = Path.cwd() / out_dir
print(contract['name'])
onnx_path = Path(contract['onnx_path'])
if not onnx_path.is_absolute():
    onnx_path = Path.cwd() / onnx_path
print(onnx_path)
print(contract['hbm_name'])
print(Path(contract['hbm_name']).stem)
inputs = contract['inputs']
print(';'.join(t['name'] for t in inputs))
print(';'.join('x'.join(str(dim) for dim in t['concrete_shape']) for t in inputs))
print(';'.join(['featuremap'] * len(inputs)))
print(';'.join(['NCHW'] * len(inputs)))
print(';'.join(contract.get('run_on_cpu', [])))
PY
  )

  name="${fields[0]}"
  onnx_path="${fields[1]}"
  hbm_name="${fields[2]}"
  prefix="${fields[3]}"
  input_names="${fields[4]}"
  input_shapes="${fields[5]}"
  input_types="${fields[6]}"
  input_layouts="${fields[7]}"
  run_on_cpu="${fields[8]}"

  if [[ ! -f "$onnx_path" ]]; then
    echo "missing ONNX: $onnx_path" >&2
    exit 1
  fi

  config_path="$config_dir/${name}.yaml"
  cat >"$config_path" <<EOF
model_parameters:
  onnx_model: "$onnx_path"
  march: "nash-p"
  working_dir: "$output_dir"
  output_model_file_prefix: "$prefix"

input_parameters:
  input_name: "$input_names"
  input_type_rt: "$input_types"
  input_type_train: "$input_types"
  input_layout_train: "$input_layouts"
  input_shape: "$input_shapes"
  norm_type: "no_preprocess"

calibration_parameters:
  calibration_type: "skip"
  run_on_cpu: "$run_on_cpu"

compiler_parameters:
  optimize_level: "${HB_COMPILE_OPTIMIZE_LEVEL:-O2}"
EOF

  echo "Compiling $name -> $hbm_name"
  "$hb_compile_bin" --config "$config_path"
done

echo "✅ HBM compilation finished."
