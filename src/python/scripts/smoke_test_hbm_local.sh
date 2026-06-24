#!/usr/bin/env bash
set -euo pipefail

PRELOAD_DEFAULT="/home/sunrise/Projects/FoundationPose-s600/build/cmake/src/csrc/libfoundationpose_bpu_core1_preload.so"
PRELOAD="${GRASPGEN_HBM_PRELOAD:-$PRELOAD_DEFAULT}"
HRT="${HRT_MODEL_EXEC:-/usr/hobot/bin/hrt_model_exec}"
HBM_DIR="${1:-models/hbm}"
INPUT_DIR="$HBM_DIR/smoke_inputs"
DISC_OUT="$HBM_DIR/smoke_outputs_disc"
GEN_OUT="$HBM_DIR/smoke_outputs_gen"

mkdir -p "$INPUT_DIR"
python3 - <<'PY'
from pathlib import Path
import numpy as np
out = Path('models/hbm/smoke_inputs')
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
rng.standard_normal((1, 2048, 3), dtype=np.float32).tofile(out / 'pc_f32_1x2048x3.bin')
rng.standard_normal((20, 6), dtype=np.float32).tofile(out / 'noisy_grasps_f32_20x6.bin')
np.array([5], dtype=np.int64).tofile(out / 'timestep_i64_1.bin')
rng.standard_normal((1, 20, 6), dtype=np.float32).tofile(out / 'grasps_f32_1x20x6.bin')
PY

if [[ -f "$PRELOAD" ]]; then
  export LD_PRELOAD="$PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "Using HBRT preload: $PRELOAD"
else
  echo "WARNING: preload not found: $PRELOAD" >&2
  echo "HBRT 4.7.5 may fail with 'iova addr not equal' without it." >&2
fi

"$HRT" model_info --model_file "$HBM_DIR/graspgen_discriminator_pointnet.hbm" >/dev/null
"$HRT" model_info --model_file "$HBM_DIR/graspgen_generator_pointnet.hbm" >/dev/null

rm -rf "$DISC_OUT" "$GEN_OUT"
"$HRT" infer \
  --model_file "$HBM_DIR/graspgen_discriminator_pointnet.hbm" \
  --model_name graspgen_discriminator_pointnet \
  --input_file "$INPUT_DIR/pc_f32_1x2048x3.bin,$INPUT_DIR/grasps_f32_1x20x6.bin" \
  --frame_count 1 \
  --enable_dump true \
  --dump_format bin \
  --dump_path "$DISC_OUT"

"$HRT" infer \
  --model_file "$HBM_DIR/graspgen_generator_pointnet.hbm" \
  --model_name graspgen_generator_pointnet \
  --input_file "$INPUT_DIR/pc_f32_1x2048x3.bin,$INPUT_DIR/noisy_grasps_f32_20x6.bin,$INPUT_DIR/timestep_i64_1.bin" \
  --frame_count 1 \
  --enable_dump true \
  --dump_format bin \
  --dump_path "$GEN_OUT"

python3 - <<'PY'
from pathlib import Path
import numpy as np
checks = [
    (Path('models/hbm/smoke_outputs_disc/model_infer_output_0_scores.bin'), np.float32),
    (Path('models/hbm/smoke_outputs_gen/model_infer_output_0_noise_pred.bin'), np.float32),
]
for path, dtype in checks:
    arr = np.fromfile(path, dtype=dtype)
    print(path)
    print(f"  values={arr.size} finite={np.isfinite(arr).all()} min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g}")
PY

echo "✅ Local HBM smoke test passed."
