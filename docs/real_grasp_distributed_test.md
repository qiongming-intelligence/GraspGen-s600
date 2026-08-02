# Distributed Real-Grasp Test: Pi Hardware + Local S600 Inference

## Goal

Run a safety-first real grasp test with:

- camera and robot arm attached to `pi@192.168.1.156`
- GraspGen / S600 HBM inference on the local S600 machine
- dry-run validation before any robot motion

The inference server returns grasp candidates only. It never authorizes robot motion.

## Safety defaults

The current implementation is dry-run-first:

- `configs/real_grasp_test.yaml` defaults to `safety.default_mode: dry_run`
- robot adapter defaults to `null`
- workspace bounds and calibration paths default to `null`
- server responses always include `safety.motion_authorized: false`
- real execution is refused if inference results are mock/dry-run, calibration is missing, workspace checks are missing, collision checks are missing, or the robot adapter is not a real hardware adapter

Do not run a physical robot motion command until a same-session dry-run has passed and the exact execution step has been explicitly approved.

## Components

### Local S600 server

```text
src/python/scripts/serve_graspgen_s600_zmq.py
```

Actions:

- `health`
- `metadata`
- `model_info`
- `infer`

The server uses `src/python/graspgen_s600_tools/runtime/hbm_graspgen_runtime.py` for config loading, point-cloud validation, HBM path validation, and dry-run-safe response formatting.

### Pi-side client/orchestrator

```text
src/python/scripts/real_grasp_pi_client.py
```

Initial safe adapters:

- `FilePointCloudCameraAdapter`
- `NullCameraAdapter`
- `NullRobotAdapter`
- `LoggingRobotAdapter`

Real camera/robot adapters should be added only after the Pi's hardware APIs are identified.

### Validation script

```text
src/python/scripts/validate_real_grasp_stack.py
```

This checks Pi SSH connectivity, point-cloud loading, preprocessing, inference round trip, and robot dry-run without commanding robot motion.

### Split-HBM smoke script

```text
src/python/scripts/smoke_test_split_hbm_local.sh
```

This independently runs `model_info` and synthetic inference for the selected split generator HBM files:

```text
pointnet_sa1_neural_gen_bpu.hbm
pointnet_sa2_neural_gen_conv1_cpu.hbm
pointnet_sa3_encoder_head_gen_all_cpu.hbm
graspgen_generator_head_temb_simplified_pred_cpu.hbm
```

## Python dependencies

The local minimal Python currently can run config loading and local mock validation with only NumPy. ZMQ serving requires:

```text
pyzmq
msgpack
msgpack-numpy
```

If missing, install them in the Python environment used by the local server and Pi client.

## Stage 0: local non-network validation

This does not touch the Pi or robot:

```bash
python3 src/python/scripts/validate_real_grasp_stack.py \
  --config configs/real_grasp_test.yaml \
  --skip-ssh \
  --local-mock \
  --synthetic-if-no-camera-file
```

Expected result:

```text
✅ Real-grasp validation passed. No robot motion was commanded.
```

## Stage 1: local split-HBM smoke

This validates that the selected HBM artifacts load and produce finite outputs with synthetic tensors:

```bash
bash src/python/scripts/smoke_test_split_hbm_local.sh models/hbm_split_gen_selected
```

Expected result:

```text
✅ Local split-HBM smoke test passed.
```

## Stage 2: start local inference server

For transport/client testing before real HBM inference is fully wired:

```bash
python3 src/python/scripts/serve_graspgen_s600_zmq.py \
  --config configs/real_grasp_test.yaml \
  --backend mock \
  --host 0.0.0.0 \
  --port 5556
```

For HBM file validation mode:

```bash
python3 src/python/scripts/serve_graspgen_s600_zmq.py \
  --config configs/real_grasp_test.yaml \
  --backend split_hbm \
  --host 0.0.0.0 \
  --port 5556
```

Note: until the full split-HBM diffusion loop is wired into the runtime, `split_hbm` validates HBM files but returns deterministic dry-run candidates with an explicit warning unless `--require-real-hbm` is used.

## Stage 3: validate from local or Pi client

With a file point cloud configured in `configs/real_grasp_test.yaml` or passed on the command line:

```bash
python3 src/python/scripts/real_grasp_pi_client.py \
  --config configs/real_grasp_test.yaml \
  --server-host <local-s600-ip> \
  --point-cloud /path/to/object_point_cloud.npy
```

For non-interactive dry-run approval only:

```bash
python3 src/python/scripts/real_grasp_pi_client.py \
  --config configs/real_grasp_test.yaml \
  --server-host <local-s600-ip> \
  --point-cloud /path/to/object_point_cloud.npy \
  --yes
```

This still does not move the robot with the default `null` or `logging` robot adapters.

## Real execution gate

The client has an `--execute` path, but the current adapters intentionally refuse real motion. Before adding a real robot adapter, fill in:

- calibration transform paths
- camera and robot frame names
- workspace bounds
- collision-check inputs
- actual robot adapter implementation
- E-stop/readiness checks for that robot

The intended command shape after those are implemented is:

```bash
python3 src/python/scripts/real_grasp_pi_client.py \
  --config configs/real_grasp_test.yaml \
  --server-host <local-s600-ip> \
  --execute \
  --i-understand-this-can-move-the-robot
```

The script still requires a same-session dry-run and final typed approval before calling `execute_grasp()`.

## Scripted Yahboom motion for visual acquisition

The default Orange Pi profile remains read-only: `YahboomArmHealthCheckAdapter` reads arm board/servo state only and still refuses execution. A separate opt-in adapter, `yahboom_arm_scripted_motion`, supports only tightly bounded scripted servo motions for camera viewpoint setup:

- `delta`: one or more CLI servo deltas, constrained by config
- `named-pose`: a named servo pose from `robot.scripted_motion.named_poses`
- `observation-pose`: the fixed pose in `calibration.observation_pose`

It does **not** execute arbitrary GraspGen 6D grasp candidates. Pure scripted motion branches before camera capture and S600 inference, then calls a dry-run first. Physical motion still requires all three gates: `--execute`, `--i-understand-this-can-move-the-robot`, and final typed `MOVE`.

The current Gemini Max eye-in-hand depth-view pose is **not** trusted as the final observation pose. On the current hardware, the DOFBOT-Pro 3D-vision example pose `[90, 120, 0, 0, 90, 90]` produced an empty depth point cloud. A later probe pose produced about 188k valid depth points, but the operator observed that the camera still was not aimed at the table. That pose is therefore recorded only as `robot.scripted_motion.named_poses.depth_probe_points_only`, while `calibration.observation_pose.enabled` is `false`:

```yaml
robot:
  scripted_motion:
    named_poses:
      depth_probe_points_only:
        servo_positions_deg:
          1: 90.0
          2: 95.0
          3: 52.0
          4: 62.0
          5: 90.0
          6: 90.0

calibration:
  observation_pose:
    enabled: false
    servo_positions_deg: {}
```

Use pose scout before promoting any pose to `calibration.observation_pose`. The scout scores the actual captured/saved Gemini Max point cloud for a broad, central, table-like plane, optional RGB color evidence, and depth coverage; it never contacts S600 and never authorizes GraspGen execution.

No-motion score of the currently saved point-cloud files:

```bash
python3 src/python/scripts/dofbot_pose_scout.py \
  --config configs/real_grasp_orangepi_gemini_max_dryrun.yaml \
  --score-only \
  --pose depth_probe_points_only \
  --report-json /tmp/dofbot_pose_scout_depth_probe.json
```

If a capture-refresh command exists on the Orange Pi, pass it with `--capture-command '...'` so the score uses a fresh point cloud. Physical scouting remains one pose per command and still requires the final typed `MOVE` gate:

```bash
python3 src/python/scripts/dofbot_pose_scout.py \
  --config configs/real_grasp_orangepi_yahboom_scripted_motion.yaml \
  --pose depth_probe_points_only \
  --capture-command '<refresh Gemini Max point-cloud npy files>' \
  --execute \
  --i-understand-this-can-move-the-robot \
  --report-json /tmp/dofbot_pose_scout_depth_probe.json
```

Fail-closed dry-run with the default non-moving config:

```bash
python3 src/python/scripts/real_grasp_pi_client.py \
  --config configs/real_grasp_orangepi_gemini_max_dryrun.yaml \
  --scripted-motion delta \
  --servo-delta 1=1.0
```

Accepted no-motion dry-run with the opt-in scripted adapter config:

```bash
python3 src/python/scripts/real_grasp_pi_client.py \
  --config configs/real_grasp_orangepi_yahboom_scripted_motion.yaml \
  --scripted-motion delta \
  --servo-delta 1=1.0
```

Expected: no camera capture, no S600 request, no servo write, and printed current/target/delta with `command_sent: false`.

Only with the operator physically present and ready to cut power, the first physical test should be a slow one-servo 1° delta:

```bash
python3 src/python/scripts/real_grasp_pi_client.py \
  --config configs/real_grasp_orangepi_yahboom_scripted_motion.yaml \
  --scripted-motion delta \
  --servo-delta 1=1.0 \
  --execute \
  --i-understand-this-can-move-the-robot
```

Then type exactly `MOVE` at the final prompt. If it succeeds, reverse it with `--servo-delta 1=-1.0` under the same gates.

## Deployment dry-run profile

For a service-oriented deployment dry run, use:

```text
configs/real_grasp_deploy_dryrun.yaml
```

This profile differs from the lab test profile in three important ways:

- it uses `runtime.backend: monolithic_hbm` so inference must run a real HBM generator loop
- it probes the Pi USB camera through `v4l2-ctl` without depending on OpenCV
- it probes the Yahboom/Dofbot arm through `Arm_Lib` read-only health checks

It still cannot command real robot motion. The arm adapter is a health-check adapter: it may read board/servo state, but `execute_grasp()` refuses motion.

### Pi cleanup for deployment

The Pi had unrelated services that can consume camera/audio/desktop/serial resources. For this deployment profile, stop and disable only the conflicting services, keeping files and projects for rollback:

```bash
systemctl --user disable --now wakeword-listener.service
sudo systemctl disable --now lightdm.service cups.service cups-browsed.service ModemManager.service triggerhappy.service
```

Keep Docker/FreshRSS/HomeAssistant data, `avahi`, and the existing `auto_ipmi_power.sh` cron jobs unless a later test proves they interfere.

Rollback:

```bash
systemctl --user enable --now wakeword-listener.service
sudo systemctl enable --now lightdm.service cups.service cups-browsed.service ModemManager.service triggerhappy.service
```

### S600 systemd service template

A template is provided at:

```text
deploy/systemd/graspgen-s600-server.service
```

Install/start shape on the S600 machine:

```bash
sudo cp deploy/systemd/graspgen-s600-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now graspgen-s600-server.service
sudo journalctl -u graspgen-s600-server.service -n 100 --no-pager
```

The template starts the server with `--backend monolithic_hbm --require-real-hbm` so it fails closed if real HBM inference cannot run.

### Pi preflight systemd service template

A template is provided at:

```text
deploy/systemd/graspgen-pi-preflight.service
```

Suggested Pi deployment directory:

```text
/home/pi/graspgen-s600-deploy
```

Copy the minimal deploy files there, then install the unit:

```bash
sudo cp deploy/systemd/graspgen-pi-preflight.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start graspgen-pi-preflight.service
sudo journalctl -u graspgen-pi-preflight.service -n 100 --no-pager
```

This is a one-shot dry-run preflight. It checks camera health, arm health, S600 server reachability, inference response shape, and robot dry-run logging. It runs on the Pi itself, so it uses `--skip-ssh` and does not include `--execute`.

## Current limitations

- The deployable real-HBM dry-run path currently uses the monolithic generator HBM, not the selected split-HBM chain.
- The split-HBM artifacts are validated and smoke-tested independently.
- The full split-HBM diffusion loop for live inference is not yet wired into `S600GraspRuntime`.
- The Pi camera adapter is a V4L2 health probe, not a calibrated RGB-D point-cloud pipeline.
- The default Yahboom/Dofbot robot adapter is read-only/health-check plus dry-run logging; it refuses real execution.
- The scripted Yahboom/Dofbot motion adapter is opt-in and only supports bounded named/delta servo scripts, not GraspGen grasp execution.
- ZMQ server/client execution requires `pyzmq` and `msgpack-numpy` in the active Python environment.
