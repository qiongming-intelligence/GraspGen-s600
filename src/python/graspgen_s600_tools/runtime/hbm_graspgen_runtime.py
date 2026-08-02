"""Runtime helpers for safe S600-backed real-grasp dry runs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PRELOAD = (
    "/home/sunrise/Projects/FoundationPose-s600/build/cmake/src/csrc/"
    "libfoundationpose_bpu_core1_preload.so"
)


@dataclass(frozen=True)
class NormalizedPointCloud:
    """Centered/resampled point cloud plus inverse transform metadata."""

    model_input: np.ndarray
    centroid: np.ndarray
    scale: float
    radius: float
    original_count: int
    original_bounds_min: np.ndarray
    original_bounds_max: np.ndarray
    sampled_indices: np.ndarray


@dataclass(frozen=True)
class InferenceResult:
    """Inference response payload before msgpack serialization."""

    grasps: np.ndarray
    confidences: np.ndarray
    ranked_indices: np.ndarray
    grasp_vectors: np.ndarray
    timing: dict[str, float]
    safety: dict[str, Any]
    normalization: dict[str, Any]
    warnings: list[str]
    backend: str
    effective_backend: str


SPLIT_GENERATOR_MODELS = {
    "sa1": (
        "pointnet_sa1_neural_gen_bpu",
        "pointnet_sa1_neural_gen_bpu/pointnet_sa1_neural_gen_bpu.hbm",
    ),
    "sa2": (
        "pointnet_sa2_neural_gen_conv1_cpu",
        "pointnet_sa2_neural_gen_conv1_cpu/pointnet_sa2_neural_gen_conv1_cpu.hbm",
    ),
    "sa3": (
        "pointnet_sa3_encoder_head_gen_all_cpu",
        "pointnet_sa3_encoder_head_gen_all_cpu/pointnet_sa3_encoder_head_gen_all_cpu.hbm",
    ),
    "head": (
        "graspgen_generator_head_temb_simplified_pred_cpu",
        "graspgen_generator_head_temb_simplified_pred_cpu/"
        "graspgen_generator_head_temb_simplified_pred_cpu.hbm",
    ),
}


MONOLITHIC_MODELS = {
    "generator": ("graspgen_generator_pointnet", "graspgen_generator_pointnet.hbm"),
    "discriminator": (
        "graspgen_discriminator_pointnet",
        "graspgen_discriminator_pointnet.hbm",
    ),
}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file with a clear dependency error."""

    cfg_path = Path(path)
    if cfg_path.suffix.lower() == ".json":
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except ImportError:
            data = _load_simple_yaml(path)
        else:
            with cfg_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def _load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the simple mapping/list YAML used by the safety config.

    This fallback keeps the dry-run scripts usable on minimal Pi images that do
    not have PyYAML installed. It intentionally supports only the subset used in
    `configs/real_grasp_test.yaml`.
    """

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"unsupported YAML line: {raw_line!r}")
        key, value_text = stripped.split(":", 1)
        key = key.strip()
        value_text = value_text.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value_text == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_simple_yaml_scalar(value_text)
    return root


def _parse_simple_yaml_scalar(text: str) -> Any:
    if text in {"null", "Null", "NULL", "~"}:
        return None
    if text in {"true", "True", "TRUE"}:
        return True
    if text in {"false", "False", "FALSE"}:
        return False
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [_parse_simple_yaml_scalar(part.strip()) for part in inner.split(",")]
    if text == "{}":
        return {}
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def get_nested(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Read a nested config value without requiring a schema library."""

    cur: Any = config
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_config_path(value: str | None, *, base: Path = PROJECT_ROOT) -> Path | None:
    """Resolve a path from config; None stays None."""

    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def validate_point_cloud(
    point_cloud: np.ndarray,
    *,
    min_points: int = 256,
    max_abs_coord_m: float = 5.0,
) -> np.ndarray:
    """Validate and return an `(N, 3)` float32 point cloud."""

    pc = np.asarray(point_cloud, dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] != 3:
        raise ValueError(f"point_cloud must have shape (N, 3), got {pc.shape}")
    if pc.shape[0] < min_points:
        raise ValueError(f"point_cloud has {pc.shape[0]} points, need at least {min_points}")
    finite = np.isfinite(pc).all(axis=1)
    if not finite.all():
        pc = pc[finite]
    if pc.shape[0] < min_points:
        raise ValueError(
            f"point_cloud has {pc.shape[0]} finite points after filtering, "
            f"need at least {min_points}"
        )
    if np.abs(pc).max() > max_abs_coord_m:
        raise ValueError(
            f"point_cloud coordinate exceeds configured max_abs_coord_m={max_abs_coord_m}"
        )
    return np.ascontiguousarray(pc, dtype=np.float32)


def normalize_point_cloud(
    point_cloud: np.ndarray,
    *,
    num_points: int = 2048,
    seed: int = 0,
    min_points: int = 256,
    max_abs_coord_m: float = 5.0,
    scale_to_unit: bool = False,
) -> NormalizedPointCloud:
    """Center, optionally scale, and resample an object point cloud."""

    pc = validate_point_cloud(
        point_cloud,
        min_points=min_points,
        max_abs_coord_m=max_abs_coord_m,
    )
    rng = np.random.default_rng(seed)
    count = pc.shape[0]
    if count >= num_points:
        sampled = rng.choice(count, size=num_points, replace=False)
    else:
        extra = rng.choice(count, size=num_points - count, replace=True)
        sampled = np.concatenate([np.arange(count), extra], axis=0)

    sampled_pc = pc[sampled]
    centroid = sampled_pc.mean(axis=0).astype(np.float32)
    centered = sampled_pc - centroid[None, :]
    radius = float(np.linalg.norm(centered, axis=1).max())
    scale = radius if scale_to_unit and radius > 1e-8 else 1.0
    model_pc = (centered / scale).astype(np.float32)

    return NormalizedPointCloud(
        model_input=np.ascontiguousarray(model_pc[None, :, :], dtype=np.float32),
        centroid=centroid,
        scale=float(scale),
        radius=float(radius),
        original_count=count,
        original_bounds_min=pc.min(axis=0).astype(np.float32),
        original_bounds_max=pc.max(axis=0).astype(np.float32),
        sampled_indices=sampled.astype(np.int64),
    )


class S600GraspRuntime:
    """Safety-oriented S600 inference facade.

    The split-HBM denoising chain is validated here, but the initial real-grasp
    path deliberately falls back to deterministic mock grasps unless a concrete
    HBM execution loop is added. Responses label the effective backend so robot
    execution can refuse mock results.
    """

    def __init__(self, config: dict[str, Any], *, backend: str | None = None) -> None:
        self.config = config
        self.backend = backend or str(get_nested(config, ("runtime", "backend"), "split_hbm"))
        self.hrt_model_exec = str(
            get_nested(config, ("runtime", "hrt_model_exec"), "/usr/hobot/bin/hrt_model_exec")
        )
        self.preload = str(get_nested(config, ("runtime", "preload"), DEFAULT_PRELOAD))
        self.hbm_root = resolve_config_path(
            get_nested(config, ("runtime", "hbm_root"), "models/hbm_split_gen_selected")
        ) or (PROJECT_ROOT / "models" / "hbm_split_gen_selected")
        self.monolithic_hbm_dir = resolve_config_path(
            get_nested(config, ("runtime", "monolithic_hbm_dir"), "models/hbm")
        ) or (PROJECT_ROOT / "models" / "hbm")
        self.work_dir = resolve_config_path(
            get_nested(config, ("runtime", "work_dir"), "/tmp/graspgen_s600_real_grasp")
        ) or Path("/tmp/graspgen_s600_real_grasp")
        self.num_points = int(get_nested(config, ("runtime", "num_points"), 2048))
        self.num_grasps = int(get_nested(config, ("runtime", "num_grasps"), 20))
        self.grasp_dim = int(get_nested(config, ("runtime", "grasp_dim"), 6))
        self.num_diffusion_steps = int(
            get_nested(config, ("runtime", "num_diffusion_steps"), 20)
        )
        self.seed = int(get_nested(config, ("runtime", "seed"), 0))
        self.scale_to_unit = bool(get_nested(config, ("runtime", "scale_to_unit"), False))
        # Training-time translation scale that the decode must undo. This is a
        # per-DATASET constant (kappa = 1 / mean(grasp_extent)) baked into the
        # checkpoint, not a per-cloud value, so it has to match the gripper the
        # weights were trained on: upstream runs/ gives 2.02217 for
        # robotiq_2f_140 and 3.27 for franka_panda. Required, with no default,
        # because a silently wrong kappa mis-scales every grasp translation.
        kappa_cfg = get_nested(config, ("runtime", "grasp_translation_kappa"), None)
        self.kappa = float(kappa_cfg) if kappa_cfg is not None else None
        self.monolithic_timestep_dtype = str(
            get_nested(
                config,
                ("runtime", "monolithic_timestep_dtype"),
                get_nested(config, ("runtime", "monolithic", "timestep_dtype"), "f32"),
            )
        )
        if self.monolithic_timestep_dtype not in {"i64", "i32", "f32"}:
            raise ValueError(
                "runtime.monolithic_timestep_dtype must be one of i64, i32, or f32; "
                f"got {self.monolithic_timestep_dtype!r}"
            )

    @classmethod
    def from_config_file(
        cls, path: str | Path, *, backend: str | None = None
    ) -> "S600GraspRuntime":
        return cls(load_yaml_config(path), backend=backend)

    def metadata(self) -> dict[str, Any]:
        """Return non-secret runtime metadata."""

        return {
            "backend": self.backend,
            "hbm_files": {key: str(path) for key, path in self.hbm_files().items()},
            "num_points": self.num_points,
            "num_grasps": self.num_grasps,
            "grasp_dim": self.grasp_dim,
            "num_diffusion_steps": self.num_diffusion_steps,
            "preload": self.preload,
            "work_dir": str(self.work_dir),
            "safety": {
                "default_mode": get_nested(
                    self.config, ("safety", "default_mode"), "dry_run"
                ),
                "require_manual_approval": bool(
                    get_nested(self.config, ("safety", "require_manual_approval"), True)
                ),
                "motion_authorized_by_server": False,
            },
        }

    def monolithic_model_name(self, key: str) -> str:
        """Return the HBM model name for a monolithic model key."""

        if key not in MONOLITHIC_MODELS:
            raise KeyError(f"unknown monolithic model key: {key}")
        configured = get_nested(self.config, ("runtime", "monolithic_model_names"), {}) or {}
        if isinstance(configured, dict) and key in configured:
            return str(configured[key])
        model_config = get_nested(self.config, ("runtime", "monolithic"), {}) or {}
        name_key = f"{key}_model_name"
        if isinstance(model_config, dict) and name_key in model_config:
            return str(model_config[name_key])
        return MONOLITHIC_MODELS[key][0]

    def hbm_files(self) -> dict[str, Path]:
        """Resolve HBM files for the selected backend."""

        if self.backend == "mock":
            return {}
        if self.backend == "split_hbm":
            configured = get_nested(self.config, ("runtime", "split_generator"), {}) or {}
            files: dict[str, Path] = {}
            for key, (_, default_rel) in SPLIT_GENERATOR_MODELS.items():
                rel = configured.get(key, default_rel) if isinstance(configured, dict) else default_rel
                files[key] = self.hbm_root / rel
            return files
        if self.backend == "monolithic_hbm":
            configured = get_nested(self.config, ("runtime", "monolithic"), {}) or {}
            files = {}
            for key, (_, default_rel) in MONOLITHIC_MODELS.items():
                rel = configured.get(key, default_rel) if isinstance(configured, dict) else default_rel
                files[key] = self.monolithic_hbm_dir / rel
            return files
        raise ValueError(f"unsupported runtime backend: {self.backend}")

    def validate_hbm_files(self) -> list[str]:
        """Validate configured HBM paths and return warnings."""

        warnings: list[str] = []
        missing = [str(path) for path in self.hbm_files().values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing HBM file(s): " + ", ".join(missing))
        if self.backend != "mock" and not Path(self.hrt_model_exec).exists():
            raise FileNotFoundError(f"hrt_model_exec not found: {self.hrt_model_exec}")
        preload_path = Path(self.preload)
        if self.backend != "mock" and not preload_path.exists():
            warnings.append(
                f"HBRT preload not found: {preload_path}; model loading may fail on S600"
            )
        return warnings

    def run_model_info(self, *, timeout_s: float = 120.0) -> dict[str, dict[str, Any]]:
        """Run `hrt_model_exec model_info` for configured HBM files."""

        self.validate_hbm_files()
        results: dict[str, dict[str, Any]] = {}
        env = os.environ.copy()
        if Path(self.preload).exists():
            env["LD_PRELOAD"] = self.preload + (
                f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
            )
        for key, path in self.hbm_files().items():
            t0 = time.monotonic()
            proc = subprocess.run(
                [self.hrt_model_exec, "model_info", "--model_file", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
                env=env,
            )
            results[key] = {
                "path": str(path),
                "returncode": proc.returncode,
                "elapsed_ms": (time.monotonic() - t0) * 1000.0,
                "output": proc.stdout,
            }
        return results

    def infer(
        self,
        point_cloud: np.ndarray,
        *,
        params: dict[str, Any] | None = None,
        request_id: str | None = None,
        require_real_hbm: bool = False,
    ) -> InferenceResult:
        """Run a dry-run-safe inference request.

        Until the full split-HBM diffusion loop is wired, non-mock backends are
        validated and then refused when `require_real_hbm` is true; otherwise a
        deterministic mock result is returned with an explicit warning.
        """

        del request_id
        params = params or {}
        t0 = time.monotonic()
        min_points = int(get_nested(self.config, ("safety", "min_points_before_resample"), 256))
        max_abs = float(get_nested(self.config, ("safety", "max_abs_coord_m"), 5.0))
        normalized = normalize_point_cloud(
            point_cloud,
            num_points=self.num_points,
            seed=int(params.get("seed", self.seed)),
            min_points=min_points,
            max_abs_coord_m=max_abs,
            scale_to_unit=bool(params.get("scale_to_unit", self.scale_to_unit)),
        )
        preprocess_ms = (time.monotonic() - t0) * 1000.0

        warnings = self.validate_hbm_files() if self.backend != "mock" else []

        t1 = time.monotonic()
        effective_backend = "mock"
        if self.backend == "monolithic_hbm":
            grasps, vectors, confidences = self._infer_monolithic_hbm(
                normalized,
                seed=int(params.get("seed", self.seed)),
                num_steps=int(params.get("num_diffusion_steps", self.num_diffusion_steps)),
            )
            effective_backend = "monolithic_hbm"
            warnings.append(
                "monolithic_hbm ran the single-step generator in a Python DDPM loop; "
                "use split_hbm for the selected production candidate once its loop is wired"
            )
        elif self.backend == "split_hbm":
            msg = (
                "split_hbm files validated, but the real split-HBM diffusion "
                "execution loop is not wired into this runtime yet; returning "
                "deterministic dry-run grasps"
            )
            if require_real_hbm:
                raise NotImplementedError(msg)
            warnings.append(msg)
            grasps, vectors, confidences = self._mock_grasps(normalized)
        else:
            grasps, vectors, confidences = self._mock_grasps(normalized)
        infer_ms = (time.monotonic() - t1) * 1000.0
        ranked = np.argsort(-confidences).astype(np.int64)
        safety = self._safety_summary(
            grasps,
            normalized=normalized,
            effective_backend=effective_backend,
        )
        envelope_warning = safety.get("translation_envelope_warning")
        if envelope_warning:
            warnings.append(str(envelope_warning))
        timing = {"preprocess_ms": preprocess_ms, "infer_ms": infer_ms}
        if hasattr(self, "_last_hbm_step_timing_ms"):
            timing["hbm_step_mean_ms"] = float(np.mean(self._last_hbm_step_timing_ms))
            timing["hbm_step_max_ms"] = float(np.max(self._last_hbm_step_timing_ms))
            timing["hbm_steps"] = int(len(self._last_hbm_step_timing_ms))
        normalization = {
            "centroid": normalized.centroid,
            "scale": normalized.scale,
            "radius": normalized.radius,
            "original_count": normalized.original_count,
            "original_bounds_min": normalized.original_bounds_min,
            "original_bounds_max": normalized.original_bounds_max,
            "model_input_shape": np.array(normalized.model_input.shape, dtype=np.int64),
        }
        return InferenceResult(
            grasps=grasps,
            confidences=confidences,
            ranked_indices=ranked,
            grasp_vectors=vectors,
            timing=timing,
            safety=safety,
            normalization=normalization,
            warnings=warnings,
            backend=self.backend,
            effective_backend=effective_backend,
        )

    def _infer_monolithic_hbm(
        self,
        normalized: NormalizedPointCloud,
        *,
        seed: int,
        num_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the monolithic single-step generator HBM in a Python DDPM loop."""

        if self.grasp_dim != 6:
            raise ValueError("monolithic_hbm runtime currently expects r3_so3 grasp_dim=6")
        files = self.hbm_files()
        generator_hbm = files.get("generator")
        if generator_hbm is None:
            raise FileNotFoundError("generator HBM is not configured")

        rng = np.random.default_rng(seed)
        sample = rng.standard_normal((self.num_grasps, self.grasp_dim), dtype=np.float32)
        train_timesteps = 100
        timesteps = _ddpm_timesteps(num_steps, num_train_timesteps=train_timesteps)
        alphas_cumprod = _ddpm_alphas_cumprod(num_train_timesteps=train_timesteps)
        train_step = max(train_timesteps // num_steps, 1)
        step_timing_ms: list[float] = []
        request_dir = self.work_dir / f"monolithic_{time.time_ns()}"
        input_dir = request_dir / "inputs"
        output_root = request_dir / "outputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        try:
            pc_path = input_dir / "pc_f32_1x2048x3.bin"
            normalized.model_input.astype(np.float32).tofile(pc_path)
            for step_index, timestep in enumerate(timesteps):
                noisy_path = input_dir / f"noisy_grasps_step{step_index:03d}_f32_20x6.bin"
                timestep_path = input_dir / (
                    f"timestep_step{step_index:03d}_{self.monolithic_timestep_dtype}_1.bin"
                )
                out_dir = output_root / f"step_{step_index:03d}"
                sample.astype(np.float32).tofile(noisy_path)
                _write_timestep_input(
                    timestep_path,
                    int(timestep),
                    self.monolithic_timestep_dtype,
                )
                t0 = time.monotonic()
                self._run_hbm_infer(
                    generator_hbm,
                    self.monolithic_model_name("generator"),
                    [pc_path, noisy_path, timestep_path],
                    out_dir,
                )
                step_timing_ms.append((time.monotonic() - t0) * 1000.0)
                noise_pred = self._read_output_array(
                    out_dir,
                    "noise_pred",
                    (self.num_grasps, self.grasp_dim),
                    allow_stride8=True,
                )
                prev_timestep = int(timestep) - train_step
                sample = _ddpm_step_epsilon(
                    sample,
                    noise_pred,
                    int(timestep),
                    alphas_cumprod,
                    prev_timestep=prev_timestep,
                )
                if not np.isfinite(sample).all():
                    raise RuntimeError(f"non-finite DDPM sample at step {step_index}")
        finally:
            if bool(get_nested(self.config, ("runtime", "keep_work_dir"), False)):
                pass
            else:
                shutil.rmtree(request_dir, ignore_errors=True)

        self._last_hbm_step_timing_ms = step_timing_ms
        grasps = _r3_so3_vectors_to_matrices(
            sample,
            centroid=normalized.centroid,
            scale=normalized.scale,
            kappa=self.kappa,
        )
        if not np.isfinite(grasps).all():
            raise RuntimeError("monolithic_hbm produced non-finite grasp matrices")
        confidences = _rank_by_center_distance(grasps, normalized.centroid)
        return grasps, sample.astype(np.float32), confidences

    def _run_hbm_infer(
        self,
        hbm_path: Path,
        model_name: str,
        input_paths: list[Path],
        out_dir: Path,
        *,
        timeout_s: float = 120.0,
    ) -> None:
        """Run `hrt_model_exec infer` and raise with combined output on failure."""

        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if Path(self.preload).exists():
            env["LD_PRELOAD"] = self.preload + (
                f":{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else ""
            )
        proc = subprocess.run(
            [
                self.hrt_model_exec,
                "infer",
                "--model_file",
                str(hbm_path),
                "--model_name",
                model_name,
                "--input_file",
                ",".join(str(path) for path in input_paths),
                "--frame_count",
                "1",
                "--enable_dump",
                "true",
                "--dump_format",
                "bin",
                "--dump_path",
                str(out_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
            env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hrt_model_exec infer failed for {model_name} rc={proc.returncode}:\n{proc.stdout}"
            )

    @staticmethod
    def _read_output_array(
        out_dir: Path,
        output_token: str,
        logical_shape: tuple[int, ...],
        *,
        allow_stride8: bool = False,
    ) -> np.ndarray:
        candidates = sorted(out_dir.glob(f"*{output_token}*.bin"))
        if not candidates:
            raise FileNotFoundError(f"missing HBM output containing {output_token!r} in {out_dir}")

        logical_size = int(np.prod(logical_shape))
        valid: list[tuple[Path, str]] = []
        for path in candidates:
            values = path.stat().st_size // np.dtype(np.float32).itemsize
            if values == logical_size:
                valid.append((path, "logical"))
            elif allow_stride8 and len(logical_shape) == 2 and logical_shape[1] == 6:
                stride_size = int(logical_shape[0] * 8)
                if values == stride_size:
                    valid.append((path, "stride8"))

        if len(valid) != 1:
            details = []
            for path in candidates:
                stat = path.stat()
                details.append(f"{path.name}: {stat.st_size} bytes")
            expected = [f"{logical_size} float32 values for {logical_shape}"]
            if allow_stride8 and len(logical_shape) == 2 and logical_shape[1] == 6:
                expected.append(f"{logical_shape[0] * 8} float32 values for stride8")
            raise ValueError(
                "ambiguous HBM output selection for "
                f"{output_token!r} in {out_dir}; expected {' or '.join(expected)}, "
                f"valid_matches={len(valid)}, candidates={details}"
            )

        path, layout = valid[0]
        flat = np.fromfile(path, dtype=np.float32)
        if layout == "logical":
            return np.ascontiguousarray(flat.reshape(logical_shape), dtype=np.float32)
        stride_shape = (logical_shape[0], 8)
        return np.ascontiguousarray(flat.reshape(stride_shape)[:, :6], dtype=np.float32)

    def _mock_grasps(
        self, normalized: NormalizedPointCloud
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate deterministic non-executable grasp candidates for dry runs."""

        count = self.num_grasps
        grasps = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
        vectors = np.zeros((count, self.grasp_dim), dtype=np.float32)
        radius = max(float(normalized.radius), 0.04)
        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False, dtype=np.float32)
        for i, angle in enumerate(angles):
            offset = np.array(
                [0.15 * radius * np.cos(angle), 0.15 * radius * np.sin(angle), 0.35 * radius],
                dtype=np.float32,
            )
            translation = normalized.centroid + offset
            grasps[i, :3, 3] = translation
            vectors[i, :3] = translation
            if self.grasp_dim >= 6:
                vectors[i, 3:6] = np.array([0.0, np.pi, angle], dtype=np.float32)
        confidences = np.linspace(0.75, 0.45, count, dtype=np.float32)
        return grasps, vectors, confidences

    def _safety_summary(
        self,
        grasps: np.ndarray,
        *,
        normalized: NormalizedPointCloud,
        effective_backend: str,
    ) -> dict[str, Any]:
        bounds = get_nested(self.config, ("safety", "workspace_bounds"), {}) or {}
        workspace_configured = self._workspace_bounds_configured(bounds)
        workspace_ok = bool(workspace_configured and self._grasps_in_workspace(grasps, bounds))
        envelope = self._translation_envelope_summary(grasps, normalized)
        summary = {
            "finite_checked": bool(np.isfinite(grasps).all()),
            "workspace_checked": bool(workspace_configured),
            "workspace_ok": workspace_ok,
            "collision_checked": False,
            "collision_ok": False,
            "requires_manual_approval": bool(
                get_nested(self.config, ("safety", "require_manual_approval"), True)
            ),
            "motion_authorized": False,
            "effective_backend": effective_backend,
            "real_hbm_inference": effective_backend != "mock",
            "translation_envelope": envelope,
        }
        if envelope.get("warning"):
            summary["translation_envelope_warning"] = envelope["warning"]
        return summary

    @staticmethod
    def _translation_envelope_summary(
        grasps: np.ndarray,
        normalized: NormalizedPointCloud,
    ) -> dict[str, Any]:
        trans = np.asarray(grasps, dtype=np.float32)[:, :3, 3]
        if trans.size == 0:
            return {
                "checked": True,
                "max_distance_from_centroid_m": 0.0,
                "median_distance_from_centroid_m": 0.0,
                "threshold_m": 0.0,
                "outside_count": 0,
            }
        centroid = normalized.centroid.astype(np.float32)
        distances = np.linalg.norm(trans - centroid[None, :], axis=1)
        radius = max(float(normalized.radius), 1e-6)
        # Robotiq palm-frame translations can be well outside the object bounds.
        # This is a diagnostic envelope only: large values are reported, not clipped.
        threshold = max(0.35, 4.0 * radius)
        outside = distances > threshold
        result: dict[str, Any] = {
            "checked": True,
            "centroid_m": [float(v) for v in centroid],
            "object_radius_m": float(normalized.radius),
            "threshold_m": float(threshold),
            "max_distance_from_centroid_m": float(distances.max()),
            "median_distance_from_centroid_m": float(np.median(distances)),
            "outside_count": int(outside.sum()),
            "count": int(distances.size),
        }
        if outside.any():
            result["warning"] = (
                "grasp translations exceed the diagnostic Robotiq palm-frame envelope "
                f"({int(outside.sum())}/{int(distances.size)} > {threshold:.3f} m); "
                "keep treating these as non-executable dry-run candidates"
            )
        return result

    @staticmethod
    def _workspace_bounds_configured(bounds: dict[str, Any]) -> bool:
        for axis in ("x", "y", "z"):
            values = bounds.get(axis) if isinstance(bounds, dict) else None
            if not isinstance(values, (list, tuple)) or len(values) != 2:
                return False
            if values[0] is None or values[1] is None:
                return False
        return True

    @staticmethod
    def _grasps_in_workspace(grasps: np.ndarray, bounds: dict[str, Any]) -> bool:
        trans = np.asarray(grasps, dtype=np.float32)[:, :3, 3]
        for idx, axis in enumerate(("x", "y", "z")):
            lo, hi = bounds[axis]
            if (trans[:, idx] < float(lo)).any() or (trans[:, idx] > float(hi)).any():
                return False
        return True


def _ddpm_alphas_cumprod(num_train_timesteps: int = 100) -> np.ndarray:
    """Return DDPM cumulative alphas for the squared-cosine beta schedule."""

    max_beta = 0.999

    def alpha_bar(t: float) -> float:
        return float(np.cos((t + 0.008) / 1.008 * np.pi / 2.0) ** 2)

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    alphas = 1.0 - np.asarray(betas, dtype=np.float64)
    return np.cumprod(alphas, axis=0).astype(np.float32)


def _ddpm_timesteps(num_steps: int, *, num_train_timesteps: int = 100) -> np.ndarray:
    """Match diffusers' leading-spaced DDPM timesteps for this fixed setup."""

    if num_steps <= 0:
        raise ValueError("num_diffusion_steps must be positive")
    if num_steps > num_train_timesteps:
        raise ValueError("num_diffusion_steps cannot exceed num_train_timesteps")
    step_ratio = num_train_timesteps // num_steps
    return (np.arange(0, num_steps, dtype=np.int64) * step_ratio)[::-1].copy()


def _ddpm_step_epsilon(
    sample: np.ndarray,
    noise_pred: np.ndarray,
    timestep: int,
    alphas_cumprod: np.ndarray,
    *,
    prev_timestep: int,
) -> np.ndarray:
    """One deterministic DDPM epsilon-prediction reverse step.

    This mirrors the mean update from diffusers' DDPMScheduler for
    prediction_type='epsilon' and variance noise set to zero. It is sufficient
    for safe deterministic S600 dry-run candidate generation.
    """

    alpha_prod_t = float(alphas_cumprod[timestep])
    alpha_prod_t_prev = float(alphas_cumprod[prev_timestep]) if prev_timestep >= 0 else 1.0
    beta_prod_t = 1.0 - alpha_prod_t
    beta_prod_t_prev = 1.0 - alpha_prod_t_prev
    current_alpha_t = alpha_prod_t / alpha_prod_t_prev
    current_beta_t = 1.0 - current_alpha_t

    pred_original_sample = (sample - np.sqrt(beta_prod_t) * noise_pred) / np.sqrt(alpha_prod_t)
    pred_original_sample = np.clip(pred_original_sample, -1.0, 1.0)
    pred_original_coeff = np.sqrt(alpha_prod_t_prev) * current_beta_t / beta_prod_t
    current_sample_coeff = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
    prev_sample = pred_original_coeff * pred_original_sample + current_sample_coeff * sample
    return np.ascontiguousarray(prev_sample.astype(np.float32))


def _r3_so3_vectors_to_matrices(
    vectors: np.ndarray,
    *,
    centroid: np.ndarray,
    scale: float,
    kappa: float | None = None,
) -> np.ndarray:
    """Convert GraspGen r3_so3 vectors to camera-frame 4x4 grasp matrices.

    The translation must be divided by `kappa` to undo the training-time scaling.
    Upstream `matrix_to_rt` multiplies the grasp translation by kappa when building
    targets, and `rt_to_matrix` divides by it on the way back
    (`mat[:, :3, 3] *= 1.0 / kappa`). Our exported ONNX returns the raw denoised
    vector and never applies that inverse, so omitting it here inflated every
    translation offset by kappa = 3.27x: measured candidate offsets from the
    centroid were 20.9-51.4mm where 6.4-15.7mm was correct, which pushed grasp
    origins outside the object and into the workspace/collision gates.

    kappa is a dataset-level constant (`kappa = 1 / mean(grasp_extent)`), NOT a
    per-cloud quantity, so it must not be confused with the point-cloud radius.
    Note also that upstream applies only `T_move_to_pc_mean`, a pure translation,
    to the input cloud - there is no unit-sphere scaling - which is why `scale`
    stays 1.0 on this path.
    """

    vec = np.asarray(vectors, dtype=np.float32)
    mats = np.tile(np.eye(4, dtype=np.float32), (vec.shape[0], 1, 1))
    translation = vec[:, :3] * float(scale)
    if kappa is not None and kappa > 0.0:
        translation = translation / float(kappa)
    mats[:, :3, 3] = translation + centroid[None, :]
    for i, rotvec in enumerate(vec[:, 3:6] * np.pi):
        mats[i, :3, :3] = _axis_angle_to_matrix(rotvec)
    return mats


def _axis_angle_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    axis = rotvec.astype(np.float64) / theta
    x, y, z = axis
    k = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    rot = eye + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)
    return rot.astype(np.float32)


def _rank_by_center_distance(grasps: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    trans = np.asarray(grasps, dtype=np.float32)[:, :3, 3]
    dist = np.linalg.norm(trans - centroid[None, :], axis=1)
    if dist.size == 0:
        return np.asarray([], dtype=np.float32)
    denom = float(dist.max() - dist.min())
    if denom < 1e-8:
        return np.ones(dist.shape, dtype=np.float32) * 0.5
    return (1.0 - (dist - dist.min()) / denom).astype(np.float32)


def _write_timestep_input(path: Path, timestep: int, dtype_name: str) -> None:
    if dtype_name == "i64":
        np.asarray([timestep], dtype=np.int64).tofile(path)
    elif dtype_name == "i32":
        np.asarray([timestep], dtype=np.int32).tofile(path)
    elif dtype_name == "f32":
        np.asarray([timestep], dtype=np.float32).tofile(path)
    else:
        raise ValueError(f"unsupported timestep dtype: {dtype_name}")


def result_to_response(result: InferenceResult, *, request_id: str | None = None) -> dict[str, Any]:
    """Convert an inference result to the project ZMQ response payload."""

    return {
        "request_id": request_id,
        "status": "ok",
        "mode": "dry_run",
        "backend": result.backend,
        "effective_backend": result.effective_backend,
        "grasps": result.grasps.astype(np.float32),
        "grasp_vectors": result.grasp_vectors.astype(np.float32),
        "confidences": result.confidences.astype(np.float32),
        "ranked_indices": result.ranked_indices.astype(np.int64),
        "num_grasps": int(result.grasps.shape[0]),
        "timing": result.timing,
        "safety": result.safety,
        "normalization": result.normalization,
        "warnings": result.warnings,
    }
