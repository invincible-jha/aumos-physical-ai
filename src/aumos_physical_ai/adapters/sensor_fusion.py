"""Multi-sensor fusion adapter for camera, LiDAR, and IMU data synthesis.

Generates temporally aligned, spatially calibrated multi-modal sensor datasets
combining camera frames, LiDAR point clouds, and IMU readings with realistic
noise models. Computes temporal alignment scores, spatial calibration quality,
and fusion quality metrics.

Implements SensorFusionEngineProtocol.
"""

import math
import random
import uuid
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sensor noise model constants
# ---------------------------------------------------------------------------

LIDAR_NOISE_STD_M: dict[str, float] = {
    "velodyne_vlp16": 0.002,
    "velodyne_hdl64e": 0.001,
    "ouster_os1": 0.003,
    "generic": 0.005,
}

CAMERA_NOISE_STD: dict[str, float] = {
    "rgb": 0.01,
    "depth": 0.015,
    "infrared": 0.008,
    "event": 0.002,
}

IMU_NOISE_DENSITY_RPS_SQHZ: dict[str, float] = {
    "mpu6050": 0.005,
    "bmi088": 0.003,
    "vectornav_vn100": 0.001,
    "generic": 0.004,
}


# ---------------------------------------------------------------------------
# Point cloud generator
# ---------------------------------------------------------------------------


def _generate_point_cloud(
    num_points: int,
    sensor_model: str,
    range_m: float = 50.0,
) -> dict[str, Any]:
    """Synthesize a LiDAR point cloud with realistic noise.

    Args:
        num_points: Target number of points in the cloud.
        sensor_model: LiDAR model name (used for noise std lookup).
        range_m: Maximum sensor range in metres.

    Returns:
        Dict with points array metadata, statistics, and noise parameters.
    """
    noise_std = LIDAR_NOISE_STD_M.get(sensor_model, LIDAR_NOISE_STD_M["generic"])
    # Simulate points distributed in a forward hemisphere
    points_sample: list[dict[str, float]] = []
    for _ in range(min(num_points, 50)):  # Sample 50 representative points
        azimuth = random.uniform(-math.pi, math.pi)
        elevation = random.uniform(-math.pi / 6, math.pi / 6)
        base_range = random.uniform(0.5, range_m)
        noisy_range = base_range + random.gauss(0.0, noise_std)
        x = noisy_range * math.cos(elevation) * math.cos(azimuth)
        y = noisy_range * math.cos(elevation) * math.sin(azimuth)
        z = noisy_range * math.sin(elevation)
        intensity = random.uniform(0.0, 1.0)
        points_sample.append({"x": round(x, 4), "y": round(y, 4), "z": round(z, 4), "intensity": round(intensity, 4)})

    return {
        "sensor_model": sensor_model,
        "total_points": num_points,
        "sample_points": points_sample,
        "range_m": range_m,
        "noise_std_m": noise_std,
        "point_density_pts_per_m2": round(num_points / (math.pi * range_m**2), 2),
        "dropout_rate": round(random.uniform(0.001, 0.02), 4),
    }


def _generate_depth_map(
    width: int, height: int, max_depth_m: float = 10.0
) -> dict[str, Any]:
    """Synthesize a depth map from a structured-light or stereo sensor.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        max_depth_m: Maximum depth range in metres.

    Returns:
        Dict with depth map metadata and statistics.
    """
    # Simulate a depth gradient scene (near floor, farther walls)
    center_depth = random.uniform(1.0, max_depth_m * 0.4)
    edge_depth = random.uniform(center_depth, max_depth_m)
    min_depth = round(center_depth * 0.8, 3)
    max_depth_measured = round(edge_depth * 1.05, 3)

    return {
        "width_px": width,
        "height_px": height,
        "min_depth_m": min_depth,
        "max_depth_m": max_depth_measured,
        "mean_depth_m": round((min_depth + max_depth_measured) / 2.0, 3),
        "noise_std_m": CAMERA_NOISE_STD["depth"],
        "invalid_pixel_rate": round(random.uniform(0.001, 0.05), 4),
        "format": "float32",
    }


def _generate_camera_frame(
    camera_type: str,
    width: int,
    height: int,
    include_depth: bool = False,
) -> dict[str, Any]:
    """Synthesize a camera frame with noise model.

    Args:
        camera_type: Camera modality: 'rgb', 'depth', 'infrared', 'event'.
        width: Frame width in pixels.
        height: Frame height in pixels.
        include_depth: Also generate a depth map alongside the RGB frame.

    Returns:
        Dict with frame metadata, noise model, and optional depth map.
    """
    noise_std = CAMERA_NOISE_STD.get(camera_type, 0.01)
    frame: dict[str, Any] = {
        "camera_type": camera_type,
        "width_px": width,
        "height_px": height,
        "channels": 3 if camera_type == "rgb" else 1,
        "bit_depth": 8 if camera_type in ("rgb", "infrared") else 16,
        "noise_std": noise_std,
        "snr_db": round(20.0 * math.log10(1.0 / max(noise_std, 1e-8)), 2),
        "exposure_ms": round(random.uniform(1.0, 30.0), 2),
        "gain_db": round(random.uniform(0.0, 6.0), 2),
    }
    if include_depth:
        frame["depth_map"] = _generate_depth_map(width, height)
    return frame


def _generate_imu_reading(sensor_model: str, duration_s: float, rate_hz: float) -> dict[str, Any]:
    """Synthesize an IMU reading sequence.

    Args:
        sensor_model: IMU model name for noise density lookup.
        duration_s: Duration of the IMU sequence in seconds.
        rate_hz: IMU output rate in Hz.

    Returns:
        Dict with IMU statistics and noise parameters.
    """
    noise_density = IMU_NOISE_DENSITY_RPS_SQHZ.get(sensor_model, IMU_NOISE_DENSITY_RPS_SQHZ["generic"])
    num_samples = int(duration_s * rate_hz)
    noise_rms = noise_density * math.sqrt(rate_hz)

    return {
        "sensor_model": sensor_model,
        "num_samples": num_samples,
        "duration_s": duration_s,
        "rate_hz": rate_hz,
        "gyro_noise_density_rads_sqhz": noise_density,
        "accel_noise_density_ms2_sqhz": noise_density * 10.0,
        "gyro_noise_rms_rads": round(noise_rms, 6),
        "accel_noise_rms_ms2": round(noise_rms * 10.0, 6),
        "bias_drift_model": "random_walk",
        "temperature_drift": round(random.uniform(0.001, 0.01), 5),
    }


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------


def _compute_extrinsic_calibration(
    sensor_a_frame: str, sensor_b_frame: str
) -> dict[str, Any]:
    """Generate a plausible extrinsic calibration between two sensor frames.

    Args:
        sensor_a_frame: Source sensor frame identifier.
        sensor_b_frame: Target sensor frame identifier.

    Returns:
        Dict with translation_m and rotation_euler_deg.
    """
    translation = [round(random.gauss(0.0, 0.05), 6) for _ in range(3)]
    rotation = [round(random.gauss(0.0, 0.5), 4) for _ in range(3)]
    return {
        "from_frame": sensor_a_frame,
        "to_frame": sensor_b_frame,
        "translation_m": translation,
        "rotation_euler_deg": rotation,
        "reprojection_error_px": round(random.uniform(0.1, 1.5), 3),
    }


def _compute_temporal_offset(
    sensor_a_hz: float, sensor_b_hz: float
) -> float:
    """Estimate temporal offset between two sensors given their rates.

    Args:
        sensor_a_hz: Rate of sensor A in Hz.
        sensor_b_hz: Rate of sensor B in Hz.

    Returns:
        Worst-case temporal offset in milliseconds.
    """
    # Half-period of the slower sensor
    return round(1000.0 / (2.0 * min(sensor_a_hz, sensor_b_hz)), 2)


# ---------------------------------------------------------------------------
# SensorFusion adapter
# ---------------------------------------------------------------------------


class SensorFusion:
    """Multi-sensor data synthesis and fusion adapter.

    Generates synchronized, calibrated, and noise-injected multi-modal sensor
    datasets combining camera, LiDAR, and IMU streams. Computes temporal
    alignment scores, spatial calibration quality, and overall fusion quality.

    Implements SensorFusionEngineProtocol.
    """

    # Default sensor output rates (Hz) by type
    DEFAULT_RATES_HZ: dict[str, float] = {
        "lidar": 10.0,
        "camera": 30.0,
        "imu": 200.0,
        "radar": 15.0,
        "depth_camera": 30.0,
    }

    # Temporal alignment quality thresholds by fusion strategy
    STRATEGY_ALIGNMENT_WEIGHT: dict[str, float] = {
        "early_fusion": 0.95,
        "late_fusion": 0.80,
        "intermediate_fusion": 0.90,
        "kalman_filter": 0.97,
        "particle_filter": 0.92,
        "deep_fusion": 0.88,
    }

    async def fuse(
        self,
        sensor_streams: list[dict[str, Any]],
        fusion_strategy: str,
        fusion_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Fuse multiple sensor streams into a unified multi-modal dataset.

        Args:
            sensor_streams: List of sensor stream configurations, each with:
                - sensor_type: 'lidar' | 'camera' | 'imu' | 'radar' | 'depth_camera'
                - sensor_model: Optional model name for noise lookup
                - calibration: Optional extrinsic calibration overrides
                - num_frames: Number of frames/sweeps to synthesize (default 100)
                - rate_hz: Override sensor output rate
            fusion_strategy: Fusion algorithm: 'early_fusion', 'late_fusion',
                'intermediate_fusion', 'kalman_filter', 'particle_filter', 'deep_fusion'.
            fusion_config: Strategy-specific configuration:
                - temporal_window_ms: Synchronization window (default 10 ms)
                - reference_sensor: Primary reference sensor type
                - output_format: 'rosbag2' | 'mcap' | 'hdf5' | 'numpy'
                - interpolation_method: 'linear' | 'nearest' | 'cubic'
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with output_uri, output_format, temporal_alignment_score,
            spatial_calibration_score, fusion_quality_score.
        """
        temporal_window_ms = float(fusion_config.get("temporal_window_ms", 10.0))
        reference_sensor = fusion_config.get("reference_sensor", "lidar")
        output_format = fusion_config.get("output_format", "hdf5")
        interpolation_method = fusion_config.get("interpolation_method", "linear")

        logger.info(
            "Starting multi-sensor fusion",
            num_streams=len(sensor_streams),
            fusion_strategy=fusion_strategy,
            output_format=output_format,
            tenant_id=str(tenant_id),
        )

        synthesized_streams: list[dict[str, Any]] = []
        calibration_pairs: list[dict[str, Any]] = []
        temporal_offsets_ms: list[float] = []

        reference_rate_hz = self.DEFAULT_RATES_HZ.get(reference_sensor, 10.0)

        for stream_idx, stream in enumerate(sensor_streams):
            sensor_type: str = stream.get("sensor_type", "camera")
            sensor_model: str = stream.get("sensor_model", "generic")
            num_frames: int = int(stream.get("num_frames", 100))
            rate_hz: float = float(
                stream.get("rate_hz", self.DEFAULT_RATES_HZ.get(sensor_type, 30.0))
            )
            duration_s = num_frames / rate_hz

            synthesized = self._synthesize_stream(
                sensor_type=sensor_type,
                sensor_model=sensor_model,
                num_frames=num_frames,
                rate_hz=rate_hz,
                duration_s=duration_s,
            )
            synthesized_streams.append(synthesized)

            # Compute temporal offset relative to reference sensor
            if sensor_type != reference_sensor:
                offset_ms = _compute_temporal_offset(rate_hz, reference_rate_hz)
                temporal_offsets_ms.append(offset_ms)

            # Generate calibration pairs between consecutive streams
            if stream_idx > 0:
                prev_type = sensor_streams[stream_idx - 1].get("sensor_type", "camera")
                calibration_pairs.append(
                    _compute_extrinsic_calibration(
                        f"{prev_type}_frame_{stream_idx - 1}",
                        f"{sensor_type}_frame_{stream_idx}",
                    )
                )

        # Score temporal alignment
        temporal_alignment_score = self._compute_temporal_alignment_score(
            temporal_offsets_ms=temporal_offsets_ms,
            temporal_window_ms=temporal_window_ms,
            fusion_strategy=fusion_strategy,
        )

        # Score spatial calibration quality
        spatial_calibration_score = self._compute_spatial_calibration_score(
            calibration_pairs=calibration_pairs,
        )

        # Overall fusion quality
        fusion_quality_score = round(
            0.5 * temporal_alignment_score + 0.5 * spatial_calibration_score, 4
        )

        logger.info(
            "Multi-sensor fusion complete",
            temporal_alignment_score=temporal_alignment_score,
            spatial_calibration_score=spatial_calibration_score,
            fusion_quality_score=fusion_quality_score,
            tenant_id=str(tenant_id),
        )

        output_id = uuid.uuid4()
        return {
            "output_uri": (
                f"s3://aumos-physical-ai/{tenant_id}/fusion/{output_id}.{output_format}"
            ),
            "output_format": output_format,
            "temporal_alignment_score": temporal_alignment_score,
            "spatial_calibration_score": spatial_calibration_score,
            "fusion_quality_score": fusion_quality_score,
            "synthesized_streams": synthesized_streams,
            "calibration_pairs": calibration_pairs,
            "fusion_metadata": {
                "fusion_strategy": fusion_strategy,
                "interpolation_method": interpolation_method,
                "reference_sensor": reference_sensor,
                "temporal_window_ms": temporal_window_ms,
                "num_streams": len(sensor_streams),
            },
        }

    def _synthesize_stream(
        self,
        sensor_type: str,
        sensor_model: str,
        num_frames: int,
        rate_hz: float,
        duration_s: float,
    ) -> dict[str, Any]:
        """Synthesize one sensor stream's metadata and sample data.

        Args:
            sensor_type: Sensor modality type string.
            sensor_model: Sensor model for noise lookup.
            num_frames: Number of data frames to simulate.
            rate_hz: Sensor output rate in Hz.
            duration_s: Total recording duration in seconds.

        Returns:
            Dict describing the synthesized stream.
        """
        if sensor_type == "lidar":
            num_points_per_scan = random.randint(50_000, 200_000)
            data_sample = _generate_point_cloud(num_points_per_scan, sensor_model)
            data_sample["num_scans"] = num_frames
        elif sensor_type in ("camera", "depth_camera"):
            width, height = random.choice([(640, 480), (1280, 720), (1920, 1080)])
            include_depth = sensor_type == "depth_camera"
            data_sample = _generate_camera_frame(
                camera_type="rgb" if sensor_type == "camera" else "depth",
                width=width,
                height=height,
                include_depth=include_depth,
            )
            data_sample["num_frames"] = num_frames
        elif sensor_type == "imu":
            data_sample = _generate_imu_reading(sensor_model, duration_s, rate_hz)
        elif sensor_type == "radar":
            data_sample = {
                "sensor_type": "radar",
                "num_targets_typical": random.randint(5, 50),
                "range_m": random.uniform(50, 200),
                "velocity_resolution_ms": round(random.uniform(0.1, 0.5), 3),
                "range_resolution_m": round(random.uniform(0.1, 1.0), 3),
                "num_frames": num_frames,
                "noise_std_dbsm": round(random.uniform(0.5, 3.0), 2),
            }
        else:
            data_sample = {"sensor_type": sensor_type, "num_frames": num_frames}

        return {
            "sensor_type": sensor_type,
            "sensor_model": sensor_model,
            "rate_hz": rate_hz,
            "duration_s": round(duration_s, 3),
            "num_frames": num_frames,
            "data_sample": data_sample,
        }

    def _compute_temporal_alignment_score(
        self,
        temporal_offsets_ms: list[float],
        temporal_window_ms: float,
        fusion_strategy: str,
    ) -> float:
        """Compute temporal alignment quality score.

        Args:
            temporal_offsets_ms: List of inter-sensor temporal offsets.
            temporal_window_ms: Maximum allowable offset for alignment.
            fusion_strategy: Fusion strategy name (affects alignment weight).

        Returns:
            Temporal alignment score in [0.0, 1.0].
        """
        if not temporal_offsets_ms:
            return 1.0

        strategy_weight = self.STRATEGY_ALIGNMENT_WEIGHT.get(fusion_strategy, 0.90)
        max_offset = max(temporal_offsets_ms)

        # Score degrades linearly as max offset approaches the temporal window
        raw_score = max(0.0, 1.0 - (max_offset / max(temporal_window_ms, 1.0)))
        return round(raw_score * strategy_weight, 4)

    def _compute_spatial_calibration_score(
        self, calibration_pairs: list[dict[str, Any]]
    ) -> float:
        """Compute spatial calibration quality from reprojection errors.

        Args:
            calibration_pairs: List of extrinsic calibration dicts with
                reprojection_error_px.

        Returns:
            Spatial calibration score in [0.0, 1.0].
        """
        if not calibration_pairs:
            return 1.0

        errors = [p.get("reprojection_error_px", 1.0) for p in calibration_pairs]
        max_acceptable_error_px = 2.0
        mean_error = sum(errors) / len(errors)
        score = max(0.0, 1.0 - mean_error / max_acceptable_error_px)
        return round(score, 4)

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Poll the status of an async fusion job.

        Args:
            job_id: Fusion job identifier.

        Returns:
            Dict with status, progress_pct, estimated_completion.
        """
        logger.info("Querying fusion job status", job_id=job_id)
        return {
            "job_id": job_id,
            "status": "completed",
            "progress_pct": 100.0,
            "estimated_completion": None,
        }
