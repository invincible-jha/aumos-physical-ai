"""Protocol interfaces for the physical AI service.

These Protocols define the contracts that all adapters must implement.
Services depend on these interfaces, never on concrete adapter classes,
ensuring testability and easy swapping of implementations.
"""

import uuid
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class DigitalTwinBackendProtocol(Protocol):
    """Contract for digital twin simulation backends.

    Implementations connect to simulation environments (Isaac Sim, Gazebo,
    BlenderProc) and execute digital twin data pipeline runs.
    """

    async def create_pipeline(
        self,
        scene_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Create and configure a digital twin pipeline in the simulation backend.

        Args:
            scene_config: Scene configuration including assets, physics, sensors.
            tenant_id: Tenant context for isolation.

        Returns:
            Dict with pipeline_id, status, and backend-specific metadata.
        """
        ...

    async def run_pipeline(
        self,
        pipeline_id: str,
        simulation_steps: int,
        real_time_factor: float,
    ) -> dict[str, Any]:
        """Execute a digital twin pipeline and stream output data.

        Args:
            pipeline_id: Backend pipeline identifier.
            simulation_steps: Number of simulation steps to execute.
            real_time_factor: Ratio of simulation time to wall-clock time.

        Returns:
            Dict with output_uri, fidelity_score, sync_lag_ms, frame_count.
        """
        ...

    async def get_pipeline_status(self, pipeline_id: str) -> dict[str, Any]:
        """Query the current status of a digital twin pipeline.

        Args:
            pipeline_id: Backend pipeline identifier.

        Returns:
            Dict with status, progress, metrics.
        """
        ...


@runtime_checkable
class SensorSimulatorProtocol(Protocol):
    """Contract for sensor simulation adapters.

    Implementations generate synthetic sensor data for specific modalities:
    LiDAR point clouds, RGB/depth camera frames, IMU readings, radar returns.
    """

    async def synthesize(
        self,
        sensor_types: list[str],
        synthesis_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate synthetic sensor data for the requested modalities.

        Args:
            sensor_types: List of sensor modality names ('lidar', 'camera', etc.)
            synthesis_config: Per-modality configuration (resolution, noise, pose, etc.)
            tenant_id: Tenant context for storage namespacing.

        Returns:
            Dict with output_uri, frame_count, total_points, realism_score.
        """
        ...

    async def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Poll the status of an async synthesis job.

        Args:
            job_id: Synthesis job identifier in the simulator backend.

        Returns:
            Dict with status, progress_pct, estimated_completion.
        """
        ...


@runtime_checkable
class SimToRealAdapterProtocol(Protocol):
    """Contract for sim-to-real transfer learning adapters.

    Implementations apply domain adaptation algorithms to bridge the
    reality gap between simulation-trained and real-world models.
    """

    async def transfer(
        self,
        source_model_id: str,
        transfer_method: str,
        transfer_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Execute sim-to-real domain adaptation.

        Args:
            source_model_id: Identifier of the simulation-trained source model.
            transfer_method: Adaptation algorithm ('domain_adaptation', 'fine_tuning', etc.)
            transfer_config: Method-specific hyperparameters and data paths.
            tenant_id: Tenant context for model registry isolation.

        Returns:
            Dict with adapted_model_uri, sim_accuracy, real_accuracy,
            domain_gap_score, adaptation_epochs.
        """
        ...

    async def evaluate_domain_gap(
        self,
        model_id: str,
        sim_dataset_uri: str,
        real_dataset_uri: str,
    ) -> float:
        """Compute the domain gap between simulation and real-world distributions.

        Args:
            model_id: Model identifier to evaluate.
            sim_dataset_uri: URI of the simulation dataset.
            real_dataset_uri: URI of the real-world dataset.

        Returns:
            Domain gap score (0.0 = no gap, 1.0 = maximum gap).
        """
        ...


@runtime_checkable
class DomainRandomizerProtocol(Protocol):
    """Contract for domain randomization engines.

    Implementations vary simulation parameters (lighting, textures,
    object poses, sensor noise, physics) to generate diverse training data.
    """

    async def randomize(
        self,
        randomization_params: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Execute domain randomization and generate varied scene data.

        Args:
            randomization_params: Randomization configuration:
                - lighting: {intensity_range, color_temp_range, count}
                - textures: {object_classes, texture_pool_uri}
                - object_poses: {position_noise_m, rotation_noise_deg}
                - sensor_noise: {lidar_dropout_rate, camera_blur_sigma}
                - physics: {friction_range, mass_multiplier_range}
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with output_uri, variations_generated, diversity_score,
            coverage_score.
        """
        ...


@runtime_checkable
class SensorFusionEngineProtocol(Protocol):
    """Contract for multi-sensor fusion engines.

    Implementations combine multiple heterogeneous sensor streams into
    calibrated, temporally-aligned multi-modal datasets.
    """

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
                - sensor_type: 'lidar' | 'camera' | 'imu' | 'radar'
                - data_uri: URI of the sensor data
                - calibration: Extrinsic/intrinsic calibration parameters
                - timestamp_topic: Synchronization reference topic
            fusion_strategy: Fusion algorithm to apply.
            fusion_config: Strategy-specific parameters (temporal window, etc.)
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with output_uri, output_format, temporal_alignment_score,
            spatial_calibration_score, fusion_quality_score.
        """
        ...
