"""Pydantic request and response models for the physical AI API.

All API schemas are strictly typed. Enums match the ORM model enums
to avoid divergence between API contract and persistence layer.
"""

import uuid
from typing import Any

from pydantic import BaseModel, Field, field_validator

from aumos_physical_ai.core.models import FusionStrategy, JobStatus, SensorType, TransferMethod


# ---------------------------------------------------------------------------
# Shared response base
# ---------------------------------------------------------------------------


class JobStatusResponse(BaseModel):
    """Minimal job status response for async operations."""

    id: uuid.UUID
    status: JobStatus

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# Digital Twin Pipeline
# ---------------------------------------------------------------------------


class CreateTwinPipelineRequest(BaseModel):
    """Request to create a new digital twin data pipeline."""

    name: str = Field(..., description="Human-readable pipeline name", min_length=1, max_length=255)
    scene_config: dict[str, Any] = Field(
        ...,
        description="Scene configuration including world_model, sensors, physics, assets",
        examples=[
            {
                "world_model": "warehouse_v2",
                "physics_engine": "isaac_sim",
                "sensors": [{"type": "lidar", "position": [0, 0, 1.5]}],
                "simulation_steps": 5000,
                "real_time_factor": 2.0,
            }
        ],
    )

    @field_validator("scene_config")
    @classmethod
    def world_model_required(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Ensure world_model is present in scene_config."""
        if "world_model" not in value:
            raise ValueError("scene_config must include 'world_model'")
        return value


class TwinPipelineResponse(BaseModel):
    """Response schema for a digital twin pipeline record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    status: JobStatus
    scene_config: dict[str, Any]
    output_uri: str | None = None
    fidelity_score: float | None = None
    sync_lag_ms: float | None = None
    simulation_steps: int | None = None
    real_time_factor: float | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class CreateTwinPipelineResponse(BaseModel):
    """Response after creating a digital twin pipeline."""

    pipeline: TwinPipelineResponse


class ListTwinPipelinesResponse(BaseModel):
    """Response for listing digital twin pipelines."""

    pipelines: list[TwinPipelineResponse]
    total: int


# ---------------------------------------------------------------------------
# Robotics Sensor Synthesis
# ---------------------------------------------------------------------------


class SynthesizeRequest(BaseModel):
    """Request to synthesize robotics sensor data.

    At least one sensor_type must be specified. The synthesis_config
    provides per-modality configuration including scene setup, noise
    models, and output resolution.
    """

    sensor_types: list[SensorType] = Field(
        ...,
        description="Sensor modalities to synthesize",
        min_length=1,
        examples=[["lidar", "camera", "imu"]],
    )
    synthesis_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-modality synthesis configuration",
        examples=[
            {
                "scene": "urban_intersection",
                "num_frames": 1000,
                "lidar": {"channels": 64, "range_m": 100, "dropout_rate": 0.01},
                "camera": {"resolution": [1920, 1080], "fov_deg": 90, "noise_snr_db": 35},
                "imu": {"sample_rate_hz": 200, "gyro_noise_density": 0.005},
            }
        ],
    )


class RoboticsJobResponse(BaseModel):
    """Response schema for a robotics synthesis job."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    status: JobStatus
    sensor_types: list[str]
    synthesis_config: dict[str, Any]
    output_uri: str | None = None
    frame_count: int | None = None
    total_points: int | None = None
    realism_score: float | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class SynthesizeResponse(BaseModel):
    """Response after submitting a synthesis request."""

    job: RoboticsJobResponse


# ---------------------------------------------------------------------------
# Sim-to-Real Transfer
# ---------------------------------------------------------------------------


class SimToRealTransferRequest(BaseModel):
    """Request to execute sim-to-real domain adaptation.

    The transfer_config must include sim_dataset_uri and real_dataset_uri
    to define the source and target domains.
    """

    source_model_id: str = Field(
        ...,
        description="Model registry ID of the simulation-trained source model",
    )
    transfer_method: TransferMethod = Field(
        ...,
        description="Domain adaptation algorithm to apply",
    )
    transfer_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific configuration",
        examples=[
            {
                "sim_dataset_uri": "s3://aumos-pai/sim/dataset_001",
                "real_dataset_uri": "s3://aumos-pai/real/dataset_001",
                "learning_rate": 0.0001,
                "adaptation_epochs": 50,
                "batch_size": 32,
            }
        ],
    )


class SimToRealTransferResponse(BaseModel):
    """Response schema for a sim-to-real transfer record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    status: JobStatus
    transfer_method: str
    source_model_id: str
    transfer_config: dict[str, Any]
    adapted_model_uri: str | None = None
    sim_accuracy: float | None = None
    real_accuracy: float | None = None
    domain_gap_score: float | None = None
    adaptation_epochs: int | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class SimToRealResponse(BaseModel):
    """Response after submitting a sim-to-real transfer request."""

    transfer: SimToRealTransferResponse


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------


class DomainRandomizationRequest(BaseModel):
    """Request to execute domain randomization.

    The randomization_params control which aspects of the simulation
    are varied and the range of variation applied.
    """

    name: str = Field(..., description="Human-readable configuration name", min_length=1, max_length=255)
    randomization_params: dict[str, Any] = Field(
        ...,
        description="Randomization parameters per aspect (lighting, textures, poses, noise, physics)",
        examples=[
            {
                "num_variations": 500,
                "lighting": {"intensity_range": [0.5, 2.0], "color_temp_range": [3000, 6500]},
                "textures": {"randomize_floor": True, "randomize_walls": True},
                "object_poses": {"position_noise_m": 0.05, "rotation_noise_deg": 15},
                "sensor_noise": {"lidar_dropout_rate": 0.02, "camera_blur_sigma": 1.5},
            }
        ],
    )


class RandomizationConfigResponse(BaseModel):
    """Response schema for a domain randomization configuration record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    status: JobStatus
    randomization_params: dict[str, Any]
    variations_generated: int | None = None
    output_uri: str | None = None
    diversity_score: float | None = None
    coverage_score: float | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class DomainRandomizationResponse(BaseModel):
    """Response after submitting a domain randomization request."""

    config: RandomizationConfigResponse


# ---------------------------------------------------------------------------
# Sensor Fusion
# ---------------------------------------------------------------------------


class SensorFusionRequest(BaseModel):
    """Request to generate a multi-sensor fused dataset.

    Requires at least 2 sensor streams. Each stream must specify its type,
    data URI, and calibration parameters.
    """

    sensor_streams: list[dict[str, Any]] = Field(
        ...,
        description="List of sensor stream configurations (at least 2 required)",
        min_length=2,
        examples=[
            [
                {
                    "sensor_type": "lidar",
                    "data_uri": "s3://aumos-pai/lidar/run_001",
                    "calibration": {"extrinsic": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1.5]]},
                },
                {
                    "sensor_type": "camera",
                    "data_uri": "s3://aumos-pai/camera/run_001",
                    "calibration": {"intrinsic": {"fx": 960, "fy": 960, "cx": 960, "cy": 540}},
                },
            ]
        ],
    )
    fusion_strategy: FusionStrategy = Field(
        ...,
        description="Fusion algorithm to apply",
    )
    fusion_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy-specific fusion configuration",
        examples=[
            {
                "temporal_window_ms": 50,
                "reference_sensor": "lidar",
                "output_format": "rosbag2",
                "interpolation_method": "linear",
            }
        ],
    )


class SensorFusionJobResponse(BaseModel):
    """Response schema for a sensor fusion job record."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    status: JobStatus
    fusion_strategy: str
    sensor_streams: list[dict[str, Any]]
    fusion_config: dict[str, Any]
    output_uri: str | None = None
    output_format: str | None = None
    temporal_alignment_score: float | None = None
    spatial_calibration_score: float | None = None
    fusion_quality_score: float | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}


class SensorFusionResponse(BaseModel):
    """Response after submitting a sensor fusion request."""

    job: SensorFusionJobResponse
