"""SQLAlchemy ORM models for the physical AI service.

All models extend AumOSModel which provides:
  - id: UUID primary key
  - tenant_id: UUID (RLS enforced via aumos-common)
  - created_at: datetime
  - updated_at: datetime
"""

import enum
import uuid

from sqlalchemy import Boolean, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from aumos_common.database import AumOSModel


class JobStatus(str, enum.Enum):
    """Execution states for all physical AI jobs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SensorType(str, enum.Enum):
    """Supported robotics sensor modalities."""

    LIDAR = "lidar"
    CAMERA = "camera"
    IMU = "imu"
    RADAR = "radar"
    ULTRASONIC = "ultrasonic"
    DEPTH_CAMERA = "depth_camera"


class TransferMethod(str, enum.Enum):
    """Sim-to-real adaptation strategies."""

    DOMAIN_ADAPTATION = "domain_adaptation"
    FINE_TUNING = "fine_tuning"
    META_LEARNING = "meta_learning"
    CYCLIC_GAN = "cyclic_gan"
    RANDOMIZATION_PRETRAIN = "randomization_pretrain"


class FusionStrategy(str, enum.Enum):
    """Multi-sensor fusion strategies."""

    EARLY_FUSION = "early_fusion"
    LATE_FUSION = "late_fusion"
    INTERMEDIATE_FUSION = "intermediate_fusion"
    KALMAN_FILTER = "kalman_filter"
    PARTICLE_FILTER = "particle_filter"
    DEEP_FUSION = "deep_fusion"


# ---------------------------------------------------------------------------
# Digital Twin Pipeline
# ---------------------------------------------------------------------------


class TwinPipeline(AumOSModel):
    """Digital twin data pipeline configuration and state.

    Tracks the full lifecycle of a digital twin pipeline — from scene
    configuration through simulation execution to real-world validation.

    Table: pai_twin_pipelines
    """

    __tablename__ = "pai_twin_pipelines"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JobStatus.PENDING, index=True
    )

    # Scene and world configuration stored as JSONB
    scene_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Output artifacts
    output_uri: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Quality and fidelity metrics
    fidelity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    sync_lag_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Simulation metadata
    simulation_steps: Mapped[int | None] = mapped_column(Integer, nullable=True)
    real_time_factor: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Robotics Synthesis Jobs
# ---------------------------------------------------------------------------


class RoboticsJob(AumOSModel):
    """Robotics sensor data synthesis job.

    Represents a single sensor synthesis run — generating synthetic
    LiDAR point clouds, camera frames, IMU readings, or multi-modal
    sensor streams from simulation environments.

    Table: pai_robotics_jobs
    """

    __tablename__ = "pai_robotics_jobs"

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JobStatus.PENDING, index=True
    )

    # Which sensor types are being synthesized
    sensor_types: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Synthesis configuration (scene, object poses, noise models, etc.)
    synthesis_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Output metadata
    output_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_points: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Quality metrics
    realism_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Sim-to-Real Transfer
# ---------------------------------------------------------------------------


class SimToRealTransfer(AumOSModel):
    """Sim-to-real transfer learning record.

    Tracks domain adaptation runs that bridge the simulation-to-reality
    gap for robotics perception models.

    Table: pai_sim2real_transfers
    """

    __tablename__ = "pai_sim2real_transfers"

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JobStatus.PENDING, index=True
    )

    transfer_method: Mapped[str] = mapped_column(String(64), nullable=False)

    # Source model from model registry
    source_model_id: Mapped[str] = mapped_column(String(255), nullable=False)

    # Transfer configuration
    transfer_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Adapted model output
    adapted_model_uri: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Performance metrics
    sim_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    real_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    domain_gap_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    adaptation_epochs: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Domain Randomization
# ---------------------------------------------------------------------------


class RandomizationConfig(AumOSModel):
    """Domain randomization configuration for sim-to-real robustness.

    Stores the randomization parameters applied to simulation environments
    to generate diverse training data and reduce domain gap.

    Table: pai_randomization_configs
    """

    __tablename__ = "pai_randomization_configs"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JobStatus.PENDING, index=True
    )

    # Randomization parameters (lighting, textures, object poses, noise, etc.)
    randomization_params: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Execution results
    variations_generated: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_uri: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Diversity metrics
    diversity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    coverage_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Sensor Fusion Jobs
# ---------------------------------------------------------------------------


class SensorFusionJob(AumOSModel):
    """Multi-sensor fusion data generation job.

    Manages the fusion of multiple sensor streams (LiDAR, camera, IMU, radar)
    into coherent multi-modal datasets for perception model training.

    Table: pai_sensor_fusion_jobs
    """

    __tablename__ = "pai_sensor_fusion_jobs"

    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default=JobStatus.PENDING, index=True
    )

    fusion_strategy: Mapped[str] = mapped_column(String(64), nullable=False)

    # Input sensor streams configuration
    sensor_streams: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)

    # Fusion configuration
    fusion_config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    # Output
    output_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_format: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Quality metrics
    temporal_alignment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    spatial_calibration_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    fusion_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
