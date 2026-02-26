"""Service-specific settings extending AumOS base config."""

from pydantic_settings import SettingsConfigDict

from aumos_common.config import AumOSSettings


class Settings(AumOSSettings):
    """Configuration for the aumos-physical-ai service.

    All standard AumOS settings (database, kafka, redis, jwt) are inherited.
    Physical AI-specific settings use the AUMOS_PHYSICAL_AI_ prefix.
    """

    service_name: str = "aumos-physical-ai"

    # BlenderProc / simulation backend
    blenderproc_url: str = "http://localhost:9090"
    blenderproc_timeout_seconds: int = 300

    # Sensor simulation
    sensor_simulator_url: str = "http://localhost:9091"
    lidar_point_cloud_max_points: int = 100_000
    camera_resolution_width: int = 1920
    camera_resolution_height: int = 1080
    imu_sample_rate_hz: int = 200

    # Sim-to-real transfer
    sim2real_model_registry_url: str = "http://localhost:8080"
    domain_gap_threshold: float = 0.15
    adaptation_epochs_default: int = 50

    # Domain randomization
    randomization_seed: int = 42
    max_randomization_variations: int = 1000

    # Sensor fusion
    fusion_output_bucket: str = "aumos-physical-ai"
    fusion_output_format: str = "rosbag2"

    # Storage
    minio_endpoint: str = "http://localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"

    model_config = SettingsConfigDict(env_prefix="AUMOS_PHYSICAL_AI_")
