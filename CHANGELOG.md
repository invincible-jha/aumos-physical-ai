# Changelog

All notable changes to `aumos-physical-ai` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-02-26

### Added
- Initial implementation of the AumOS Physical AI service
- `DigitalTwinPipelineService` — BlenderProc-backed digital twin pipeline orchestration
- `RoboticsSynthService` — multi-modal sensor data synthesis (LiDAR, camera, IMU, radar)
- `SimToRealService` — sim-to-real domain adaptation (DANN, fine-tuning, meta-learning, CycleGAN)
- `DomainRandomizationService` — scene randomization for training data diversity
- `SensorFusionService` — multi-sensor data fusion with temporal alignment and spatial calibration
- REST API: 7 endpoints under `/api/v1/physical/`
- DB models: `pai_twin_pipelines`, `pai_robotics_jobs`, `pai_sim2real_transfers`,
  `pai_randomization_configs`, `pai_sensor_fusion_jobs`
- Hexagonal architecture: `api/`, `core/`, `adapters/`
- Protocol interfaces for all external dependencies (testability, swap-ability)
- Kafka event publishing for all lifecycle transitions
- Docker and docker-compose configuration for local development
- Full CI pipeline via GitHub Actions
