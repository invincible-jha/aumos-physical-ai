"""Business logic services for the physical AI service.

Eight services covering the full physical AI stack:
  - DigitalTwinPipelineService: digital twin data pipeline orchestration
  - RoboticsSynthService: multi-modal robotics sensor data synthesis
  - SimToRealService: sim-to-real transfer learning and domain adaptation
  - DomainRandomizationService: scene randomization for training data diversity
  - SensorFusionService: multi-sensor data fusion and calibration
  - MotionPlanningService: collision-free trajectory generation (A*, RRT, B-spline)
  - GraspingSimulationService: robotic grasp pose generation and quality scoring
  - PhysicsSimulationService: rigid-body forward dynamics simulation

Services are framework-agnostic — they receive dependencies via constructor
injection and delegate all I/O to adapters.
"""

import uuid
from typing import Any

from aumos_common.errors import NotFoundError, ValidationError
from aumos_common.events import EventPublisher, Topics
from aumos_common.observability import get_logger

from aumos_physical_ai.core.interfaces import (
    DigitalTwinBackendProtocol,
    DomainRandomizerProtocol,
    GraspingSimulatorProtocol,
    MotionPlannerProtocol,
    PhysicsEngineAdapterProtocol,
    PhysicalSafetyValidatorProtocol,
    SensorFusionEngineProtocol,
    SensorSimulatorProtocol,
    SimToRealAdapterProtocol,
)
from aumos_physical_ai.core.models import (
    JobStatus,
    RandomizationConfig,
    RoboticsJob,
    SensorFusionJob,
    SimToRealTransfer,
    TwinPipeline,
)

logger = get_logger(__name__)


class DigitalTwinPipelineService:
    """Orchestrates digital twin data pipelines.

    Connects to simulation backends (Isaac Sim, Gazebo, BlenderProc) to
    create and run digital twin pipelines that generate physics-accurate
    synthetic environments synchronized with real-world counterparts.
    """

    def __init__(
        self,
        twin_backend: DigitalTwinBackendProtocol,
        pipeline_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize the digital twin pipeline service.

        Args:
            twin_backend: Simulation backend adapter (Isaac Sim, Gazebo, etc.)
            pipeline_repository: Repository for TwinPipeline persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._backend = twin_backend
        self._pipeline_repo = pipeline_repository
        self._publisher = event_publisher

    async def create_pipeline(
        self,
        tenant_id: uuid.UUID,
        name: str,
        scene_config: dict[str, Any],
    ) -> TwinPipeline:
        """Create a new digital twin pipeline.

        Registers the pipeline in the DB, provisions it in the simulation
        backend, and publishes a lifecycle event.

        Args:
            tenant_id: Tenant creating the pipeline.
            name: Human-readable pipeline name.
            scene_config: Scene configuration (assets, physics, sensors, world).

        Returns:
            TwinPipeline record in PENDING state.

        Raises:
            ValidationError: If scene_config is missing required fields.
        """
        if not scene_config.get("world_model"):
            raise ValidationError("scene_config must include 'world_model'")

        pipeline = await self._pipeline_repo.create(
            TwinPipeline(
                tenant_id=tenant_id,
                name=name,
                status=JobStatus.PENDING,
                scene_config=scene_config,
            )
        )

        await self._publisher.publish(
            Topics.PHYSICAL_AI_TWIN_CREATED,
            {"tenant_id": str(tenant_id), "pipeline_id": str(pipeline.id), "name": name},
        )

        logger.info(
            "Digital twin pipeline created",
            pipeline_id=str(pipeline.id),
            name=name,
            tenant_id=str(tenant_id),
        )

        try:
            pipeline = await self._pipeline_repo.update_status(pipeline.id, JobStatus.RUNNING)

            backend_result = await self._backend.create_pipeline(
                scene_config=scene_config,
                tenant_id=tenant_id,
            )

            simulation_steps = scene_config.get("simulation_steps", 1000)
            real_time_factor = scene_config.get("real_time_factor", 1.0)

            run_result = await self._backend.run_pipeline(
                pipeline_id=backend_result["pipeline_id"],
                simulation_steps=simulation_steps,
                real_time_factor=real_time_factor,
            )

            pipeline = await self._pipeline_repo.update(
                pipeline.id,
                status=JobStatus.COMPLETED,
                output_uri=run_result.get("output_uri"),
                fidelity_score=run_result.get("fidelity_score"),
                sync_lag_ms=run_result.get("sync_lag_ms"),
                simulation_steps=simulation_steps,
                real_time_factor=real_time_factor,
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_TWIN_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "pipeline_id": str(pipeline.id),
                    "output_uri": run_result.get("output_uri"),
                    "fidelity_score": run_result.get("fidelity_score"),
                },
            )

            logger.info(
                "Digital twin pipeline completed",
                pipeline_id=str(pipeline.id),
                fidelity_score=run_result.get("fidelity_score"),
                output_uri=run_result.get("output_uri"),
            )
            return pipeline

        except Exception as exc:
            logger.error("Digital twin pipeline failed", pipeline_id=str(pipeline.id), error=str(exc))
            pipeline = await self._pipeline_repo.update(
                pipeline.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise

    async def list_pipelines(
        self,
        tenant_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> list[TwinPipeline]:
        """List digital twin pipelines for a tenant.

        Args:
            tenant_id: Tenant context.
            limit: Maximum pipelines to return.
            offset: Pagination offset.

        Returns:
            List of TwinPipeline records ordered by created_at descending.
        """
        return await self._pipeline_repo.list_by_tenant(
            tenant_id=tenant_id, limit=limit, offset=offset
        )


class RoboticsSynthService:
    """Synthesizes multi-modal robotics sensor data.

    Generates synthetic LiDAR point clouds, RGB/depth camera frames,
    IMU readings, radar returns, and other sensor modalities from
    simulation environments for robotics perception model training.
    """

    def __init__(
        self,
        sensor_simulator: SensorSimulatorProtocol,
        job_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize the robotics synthesis service.

        Args:
            sensor_simulator: Sensor simulation adapter.
            job_repository: Repository for RoboticsJob persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._simulator = sensor_simulator
        self._job_repo = job_repository
        self._publisher = event_publisher

    async def synthesize(
        self,
        tenant_id: uuid.UUID,
        sensor_types: list[str],
        synthesis_config: dict[str, Any],
    ) -> RoboticsJob:
        """Launch a robotics sensor data synthesis job.

        Args:
            tenant_id: Tenant requesting the synthesis.
            sensor_types: List of sensor modalities to synthesize.
            synthesis_config: Per-modality configuration parameters.

        Returns:
            RoboticsJob record (initially RUNNING, async completion via events).

        Raises:
            ValidationError: If sensor_types is empty or contains unsupported types.
        """
        supported = {"lidar", "camera", "imu", "radar", "ultrasonic", "depth_camera"}
        invalid = set(sensor_types) - supported
        if invalid:
            raise ValidationError(
                f"Unsupported sensor types: {invalid}. Supported: {supported}"
            )

        if not sensor_types:
            raise ValidationError("At least one sensor_type must be specified")

        job = await self._job_repo.create(
            RoboticsJob(
                tenant_id=tenant_id,
                status=JobStatus.PENDING,
                sensor_types=sensor_types,
                synthesis_config=synthesis_config,
            )
        )

        await self._publisher.publish(
            Topics.PHYSICAL_AI_SYNTH_STARTED,
            {
                "tenant_id": str(tenant_id),
                "job_id": str(job.id),
                "sensor_types": sensor_types,
            },
        )

        logger.info(
            "Robotics synthesis started",
            job_id=str(job.id),
            sensor_types=sensor_types,
            tenant_id=str(tenant_id),
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.RUNNING)

            result = await self._simulator.synthesize(
                sensor_types=sensor_types,
                synthesis_config=synthesis_config,
                tenant_id=tenant_id,
            )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=result.get("output_uri"),
                frame_count=result.get("frame_count"),
                total_points=result.get("total_points"),
                realism_score=result.get("realism_score"),
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_SYNTH_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "job_id": str(job.id),
                    "output_uri": result.get("output_uri"),
                    "frame_count": result.get("frame_count"),
                    "realism_score": result.get("realism_score"),
                },
            )

            logger.info(
                "Robotics synthesis completed",
                job_id=str(job.id),
                frame_count=result.get("frame_count"),
                realism_score=result.get("realism_score"),
            )
            return job

        except Exception as exc:
            logger.error("Robotics synthesis failed", job_id=str(job.id), error=str(exc))
            job = await self._job_repo.update(
                job.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise

    async def get_job(self, job_id: uuid.UUID, tenant_id: uuid.UUID) -> RoboticsJob:
        """Retrieve a robotics synthesis job by ID.

        Args:
            job_id: Job identifier.
            tenant_id: Tenant context for authorization.

        Returns:
            RoboticsJob record.

        Raises:
            NotFoundError: If the job does not exist for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job or job.tenant_id != tenant_id:
            raise NotFoundError(f"Robotics job {job_id} not found")
        return job


class SimToRealService:
    """Applies sim-to-real transfer learning to bridge the domain gap.

    Manages domain adaptation workflows that adapt simulation-trained
    perception models for real-world deployment. Supports multiple
    adaptation strategies: domain adaptation, fine-tuning, meta-learning,
    CycleGAN-based image translation, and randomization pretraining.
    """

    def __init__(
        self,
        sim2real_adapter: SimToRealAdapterProtocol,
        transfer_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize the sim-to-real service.

        Args:
            sim2real_adapter: Domain adaptation algorithm adapter.
            transfer_repository: Repository for SimToRealTransfer persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._adapter = sim2real_adapter
        self._transfer_repo = transfer_repository
        self._publisher = event_publisher

    async def transfer(
        self,
        tenant_id: uuid.UUID,
        source_model_id: str,
        transfer_method: str,
        transfer_config: dict[str, Any],
    ) -> SimToRealTransfer:
        """Execute a sim-to-real domain adaptation transfer.

        Args:
            tenant_id: Tenant requesting the transfer.
            source_model_id: Model registry ID of the simulation-trained model.
            transfer_method: Adaptation algorithm to apply.
            transfer_config: Method-specific hyperparameters:
                - sim_dataset_uri: Simulation training data URI.
                - real_dataset_uri: Real-world target domain data URI.
                - learning_rate: Adaptation learning rate.
                - adaptation_epochs: Number of adaptation epochs.
                - discriminator_layers: For adversarial methods.

        Returns:
            SimToRealTransfer record with adapted model URI and metrics.

        Raises:
            ValidationError: If transfer_method is unsupported.
        """
        supported_methods = {
            "domain_adaptation", "fine_tuning", "meta_learning",
            "cyclic_gan", "randomization_pretrain",
        }
        if transfer_method not in supported_methods:
            raise ValidationError(
                f"Unsupported transfer_method '{transfer_method}'. "
                f"Supported: {supported_methods}"
            )

        transfer = await self._transfer_repo.create(
            SimToRealTransfer(
                tenant_id=tenant_id,
                status=JobStatus.PENDING,
                transfer_method=transfer_method,
                source_model_id=source_model_id,
                transfer_config=transfer_config,
            )
        )

        logger.info(
            "Sim-to-real transfer started",
            transfer_id=str(transfer.id),
            transfer_method=transfer_method,
            source_model_id=source_model_id,
            tenant_id=str(tenant_id),
        )

        try:
            transfer = await self._transfer_repo.update_status(transfer.id, JobStatus.RUNNING)

            result = await self._adapter.transfer(
                source_model_id=source_model_id,
                transfer_method=transfer_method,
                transfer_config=transfer_config,
                tenant_id=tenant_id,
            )

            transfer = await self._transfer_repo.update(
                transfer.id,
                status=JobStatus.COMPLETED,
                adapted_model_uri=result.get("adapted_model_uri"),
                sim_accuracy=result.get("sim_accuracy"),
                real_accuracy=result.get("real_accuracy"),
                domain_gap_score=result.get("domain_gap_score"),
                adaptation_epochs=result.get("adaptation_epochs"),
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_TRANSFER_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "transfer_id": str(transfer.id),
                    "adapted_model_uri": result.get("adapted_model_uri"),
                    "domain_gap_score": result.get("domain_gap_score"),
                    "real_accuracy": result.get("real_accuracy"),
                },
            )

            logger.info(
                "Sim-to-real transfer completed",
                transfer_id=str(transfer.id),
                domain_gap_score=result.get("domain_gap_score"),
                real_accuracy=result.get("real_accuracy"),
            )
            return transfer

        except Exception as exc:
            logger.error("Sim-to-real transfer failed", transfer_id=str(transfer.id), error=str(exc))
            transfer = await self._transfer_repo.update(
                transfer.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise


class DomainRandomizationService:
    """Applies domain randomization to simulation environments.

    Generates diverse training data by randomly varying scene parameters:
    lighting conditions, surface textures, object poses, sensor noise
    models, and physics properties. Reduces domain gap by exposing
    models to a wide distribution of visual and physical conditions.
    """

    def __init__(
        self,
        randomizer: DomainRandomizerProtocol,
        config_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize the domain randomization service.

        Args:
            randomizer: Domain randomization engine adapter.
            config_repository: Repository for RandomizationConfig persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._randomizer = randomizer
        self._config_repo = config_repository
        self._publisher = event_publisher

    async def randomize(
        self,
        tenant_id: uuid.UUID,
        name: str,
        randomization_params: dict[str, Any],
    ) -> RandomizationConfig:
        """Execute domain randomization to generate diverse training data.

        Args:
            tenant_id: Tenant requesting the randomization.
            name: Human-readable configuration name.
            randomization_params: Randomization parameters:
                - lighting: {intensity_range, color_temp_range, num_lights}
                - textures: {object_classes, texture_pool_uri, randomize_floor}
                - object_poses: {position_noise_m, rotation_noise_deg, scale_range}
                - sensor_noise: {lidar_dropout_rate, camera_blur_sigma, imu_noise_density}
                - physics: {friction_range, restitution_range, mass_multiplier_range}
                - num_variations: Number of randomized scenes to generate.

        Returns:
            RandomizationConfig record with output URI and diversity metrics.
        """
        config = await self._config_repo.create(
            RandomizationConfig(
                tenant_id=tenant_id,
                name=name,
                status=JobStatus.PENDING,
                randomization_params=randomization_params,
            )
        )

        logger.info(
            "Domain randomization started",
            config_id=str(config.id),
            name=name,
            num_variations=randomization_params.get("num_variations"),
            tenant_id=str(tenant_id),
        )

        try:
            config = await self._config_repo.update_status(config.id, JobStatus.RUNNING)

            result = await self._randomizer.randomize(
                randomization_params=randomization_params,
                tenant_id=tenant_id,
            )

            config = await self._config_repo.update(
                config.id,
                status=JobStatus.COMPLETED,
                variations_generated=result.get("variations_generated"),
                output_uri=result.get("output_uri"),
                diversity_score=result.get("diversity_score"),
                coverage_score=result.get("coverage_score"),
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_RANDOMIZATION_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "config_id": str(config.id),
                    "variations_generated": result.get("variations_generated"),
                    "diversity_score": result.get("diversity_score"),
                    "output_uri": result.get("output_uri"),
                },
            )

            logger.info(
                "Domain randomization completed",
                config_id=str(config.id),
                variations_generated=result.get("variations_generated"),
                diversity_score=result.get("diversity_score"),
            )
            return config

        except Exception as exc:
            logger.error("Domain randomization failed", config_id=str(config.id), error=str(exc))
            config = await self._config_repo.update(
                config.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise


class SensorFusionService:
    """Fuses multiple heterogeneous sensor streams into unified datasets.

    Manages multi-sensor fusion workflows that temporally align and
    spatially calibrate LiDAR, camera, IMU, and radar data into
    coherent multi-modal training datasets for robotics perception.
    """

    def __init__(
        self,
        fusion_engine: SensorFusionEngineProtocol,
        job_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialize the sensor fusion service.

        Args:
            fusion_engine: Multi-sensor fusion engine adapter.
            job_repository: Repository for SensorFusionJob persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._fusion_engine = fusion_engine
        self._job_repo = job_repository
        self._publisher = event_publisher

    async def generate_fusion(
        self,
        tenant_id: uuid.UUID,
        sensor_streams: list[dict[str, Any]],
        fusion_strategy: str,
        fusion_config: dict[str, Any],
    ) -> SensorFusionJob:
        """Generate a multi-sensor fused dataset.

        Args:
            tenant_id: Tenant requesting the fusion.
            sensor_streams: List of sensor stream configurations (type, URI, calibration).
            fusion_strategy: Fusion algorithm to apply.
            fusion_config: Strategy-specific configuration:
                - temporal_window_ms: Time window for temporal alignment.
                - reference_sensor: Primary sensor for calibration reference.
                - output_format: 'rosbag2' | 'mcap' | 'hdf5' | 'numpy'.
                - interpolation_method: 'linear' | 'nearest' | 'cubic'.

        Returns:
            SensorFusionJob record with output URI and quality metrics.

        Raises:
            ValidationError: If fewer than 2 sensor streams are provided.
        """
        if len(sensor_streams) < 2:
            raise ValidationError("At least 2 sensor streams are required for fusion")

        supported_strategies = {
            "early_fusion", "late_fusion", "intermediate_fusion",
            "kalman_filter", "particle_filter", "deep_fusion",
        }
        if fusion_strategy not in supported_strategies:
            raise ValidationError(
                f"Unsupported fusion_strategy '{fusion_strategy}'. "
                f"Supported: {supported_strategies}"
            )

        job = await self._job_repo.create(
            SensorFusionJob(
                tenant_id=tenant_id,
                status=JobStatus.PENDING,
                fusion_strategy=fusion_strategy,
                sensor_streams=sensor_streams,
                fusion_config=fusion_config,
            )
        )

        logger.info(
            "Sensor fusion started",
            job_id=str(job.id),
            fusion_strategy=fusion_strategy,
            num_streams=len(sensor_streams),
            tenant_id=str(tenant_id),
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.RUNNING)

            result = await self._fusion_engine.fuse(
                sensor_streams=sensor_streams,
                fusion_strategy=fusion_strategy,
                fusion_config=fusion_config,
                tenant_id=tenant_id,
            )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=result.get("output_uri"),
                output_format=result.get("output_format"),
                temporal_alignment_score=result.get("temporal_alignment_score"),
                spatial_calibration_score=result.get("spatial_calibration_score"),
                fusion_quality_score=result.get("fusion_quality_score"),
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_FUSION_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "job_id": str(job.id),
                    "fusion_strategy": fusion_strategy,
                    "output_uri": result.get("output_uri"),
                    "fusion_quality_score": result.get("fusion_quality_score"),
                },
            )

            logger.info(
                "Sensor fusion completed",
                job_id=str(job.id),
                fusion_quality_score=result.get("fusion_quality_score"),
                output_uri=result.get("output_uri"),
            )
            return job

        except Exception as exc:
            logger.error("Sensor fusion failed", job_id=str(job.id), error=str(exc))
            job = await self._job_repo.update(
                job.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise


class MotionPlanningService:
    """Generates collision-free motion planning datasets for robotic manipulators.

    Orchestrates the MotionPlannerProtocol adapter (A*, RRT, B-spline) and
    optionally validates each plan through a PhysicalSafetyValidatorProtocol
    before persisting results. Uses RoboticsJob records for persistence since
    motion planning datasets are a specialisation of robotics synthesis jobs.
    """

    def __init__(
        self,
        motion_planner: MotionPlannerProtocol,
        safety_validator: PhysicalSafetyValidatorProtocol,
        job_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise the motion planning service.

        Args:
            motion_planner: Motion planning algorithm adapter.
            safety_validator: ISO 10218 safety validation adapter.
            job_repository: Repository for RoboticsJob persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._planner = motion_planner
        self._validator = safety_validator
        self._job_repo = job_repository
        self._publisher = event_publisher

    async def plan(
        self,
        tenant_id: uuid.UUID,
        planning_config: dict[str, Any],
        validate_safety: bool = True,
    ) -> RoboticsJob:
        """Generate a motion planning dataset with optional safety validation.

        Creates a RoboticsJob record, runs the motion planner, and if
        validate_safety is True runs ISO 10218 checks on the resulting
        trajectories. Publishes Kafka events on start and completion.

        Args:
            tenant_id: Tenant requesting the planning run.
            planning_config: Configuration forwarded to the motion planner adapter:
                - start_pose: 6-DOF start configuration.
                - goal_pose: 6-DOF goal configuration.
                - algorithm: 'astar' | 'rrt'.
                - num_scenarios: Number of scenarios to generate.
                - velocity_profile: 'trapezoidal' | 'constant'.
                - smooth_trajectory: Apply B-spline smoothing.
                - export_format: 'csv' | 'json'.
            validate_safety: If True, run ISO 10218 validation on the
                generated trajectories and attach safety results.

        Returns:
            RoboticsJob record in COMPLETED or FAILED state.

        Raises:
            ValidationError: If required planning_config keys are missing.
        """
        if "start_pose" not in planning_config or "goal_pose" not in planning_config:
            from aumos_common.errors import ValidationError  # local import to match pattern
            raise ValidationError("planning_config must include 'start_pose' and 'goal_pose'")

        job = await self._job_repo.create(
            RoboticsJob(
                tenant_id=tenant_id,
                status=JobStatus.PENDING,
                sensor_types=["motion_planning"],
                synthesis_config=planning_config,
            )
        )

        await self._publisher.publish(
            Topics.PHYSICAL_AI_SYNTH_STARTED,
            {
                "tenant_id": str(tenant_id),
                "job_id": str(job.id),
                "job_type": "motion_planning",
                "algorithm": planning_config.get("algorithm", "rrt"),
            },
        )

        logger.info(
            "Motion planning job started",
            job_id=str(job.id),
            algorithm=planning_config.get("algorithm", "rrt"),
            num_scenarios=planning_config.get("num_scenarios", 1),
            tenant_id=str(tenant_id),
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.RUNNING)

            plan_result = await self._planner.generate_dataset(
                planning_config=planning_config,
                tenant_id=tenant_id,
            )

            safety_result: dict[str, Any] = {}
            if validate_safety and plan_result.get("scenarios_generated", 0) > 0:
                safety_config = {
                    "trajectory_points": planning_config.get("goal_pose", []),
                    "workspace_boundary": planning_config.get("workspace_boundary", {}),
                    "max_velocity_ms": planning_config.get("max_velocity_ms", 1.5),
                    "max_acceleration_ms2": planning_config.get("max_acceleration_ms2", 3.0),
                    "collision_proximity_m": planning_config.get("collision_proximity_m", 0.05),
                    "obstacles": planning_config.get("obstacle_map", []),
                    "run_iso_check": True,
                }
                safety_result = await self._validator.validate_motion_plan(
                    validation_config=safety_config,
                    tenant_id=tenant_id,
                )
                logger.info(
                    "Motion plan safety validation complete",
                    job_id=str(job.id),
                    overall_safe=safety_result.get("overall_safe"),
                    safety_score=safety_result.get("safety_score"),
                )

            combined_config = {
                **planning_config,
                "safety_validation": safety_result,
            }

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=plan_result.get("output_uri"),
                realism_score=plan_result.get("success_rate"),
                synthesis_config=combined_config,
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_SYNTH_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "job_id": str(job.id),
                    "job_type": "motion_planning",
                    "output_uri": plan_result.get("output_uri"),
                    "success_rate": plan_result.get("success_rate"),
                    "overall_safe": safety_result.get("overall_safe", True),
                },
            )

            logger.info(
                "Motion planning job completed",
                job_id=str(job.id),
                scenarios_generated=plan_result.get("scenarios_generated"),
                success_rate=plan_result.get("success_rate"),
                overall_safe=safety_result.get("overall_safe", True),
            )
            return job

        except Exception as exc:
            logger.error("Motion planning job failed", job_id=str(job.id), error=str(exc))
            job = await self._job_repo.update(
                job.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise

    async def get_job(self, job_id: uuid.UUID, tenant_id: uuid.UUID) -> RoboticsJob:
        """Retrieve a motion planning job by ID.

        Args:
            job_id: Job identifier.
            tenant_id: Tenant context for authorization.

        Returns:
            RoboticsJob record.

        Raises:
            NotFoundError: If the job does not exist for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job or job.tenant_id != tenant_id:
            raise NotFoundError(f"Motion planning job {job_id} not found")
        return job


class GraspingSimulationService:
    """Generates robotic grasping scenario datasets.

    Orchestrates the GraspingSimulatorProtocol adapter to produce grasp pose
    libraries, force-closure evaluations, and epsilon/volume quality metrics
    for training robotic manipulation models. Uses RoboticsJob for persistence.
    """

    def __init__(
        self,
        grasping_simulator: GraspingSimulatorProtocol,
        job_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise the grasping simulation service.

        Args:
            grasping_simulator: Grasping simulation adapter.
            job_repository: Repository for RoboticsJob persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._simulator = grasping_simulator
        self._job_repo = job_repository
        self._publisher = event_publisher

    async def generate_scenarios(
        self,
        tenant_id: uuid.UUID,
        scenario_config: dict[str, Any],
    ) -> RoboticsJob:
        """Generate a robotic grasping scenario dataset.

        Args:
            tenant_id: Tenant requesting the grasping scenarios.
            scenario_config: Grasping configuration forwarded to the adapter:
                - num_scenarios: Number of scenarios to generate.
                - gripper_type: 'parallel_jaw' | 'three_finger' | 'dexterous_hand'
                    | 'suction_cup'.
                - object_types: List of object geometry types to sample.
                - grasp_candidates_per_object: Grasp attempts per object.
                - friction_coefficient: Contact friction coefficient.

        Returns:
            RoboticsJob record in COMPLETED or FAILED state.

        Raises:
            ValidationError: If gripper_type is not supported.
        """
        supported_grippers = {"parallel_jaw", "three_finger", "dexterous_hand", "suction_cup"}
        gripper = scenario_config.get("gripper_type", "parallel_jaw")
        if gripper not in supported_grippers:
            from aumos_common.errors import ValidationError
            raise ValidationError(
                f"Unsupported gripper_type '{gripper}'. Supported: {supported_grippers}"
            )

        job = await self._job_repo.create(
            RoboticsJob(
                tenant_id=tenant_id,
                status=JobStatus.PENDING,
                sensor_types=["grasping"],
                synthesis_config=scenario_config,
            )
        )

        await self._publisher.publish(
            Topics.PHYSICAL_AI_SYNTH_STARTED,
            {
                "tenant_id": str(tenant_id),
                "job_id": str(job.id),
                "job_type": "grasping_simulation",
                "gripper_type": gripper,
            },
        )

        logger.info(
            "Grasping simulation job started",
            job_id=str(job.id),
            gripper_type=gripper,
            num_scenarios=scenario_config.get("num_scenarios", 100),
            tenant_id=str(tenant_id),
        )

        try:
            job = await self._job_repo.update_status(job.id, JobStatus.RUNNING)

            result = await self._simulator.generate_scenarios(
                scenario_config=scenario_config,
                tenant_id=tenant_id,
            )

            job = await self._job_repo.update(
                job.id,
                status=JobStatus.COMPLETED,
                output_uri=result.get("output_uri"),
                realism_score=result.get("success_rate"),
                synthesis_config={
                    **scenario_config,
                    "mean_epsilon_metric": result.get("mean_epsilon_metric"),
                    "mean_volume_metric": result.get("mean_volume_metric"),
                    "gripper_config": result.get("gripper_config"),
                },
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_SYNTH_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "job_id": str(job.id),
                    "job_type": "grasping_simulation",
                    "output_uri": result.get("output_uri"),
                    "success_rate": result.get("success_rate"),
                    "mean_epsilon_metric": result.get("mean_epsilon_metric"),
                },
            )

            logger.info(
                "Grasping simulation job completed",
                job_id=str(job.id),
                scenarios_generated=result.get("scenarios_generated"),
                success_rate=result.get("success_rate"),
                mean_epsilon_metric=result.get("mean_epsilon_metric"),
            )
            return job

        except Exception as exc:
            logger.error("Grasping simulation job failed", job_id=str(job.id), error=str(exc))
            job = await self._job_repo.update(
                job.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise

    async def get_job(self, job_id: uuid.UUID, tenant_id: uuid.UUID) -> RoboticsJob:
        """Retrieve a grasping simulation job by ID.

        Args:
            job_id: Job identifier.
            tenant_id: Tenant context for authorization.

        Returns:
            RoboticsJob record.

        Raises:
            NotFoundError: If the job does not exist for this tenant.
        """
        job = await self._job_repo.get_by_id(job_id)
        if not job or job.tenant_id != tenant_id:
            raise NotFoundError(f"Grasping simulation job {job_id} not found")
        return job


class PhysicsSimulationService:
    """Executes rigid-body physics simulations for robot environment modelling.

    Orchestrates the PhysicsEngineAdapterProtocol to run forward dynamics
    simulations with collision detection, joint constraints, and energy
    tracking. Results are stored in TwinPipeline records because physics
    simulations are a specialisation of digital twin pipelines.
    """

    def __init__(
        self,
        physics_engine: PhysicsEngineAdapterProtocol,
        pipeline_repository: Any,
        event_publisher: EventPublisher,
    ) -> None:
        """Initialise the physics simulation service.

        Args:
            physics_engine: Rigid-body physics engine adapter.
            pipeline_repository: Repository for TwinPipeline persistence.
            event_publisher: Kafka publisher for physical AI lifecycle events.
        """
        self._engine = physics_engine
        self._pipeline_repo = pipeline_repository
        self._publisher = event_publisher

    async def run_simulation(
        self,
        tenant_id: uuid.UUID,
        name: str,
        simulation_config: dict[str, Any],
    ) -> TwinPipeline:
        """Run a rigid-body physics simulation.

        Creates a TwinPipeline record to track the simulation, delegates to
        the physics engine adapter, and publishes lifecycle Kafka events.

        Args:
            tenant_id: Tenant requesting the simulation.
            name: Human-readable simulation name.
            simulation_config: Simulation parameters forwarded to the adapter:
                - bodies: List of rigid body dicts.
                - joints: List of joint constraint dicts.
                - gravity_ms2: Gravitational acceleration.
                - dt_s: Integration timestep in seconds.
                - num_steps: Total simulation steps to run.
                - export_trajectory: Include per-step state history.

        Returns:
            TwinPipeline record in COMPLETED or FAILED state.

        Raises:
            ValidationError: If simulation_config is missing required fields.
        """
        if not simulation_config.get("bodies"):
            from aumos_common.errors import ValidationError
            raise ValidationError("simulation_config must include at least one 'bodies' entry")

        pipeline = await self._pipeline_repo.create(
            TwinPipeline(
                tenant_id=tenant_id,
                name=name,
                status=JobStatus.PENDING,
                scene_config=simulation_config,
            )
        )

        await self._publisher.publish(
            Topics.PHYSICAL_AI_TWIN_CREATED,
            {
                "tenant_id": str(tenant_id),
                "pipeline_id": str(pipeline.id),
                "name": name,
                "type": "physics_simulation",
            },
        )

        logger.info(
            "Physics simulation started",
            pipeline_id=str(pipeline.id),
            num_bodies=len(simulation_config.get("bodies", [])),
            num_steps=simulation_config.get("num_steps", 1000),
            tenant_id=str(tenant_id),
        )

        try:
            pipeline = await self._pipeline_repo.update_status(pipeline.id, JobStatus.RUNNING)

            result = await self._engine.run_simulation(
                simulation_config=simulation_config,
                tenant_id=tenant_id,
            )

            energy_conservation_error: float = result.get("energy_conservation_error", 0.0)

            pipeline = await self._pipeline_repo.update(
                pipeline.id,
                status=JobStatus.COMPLETED,
                output_uri=result.get("output_uri"),
                fidelity_score=max(0.0, 1.0 - energy_conservation_error),
                simulation_steps=result.get("total_steps"),
                real_time_factor=simulation_config.get("dt_s", 0.001),
                scene_config={
                    **simulation_config,
                    "sim_time_s": result.get("sim_time_s"),
                    "collision_events": result.get("collision_events", 0),
                    "energy_conservation_error": energy_conservation_error,
                },
            )

            await self._publisher.publish(
                Topics.PHYSICAL_AI_TWIN_COMPLETED,
                {
                    "tenant_id": str(tenant_id),
                    "pipeline_id": str(pipeline.id),
                    "output_uri": result.get("output_uri"),
                    "total_steps": result.get("total_steps"),
                    "energy_conservation_error": energy_conservation_error,
                },
            )

            logger.info(
                "Physics simulation completed",
                pipeline_id=str(pipeline.id),
                total_steps=result.get("total_steps"),
                sim_time_s=result.get("sim_time_s"),
                energy_conservation_error=energy_conservation_error,
            )
            return pipeline

        except Exception as exc:
            logger.error(
                "Physics simulation failed", pipeline_id=str(pipeline.id), error=str(exc)
            )
            pipeline = await self._pipeline_repo.update(
                pipeline.id, status=JobStatus.FAILED, error_message=str(exc)
            )
            raise

    async def get_simulation(
        self, pipeline_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> TwinPipeline:
        """Retrieve a physics simulation pipeline record by ID.

        Args:
            pipeline_id: Pipeline identifier.
            tenant_id: Tenant context for authorization.

        Returns:
            TwinPipeline record.

        Raises:
            NotFoundError: If the pipeline does not exist for this tenant.
        """
        pipeline = await self._pipeline_repo.get_by_id(pipeline_id)
        if not pipeline or pipeline.tenant_id != tenant_id:
            raise NotFoundError(f"Physics simulation {pipeline_id} not found")
        return pipeline
