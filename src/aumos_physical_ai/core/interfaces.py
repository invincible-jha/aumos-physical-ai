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


# ---------------------------------------------------------------------------
# New adapter protocols for motion planning, grasping, safety, and physics
# ---------------------------------------------------------------------------


@runtime_checkable
class MotionPlannerProtocol(Protocol):
    """Contract for robotic motion planning adapters.

    Implementations generate collision-free joint-space and task-space
    trajectories for robotic manipulators using algorithms such as A*,
    RRT, and B-spline smoothing.
    """

    async def generate_dataset(
        self,
        planning_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate a motion-planning dataset from the supplied configuration.

        Args:
            planning_config: Planning parameters including:
                - start_pose: List[float] — 6-DOF start configuration.
                - goal_pose: List[float] — 6-DOF goal configuration.
                - algorithm: 'astar' | 'rrt' — path planning algorithm.
                - obstacle_map: List of obstacle dicts (center, half_extents).
                - velocity_profile: 'trapezoidal' | 'constant'.
                - smooth_trajectory: bool — apply B-spline smoothing.
                - num_scenarios: int — number of planning scenarios.
                - export_format: 'csv' | 'json'.
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with output_uri, scenarios_generated, success_rate,
            mean_path_length, mean_planning_time_ms, export_format.
        """
        ...


@runtime_checkable
class GraspingSimulatorProtocol(Protocol):
    """Contract for robotic grasping simulation adapters.

    Implementations generate diverse grasp poses, evaluate force closure,
    and compute grasp quality metrics (epsilon and volume) for training
    robotic manipulation models.
    """

    async def generate_scenarios(
        self,
        scenario_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate grasping simulation scenarios.

        Args:
            scenario_config: Scenario configuration including:
                - num_scenarios: int — number of grasping scenarios.
                - gripper_type: 'parallel_jaw' | 'three_finger' | 'dexterous_hand'
                    | 'suction_cup'.
                - object_types: List[str] — object geometry types to sample.
                - grasp_candidates_per_object: int — grasp attempts per object.
                - friction_coefficient: float — contact friction coefficient.
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with output_uri, scenarios_generated, success_rate,
            mean_epsilon_metric, mean_volume_metric, gripper_config.
        """
        ...


@runtime_checkable
class PhysicalSafetyValidatorProtocol(Protocol):
    """Contract for ISO 10218 robotic safety validation adapters.

    Implementations check motion plans against workspace boundaries,
    velocity limits, collision proximity thresholds, and emergency stop
    scenarios per ISO 10218-1:2011 and ISO 10218-2:2011.
    """

    async def validate_motion_plan(
        self,
        validation_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Validate a motion plan against physical safety requirements.

        Args:
            validation_config: Validation parameters including:
                - trajectory_points: List of 6-DOF joint configurations.
                - workspace_boundary: Dict with min/max bounds per axis (m).
                - max_velocity_ms: float — maximum allowable end-effector speed.
                - max_acceleration_ms2: float — maximum allowable acceleration.
                - collision_proximity_m: float — minimum clearance from obstacles.
                - obstacles: List of obstacle dicts (center, half_extents).
                - run_iso_check: bool — run ISO 10218 clause checks.
            tenant_id: Tenant context for audit logging.

        Returns:
            Dict with overall_safe, safety_score, tests (list of test results),
            iso_compliance, findings, recommendations.
        """
        ...


@runtime_checkable
class PhysicsEngineAdapterProtocol(Protocol):
    """Contract for rigid-body physics simulation adapters.

    Implementations execute forward dynamics simulations for articulated
    rigid bodies, detecting ground plane collisions, enforcing joint limits,
    and tracking kinetic and potential energy conservation.
    """

    async def run_simulation(
        self,
        simulation_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Run a rigid-body physics simulation.

        Args:
            simulation_config: Simulation parameters including:
                - bodies: List of rigid body dicts (name, mass_kg, position,
                    velocity, inertia_tensor, shape, dimensions).
                - joints: List of joint dicts (body_a, body_b, type,
                    axis, limits_rad, stiffness, damping).
                - gravity_ms2: float — gravitational acceleration (default 9.81).
                - dt_s: float — integration timestep in seconds.
                - num_steps: int — total simulation steps to execute.
                - export_trajectory: bool — include per-step state in output.
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with output_uri, total_steps, sim_time_s, final_state,
            energy_conservation_error, collision_events, joints_enforced.
        """
        ...
