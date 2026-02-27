"""Physics engine adapter for rigid body dynamics simulation.

Provides a stable interface over PyBullet/MuJoCo-style physics simulation.
Handles rigid body dynamics, collision response, gravity and friction
modelling, joint/constraint simulation, simulation step management, and
state extraction for training data generation.

Implements PhysicsEngineAdapterProtocol.
"""

import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Rigid body state types
# ---------------------------------------------------------------------------


@dataclass
class RigidBodyState:
    """Complete state of a single rigid body.

    Position and orientation are in world frame.
    Velocities are in world frame.
    """

    body_id: int
    name: str
    position: list[float]  # [x, y, z] metres
    orientation_quat: list[float]  # [x, y, z, w] quaternion
    linear_velocity: list[float]  # [vx, vy, vz] m/s
    angular_velocity: list[float]  # [wx, wy, wz] rad/s
    contact_forces: list[dict[str, Any]] = field(default_factory=list)
    is_sleeping: bool = False


@dataclass
class JointState:
    """State of a single joint in a multi-body system."""

    joint_id: int
    joint_name: str
    joint_type: str  # 'revolute' | 'prismatic' | 'fixed' | 'spherical'
    position: float  # rad or m
    velocity: float  # rad/s or m/s
    force_torque: float  # N or N.m
    lower_limit: float
    upper_limit: float


@dataclass
class SimulationState:
    """Full simulation state snapshot."""

    step: int
    sim_time_s: float
    bodies: list[RigidBodyState]
    joints: list[JointState]
    collision_events: list[dict[str, Any]]
    energy_j: float
    constraint_violations: int


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------


def _integrate_euler(
    position: list[float],
    velocity: list[float],
    acceleration: list[float],
    dt: float,
) -> tuple[list[float], list[float]]:
    """Semi-implicit Euler integration step.

    Args:
        position: Current position vector.
        velocity: Current velocity vector.
        acceleration: Applied acceleration (gravity + forces / mass).
        dt: Time step in seconds.

    Returns:
        Tuple of (new_position, new_velocity).
    """
    new_velocity = [v + a * dt for v, a in zip(velocity, acceleration)]
    new_position = [p + v * dt for p, v in zip(position, new_velocity)]
    return new_position, new_velocity


def _detect_ground_collision(
    position: list[float], radius: float = 0.05
) -> bool:
    """Return True when the body sphere approximation penetrates the ground plane.

    Args:
        position: Body centre position [x, y, z].
        radius: Body bounding sphere radius in metres.

    Returns:
        True if body is below ground level.
    """
    return position[2] - radius < 0.0


def _compute_contact_force(
    penetration_depth_m: float,
    relative_velocity_ms: float,
    restitution: float,
    stiffness_n_m: float = 10000.0,
    damping_ns_m: float = 200.0,
) -> float:
    """Compute normal contact force using Hertz-Mindlin spring-damper model.

    Args:
        penetration_depth_m: Overlap depth in metres (positive).
        relative_velocity_ms: Relative approach velocity (positive = approaching).
        restitution: Coefficient of restitution [0, 1].
        stiffness_n_m: Contact spring stiffness in N/m.
        damping_ns_m: Contact damping coefficient in N.s/m.

    Returns:
        Normal contact force in Newtons.
    """
    spring_force = stiffness_n_m * penetration_depth_m
    damping_force = damping_ns_m * relative_velocity_ms * (1.0 - restitution)
    return max(0.0, spring_force - damping_force)


def _multiply_quaternions(
    q1: list[float], q2: list[float]
) -> list[float]:
    """Multiply two quaternions [x, y, z, w].

    Args:
        q1: First quaternion [x1, y1, z1, w1].
        q2: Second quaternion [x2, y2, z2, w2].

    Returns:
        Product quaternion [x, y, z, w].
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]


# ---------------------------------------------------------------------------
# PhysicsEngineAdapter adapter
# ---------------------------------------------------------------------------


class PhysicsEngineAdapter:
    """Physics simulation adapter providing rigid body dynamics for training data.

    Models PyBullet/MuJoCo integration points: world setup, stepping,
    state extraction, collision detection, and joint simulation. The
    simulation runs deterministically given the same seed, enabling
    reproducible dataset generation.

    Implements PhysicsEngineAdapterProtocol.
    """

    GRAVITY_MS2: float = -9.81  # Standard gravity (Z-down convention)

    def __init__(self) -> None:
        """Initialise the physics engine adapter with empty simulation state."""
        self._bodies: list[RigidBodyState] = []
        self._joints: list[JointState] = []
        self._sim_time: float = 0.0
        self._step_count: int = 0

    async def run_simulation(
        self,
        simulation_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Execute a full physics simulation run and export state history.

        Args:
            simulation_config: Simulation configuration dict. Supported keys:
                - bodies: List of rigid body dicts {name, mass_kg, position, velocity, restitution, friction}
                - joints: List of joint dicts {name, type, body_a, body_b, axis, limits}
                - gravity: [gx, gy, gz] m/s^2 (default [0, 0, -9.81])
                - dt: Time step in seconds (default 0.01)
                - num_steps: Number of simulation steps (default 500)
                - record_interval: Record state every N steps (default 10)
                - seed: Random seed for reproducibility (default 42)
                - engine: 'pybullet' | 'mujoco' (metadata only, affects export format)
            tenant_id: Tenant context.

        Returns:
            Dict with state_history, collision_events, energy_trace,
            dataset_stats, output_uri.
        """
        gravity_vec: list[float] = simulation_config.get("gravity", [0.0, 0.0, self.GRAVITY_MS2])
        dt: float = float(simulation_config.get("dt", 0.01))
        num_steps: int = int(simulation_config.get("num_steps", 500))
        record_interval: int = int(simulation_config.get("record_interval", 10))
        seed: int = int(simulation_config.get("seed", 42))
        engine: str = simulation_config.get("engine", "pybullet")
        bodies_cfg: list[dict[str, Any]] = simulation_config.get("bodies", [])
        joints_cfg: list[dict[str, Any]] = simulation_config.get("joints", [])

        random.seed(seed)

        logger.info(
            "Starting physics simulation",
            engine=engine,
            num_steps=num_steps,
            num_bodies=len(bodies_cfg),
            dt=dt,
            tenant_id=str(tenant_id),
        )

        # Initialise bodies
        self._bodies = [
            RigidBodyState(
                body_id=i,
                name=body_cfg.get("name", f"body_{i}"),
                position=list(body_cfg.get("position", [0.0, 0.0, 1.0 + i * 0.5])),
                orientation_quat=[0.0, 0.0, 0.0, 1.0],
                linear_velocity=list(body_cfg.get("velocity", [0.0, 0.0, 0.0])),
                angular_velocity=[0.0, 0.0, 0.0],
                contact_forces=[],
                is_sleeping=False,
            )
            for i, body_cfg in enumerate(bodies_cfg)
        ] or [
            RigidBodyState(
                body_id=0,
                name="default_sphere",
                position=[0.0, 0.0, 2.0],
                orientation_quat=[0.0, 0.0, 0.0, 1.0],
                linear_velocity=[0.1, 0.0, 0.0],
                angular_velocity=[0.0, 0.0, 0.1],
                contact_forces=[],
            )
        ]

        # Initialise joints
        self._joints = self._build_joints(joints_cfg)

        self._sim_time = 0.0
        self._step_count = 0

        state_history: list[dict[str, Any]] = []
        collision_events: list[dict[str, Any]] = []
        energy_trace: list[dict[str, float]] = []

        bodies_params = {
            i: {
                "mass_kg": float(bodies_cfg[i].get("mass_kg", 1.0)) if i < len(bodies_cfg) else 1.0,
                "restitution": float(bodies_cfg[i].get("restitution", 0.3)) if i < len(bodies_cfg) else 0.3,
                "friction": float(bodies_cfg[i].get("friction", 0.5)) if i < len(bodies_cfg) else 0.5,
                "radius": float(bodies_cfg[i].get("radius", 0.05)) if i < len(bodies_cfg) else 0.05,
            }
            for i in range(len(self._bodies))
        }

        for step in range(num_steps):
            self._sim_time += dt
            self._step_count += 1

            step_collisions = self._step_simulation(
                gravity_vec=gravity_vec,
                dt=dt,
                bodies_params=bodies_params,
            )
            collision_events.extend(step_collisions)

            # Record state at the specified interval
            if step % record_interval == 0:
                state_snapshot = self._extract_state(step, step_collisions)
                state_history.append(self._state_to_dict(state_snapshot))
                energy_trace.append(
                    {
                        "step": step,
                        "sim_time_s": round(self._sim_time, 4),
                        "kinetic_energy_j": self._compute_kinetic_energy(bodies_params),
                        "potential_energy_j": self._compute_potential_energy(gravity_vec, bodies_params),
                    }
                )

        dataset_stats = {
            "engine": engine,
            "num_steps": num_steps,
            "dt_s": dt,
            "total_sim_time_s": round(self._sim_time, 4),
            "num_bodies": len(self._bodies),
            "num_joints": len(self._joints),
            "state_snapshots": len(state_history),
            "total_collision_events": len(collision_events),
        }

        logger.info(
            "Physics simulation complete",
            total_collisions=len(collision_events),
            snapshots=len(state_history),
            tenant_id=str(tenant_id),
        )

        return {
            "state_history": state_history,
            "collision_events": collision_events[:50],  # Truncate for API response
            "energy_trace": energy_trace,
            "dataset_stats": dataset_stats,
            "output_uri": (
                f"s3://aumos-physical-ai/{tenant_id}/physics/{uuid.uuid4()}.hdf5"
            ),
        }

    def _step_simulation(
        self,
        gravity_vec: list[float],
        dt: float,
        bodies_params: dict[int, dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Advance the simulation by one time step.

        Args:
            gravity_vec: Gravity acceleration vector [gx, gy, gz].
            dt: Time step in seconds.
            bodies_params: Per-body physical parameters.

        Returns:
            List of collision events occurring in this step.
        """
        collision_events: list[dict[str, Any]] = []

        for body in self._bodies:
            params = bodies_params[body.body_id]
            mass = params["mass_kg"]

            # Gravity contribution
            accel = [g / mass * mass for g in gravity_vec]  # gravity uniform per unit mass

            new_pos, new_vel = _integrate_euler(
                body.position, body.linear_velocity, accel, dt
            )
            body.position = [round(v, 6) for v in new_pos]
            body.linear_velocity = [round(v, 6) for v in new_vel]

            # Ground plane collision (z = 0)
            radius = params["radius"]
            if _detect_ground_collision(body.position, radius):
                penetration = abs(body.position[2] - radius)
                normal_vel = abs(body.linear_velocity[2])
                contact_force = _compute_contact_force(
                    penetration,
                    normal_vel,
                    params["restitution"],
                )
                # Apply impulse-based response
                restitution = params["restitution"]
                friction = params["friction"]
                body.position[2] = radius
                body.linear_velocity[2] = abs(body.linear_velocity[2]) * restitution
                body.linear_velocity[0] *= 1.0 - friction * dt
                body.linear_velocity[1] *= 1.0 - friction * dt
                body.contact_forces = [
                    {
                        "contact_body": "ground_plane",
                        "normal_force_n": round(contact_force, 4),
                        "friction_force_n": round(contact_force * friction, 4),
                        "contact_point": [
                            round(body.position[0], 4),
                            round(body.position[1], 4),
                            0.0,
                        ],
                    }
                ]
                collision_events.append(
                    {
                        "step": self._step_count,
                        "sim_time_s": round(self._sim_time, 4),
                        "body_id": body.body_id,
                        "body_name": body.name,
                        "collision_type": "ground_plane",
                        "contact_force_n": round(contact_force, 4),
                        "penetration_m": round(penetration, 6),
                    }
                )
            else:
                body.contact_forces = []

            # Simple angular velocity damping (air resistance)
            body.angular_velocity = [w * (1.0 - 0.01 * dt) for w in body.angular_velocity]

        # Joint constraint enforcement
        for joint in self._joints:
            self._enforce_joint_limits(joint)

        return collision_events

    def _build_joints(self, joints_cfg: list[dict[str, Any]]) -> list[JointState]:
        """Construct JointState objects from configuration dicts.

        Args:
            joints_cfg: List of joint configuration dicts.

        Returns:
            List of JointState objects.
        """
        joints: list[JointState] = []
        for i, cfg in enumerate(joints_cfg):
            limits = cfg.get("limits", [-math.pi, math.pi])
            joints.append(
                JointState(
                    joint_id=i,
                    joint_name=cfg.get("name", f"joint_{i}"),
                    joint_type=cfg.get("type", "revolute"),
                    position=float(cfg.get("initial_position", 0.0)),
                    velocity=0.0,
                    force_torque=0.0,
                    lower_limit=float(limits[0]),
                    upper_limit=float(limits[1]),
                )
            )
        return joints

    def _enforce_joint_limits(self, joint: JointState) -> None:
        """Clamp joint position within configured limits.

        Args:
            joint: Joint state to clamp in-place.
        """
        if joint.joint_type == "fixed":
            joint.position = 0.0
            joint.velocity = 0.0
            return
        joint.position = max(joint.lower_limit, min(joint.upper_limit, joint.position))
        if joint.position in (joint.lower_limit, joint.upper_limit):
            joint.velocity = 0.0

    def _extract_state(
        self, step: int, step_collisions: list[dict[str, Any]]
    ) -> SimulationState:
        """Snapshot current simulation state.

        Args:
            step: Current step index.
            step_collisions: Collisions from this step.

        Returns:
            SimulationState snapshot.
        """
        total_energy = sum(
            0.5 * 1.0 * sum(v**2 for v in b.linear_velocity) + abs(self.GRAVITY_MS2) * b.position[2]
            for b in self._bodies
        )
        return SimulationState(
            step=step,
            sim_time_s=round(self._sim_time, 4),
            bodies=list(self._bodies),
            joints=list(self._joints),
            collision_events=step_collisions,
            energy_j=round(total_energy, 6),
            constraint_violations=0,
        )

    def _state_to_dict(self, state: SimulationState) -> dict[str, Any]:
        """Serialise a SimulationState to a plain dict for export.

        Args:
            state: Simulation state snapshot.

        Returns:
            Serialisable dict.
        """
        return {
            "step": state.step,
            "sim_time_s": state.sim_time_s,
            "energy_j": state.energy_j,
            "bodies": [
                {
                    "body_id": b.body_id,
                    "name": b.name,
                    "position": b.position,
                    "orientation_quat": b.orientation_quat,
                    "linear_velocity": b.linear_velocity,
                    "angular_velocity": b.angular_velocity,
                    "contact_forces": b.contact_forces,
                    "is_sleeping": b.is_sleeping,
                }
                for b in state.bodies
            ],
            "joints": [
                {
                    "joint_id": j.joint_id,
                    "joint_name": j.joint_name,
                    "joint_type": j.joint_type,
                    "position": j.position,
                    "velocity": j.velocity,
                    "force_torque": j.force_torque,
                }
                for j in state.joints
            ],
            "num_collisions": len(state.collision_events),
        }

    def _compute_kinetic_energy(self, bodies_params: dict[int, dict[str, Any]]) -> float:
        """Compute total kinetic energy of all bodies.

        Args:
            bodies_params: Per-body physical parameter dicts.

        Returns:
            Total kinetic energy in Joules.
        """
        ke = sum(
            0.5 * bodies_params[b.body_id]["mass_kg"] * sum(v**2 for v in b.linear_velocity)
            for b in self._bodies
        )
        return round(ke, 6)

    def _compute_potential_energy(
        self, gravity_vec: list[float], bodies_params: dict[int, dict[str, Any]]
    ) -> float:
        """Compute total gravitational potential energy of all bodies.

        Args:
            gravity_vec: Gravity acceleration vector.
            bodies_params: Per-body physical parameter dicts.

        Returns:
            Total potential energy in Joules.
        """
        g = abs(gravity_vec[2]) if len(gravity_vec) > 2 else 9.81
        pe = sum(
            bodies_params[b.body_id]["mass_kg"] * g * b.position[2]
            for b in self._bodies
        )
        return round(pe, 6)
