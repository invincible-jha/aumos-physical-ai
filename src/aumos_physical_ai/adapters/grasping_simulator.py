"""Robotic grasping scenario simulator adapter.

Generates diverse grasping datasets: grasp pose candidates, object geometry
models, force closure analysis, success probability estimation, grasp quality
metrics (epsilon and volume metrics), multi-finger grasp configurations, and
scenario diversity control.
"""

import math
import random
import uuid
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def _rotation_matrix_from_axis_angle(
    axis: tuple[float, float, float], angle_rad: float
) -> list[list[float]]:
    """Compute a 3x3 rotation matrix from axis-angle representation.

    Args:
        axis: Unit vector (x, y, z).
        angle_rad: Rotation angle in radians.

    Returns:
        3x3 rotation matrix as a nested list.
    """
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    t = 1.0 - c
    return [
        [t * x * x + c,     t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c,     t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def _sample_unit_sphere() -> tuple[float, float, float]:
    """Sample a uniformly distributed point on the unit sphere.

    Returns:
        Normalised direction vector (x, y, z).
    """
    while True:
        x = random.gauss(0.0, 1.0)
        y = random.gauss(0.0, 1.0)
        z = random.gauss(0.0, 1.0)
        norm = math.sqrt(x**2 + y**2 + z**2)
        if norm > 1e-8:
            return (x / norm, y / norm, z / norm)


# ---------------------------------------------------------------------------
# Grasp quality metrics
# ---------------------------------------------------------------------------


def _compute_epsilon_metric(contact_normals: list[tuple[float, float, float]], friction_coeff: float) -> float:
    """Compute the epsilon (largest minimum resisted wrench) quality metric.

    Uses a simplified linear approximation: epsilon is proportional to
    the minimum projection of contact normals onto the average grasp axis
    scaled by friction coefficient.

    Args:
        contact_normals: List of contact point surface normals.
        friction_coeff: Coulomb friction coefficient at each contact.

    Returns:
        Epsilon metric in [0.0, 1.0].
    """
    if not contact_normals:
        return 0.0
    avg_normal = (
        sum(n[0] for n in contact_normals) / len(contact_normals),
        sum(n[1] for n in contact_normals) / len(contact_normals),
        sum(n[2] for n in contact_normals) / len(contact_normals),
    )
    norm = math.sqrt(sum(v**2 for v in avg_normal)) or 1.0
    avg_unit = tuple(v / norm for v in avg_normal)

    projections = [
        abs(n[0] * avg_unit[0] + n[1] * avg_unit[1] + n[2] * avg_unit[2])
        for n in contact_normals
    ]
    epsilon = min(projections) * friction_coeff
    return round(min(epsilon, 1.0), 4)


def _compute_volume_metric(
    contact_positions: list[tuple[float, float, float]],
    object_centroid: tuple[float, float, float],
) -> float:
    """Compute the volume quality metric (fraction of object centroid enclosed).

    Approximated as the ratio of the triangle area formed by the contacts
    relative to the object's bounding-sphere area.

    Args:
        contact_positions: World positions of grasp contact points.
        object_centroid: Centroid of the grasped object.

    Returns:
        Volume metric in [0.0, 1.0].
    """
    if len(contact_positions) < 2:
        return 0.0
    centroid_distances = [
        math.sqrt(
            (cp[0] - object_centroid[0]) ** 2
            + (cp[1] - object_centroid[1]) ** 2
            + (cp[2] - object_centroid[2]) ** 2
        )
        for cp in contact_positions
    ]
    avg_dist = sum(centroid_distances) / len(centroid_distances)
    spread = max(centroid_distances) - min(centroid_distances)
    volume_metric = 1.0 - (spread / max(avg_dist + spread, 1e-6))
    return round(max(0.0, min(volume_metric, 1.0)), 4)


# ---------------------------------------------------------------------------
# GraspingSimulator adapter
# ---------------------------------------------------------------------------


class GraspingSimulator:
    """Robotic grasping scenario dataset generator.

    Produces diverse grasping scenarios for training robot manipulation
    policies: object geometry models, grasp pose sampling, force closure
    analysis, quality metrics, and multi-finger configuration synthesis.

    Implements GraspingSimulatorProtocol.
    """

    # Supported gripper types and their finger counts
    GRIPPER_CONFIGS: dict[str, dict[str, Any]] = {
        "parallel_jaw": {"num_fingers": 2, "max_width_m": 0.08, "max_force_n": 100.0},
        "three_finger": {"num_fingers": 3, "max_width_m": 0.12, "max_force_n": 150.0},
        "dexterous_hand": {"num_fingers": 5, "max_width_m": 0.20, "max_force_n": 50.0},
        "suction_cup": {"num_fingers": 1, "max_width_m": 0.04, "max_force_n": 30.0},
    }

    # Object geometry primitives
    GEOMETRY_PRIMITIVES: list[str] = ["box", "cylinder", "sphere", "cone", "capsule"]

    async def generate_scenarios(
        self,
        scenario_config: dict[str, Any],
        tenant_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Generate a batch of grasping scenarios.

        Args:
            scenario_config: Configuration dict. Supported keys:
                - num_scenarios: Number of scenarios to generate (default 50)
                - gripper_type: 'parallel_jaw' | 'three_finger' | 'dexterous_hand' | 'suction_cup'
                - object_classes: List of object category names (default generic list)
                - friction_coefficient: Contact friction (default 0.5)
                - min_success_probability: Filter threshold (default 0.0, keep all)
                - geometry_type: 'primitive' | 'mesh' (default 'primitive')
                - diversity_control: 'low' | 'medium' | 'high' (default 'medium')
                - num_grasp_candidates: Candidates per object (default 10)
            tenant_id: Tenant context for output namespacing.

        Returns:
            Dict with scenarios list, quality_stats, dataset_summary, output_uri.
        """
        num_scenarios = int(scenario_config.get("num_scenarios", 50))
        gripper_type = scenario_config.get("gripper_type", "parallel_jaw")
        object_classes: list[str] = scenario_config.get(
            "object_classes",
            ["mug", "bottle", "box", "tool", "cylinder", "sphere"],
        )
        friction_coeff = float(scenario_config.get("friction_coefficient", 0.5))
        min_success_prob = float(scenario_config.get("min_success_probability", 0.0))
        geometry_type = scenario_config.get("geometry_type", "primitive")
        diversity_control = scenario_config.get("diversity_control", "medium")
        num_candidates = int(scenario_config.get("num_grasp_candidates", 10))

        gripper_cfg = self.GRIPPER_CONFIGS.get(
            gripper_type, self.GRIPPER_CONFIGS["parallel_jaw"]
        )

        logger.info(
            "Generating grasping scenarios",
            num_scenarios=num_scenarios,
            gripper_type=gripper_type,
            num_candidates=num_candidates,
            tenant_id=str(tenant_id),
        )

        scenarios: list[dict[str, Any]] = []
        total_epsilon = 0.0
        total_volume = 0.0
        total_success = 0.0
        kept_count = 0

        diversity_seed_step = {"low": 10, "medium": 3, "high": 1}.get(diversity_control, 3)

        for scenario_idx in range(num_scenarios):
            random.seed(scenario_idx * diversity_seed_step)

            object_class = random.choice(object_classes)
            object_geometry = self._generate_object_geometry(object_class, geometry_type)
            centroid = tuple(object_geometry["centroid"])

            grasp_candidates: list[dict[str, Any]] = []
            for candidate_idx in range(num_candidates):
                pose = self._sample_grasp_pose(object_geometry, gripper_cfg)
                contact_positions = self._compute_contact_positions(pose, gripper_cfg, object_geometry)
                contact_normals = [_sample_unit_sphere() for _ in contact_positions]

                epsilon = _compute_epsilon_metric(contact_normals, friction_coeff)
                volume_metric = _compute_volume_metric(
                    contact_positions, centroid  # type: ignore[arg-type]
                )
                success_prob = self._estimate_success_probability(
                    epsilon, volume_metric, pose, gripper_cfg
                )

                grasp_candidates.append(
                    {
                        "candidate_id": candidate_idx,
                        "pose": pose,
                        "contact_positions": contact_positions,
                        "contact_normals": contact_normals,
                        "epsilon_metric": epsilon,
                        "volume_metric": volume_metric,
                        "success_probability": success_prob,
                        "force_closure": epsilon > 0.1 and volume_metric > 0.1,
                    }
                )

            # Filter by minimum success probability
            valid_grasps = [
                g for g in grasp_candidates if g["success_probability"] >= min_success_prob
            ]

            if valid_grasps:
                best_grasp = max(valid_grasps, key=lambda g: g["success_probability"])
                avg_eps = sum(g["epsilon_metric"] for g in grasp_candidates) / len(grasp_candidates)
                avg_vol = sum(g["volume_metric"] for g in grasp_candidates) / len(grasp_candidates)
                avg_suc = sum(g["success_probability"] for g in grasp_candidates) / len(grasp_candidates)
                total_epsilon += avg_eps
                total_volume += avg_vol
                total_success += avg_suc
                kept_count += 1

                scenarios.append(
                    {
                        "scenario_id": scenario_idx,
                        "object_class": object_class,
                        "object_geometry": object_geometry,
                        "gripper_type": gripper_type,
                        "grasp_candidates": grasp_candidates,
                        "best_grasp": best_grasp,
                        "valid_grasp_count": len(valid_grasps),
                    }
                )

        quality_stats = {
            "scenarios_generated": num_scenarios,
            "scenarios_with_valid_grasps": kept_count,
            "avg_epsilon_metric": round(total_epsilon / max(kept_count, 1), 4),
            "avg_volume_metric": round(total_volume / max(kept_count, 1), 4),
            "avg_success_probability": round(total_success / max(kept_count, 1), 4),
        }

        logger.info(
            "Grasping scenario generation complete",
            kept_count=kept_count,
            avg_success=quality_stats["avg_success_probability"],
            tenant_id=str(tenant_id),
        )

        return {
            "scenarios": scenarios,
            "quality_stats": quality_stats,
            "dataset_summary": {
                "gripper_type": gripper_type,
                "num_fingers": gripper_cfg["num_fingers"],
                "geometry_type": geometry_type,
                "diversity_control": diversity_control,
                "friction_coefficient": friction_coeff,
            },
            "output_uri": (
                f"s3://aumos-physical-ai/{tenant_id}/grasping/{uuid.uuid4()}.json"
            ),
        }

    def _generate_object_geometry(
        self, object_class: str, geometry_type: str
    ) -> dict[str, Any]:
        """Generate a geometry model for an object.

        Args:
            object_class: Semantic object category name.
            geometry_type: 'primitive' or 'mesh'.

        Returns:
            Dict with shape, dimensions, centroid, mass, surface_area.
        """
        primitive = random.choice(self.GEOMETRY_PRIMITIVES)
        scale = random.uniform(0.03, 0.20)
        centroid = [round(random.uniform(-0.1, 0.1), 4) for _ in range(3)]
        centroid[2] = round(abs(centroid[2]) + scale / 2, 4)

        return {
            "object_class": object_class,
            "shape": primitive if geometry_type == "primitive" else "mesh",
            "dimensions": {
                "width": round(scale, 4),
                "height": round(scale * random.uniform(0.5, 2.5), 4),
                "depth": round(scale, 4),
            },
            "centroid": centroid,
            "mass_kg": round(scale * 5.0 * random.uniform(0.5, 2.0), 4),
            "surface_area_m2": round(6.0 * scale**2, 6),
            "friction_coefficient": round(random.uniform(0.3, 0.8), 3),
        }

    def _sample_grasp_pose(
        self,
        object_geometry: dict[str, Any],
        gripper_cfg: dict[str, Any],
    ) -> dict[str, Any]:
        """Sample a grasp pose (approach direction, orientation, width, depth).

        Args:
            object_geometry: Object geometry model.
            gripper_cfg: Gripper configuration dict.

        Returns:
            Dict with position, approach_vector, rotation_matrix, width_m, depth_m.
        """
        centroid = object_geometry["centroid"]
        approach_dir = _sample_unit_sphere()
        obj_height = object_geometry["dimensions"]["height"]
        approach_dist = obj_height / 2.0 + 0.05

        position = [
            centroid[0] + approach_dir[0] * approach_dist,
            centroid[1] + approach_dir[1] * approach_dist,
            centroid[2] + approach_dir[2] * approach_dist,
        ]

        angle = random.uniform(0.0, 2.0 * math.pi)
        rotation = _rotation_matrix_from_axis_angle(approach_dir, angle)

        obj_width = object_geometry["dimensions"]["width"]
        grasp_width = min(obj_width * random.uniform(1.0, 1.3), gripper_cfg["max_width_m"])
        grasp_depth = min(obj_height * random.uniform(0.3, 0.8), 0.10)

        return {
            "position": [round(p, 6) for p in position],
            "approach_vector": list(approach_dir),
            "rotation_matrix": [[round(v, 6) for v in row] for row in rotation],
            "width_m": round(grasp_width, 4),
            "depth_m": round(grasp_depth, 4),
            "approach_distance_m": round(approach_dist, 4),
        }

    def _compute_contact_positions(
        self,
        pose: dict[str, Any],
        gripper_cfg: dict[str, Any],
        object_geometry: dict[str, Any],
    ) -> list[tuple[float, float, float]]:
        """Compute the world-space positions of gripper finger contacts.

        Args:
            pose: Grasp pose dict.
            gripper_cfg: Gripper configuration dict.
            object_geometry: Object geometry model.

        Returns:
            List of (x, y, z) contact point positions.
        """
        num_fingers = gripper_cfg["num_fingers"]
        centroid = object_geometry["centroid"]
        width = pose["width_m"]
        contact_positions: list[tuple[float, float, float]] = []

        for finger_idx in range(num_fingers):
            angle = 2.0 * math.pi * finger_idx / num_fingers
            offset_x = (width / 2.0) * math.cos(angle)
            offset_y = (width / 2.0) * math.sin(angle)
            contact_positions.append(
                (
                    round(centroid[0] + offset_x, 6),
                    round(centroid[1] + offset_y, 6),
                    round(centroid[2], 6),
                )
            )

        return contact_positions

    def _estimate_success_probability(
        self,
        epsilon: float,
        volume_metric: float,
        pose: dict[str, Any],
        gripper_cfg: dict[str, Any],
    ) -> float:
        """Estimate grasp success probability from quality metrics.

        Args:
            epsilon: Epsilon quality metric.
            volume_metric: Volume quality metric.
            pose: Grasp pose dict.
            gripper_cfg: Gripper configuration dict.

        Returns:
            Success probability in [0.0, 1.0].
        """
        # Penalise grasps near the gripper width limit
        width_utilisation = pose["width_m"] / gripper_cfg["max_width_m"]
        width_penalty = max(0.0, width_utilisation - 0.9) * 5.0

        base_probability = 0.4 * epsilon + 0.4 * volume_metric + 0.2 * (1.0 - width_penalty)
        noise = random.gauss(0.0, 0.03)
        return round(max(0.0, min(base_probability + noise, 1.0)), 4)
